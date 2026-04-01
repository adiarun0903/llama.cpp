#include "ggml-turboq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace llama_turboq {

namespace {

static constexpr float turboq_codebook_2bit[] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

static constexpr float turboq_codebook_3bit[] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};

static constexpr float turboq_codebook_4bit[] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static inline uint64_t splitmix64(uint64_t & state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint32_t next_power_of_two(uint32_t value) {
    uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static void hadamard_inplace(float * data, uint32_t n) {
    for (uint32_t step = 1; step < n; step <<= 1) {
        const uint32_t jump = step << 1;
        for (uint32_t i = 0; i < n; i += jump) {
            for (uint32_t j = 0; j < step; ++j) {
                const float a = data[i + j];
                const float b = data[i + j + step];
                data[i + j] = a + b;
                data[i + j + step] = a - b;
            }
        }
    }
}

} // namespace

const codebook & get_codebook(uint32_t bits) {
    static constexpr codebook codebooks[] = {
        { 2, turboq_codebook_2bit },
        { 3, turboq_codebook_3bit },
        { 4, turboq_codebook_4bit },
    };

    for (const auto & codebook : codebooks) {
        if (codebook.bits == bits) {
            return codebook;
        }
    }

    throw std::runtime_error("unsupported TurboQ bitwidth");
}

uint32_t salt_for_surface_kind(surface_kind kind) {
    switch (kind) {
        case surface_kind::attn_k:      return 0x4b;
        case surface_kind::attn_v:      return 0x56;
        case surface_kind::recurrent_r: return 0x52;
        case surface_kind::recurrent_s: return 0x53;
    }

    throw std::runtime_error("unsupported TurboQ surface kind");
}

rotation_plan make_rotation_plan(uint32_t seed, surface_kind kind, uint32_t layer, uint32_t dim) {
    return make_rotation_plan(seed, layer, dim, salt_for_surface_kind(kind));
}

rotation_plan make_rotation_plan(uint32_t seed, uint32_t layer, uint32_t dim, uint32_t salt) {
    rotation_plan plan;
    plan.dim = dim;
    plan.padded_dim = next_power_of_two(dim);
    plan.perm.resize(dim);
    plan.inv_perm.resize(dim);
    plan.sign.resize(dim);

    std::iota(plan.perm.begin(), plan.perm.end(), 0u);

    uint64_t state = (uint64_t(seed) << 32) ^ (uint64_t(layer) << 8) ^ uint64_t(salt);

    for (size_t i = plan.perm.size(); i > 1; --i) {
        const size_t j = splitmix64(state) % i;
        std::swap(plan.perm[i - 1], plan.perm[j]);
    }

    for (uint32_t i = 0; i < dim; ++i) {
        plan.inv_perm[plan.perm[i]] = i;
        plan.sign[i] = (splitmix64(state) & 1ULL) ? 1 : -1;
    }

    return plan;
}

size_t div_round_up(size_t num, size_t den) {
    return (num + den - 1) / den;
}

size_t bitpacked_bytes(size_t count, size_t bits) {
    return div_round_up(count * bits, size_t(8));
}

void pack_bits(uint8_t * dst, size_t index, uint32_t bits, uint32_t value) {
    const size_t bit_index = index * bits;
    const size_t byte_index = bit_index / 8;
    const uint32_t bit_offset = bit_index % 8;
    const uint32_t mask = (1u << bits) - 1u;

    uint32_t wide = uint32_t(dst[byte_index]);
    if (bit_offset + bits > 8) {
        wide |= uint32_t(dst[byte_index + 1]) << 8;
    }

    wide &= ~(mask << bit_offset);
    wide |= (value & mask) << bit_offset;

    dst[byte_index] = uint8_t(wide & 0xffu);
    if (bit_offset + bits > 8) {
        dst[byte_index + 1] = uint8_t((wide >> 8) & 0xffu);
    }
}

uint32_t unpack_bits(const uint8_t * src, size_t index, uint32_t bits) {
    const size_t bit_index = index * bits;
    const size_t byte_index = bit_index / 8;
    const uint32_t bit_offset = bit_index % 8;
    const uint32_t mask = (1u << bits) - 1u;

    uint32_t wide = uint32_t(src[byte_index]);
    if (bit_offset + bits > 8) {
        wide |= uint32_t(src[byte_index + 1]) << 8;
    }

    return (wide >> bit_offset) & mask;
}

void set_sign_bit(uint8_t * dst, size_t index, bool value) {
    const size_t byte_index = index / 8;
    const uint32_t bit = 1u << (index % 8);
    if (value) {
        dst[byte_index] |= uint8_t(bit);
    } else {
        dst[byte_index] &= ~uint8_t(bit);
    }
}

float get_sign_value(const uint8_t * src, size_t index) {
    const size_t byte_index = index / 8;
    const uint32_t bit = 1u << (index % 8);
    return (src[byte_index] & bit) ? 1.0f : -1.0f;
}

void rotate_vector(const rotation_plan & plan, const float * src, std::vector<float> & dst) {
    dst.assign(plan.padded_dim, 0.0f);

    for (uint32_t i = 0; i < plan.dim; ++i) {
        dst[i] = float(plan.sign[i]) * src[plan.perm[i]];
    }

    hadamard_inplace(dst.data(), plan.padded_dim);

    const float scale = 1.0f / std::sqrt(float(plan.padded_dim));
    for (float & value : dst) {
        value *= scale;
    }
}

void inverse_rotate_vector(const rotation_plan & plan, std::vector<float> & rotated, float * dst) {
    hadamard_inplace(rotated.data(), plan.padded_dim);

    const float scale = 1.0f / std::sqrt(float(plan.padded_dim));
    for (float & value : rotated) {
        value *= scale;
    }

    for (uint32_t out = 0; out < plan.dim; ++out) {
        const uint32_t src_index = plan.inv_perm[out];
        dst[out] = float(plan.sign[src_index]) * rotated[src_index];
    }
}

uint32_t quantize_scalar(const codebook & codebook, float value) {
    const uint32_t n_codes = 1u << codebook.bits;
    uint32_t best = 0;
    float best_err = std::numeric_limits<float>::infinity();

    for (uint32_t i = 0; i < n_codes; ++i) {
        const float err = std::fabs(value - codebook.values[i]);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }

    return best;
}

void encode_stage1_codes(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst) {
    std::vector<float> rotated;
    rotate_vector(plan, src, rotated);

    std::memset(code_dst, 0, bitpacked_bytes(plan.padded_dim, codebook.bits));
    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        pack_bits(code_dst, i, codebook.bits, quantize_scalar(codebook, rotated[i]));
    }
}

void decode_stage1_codes(
        const rotation_plan & plan,
        const codebook & codebook,
        const uint8_t * code_src,
        float * dst) {
    std::vector<float> rotated(plan.padded_dim);
    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        rotated[i] = codebook.values[unpack_bits(code_src, i, codebook.bits)];
    }
    inverse_rotate_vector(plan, rotated, dst);
}

void encode_stage1_normed(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst,
        ggml_fp16_t * norm_dst) {
    std::vector<float> rotated;
    rotate_vector(plan, src, rotated);

    std::memset(code_dst, 0, bitpacked_bytes(plan.padded_dim, codebook.bits));

    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        norm_sq += rotated[i] * rotated[i];
    }

    const float norm = std::sqrt(norm_sq / float(std::max(1u, plan.padded_dim)));
    *norm_dst = ggml_fp32_to_fp16(norm);

    if (norm <= 0.0f) {
        return;
    }

    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        pack_bits(code_dst, i, codebook.bits, quantize_scalar(codebook, rotated[i] / norm));
    }
}

void decode_stage1_normed(
        const rotation_plan & plan,
        const codebook & codebook,
        const uint8_t * code_src,
        ggml_fp16_t norm,
        float * dst) {
    std::vector<float> rotated(plan.padded_dim);
    const float scale = ggml_fp16_to_fp32(norm);
    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        rotated[i] = codebook.values[unpack_bits(code_src, i, codebook.bits)] * scale;
    }
    inverse_rotate_vector(plan, rotated, dst);
}

void encode_stage1_residual(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst,
        uint8_t * sign_dst,
        ggml_fp16_t * norm_dst) {
    std::vector<float> rotated;
    rotate_vector(plan, src, rotated);

    std::memset(code_dst, 0, bitpacked_bytes(plan.padded_dim, codebook.bits));
    if (sign_dst != nullptr) {
        std::memset(sign_dst, 0, bitpacked_bytes(plan.padded_dim, 1));
    }

    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
        const uint32_t code = quantize_scalar(codebook, rotated[i]);
        pack_bits(code_dst, i, codebook.bits, code);

        if (sign_dst != nullptr && norm_dst != nullptr) {
            const float residual = rotated[i] - codebook.values[code];
            set_sign_bit(sign_dst, i, residual >= 0.0f);
            norm_sq += residual * residual;
        }
    }

    if (norm_dst != nullptr) {
        *norm_dst = ggml_fp32_to_fp16(std::sqrt(norm_sq));
    }
}

} // namespace llama_turboq
