#include "../src/llama-turboq-codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#undef NDEBUG
#include <cassert>

static bool rotation_plan_equal(const llama_turboq::rotation_plan & lhs, const llama_turboq::rotation_plan & rhs) {
    return lhs.dim == rhs.dim &&
           lhs.padded_dim == rhs.padded_dim &&
           lhs.perm == rhs.perm &&
           lhs.inv_perm == rhs.inv_perm &&
           lhs.sign == rhs.sign;
}

static std::vector<float> make_signal(uint32_t dim, float scale = 1.0f) {
    std::vector<float> values(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        values[i] = scale * (
                std::sin(0.17f * float(i + 1)) +
                0.5f * std::cos(0.11f * float(i + 3)) +
                0.1f * float((int(i) % 7) - 3));
    }
    return values;
}

static float rms(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += double(value) * double(value);
    }
    return float(std::sqrt(sum / std::max<size_t>(size_t(1), values.size())));
}

static float rel_rms_error(const std::vector<float> & ref, const std::vector<float> & got) {
    assert(ref.size() == got.size());

    double err_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = double(ref[i]) - double(got[i]);
        err_sq += diff * diff;
    }

    const float denom = std::max(1.0e-6f, rms(ref));
    return float(std::sqrt(err_sq / std::max<size_t>(size_t(1), ref.size()))) / denom;
}

int main() {
    {
        const auto plan_a = llama_turboq::make_rotation_plan(7, 3, 96, 0x52);
        const auto plan_b = llama_turboq::make_rotation_plan(7, 3, 96, 0x52);
        const auto plan_c = llama_turboq::make_rotation_plan(8, 3, 96, 0x52);

        assert(rotation_plan_equal(plan_a, plan_b));
        assert(!rotation_plan_equal(plan_a, plan_c));

        std::vector<uint32_t> sorted = plan_a.perm;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < plan_a.dim; ++i) {
            assert(sorted[i] == i);
            assert(plan_a.inv_perm[plan_a.perm[i]] == i);
            assert(plan_a.sign[i] == -1 || plan_a.sign[i] == 1);
        }
    }

    {
        struct surface_case {
            llama_turboq::surface_kind kind;
            uint32_t expected_salt;
        };

        const std::array<surface_case, 4> cases = {{
            { llama_turboq::surface_kind::attn_k,      0x4b },
            { llama_turboq::surface_kind::attn_v,      0x56 },
            { llama_turboq::surface_kind::recurrent_r, 0x52 },
            { llama_turboq::surface_kind::recurrent_s, 0x53 },
        }};

        for (const auto & tc : cases) {
            assert(llama_turboq::salt_for_surface_kind(tc.kind) == tc.expected_salt);

            const auto via_kind = llama_turboq::make_rotation_plan(23, tc.kind, 11, 80);
            const auto via_salt = llama_turboq::make_rotation_plan(23, 11, 80, tc.expected_salt);
            assert(rotation_plan_equal(via_kind, via_salt));
        }

        const auto attn_k = llama_turboq::make_rotation_plan(23, llama_turboq::surface_kind::attn_k, 11, 80);
        const auto attn_v = llama_turboq::make_rotation_plan(23, llama_turboq::surface_kind::attn_v, 11, 80);
        assert(!rotation_plan_equal(attn_k, attn_v));
    }

    for (uint32_t bits : { 2u, 3u, 4u }) {
        const size_t count = 23;
        std::vector<uint8_t> packed(llama_turboq::bitpacked_bytes(count, bits), 0);
        const uint32_t mask = (1u << bits) - 1u;

        for (size_t i = 0; i < count; ++i) {
            const uint32_t value = uint32_t((i * 5 + 3) & mask);
            llama_turboq::pack_bits(packed.data(), i, bits, value);
        }

        for (size_t i = 0; i < count; ++i) {
            const uint32_t value = uint32_t((i * 5 + 3) & mask);
            assert(llama_turboq::unpack_bits(packed.data(), i, bits) == value);
        }
    }

    {
        const size_t count = 19;
        std::vector<uint8_t> signs(llama_turboq::bitpacked_bytes(count, 1), 0);
        for (size_t i = 0; i < count; ++i) {
            llama_turboq::set_sign_bit(signs.data(), i, (i % 3) != 0);
        }
        for (size_t i = 0; i < count; ++i) {
            const float expected = (i % 3) != 0 ? 1.0f : -1.0f;
            assert(llama_turboq::get_sign_value(signs.data(), i) == expected);
        }
    }

    const auto plan = llama_turboq::make_rotation_plan(13, 5, 96, 0x52);
    const auto signal = make_signal(plan.dim, 1.0f);

    {
        std::vector<float> decoded(plan.dim, 0.0f);
        std::vector<uint8_t> codes(llama_turboq::bitpacked_bytes(plan.padded_dim, 4), 0);
        ggml_fp16_t norm = ggml_fp32_to_fp16(0.0f);

        std::vector<float> zeros(plan.dim, 0.0f);
        llama_turboq::encode_stage1_normed(plan, llama_turboq::get_codebook(4), zeros.data(), codes.data(), &norm);
        assert(ggml_fp16_to_fp32(norm) == 0.0f);

        llama_turboq::decode_stage1_normed(plan, llama_turboq::get_codebook(4), codes.data(), norm, decoded.data());
        for (float value : decoded) {
            assert(value == 0.0f);
        }
    }

    std::array<float, 3> rel_errors = { 0.0f, 0.0f, 0.0f };
    size_t rel_index = 0;

    for (uint32_t bits : { 2u, 3u, 4u }) {
        const auto & codebook = llama_turboq::get_codebook(bits);
        std::vector<uint8_t> codes(llama_turboq::bitpacked_bytes(plan.padded_dim, bits), 0);
        std::vector<float> decoded(plan.dim, 0.0f);
        ggml_fp16_t norm = ggml_fp32_to_fp16(0.0f);

        llama_turboq::encode_stage1_normed(plan, codebook, signal.data(), codes.data(), &norm);
        assert(ggml_fp16_to_fp32(norm) > 0.0f);

        llama_turboq::decode_stage1_normed(plan, codebook, codes.data(), norm, decoded.data());

        const float rel = rel_rms_error(signal, decoded);
        rel_errors[rel_index++] = rel;

        const float upper_bound = bits == 2 ? 1.35f : (bits == 3 ? 1.00f : 0.80f);
        assert(std::isfinite(rel));
        assert(rel < upper_bound);
    }

    assert(rel_errors[2] <= rel_errors[1] + 1.0e-4f);
    assert(rel_errors[1] <= rel_errors[0] + 1.0e-4f);

    {
        const auto & codebook = llama_turboq::get_codebook(3);
        std::vector<uint8_t> codes(llama_turboq::bitpacked_bytes(plan.padded_dim, codebook.bits), 0);
        std::vector<uint8_t> signs(llama_turboq::bitpacked_bytes(plan.padded_dim, 1), 0);
        ggml_fp16_t residual_norm = ggml_fp32_to_fp16(0.0f);

        llama_turboq::encode_stage1_residual(plan, codebook, signal.data(), codes.data(), signs.data(), &residual_norm);
        assert(ggml_fp16_to_fp32(residual_norm) >= 0.0f);

        const auto query = make_signal(plan.dim, 0.7f);
        std::vector<float> decoded(plan.dim, 0.0f);
        std::vector<float> rotated_q;

        llama_turboq::decode_stage1_codes(plan, codebook, codes.data(), decoded.data());
        llama_turboq::rotate_vector(plan, query.data(), rotated_q);

        float actual = 0.0f;
        float stage1 = 0.0f;
        for (uint32_t i = 0; i < plan.dim; ++i) {
            actual += signal[i] * query[i];
            stage1 += decoded[i] * query[i];
        }

        float sketch_dot = 0.0f;
        for (uint32_t i = 0; i < plan.padded_dim; ++i) {
            sketch_dot += rotated_q[i] * llama_turboq::get_sign_value(signs.data(), i);
        }

        const float corrected = stage1 + ggml_fp16_to_fp32(residual_norm) * sketch_dot / std::sqrt(float(plan.padded_dim));
        assert(std::fabs(actual - corrected) <= std::fabs(actual - stage1) + 1.0e-4f);
    }

    std::printf("test-turboq-codec: all tests OK\n");
    return 0;
}
