#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace llama_turboq {

enum class surface_kind : uint32_t {
    attn_k,
    attn_v,
    recurrent_r,
    recurrent_s,
};

struct codebook {
    uint32_t bits = 0;
    const float * values = nullptr;
};

struct rotation_plan {
    uint32_t dim = 0;
    uint32_t padded_dim = 0;

    std::vector<uint32_t> perm;
    std::vector<uint32_t> inv_perm;
    std::vector<int8_t> sign;
};

const codebook & get_codebook(uint32_t bits);

uint32_t salt_for_surface_kind(surface_kind kind);
rotation_plan make_rotation_plan(uint32_t seed, surface_kind kind, uint32_t layer, uint32_t dim);
rotation_plan make_rotation_plan(uint32_t seed, uint32_t layer, uint32_t dim, uint32_t salt);

size_t div_round_up(size_t num, size_t den);
size_t bitpacked_bytes(size_t count, size_t bits);

void pack_bits(uint8_t * dst, size_t index, uint32_t bits, uint32_t value);
uint32_t unpack_bits(const uint8_t * src, size_t index, uint32_t bits);

void set_sign_bit(uint8_t * dst, size_t index, bool value);
float get_sign_value(const uint8_t * src, size_t index);

void rotate_vector(const rotation_plan & plan, const float * src, std::vector<float> & dst);
void inverse_rotate_vector(const rotation_plan & plan, std::vector<float> & rotated, float * dst);

uint32_t quantize_scalar(const codebook & codebook, float value);

void encode_stage1_codes(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst);

void decode_stage1_codes(
        const rotation_plan & plan,
        const codebook & codebook,
        const uint8_t * code_src,
        float * dst);

void encode_stage1_normed(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst,
        ggml_fp16_t * norm_dst);

void decode_stage1_normed(
        const rotation_plan & plan,
        const codebook & codebook,
        const uint8_t * code_src,
        ggml_fp16_t norm,
        float * dst);

void encode_stage1_residual(
        const rotation_plan & plan,
        const codebook & codebook,
        const float * src,
        uint8_t * code_dst,
        uint8_t * sign_dst,
        ggml_fp16_t * norm_dst);

} // namespace llama_turboq
