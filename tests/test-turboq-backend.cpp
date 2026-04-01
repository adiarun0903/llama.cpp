#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "../src/llama-turboq-codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace {

struct attn_layout {
    int32_t seed;
    int32_t layer_index;
    int32_t bits;
    int32_t dim;
    int32_t n_heads;
    int32_t n_phys_rows;
    int32_t n_logical_rows;
};

struct kcorr_layout {
    int32_t seed;
    int32_t layer_index;
    int32_t bits;
    int32_t dim;
    int32_t n_kv_heads;
    int32_t n_q_heads;
    int32_t n_tokens;
    int32_t n_phys_rows;
    int32_t n_logical_rows;
};

struct recurrent_layout {
    int32_t seed;
    int32_t layer_index;
    int32_t bits;
    int32_t dim;
    int32_t n_phys_rows;
    int32_t n_logical_rows;
};

static ggml_context_ptr make_ctx(size_t tensor_capacity = 64, size_t graph_nodes = 32) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * tensor_capacity + ggml_graph_overhead_custom(graph_nodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);
    return ctx;
}

static ggml_backend_ptr make_cpu_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);
    return ggml_backend_ptr(backend);
}

template <typename T>
static void tensor_set(ggml_tensor * tensor, const std::vector<T> & data) {
    assert(data.size() * sizeof(T) == ggml_nbytes(tensor));
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

template <typename T>
static std::vector<T> tensor_get(ggml_tensor * tensor) {
    std::vector<T> data(ggml_nbytes(tensor) / sizeof(T));
    ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
    return data;
}

static ggml_backend_buffer_ptr alloc_and_compute(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    assert(buf != nullptr);

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);
    return buf;
}

static std::vector<float> make_signal(uint32_t dim, float scale, float phase) {
    std::vector<float> values(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        values[i] = scale * (
                std::sin(0.19f * float(i + 1) + phase) +
                0.35f * std::cos(0.13f * float(i + 3) - phase) +
                0.08f * float((int(i) % 5) - 2));
    }
    return values;
}

static float abs_diff(float a, float b) {
    return std::fabs(a - b);
}

static void expect_near(float actual, float expected, float tol, const char * label) {
    if (abs_diff(actual, expected) > tol) {
        std::fprintf(stderr, "%s: expected %.9f, got %.9f (tol %.9f)\n", label, expected, actual, tol);
        std::abort();
    }
}

static void expect_true(bool value, const char * label) {
    if (!value) {
        std::fprintf(stderr, "%s: expected true\n", label);
        std::abort();
    }
}

static inline size_t attn_code_offset(int32_t code_head_bytes, int32_t n_heads, int32_t row, int32_t head) {
    return size_t(row) * size_t(code_head_bytes) * size_t(n_heads) + size_t(head) * size_t(code_head_bytes);
}

static inline size_t attn_norm_offset(int32_t n_heads, int32_t row, int32_t head) {
    return size_t(row) * size_t(n_heads) + size_t(head);
}

static inline size_t recurrent_code_offset(int32_t code_row_bytes, int32_t row) {
    return size_t(row) * size_t(code_row_bytes);
}

static std::vector<int32_t> make_row_map(int32_t n_rows) {
    return std::vector<int32_t>(size_t(n_rows) * GGML_TURBOQ_ROW_FIELD_COUNT, 0);
}

static void set_row_map_entry(
        std::vector<int32_t> & row_map,
        int32_t row,
        int32_t logical_row,
        int32_t src_row,
        int32_t dst_row,
        int32_t flags) {
    row_map[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_LOGICAL] = logical_row;
    row_map[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC]     = src_row;
    row_map[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_DST]     = dst_row;
    row_map[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS]   = flags;
}

static float fp16_roundtrip(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static void test_builder_shapes_and_support(ggml_backend_t backend) {
    auto ctx = make_ctx();

    constexpr int32_t dim          = 13;
    constexpr int32_t n_heads      = 2;
    constexpr int32_t n_q_heads    = 4;
    constexpr int32_t n_tokens     = 3;
    constexpr int32_t n_phys_rows  = 5;
    constexpr int32_t n_logical    = 4;
    constexpr int32_t bits         = 3;
    constexpr int32_t seed         = 17;
    constexpr int32_t layer_index  = 2;

    const auto attn_plan = llama_turboq::make_rotation_plan(seed, llama_turboq::surface_kind::attn_k, layer_index, dim);
    const int32_t code_head_bytes = int32_t(llama_turboq::bitpacked_bytes(attn_plan.padded_dim, bits));
    const int32_t sign_head_bytes = int32_t(llama_turboq::bitpacked_bytes(attn_plan.padded_dim, 1));

    ggml_tensor * row_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, n_logical);
    ggml_tensor * codes   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, code_head_bytes, n_heads, n_phys_rows);
    ggml_tensor * signs   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, sign_head_bytes, n_heads, n_phys_rows);
    ggml_tensor * norms   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n_heads, n_phys_rows);
    ggml_tensor * dep     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * q       = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, n_q_heads, n_tokens);

    ggml_tensor * attn_decode = ggml_turboq_attn_decode(
            ctx.get(), row_map, codes, signs, norms, dep,
            int32_t(llama_turboq::surface_kind::attn_k), seed, layer_index, bits, dim, n_heads);
    ggml_tensor * attn_kcorr = ggml_turboq_attn_kcorr(
            ctx.get(), q, row_map, signs, norms, dep,
            seed, layer_index, bits, dim, n_heads);

    const auto recurrent_plan = llama_turboq::make_rotation_plan(seed + 1, llama_turboq::surface_kind::recurrent_r, layer_index + 1, dim);
    const int32_t recurrent_code_bytes = int32_t(llama_turboq::bitpacked_bytes(recurrent_plan.padded_dim, bits));
    ggml_tensor * rec_codes = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, recurrent_code_bytes, n_phys_rows);
    ggml_tensor * rec_norms = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1, n_phys_rows);
    ggml_tensor * values    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, dim, n_logical);

    ggml_tensor * recurrent_load = ggml_turboq_recurrent_load(
            ctx.get(), row_map, rec_codes, rec_norms,
            int32_t(llama_turboq::surface_kind::recurrent_r), seed + 1, layer_index + 1, bits, dim);
    ggml_tensor * recurrent_store = ggml_turboq_recurrent_store(
            ctx.get(), values, row_map, rec_codes, rec_norms,
            int32_t(llama_turboq::surface_kind::recurrent_s), seed + 2, layer_index + 2, bits, dim);

    assert(attn_decode->op == GGML_OP_TURBOQ_ATTN_DECODE);
    assert(attn_decode->type == GGML_TYPE_F16);
    assert(attn_decode->ne[0] == dim);
    assert(attn_decode->ne[1] == n_heads);
    assert(attn_decode->ne[2] == n_logical);
    assert(attn_decode->src[0] == row_map);
    assert(attn_decode->src[1] == codes);
    assert(attn_decode->src[2] == signs);
    assert(attn_decode->src[3] == norms);

    assert(attn_kcorr->op == GGML_OP_TURBOQ_ATTN_KCORR);
    assert(attn_kcorr->type == GGML_TYPE_F32);
    assert(attn_kcorr->ne[0] == n_logical);
    assert(attn_kcorr->ne[1] == n_tokens);
    assert(attn_kcorr->ne[2] == n_q_heads);
    assert(attn_kcorr->src[0] == q);
    assert(attn_kcorr->src[1] == row_map);

    assert(recurrent_load->op == GGML_OP_TURBOQ_RECURRENT_LOAD);
    assert(recurrent_load->type == GGML_TYPE_F32);
    assert(recurrent_load->ne[0] == dim);
    assert(recurrent_load->ne[1] == n_logical);
    assert(recurrent_load->src[0] == row_map);
    assert(recurrent_load->src[1] == rec_codes);
    assert(recurrent_load->src[2] == rec_norms);

    assert(recurrent_store->op == GGML_OP_TURBOQ_RECURRENT_STORE);
    assert(recurrent_store->type == GGML_TYPE_I32);
    assert(recurrent_store->ne[0] == 1);
    assert(recurrent_store->src[0] == values);
    assert(recurrent_store->src[1] == row_map);
    assert(recurrent_store->src[2] == rec_codes);
    assert(recurrent_store->src[3] == rec_norms);

    expect_true(ggml_backend_supports_op(backend, attn_decode), "CPU backend supports TURBOQ_ATTN_DECODE");
    expect_true(ggml_backend_supports_op(backend, attn_kcorr), "CPU backend supports TURBOQ_ATTN_KCORR");
    expect_true(ggml_backend_supports_op(backend, recurrent_load), "CPU backend supports TURBOQ_RECURRENT_LOAD");
    expect_true(ggml_backend_supports_op(backend, recurrent_store), "CPU backend supports TURBOQ_RECURRENT_STORE");
}

static void test_attn_decode_surface(ggml_backend_t backend, llama_turboq::surface_kind kind) {
    auto ctx = make_ctx();

    const attn_layout spec = {
        /*seed         =*/ kind == llama_turboq::surface_kind::attn_k ? 19 : 23,
        /*layer_index  =*/ kind == llama_turboq::surface_kind::attn_k ? 3  : 4,
        /*bits         =*/ kind == llama_turboq::surface_kind::attn_k ? 3  : 4,
        /*dim          =*/ 13,
        /*n_heads      =*/ 2,
        /*n_phys_rows  =*/ 3,
        /*n_logical_rows=*/ 4,
    };

    const auto plan = llama_turboq::make_rotation_plan(spec.seed, kind, spec.layer_index, spec.dim);
    const auto & codebook = llama_turboq::get_codebook(spec.bits);
    const int32_t code_head_bytes = int32_t(llama_turboq::bitpacked_bytes(plan.padded_dim, spec.bits));
    const int32_t sign_head_bytes = int32_t(llama_turboq::bitpacked_bytes(plan.padded_dim, 1));

    ggml_tensor * row_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, spec.n_logical_rows);
    ggml_tensor * codes   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, code_head_bytes, spec.n_heads, spec.n_phys_rows);
    ggml_tensor * signs   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, sign_head_bytes, spec.n_heads, spec.n_phys_rows);
    ggml_tensor * norms   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, spec.n_heads, spec.n_phys_rows);

    ggml_tensor * decode = ggml_turboq_attn_decode(
            ctx.get(), row_map, codes,
            kind == llama_turboq::surface_kind::attn_k ? signs : nullptr,
            kind == llama_turboq::surface_kind::attn_k ? norms : nullptr,
            nullptr,
            int32_t(kind), spec.seed, spec.layer_index, spec.bits, spec.dim, spec.n_heads);

    std::vector<int32_t> row_map_data = make_row_map(spec.n_logical_rows);
    set_row_map_entry(row_map_data, 0, 0, 2, 2, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 1, 1, -1, -1, 0);
    set_row_map_entry(row_map_data, 2, 2, 0, 0, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 3, 3, 1, 1, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);

    std::vector<uint8_t> codes_data(size_t(code_head_bytes) * spec.n_heads * spec.n_phys_rows, 0);
    std::vector<uint8_t> signs_data(size_t(sign_head_bytes) * spec.n_heads * spec.n_phys_rows, 0);
    std::vector<ggml_fp16_t> norms_data(size_t(spec.n_heads) * spec.n_phys_rows, ggml_fp32_to_fp16(0.0f));

    for (int32_t row = 0; row < spec.n_phys_rows; ++row) {
        for (int32_t head = 0; head < spec.n_heads; ++head) {
            const auto signal = make_signal(spec.dim, 0.75f + 0.1f * float(head), 0.2f * float(row + 1));
            uint8_t * code_ptr = codes_data.data() + attn_code_offset(code_head_bytes, spec.n_heads, row, head);

            if (kind == llama_turboq::surface_kind::attn_k) {
                uint8_t * sign_ptr = signs_data.data() + attn_code_offset(sign_head_bytes, spec.n_heads, row, head);
                ggml_fp16_t * norm_ptr = &norms_data[attn_norm_offset(spec.n_heads, row, head)];
                llama_turboq::encode_stage1_residual(plan, codebook, signal.data(), code_ptr, sign_ptr, norm_ptr);
            } else {
                llama_turboq::encode_stage1_codes(plan, codebook, signal.data(), code_ptr);
            }
        }
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    assert(buf != nullptr);

    tensor_set(row_map, row_map_data);
    tensor_set(codes, codes_data);
    tensor_set(signs, signs_data);
    tensor_set(norms, norms_data);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(gf, decode);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);

    const auto out = tensor_get<ggml_fp16_t>(decode);
    std::vector<float> expected(spec.dim, 0.0f);

    for (int32_t row = 0; row < spec.n_logical_rows; ++row) {
        const int32_t src_row = row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC];
        const bool has_src = (row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS] & GGML_TURBOQ_ROW_FLAG_HAS_SRC) != 0;

        for (int32_t head = 0; head < spec.n_heads; ++head) {
            if (has_src && src_row >= 0) {
                const uint8_t * code_ptr = codes_data.data() + attn_code_offset(code_head_bytes, spec.n_heads, src_row, head);
                llama_turboq::decode_stage1_codes(plan, codebook, code_ptr, expected.data());
            } else {
                std::fill(expected.begin(), expected.end(), 0.0f);
            }

            for (int32_t d = 0; d < spec.dim; ++d) {
                const size_t out_idx = size_t(d) + size_t(spec.dim) * (size_t(head) + size_t(spec.n_heads) * size_t(row));
                expect_near(
                        ggml_fp16_to_fp32(out[out_idx]),
                        fp16_roundtrip(expected[d]),
                        5.0e-4f,
                        kind == llama_turboq::surface_kind::attn_k ? "attn decode K" : "attn decode V");
            }
        }
    }
}

static void test_attn_kcorr(ggml_backend_t backend) {
    auto ctx = make_ctx();

    const kcorr_layout spec = {
        /*seed         =*/ 31,
        /*layer_index  =*/ 5,
        /*bits         =*/ 3,
        /*dim          =*/ 13,
        /*n_kv_heads   =*/ 2,
        /*n_q_heads    =*/ 4,
        /*n_tokens     =*/ 3,
        /*n_phys_rows  =*/ 3,
        /*n_logical_rows=*/ 4,
    };

    const auto plan = llama_turboq::make_rotation_plan(spec.seed, llama_turboq::surface_kind::attn_k, spec.layer_index, spec.dim);
    const int32_t sign_head_bytes = int32_t(llama_turboq::bitpacked_bytes(plan.padded_dim, 1));
    const int32_t n_gqa = spec.n_q_heads / spec.n_kv_heads;
    const float sketch_scale = 1.0f / std::sqrt(float(plan.padded_dim));
    const auto & codebook = llama_turboq::get_codebook(spec.bits);

    ggml_tensor * q       = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, spec.dim, spec.n_q_heads, spec.n_tokens);
    ggml_tensor * row_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, spec.n_logical_rows);
    ggml_tensor * signs   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, sign_head_bytes, spec.n_kv_heads, spec.n_phys_rows);
    ggml_tensor * norms   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, spec.n_kv_heads, spec.n_phys_rows);

    ggml_tensor * kcorr = ggml_turboq_attn_kcorr(
            ctx.get(), q, row_map, signs, norms, nullptr,
            spec.seed, spec.layer_index, spec.bits, spec.dim, spec.n_kv_heads);

    std::vector<int32_t> row_map_data = make_row_map(spec.n_logical_rows);
    set_row_map_entry(row_map_data, 0, 0, 2, 2, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 1, 1, -1, -1, 0);
    set_row_map_entry(row_map_data, 2, 2, 0, 0, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 3, 3, 1, 1, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);

    std::vector<float> q_data(size_t(spec.dim) * spec.n_q_heads * spec.n_tokens, 0.0f);
    std::vector<uint8_t> signs_data(size_t(sign_head_bytes) * spec.n_kv_heads * spec.n_phys_rows, 0);
    std::vector<ggml_fp16_t> norms_data(size_t(spec.n_kv_heads) * spec.n_phys_rows, ggml_fp32_to_fp16(0.0f));

    for (int32_t token = 0; token < spec.n_tokens; ++token) {
        for (int32_t q_head = 0; q_head < spec.n_q_heads; ++q_head) {
            const auto signal = make_signal(spec.dim, 0.65f + 0.05f * float(q_head), 0.15f * float(token + 1));
            for (int32_t d = 0; d < spec.dim; ++d) {
                q_data[size_t(d) + size_t(spec.dim) * (size_t(q_head) + size_t(spec.n_q_heads) * size_t(token))] = signal[d];
            }
        }
    }

    for (int32_t row = 0; row < spec.n_phys_rows; ++row) {
        for (int32_t head = 0; head < spec.n_kv_heads; ++head) {
            const auto signal = make_signal(spec.dim, 0.9f + 0.08f * float(row), 0.11f * float(head + 1));
            std::vector<uint8_t> scratch_codes(llama_turboq::bitpacked_bytes(plan.padded_dim, codebook.bits), 0);
            uint8_t * sign_ptr = signs_data.data() + attn_code_offset(sign_head_bytes, spec.n_kv_heads, row, head);
            ggml_fp16_t * norm_ptr = &norms_data[attn_norm_offset(spec.n_kv_heads, row, head)];
            llama_turboq::encode_stage1_residual(plan, codebook, signal.data(), scratch_codes.data(), sign_ptr, norm_ptr);
        }
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    assert(buf != nullptr);

    tensor_set(q, q_data);
    tensor_set(row_map, row_map_data);
    tensor_set(signs, signs_data);
    tensor_set(norms, norms_data);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(gf, kcorr);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);

    const auto out = tensor_get<float>(kcorr);
    std::vector<float> q_head_values(spec.dim, 0.0f);
    std::vector<float> rotated_q;

    for (int32_t token = 0; token < spec.n_tokens; ++token) {
        for (int32_t q_head = 0; q_head < spec.n_q_heads; ++q_head) {
            for (int32_t d = 0; d < spec.dim; ++d) {
                q_head_values[d] = q_data[size_t(d) + size_t(spec.dim) * (size_t(q_head) + size_t(spec.n_q_heads) * size_t(token))];
            }

            llama_turboq::rotate_vector(plan, q_head_values.data(), rotated_q);
            const int32_t kv_head = q_head / n_gqa;

            for (int32_t row = 0; row < spec.n_logical_rows; ++row) {
                const int32_t src_row = row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC];
                const bool has_src = (row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS] & GGML_TURBOQ_ROW_FLAG_HAS_SRC) != 0;

                float expected = 0.0f;
                if (has_src && src_row >= 0) {
                    const uint8_t * sign_ptr = signs_data.data() + attn_code_offset(sign_head_bytes, spec.n_kv_heads, src_row, kv_head);
                    const float norm = ggml_fp16_to_fp32(norms_data[attn_norm_offset(spec.n_kv_heads, src_row, kv_head)]);
                    float dot = 0.0f;
                    for (uint32_t i = 0; i < plan.padded_dim; ++i) {
                        dot += rotated_q[i] * llama_turboq::get_sign_value(sign_ptr, i);
                    }
                    expected = norm * dot * sketch_scale;
                }

                const size_t out_idx = size_t(row) + size_t(spec.n_logical_rows) * (size_t(token) + size_t(spec.n_tokens) * size_t(q_head));
                expect_near(out[out_idx], expected, 1.0e-5f, "attn kcorr");
            }
        }
    }
}

static void test_recurrent_load(ggml_backend_t backend) {
    auto ctx = make_ctx();

    const recurrent_layout spec = {
        /*seed         =*/ 41,
        /*layer_index  =*/ 6,
        /*bits         =*/ 4,
        /*dim          =*/ 11,
        /*n_phys_rows  =*/ 3,
        /*n_logical_rows=*/ 4,
    };

    const auto plan = llama_turboq::make_rotation_plan(spec.seed, llama_turboq::surface_kind::recurrent_r, spec.layer_index, spec.dim);
    const auto & codebook = llama_turboq::get_codebook(spec.bits);
    const int32_t code_row_bytes = int32_t(llama_turboq::bitpacked_bytes(plan.padded_dim, spec.bits));

    ggml_tensor * row_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, spec.n_logical_rows);
    ggml_tensor * codes   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, code_row_bytes, spec.n_phys_rows);
    ggml_tensor * norms   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1, spec.n_phys_rows);

    ggml_tensor * load = ggml_turboq_recurrent_load(
            ctx.get(), row_map, codes, norms,
            int32_t(llama_turboq::surface_kind::recurrent_r), spec.seed, spec.layer_index, spec.bits, spec.dim);

    std::vector<int32_t> row_map_data = make_row_map(spec.n_logical_rows);
    set_row_map_entry(row_map_data, 0, 0, 2, 2, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 1, 1, -1, -1, 0);
    set_row_map_entry(row_map_data, 2, 2, 0, 0, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);
    set_row_map_entry(row_map_data, 3, 3, 1, 1, GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST);

    std::vector<uint8_t> codes_data(size_t(code_row_bytes) * spec.n_phys_rows, 0);
    std::vector<ggml_fp16_t> norms_data(spec.n_phys_rows, ggml_fp32_to_fp16(0.0f));

    for (int32_t row = 0; row < spec.n_phys_rows; ++row) {
        const auto signal = make_signal(spec.dim, 0.85f + 0.1f * float(row), 0.07f * float(row + 1));
        uint8_t * code_ptr = codes_data.data() + recurrent_code_offset(code_row_bytes, row);
        llama_turboq::encode_stage1_normed(plan, codebook, signal.data(), code_ptr, &norms_data[row]);
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    assert(buf != nullptr);

    tensor_set(row_map, row_map_data);
    tensor_set(codes, codes_data);
    tensor_set(norms, norms_data);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(gf, load);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);

    const auto out = tensor_get<float>(load);
    std::vector<float> expected(spec.dim, 0.0f);

    for (int32_t row = 0; row < spec.n_logical_rows; ++row) {
        const int32_t src_row = row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC];
        const bool has_src = (row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS] & GGML_TURBOQ_ROW_FLAG_HAS_SRC) != 0;

        if (has_src && src_row >= 0) {
            const uint8_t * code_ptr = codes_data.data() + recurrent_code_offset(code_row_bytes, src_row);
            llama_turboq::decode_stage1_normed(plan, codebook, code_ptr, ggml_fp16_to_fp32(norms_data[src_row]), expected.data());
        } else {
            std::fill(expected.begin(), expected.end(), 0.0f);
        }

        for (int32_t d = 0; d < spec.dim; ++d) {
            const size_t out_idx = size_t(d) + size_t(spec.dim) * size_t(row);
            expect_near(out[out_idx], expected[d], 1.0e-5f, "recurrent load");
        }
    }
}

static void test_recurrent_store(ggml_backend_t backend) {
    auto ctx = make_ctx();

    const recurrent_layout spec = {
        /*seed         =*/ 47,
        /*layer_index  =*/ 7,
        /*bits         =*/ 2,
        /*dim          =*/ 11,
        /*n_phys_rows  =*/ 4,
        /*n_logical_rows=*/ 4,
    };

    const auto plan = llama_turboq::make_rotation_plan(spec.seed, llama_turboq::surface_kind::recurrent_s, spec.layer_index, spec.dim);
    const auto & codebook = llama_turboq::get_codebook(spec.bits);
    const int32_t code_row_bytes = int32_t(llama_turboq::bitpacked_bytes(plan.padded_dim, spec.bits));

    ggml_tensor * values  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, spec.dim, spec.n_logical_rows);
    ggml_tensor * row_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, spec.n_logical_rows);
    ggml_tensor * codes   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, code_row_bytes, spec.n_phys_rows);
    ggml_tensor * norms   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1, spec.n_phys_rows);

    ggml_tensor * store = ggml_turboq_recurrent_store(
            ctx.get(), values, row_map, codes, norms,
            int32_t(llama_turboq::surface_kind::recurrent_s), spec.seed, spec.layer_index, spec.bits, spec.dim);

    std::vector<int32_t> row_map_data = make_row_map(spec.n_logical_rows);
    set_row_map_entry(row_map_data, 0, 0, -1, 2, GGML_TURBOQ_ROW_FLAG_HAS_DST | GGML_TURBOQ_ROW_FLAG_DIRTY);
    set_row_map_entry(row_map_data, 1, 1, -1, -1, 0);
    set_row_map_entry(row_map_data, 2, 2, -1, 0, GGML_TURBOQ_ROW_FLAG_HAS_DST | GGML_TURBOQ_ROW_FLAG_DIRTY);
    set_row_map_entry(row_map_data, 3, 3, -1, 1, GGML_TURBOQ_ROW_FLAG_HAS_DST | GGML_TURBOQ_ROW_FLAG_DIRTY);

    std::vector<float> values_data(size_t(spec.dim) * spec.n_logical_rows, 0.0f);
    for (int32_t row = 0; row < spec.n_logical_rows; ++row) {
        const auto signal = make_signal(spec.dim, 0.7f + 0.12f * float(row), 0.1f * float(row + 1));
        for (int32_t d = 0; d < spec.dim; ++d) {
            values_data[size_t(d) + size_t(spec.dim) * size_t(row)] = signal[d];
        }
    }

    std::vector<uint8_t> codes_data(size_t(code_row_bytes) * spec.n_phys_rows, 0);
    std::vector<ggml_fp16_t> norms_data(spec.n_phys_rows, ggml_fp32_to_fp16(0.0f));

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    assert(buf != nullptr);

    tensor_set(values, values_data);
    tensor_set(row_map, row_map_data);
    tensor_set(codes, codes_data);
    tensor_set(norms, norms_data);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(gf, store);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);

    const auto out_status = tensor_get<int32_t>(store);
    assert(out_status.size() == 1);
    assert(out_status[0] == 1);

    const auto stored_codes = tensor_get<uint8_t>(codes);
    const auto stored_norms = tensor_get<ggml_fp16_t>(norms);

    std::vector<uint8_t> expected_codes(size_t(code_row_bytes) * spec.n_phys_rows, 0);
    std::vector<ggml_fp16_t> expected_norms(spec.n_phys_rows, ggml_fp32_to_fp16(0.0f));

    for (int32_t row = 0; row < spec.n_logical_rows; ++row) {
        const int32_t dst_row = row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_DST];
        const bool has_dst = (row_map_data[size_t(row) * GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS] & GGML_TURBOQ_ROW_FLAG_HAS_DST) != 0;
        if (!has_dst || dst_row < 0) {
            continue;
        }

        std::vector<float> signal(spec.dim, 0.0f);
        for (int32_t d = 0; d < spec.dim; ++d) {
            signal[d] = values_data[size_t(d) + size_t(spec.dim) * size_t(row)];
        }

        uint8_t * code_ptr = expected_codes.data() + recurrent_code_offset(code_row_bytes, dst_row);
        llama_turboq::encode_stage1_normed(plan, codebook, signal.data(), code_ptr, &expected_norms[dst_row]);
    }

    assert(stored_codes == expected_codes);
    assert(stored_norms == expected_norms);
}

} // namespace

int main() {
    ggml_backend_load_all();
    auto backend = make_cpu_backend();

    test_builder_shapes_and_support(backend.get());
    test_attn_decode_surface(backend.get(), llama_turboq::surface_kind::attn_k);
    test_attn_decode_surface(backend.get(), llama_turboq::surface_kind::attn_v);
    test_attn_kcorr(backend.get());
    test_recurrent_load(backend.get());
    test_recurrent_store(backend.get());

    std::printf("test-turboq-backend: all tests OK\n");
    return 0;
}
