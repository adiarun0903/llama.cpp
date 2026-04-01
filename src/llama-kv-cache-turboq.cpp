#include "llama-kv-cache-turboq.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

static float turboq_tensor_get_f32(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3 = 0) {
    const char * base = reinterpret_cast<const char *>(t->data) + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];

    switch (t->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(base));
        default:
            GGML_ABORT("unsupported tensor type");
    }
}

static inline void turboq_tensor_set_f16(ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float value) {
    char * base = reinterpret_cast<char *>(t->data) + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    *reinterpret_cast<ggml_fp16_t *>(base) = ggml_fp32_to_fp16(value);
}

static inline void turboq_tensor_set_f32(ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float value) {
    char * base = reinterpret_cast<char *>(t->data) + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    *reinterpret_cast<float *>(base) = value;
}

template<typename T>
static T * turboq_tensor_host_data(ggml_tensor * tensor) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(tensor->buffer));
    return reinterpret_cast<T *>(tensor->data);
}

template<typename T>
static const T * turboq_tensor_host_data(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(tensor->buffer));
    return reinterpret_cast<const T *>(tensor->data);
}

static void turboq_zero_status(ggml_tensor * dst, int ith) {
    if (ith == 0) {
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }
}

template<typename T>
static void turboq_write_rows(
        llama_io_write_i & io,
        const ggml_tensor * tensor,
        size_t row_bytes,
        const std::vector<std::pair<uint32_t, uint32_t>> & ranges) {
    const auto * base = reinterpret_cast<const uint8_t *>(turboq_tensor_host_data<T>(tensor));

    for (const auto & range : ranges) {
        const size_t range_size = range.second - range.first;
        io.write(base + range.first * row_bytes, range_size * row_bytes);
    }
}

template<typename T>
static void turboq_read_rows(llama_io_read_i & io, ggml_tensor * tensor, size_t row_bytes, uint32_t cell_count, const llama_kv_cache::slot_info & sinfo) {
    auto * base = reinterpret_cast<uint8_t *>(turboq_tensor_host_data<T>(tensor));

    if (cell_count == 0) {
        return;
    }

    if (sinfo.is_contiguous()) {
        io.read_to(base + sinfo.head() * row_bytes, cell_count * row_bytes);
        return;
    }

    std::vector<uint8_t> packed(cell_count * row_bytes);
    io.read_to(packed.data(), packed.size());

    GGML_ASSERT(sinfo.n_stream() == 1);
    for (uint32_t i = 0; i < cell_count; ++i) {
        std::memcpy(base + sinfo.idxs[0][i] * row_bytes, packed.data() + i * row_bytes, row_bytes);
    }
}

static void turboq_op_encode_k(ggml_tensor * dst, int ith, int nth, void * userdata) {
    turboq_zero_status(dst, ith);

    const auto * ud = static_cast<const llama_kv_cache_turboq::turboq_op_userdata *>(userdata);
    const auto & layer = *ud->layer;
    const auto & codebook = llama_turboq::get_codebook(layer.k_bits);

    const ggml_tensor * k_cur  = dst->src[0];
    const ggml_tensor * k_idxs = dst->src[1];

    GGML_ASSERT(k_idxs != nullptr && k_idxs->type == GGML_TYPE_I64);

    const int64_t n_tokens = k_cur->ne[2];
    const int64_t start = (n_tokens * ith) / nth;
    const int64_t end   = (n_tokens * (ith + 1)) / nth;
    const auto * idxs = reinterpret_cast<const int64_t *>(k_idxs->data);

    std::vector<float> src(layer.n_embd_head_k);
    uint8_t * k_codes = turboq_tensor_host_data<uint8_t>(layer.k_codes);
    uint8_t * k_signs = turboq_tensor_host_data<uint8_t>(layer.k_signs);
    auto * k_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.k_norms);

    for (int64_t token = start; token < end; ++token) {
        const uint32_t cell = uint32_t(idxs[token]);
        uint8_t * row_codes = k_codes + cell * layer.k_code_row_bytes;
        uint8_t * row_signs = k_signs + cell * layer.k_sign_row_bytes;
        auto * row_norms = k_norms + cell * layer.n_head_kv;

        for (uint32_t head = 0; head < layer.n_head_kv; ++head) {
            for (uint32_t d = 0; d < layer.n_embd_head_k; ++d) {
                src[d] = turboq_tensor_get_f32(k_cur, d, head, token);
            }

            llama_turboq::encode_stage1_residual(
                    layer.rot_k,
                    codebook,
                    src.data(),
                    row_codes + head * layer.k_code_head_bytes,
                    row_signs + head * layer.k_sign_head_bytes,
                    row_norms + head);
        }
    }
}

static void turboq_op_encode_v(ggml_tensor * dst, int ith, int nth, void * userdata) {
    turboq_zero_status(dst, ith);

    const auto * ud = static_cast<const llama_kv_cache_turboq::turboq_op_userdata *>(userdata);
    const auto & layer = *ud->layer;
    const auto & codebook = llama_turboq::get_codebook(layer.v_bits);

    const ggml_tensor * v_cur  = dst->src[0];
    const ggml_tensor * v_idxs = dst->src[1];

    GGML_ASSERT(v_idxs != nullptr && v_idxs->type == GGML_TYPE_I64);

    const int64_t n_tokens = v_cur->ne[2];
    const int64_t start = (n_tokens * ith) / nth;
    const int64_t end   = (n_tokens * (ith + 1)) / nth;
    const auto * idxs = reinterpret_cast<const int64_t *>(v_idxs->data);

    std::vector<float> src(layer.n_embd_head_v);
    uint8_t * v_codes = turboq_tensor_host_data<uint8_t>(layer.v_codes);

    for (int64_t token = start; token < end; ++token) {
        const uint32_t cell = uint32_t(idxs[token]);
        uint8_t * row_codes = v_codes + cell * layer.v_code_row_bytes;

        for (uint32_t head = 0; head < layer.n_head_kv; ++head) {
            for (uint32_t d = 0; d < layer.n_embd_head_v; ++d) {
                src[d] = turboq_tensor_get_f32(v_cur, d, head, token);
            }

            llama_turboq::encode_stage1_codes(
                    layer.rot_v,
                    codebook,
                    src.data(),
                    row_codes + head * layer.v_code_head_bytes);
        }
    }
}

static void turboq_op_decode_k(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = static_cast<const llama_kv_cache_turboq::turboq_op_userdata *>(userdata);
    const auto & layer = *ud->layer;
    const auto & codebook = llama_turboq::get_codebook(layer.k_bits);

    const int64_t total = int64_t(ud->n_kv) * layer.n_head_kv;
    const int64_t start = (total * ith) / nth;
    const int64_t end   = (total * (ith + 1)) / nth;

    std::vector<float> decoded(layer.n_embd_head_k);
    const uint8_t * k_codes = turboq_tensor_host_data<uint8_t>(layer.k_codes);

    for (int64_t row = start; row < end; ++row) {
        const uint32_t cell = row / layer.n_head_kv;
        const uint32_t head = row % layer.n_head_kv;

        if (ud->cache->cell_is_empty(cell)) {
            for (uint32_t d = 0; d < layer.n_embd_head_k; ++d) {
                turboq_tensor_set_f16(dst, d, head, cell, 0, 0.0f);
            }
            continue;
        }

        const uint8_t * head_codes = k_codes + cell * layer.k_code_row_bytes + head * layer.k_code_head_bytes;
        llama_turboq::decode_stage1_codes(layer.rot_k, codebook, head_codes, decoded.data());
        for (uint32_t d = 0; d < layer.n_embd_head_k; ++d) {
            turboq_tensor_set_f16(dst, d, head, cell, 0, decoded[d]);
        }
    }
}

static void turboq_op_decode_v(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = static_cast<const llama_kv_cache_turboq::turboq_op_userdata *>(userdata);
    const auto & layer = *ud->layer;
    const auto & codebook = llama_turboq::get_codebook(layer.v_bits);

    const int64_t total = int64_t(ud->n_kv) * layer.n_head_kv;
    const int64_t start = (total * ith) / nth;
    const int64_t end   = (total * (ith + 1)) / nth;

    std::vector<float> decoded(layer.n_embd_head_v);
    const uint8_t * v_codes = turboq_tensor_host_data<uint8_t>(layer.v_codes);

    for (int64_t row = start; row < end; ++row) {
        const uint32_t cell = row / layer.n_head_kv;
        const uint32_t head = row % layer.n_head_kv;

        if (ud->cache->cell_is_empty(cell)) {
            for (uint32_t d = 0; d < layer.n_embd_head_v; ++d) {
                turboq_tensor_set_f16(dst, d, head, cell, 0, 0.0f);
            }
            continue;
        }

        const uint8_t * head_codes = v_codes + cell * layer.v_code_row_bytes + head * layer.v_code_head_bytes;
        llama_turboq::decode_stage1_codes(layer.rot_v, codebook, head_codes, decoded.data());
        for (uint32_t d = 0; d < layer.n_embd_head_v; ++d) {
            turboq_tensor_set_f16(dst, d, head, cell, 0, decoded[d]);
        }
    }
}

static void turboq_op_k_corr(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * ud = static_cast<const llama_kv_cache_turboq::turboq_op_userdata *>(userdata);
    const auto & layer = *ud->layer;
    const ggml_tensor * q = dst->src[0];

    const int64_t n_tokens = q->ne[2];
    const int64_t n_head_q = q->ne[1];
    const int64_t total = n_tokens * n_head_q;
    const int64_t start = (total * ith) / nth;
    const int64_t end   = (total * (ith + 1)) / nth;

    std::vector<float> src(layer.n_embd_head_k);
    std::vector<float> rotated_q;
    const float sketch_scale = 1.0f / std::sqrt(float(layer.n_rot_k));
    const uint8_t * k_signs = turboq_tensor_host_data<uint8_t>(layer.k_signs);
    const auto * k_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.k_norms);

    for (int64_t pair = start; pair < end; ++pair) {
        const uint32_t token = pair / n_head_q;
        const uint32_t q_head = pair % n_head_q;
        const uint32_t kv_head = q_head / std::max(1u, layer.n_gqa);

        for (uint32_t d = 0; d < layer.n_embd_head_k; ++d) {
            src[d] = turboq_tensor_get_f32(q, d, q_head, token);
        }

        llama_turboq::rotate_vector(layer.rot_k, src.data(), rotated_q);

        for (uint32_t cell = 0; cell < ud->n_kv; ++cell) {
            float corr = 0.0f;

            if (!ud->cache->cell_is_empty(cell)) {
                const uint8_t * signs = k_signs + cell * layer.k_sign_row_bytes + kv_head * layer.k_sign_head_bytes;
                const float norm = ggml_fp16_to_fp32(k_norms[cell * layer.n_head_kv + kv_head]);

                if (norm > 0.0f) {
                    float dot = 0.0f;
                    for (uint32_t i = 0; i < layer.n_rot_k; ++i) {
                        dot += rotated_q[i] * llama_turboq::get_sign_value(signs, i);
                    }
                    corr = norm * dot * sketch_scale;
                }
            }

            turboq_tensor_set_f32(dst, cell, token, q_head, 0, corr);
        }
    }
}

} // namespace

llama_kv_cache_turboq::llama_kv_cache_turboq(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
        const llama_turboq_memory_params & params,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse) :
    llama_kv_cache(
            model,
            type_k,
            type_v,
            false,
            offload,
            unified,
            kv_size,
            n_seq_max,
            n_pad,
            0,
            LLAMA_SWA_TYPE_NONE,
            filter,
            reuse,
            false),
    turboq_params(params) {
    GGML_ASSERT(this->n_stream == 1);

    turboq_layers.resize(layers.size());

    ggml_init_params ggml_params = {
        /*.mem_size   =*/ size_t((1u + 4u*turboq_layers.size())*ggml_tensor_overhead()),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr turboq_ctx(ggml_init(ggml_params));
    if (!turboq_ctx) {
        throw std::runtime_error("failed to create TurboQ packed storage context");
    }

    turboq_row_map = ggml_new_tensor_2d(turboq_ctx.get(), GGML_TYPE_I32, GGML_TURBOQ_ROW_FIELD_COUNT, kv_size);
    ggml_format_name(turboq_row_map, "turboq_attn_row_map");

    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        auto & layer = turboq_layers[ikv];
        layer.il = layers[ikv].il;
        layer.n_head_kv = hparams.n_head_kv(layer.il);
        layer.n_head = hparams.n_head(layer.il);
        layer.n_gqa = std::max(1u, layer.n_head / std::max(1u, layer.n_head_kv));
        layer.n_embd_head_k = hparams.n_embd_head_k(layer.il);
        layer.n_embd_head_v = hparams.n_embd_head_v(layer.il);
        layer.n_embd_k_gqa = hparams.n_embd_k_gqa(layer.il);
        layer.n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
        layer.rot_k = llama_turboq::make_rotation_plan(turboq_params.seed, llama_turboq::surface_kind::attn_k, layer.il, layer.n_embd_head_k);
        layer.rot_v = llama_turboq::make_rotation_plan(turboq_params.seed, llama_turboq::surface_kind::attn_v, layer.il, layer.n_embd_head_v);
        layer.n_rot_k = layer.rot_k.padded_dim;
        layer.n_rot_v = layer.rot_v.padded_dim;
        layer.k_bits = turboq_params.attn_k_bits;
        layer.v_bits = turboq_params.attn_v_bits;
        layer.k_code_head_bytes = llama_turboq::bitpacked_bytes(layer.n_rot_k, turboq_params.attn_k_bits);
        layer.k_sign_head_bytes = llama_turboq::bitpacked_bytes(layer.n_rot_k, 1);
        layer.v_code_head_bytes = llama_turboq::bitpacked_bytes(layer.n_rot_v, turboq_params.attn_v_bits);
        layer.k_code_row_bytes = layer.n_head_kv * layer.k_code_head_bytes;
        layer.k_sign_row_bytes = layer.n_head_kv * layer.k_sign_head_bytes;
        layer.v_code_row_bytes = layer.n_head_kv * layer.v_code_head_bytes;

        layer.k_codes = ggml_new_tensor_3d(turboq_ctx.get(), GGML_TYPE_I8, layer.k_code_head_bytes, layer.n_head_kv, kv_size);
        layer.k_signs = ggml_new_tensor_3d(turboq_ctx.get(), GGML_TYPE_I8, layer.k_sign_head_bytes, layer.n_head_kv, kv_size);
        layer.k_norms = ggml_new_tensor_2d(turboq_ctx.get(), GGML_TYPE_F16, layer.n_head_kv, kv_size);
        layer.v_codes = ggml_new_tensor_3d(turboq_ctx.get(), GGML_TYPE_I8, layer.v_code_head_bytes, layer.n_head_kv, kv_size);

        ggml_format_name(layer.k_codes, "turboq_k_codes_l%d", layer.il);
        ggml_format_name(layer.k_signs, "turboq_k_signs_l%d", layer.il);
        ggml_format_name(layer.k_norms, "turboq_k_norms_l%d", layer.il);
        ggml_format_name(layer.v_codes, "turboq_v_codes_l%d", layer.il);
    }

    ggml_backend_buffer_t turboq_buf = ggml_backend_alloc_ctx_tensors_from_buft(turboq_ctx.get(), ggml_backend_cpu_buffer_type());
    if (!turboq_buf) {
        throw std::runtime_error("failed to allocate TurboQ packed storage buffer");
    }

    ggml_backend_buffer_clear(turboq_buf, 0);
    turboq_ctxs_bufs.emplace_back(std::move(turboq_ctx), turboq_buf);

    auto * row_map = turboq_tensor_host_data<int32_t>(turboq_row_map);
    for (uint32_t row = 0; row < kv_size; ++row) {
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_LOGICAL] = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC]     = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_DST]     = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS]   = GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST;
    }

    clear(true);

    LLAMA_LOG_INFO("%s: TurboQ packed KV buffer size = %8.2f MiB (k_bits=%u, v_bits=%u, seed=%u)\n",
            __func__,
            total_packed_size() / 1024.0 / 1024.0,
            turboq_params.attn_k_bits,
            turboq_params.attn_v_bits,
            turboq_params.seed);
}

bool llama_kv_cache_turboq::get_can_shift() const {
    return false;
}

void llama_kv_cache_turboq::clear(bool data) {
    llama_kv_cache::clear(false);

    if (!data) {
        return;
    }

    for (auto & [_, buf] : turboq_ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }

    auto * row_map = turboq_tensor_host_data<int32_t>(turboq_row_map);
    for (uint32_t row = 0; row < turboq_row_map->ne[1]; ++row) {
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_LOGICAL] = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_SRC]     = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_DST]     = int32_t(row);
        row_map[row*GGML_TURBOQ_ROW_FIELD_COUNT + GGML_TURBOQ_ROW_FIELD_FLAGS]   = GGML_TURBOQ_ROW_FLAG_HAS_SRC | GGML_TURBOQ_ROW_FLAG_HAS_DST;
    }
}

int32_t llama_kv_cache_turboq::find_empty_cell() const {
    GGML_ASSERT(n_stream == 1);

    const auto & cells = v_cells[0];
    uint32_t cell = v_heads[0];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cell >= cells.size()) {
            cell -= cells.size();
        }
        if (cells.is_empty(cell)) {
            return int32_t(cell);
        }
        cell += 1;
    }

    return -1;
}

void llama_kv_cache_turboq::clone_cell_data(uint32_t src_cell, uint32_t dst_cell) {
    for (auto & layer : turboq_layers) {
        std::memcpy(turboq_tensor_host_data<uint8_t>(layer.k_codes) + dst_cell * layer.k_code_row_bytes,
                turboq_tensor_host_data<uint8_t>(layer.k_codes) + src_cell * layer.k_code_row_bytes,
                layer.k_code_row_bytes);
        std::memcpy(turboq_tensor_host_data<uint8_t>(layer.k_signs) + dst_cell * layer.k_sign_row_bytes,
                turboq_tensor_host_data<uint8_t>(layer.k_signs) + src_cell * layer.k_sign_row_bytes,
                layer.k_sign_row_bytes);
        std::memcpy(turboq_tensor_host_data<ggml_fp16_t>(layer.k_norms) + dst_cell * layer.n_head_kv,
                turboq_tensor_host_data<ggml_fp16_t>(layer.k_norms) + src_cell * layer.n_head_kv,
                layer.n_head_kv * sizeof(ggml_fp16_t));
        std::memcpy(turboq_tensor_host_data<uint8_t>(layer.v_codes) + dst_cell * layer.v_code_row_bytes,
                turboq_tensor_host_data<uint8_t>(layer.v_codes) + src_cell * layer.v_code_row_bytes,
                layer.v_code_row_bytes);
    }
}

int32_t llama_kv_cache_turboq::detach_seq_cell(llama_kv_cells & cells, llama_seq_id seq_id, uint32_t src_cell) {
    if (!cells.seq_has(src_cell, seq_id) || cells.seq_count(src_cell) <= 1) {
        return int32_t(src_cell);
    }

    const int32_t dst_cell = find_empty_cell();
    GGML_ASSERT(dst_cell >= 0 && "TurboQ attention copy-on-write ran out of cells");

    llama_pos pos = cells.pos_get(src_cell);
    const llama_pos shift = cells.get_shift(src_cell);
    const llama_kv_cell_ext ext = cells.ext_get(src_cell);

    if (shift != 0) {
        pos -= shift;
        GGML_ASSERT(pos >= 0);
    }

    cells.pos_set(dst_cell, pos);
    cells.seq_add(dst_cell, seq_id);
    if (shift != 0) {
        cells.pos_add(dst_cell, shift);
    }
    cells.ext_set(dst_cell, ext);
    clone_cell_data(src_cell, dst_cell);
    cells.seq_rm(src_cell, seq_id);

    return dst_cell;
}

void llama_kv_cache_turboq::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(n_stream == 1);

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 == p1) {
        return;
    }

    std::vector<uint32_t> targets;
    targets.reserve(cells.size());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id)) {
            targets.push_back(i);
        }
    }

    uint32_t new_head = cells.size();
    for (const uint32_t src_cell : targets) {
        const int32_t dst_cell = detach_seq_cell(cells, seq_id, src_cell);
        if (dst_cell >= 0 && cells.seq_has(dst_cell, seq_id) && cells.pos_add(dst_cell, shift)) {
            if (new_head == cells.size()) {
                new_head = dst_cell;
            }
        }
    }

    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache_turboq::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(n_stream == 1);

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 == p1) {
        return;
    }

    std::vector<uint32_t> targets;
    targets.reserve(cells.size());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id)) {
            targets.push_back(i);
        }
    }

    for (const uint32_t src_cell : targets) {
        const int32_t dst_cell = detach_seq_cell(cells, seq_id, src_cell);
        if (dst_cell >= 0 && cells.seq_has(dst_cell, seq_id)) {
            cells.pos_div(dst_cell, d);
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_turboq::memory_breakdown() const {
    return {
        { ggml_backend_cpu_buffer_type(), total_packed_size() },
    };
}

void llama_kv_cache_turboq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };
        uint32_t cell_count = 0;
        const auto & cells = v_cells[s];
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else if (cell_range_begin != cells.size()) {
                cr.data.emplace_back(cell_range_begin, i);
                cell_range_begin = cells.size();
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        io.write(&cell_count, sizeof(cell_count));
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache_turboq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t n_stream_cur = 0;
    io.read_to(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count = 0;
        io.read_to(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream.at(seq_id);
        slot_info sinfo;

        bool ok = true;
        ok = ok && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        ok = ok && state_read_data(io, strm, cell_count, sinfo);

        if (!ok) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore TurboQ kv cache");
        }
    }
}

ggml_tensor * llama_kv_cache_turboq::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, ggml_tensor * dep) const {
    GGML_UNUSED(sinfo);

    const auto & layer = layer_for(il);
    ggml_tensor * row_map = ggml_view_2d(ctx, turboq_row_map, GGML_TURBOQ_ROW_FIELD_COUNT, n_kv, turboq_row_map->nb[1], 0);

    return ggml_turboq_attn_decode(
            ctx,
            row_map,
            layer.k_codes,
            layer.k_signs,
            layer.k_norms,
            dep,
            int32_t(llama_turboq::surface_kind::attn_k),
            int32_t(turboq_params.seed),
            int32_t(layer.il),
            int32_t(layer.k_bits),
            int32_t(layer.n_embd_head_k),
            int32_t(layer.n_head_kv));
}

ggml_tensor * llama_kv_cache_turboq::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, ggml_tensor * dep) const {
    GGML_UNUSED(sinfo);

    const auto & layer = layer_for(il);
    ggml_tensor * row_map = ggml_view_2d(ctx, turboq_row_map, GGML_TURBOQ_ROW_FIELD_COUNT, n_kv, turboq_row_map->nb[1], 0);

    return ggml_turboq_attn_decode(
            ctx,
            row_map,
            layer.v_codes,
            nullptr,
            nullptr,
            dep,
            int32_t(llama_turboq::surface_kind::attn_v),
            int32_t(turboq_params.seed),
            int32_t(layer.il),
            int32_t(layer.v_bits),
            int32_t(layer.n_embd_head_v),
            int32_t(layer.n_head_kv));
}

ggml_tensor * llama_kv_cache_turboq::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const auto & layer = layer_for(il);
    ggml_tensor * args[] = { k_cur, k_idxs };

    return ggml_custom_4d(
            ctx,
            GGML_TYPE_I32,
            1, 1, 1, 1,
            args,
            2,
            turboq_op_encode_k,
            GGML_N_TASKS_MAX,
            const_cast<turboq_op_userdata *>(make_op_userdata(layer, 0)));
}

ggml_tensor * llama_kv_cache_turboq::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const auto & layer = layer_for(il);
    ggml_tensor * args[] = { v_cur, v_idxs };

    return ggml_custom_4d(
            ctx,
            GGML_TYPE_I32,
            1, 1, 1, 1,
            args,
            2,
            turboq_op_encode_v,
            GGML_N_TASKS_MAX,
            const_cast<turboq_op_userdata *>(make_op_userdata(layer, 0)));
}

ggml_tensor * llama_kv_cache_turboq::get_k_corr(
        ggml_context * ctx,
         ggml_tensor * q,
              int32_t   il,
        const slot_info & sinfo,
         ggml_tensor * dep) const {
    const auto & layer = layer_for(il);
    const uint32_t n_kv = get_n_kv(sinfo);
    ggml_tensor * row_map = ggml_view_2d(ctx, turboq_row_map, GGML_TURBOQ_ROW_FIELD_COUNT, n_kv, turboq_row_map->nb[1], 0);

    return ggml_turboq_attn_kcorr(
            ctx,
            q,
            row_map,
            layer.k_signs,
            layer.k_norms,
            dep,
            int32_t(turboq_params.seed),
            int32_t(layer.il),
            int32_t(layer.k_bits),
            int32_t(layer.n_embd_head_k),
            int32_t(layer.n_head_kv));
}

bool llama_kv_cache_turboq::prefers_flash_attn() const {
    return false;
}

bool llama_kv_cache_turboq::force_cpu_kqv() const {
    return true;
}

bool llama_kv_cache_turboq::cell_is_empty(uint32_t cell) const {
    GGML_ASSERT(n_stream == 1);
    return v_cells[0].is_empty(cell);
}

const llama_kv_cache_turboq::turboq_layer_storage & llama_kv_cache_turboq::layer_for(int32_t il) const {
    return turboq_layers.at(map_layer_ids.at(il));
}

llama_kv_cache_turboq::turboq_layer_storage & llama_kv_cache_turboq::layer_for(int32_t il) {
    return turboq_layers.at(map_layer_ids.at(il));
}

const llama_kv_cache_turboq::turboq_op_userdata * llama_kv_cache_turboq::make_op_userdata(const turboq_layer_storage & layer, uint32_t n_kv) const {
    op_userdata.push_back({ this, const_cast<turboq_layer_storage *>(&layer), n_kv });
    return &op_userdata.back();
}

size_t llama_kv_cache_turboq::total_packed_size() const {
    size_t total = 0;
    for (const auto & layer : turboq_layers) {
        total += ggml_nbytes(layer.k_codes);
        total += ggml_nbytes(layer.k_signs);
        total += ggml_nbytes(layer.k_norms);
        total += ggml_nbytes(layer.v_codes);
    }
    total += ggml_nbytes(turboq_row_map);
    return total;
}

void llama_kv_cache_turboq::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const uint32_t version = 1;
    const uint32_t n_layer = turboq_layers.size();

    io.write(&version, sizeof(version));
    io.write(&turboq_params.attn_k_bits, sizeof(turboq_params.attn_k_bits));
    io.write(&turboq_params.attn_v_bits, sizeof(turboq_params.attn_v_bits));
    io.write(&turboq_params.attn_k_residual_bits, sizeof(turboq_params.attn_k_residual_bits));
    io.write(&turboq_params.seed, sizeof(turboq_params.seed));
    io.write(&turboq_params.rotation, sizeof(turboq_params.rotation));
    io.write(&n_layer, sizeof(n_layer));

    for (const auto & layer : turboq_layers) {
        io.write(&layer.il, sizeof(layer.il));
        io.write(&layer.n_head_kv, sizeof(layer.n_head_kv));
        io.write(&layer.k_code_row_bytes, sizeof(layer.k_code_row_bytes));
        io.write(&layer.k_sign_row_bytes, sizeof(layer.k_sign_row_bytes));
        io.write(&layer.v_code_row_bytes, sizeof(layer.v_code_row_bytes));

        turboq_write_rows<uint8_t>(io, layer.k_codes, layer.k_code_row_bytes, cr.data);
        turboq_write_rows<uint8_t>(io, layer.k_signs, layer.k_sign_row_bytes, cr.data);
        turboq_write_rows<ggml_fp16_t>(io, layer.k_norms, size_t(layer.n_head_kv) * sizeof(ggml_fp16_t), cr.data);
        turboq_write_rows<uint8_t>(io, layer.v_codes, layer.v_code_row_bytes, cr.data);
    }
}

bool llama_kv_cache_turboq::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    GGML_UNUSED(strm);

    uint32_t version = 0;
    uint32_t k_bits = 0;
    uint32_t v_bits = 0;
    uint32_t residual_bits = 0;
    uint32_t seed = 0;
    llama_turboq_rotation_type rotation = LLAMA_TURBOQ_ROTATION_TYPE_HADAMARD_PERMUTE_SIGN;
    uint32_t n_layer = 0;

    io.read_to(&version, sizeof(version));
    io.read_to(&k_bits, sizeof(k_bits));
    io.read_to(&v_bits, sizeof(v_bits));
    io.read_to(&residual_bits, sizeof(residual_bits));
    io.read_to(&seed, sizeof(seed));
    io.read_to(&rotation, sizeof(rotation));
    io.read_to(&n_layer, sizeof(n_layer));

    if (version != 1 ||
        k_bits != turboq_params.attn_k_bits ||
        v_bits != turboq_params.attn_v_bits ||
        residual_bits != turboq_params.attn_k_residual_bits ||
        seed != turboq_params.seed ||
        rotation != turboq_params.rotation ||
        n_layer != turboq_layers.size()) {
        LLAMA_LOG_ERROR("%s: incompatible TurboQ KV state\n", __func__);
        return false;
    }

    for (auto & layer : turboq_layers) {
        uint32_t il = 0;
        uint32_t n_head_kv = 0;
        size_t k_code_row_bytes = 0;
        size_t k_sign_row_bytes = 0;
        size_t v_code_row_bytes = 0;

        io.read_to(&il, sizeof(il));
        io.read_to(&n_head_kv, sizeof(n_head_kv));
        io.read_to(&k_code_row_bytes, sizeof(k_code_row_bytes));
        io.read_to(&k_sign_row_bytes, sizeof(k_sign_row_bytes));
        io.read_to(&v_code_row_bytes, sizeof(v_code_row_bytes));

        if (il != layer.il ||
            n_head_kv != layer.n_head_kv ||
            k_code_row_bytes != layer.k_code_row_bytes ||
            k_sign_row_bytes != layer.k_sign_row_bytes ||
            v_code_row_bytes != layer.v_code_row_bytes) {
            LLAMA_LOG_ERROR("%s: mismatched TurboQ layer payload\n", __func__);
            return false;
        }

        turboq_read_rows<uint8_t>(io, layer.k_codes, layer.k_code_row_bytes, cell_count, sinfo);
        turboq_read_rows<uint8_t>(io, layer.k_signs, layer.k_sign_row_bytes, cell_count, sinfo);
        turboq_read_rows<ggml_fp16_t>(io, layer.k_norms, size_t(layer.n_head_kv) * sizeof(ggml_fp16_t), cell_count, sinfo);
        turboq_read_rows<uint8_t>(io, layer.v_codes, layer.v_code_row_bytes, cell_count, sinfo);
    }

    return true;
}
