#include "llama-memory-turboq.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

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

static constexpr uint32_t LLAMA_TURBOQ_STATE_MAGIC   = 0x3271746du; // "mtq2"
static constexpr uint32_t LLAMA_TURBOQ_STATE_VERSION = 2;
static constexpr uint32_t LLAMA_TURBOQ_STATE_HAS_RECURRENT = 1u << 0;

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
static void turboq_read_rows(
        llama_io_read_i & io,
        ggml_tensor * tensor,
        size_t row_bytes,
        uint32_t cell_count,
        uint32_t head) {
    auto * base = reinterpret_cast<uint8_t *>(turboq_tensor_host_data<T>(tensor));
    if (cell_count == 0) {
        return;
    }

    io.read_to(base + head * row_bytes, cell_count * row_bytes);
}

} // namespace

class llama_memory_turboq_recurrent_context : public llama_memory_recurrent_context_i {
public:
    llama_memory_turboq_recurrent_context(llama_memory_turboq * mem, bool full);

    bool apply(const llama_ubatch & ubatch);
    bool flush();

    uint32_t get_n_rs() const override;
    uint32_t get_head() const override;
    int32_t  get_rs_z() const override;
    uint32_t get_size() const override;

    ggml_tensor * get_r_l(int32_t il) const override;
    ggml_tensor * get_s_l(int32_t il) const override;

    int32_t s_copy(int i) const override;

    const void * graph_reuse_key() const override;
    bool turboq_get_surface(int32_t il, bool is_r, turboq_surface & out) const override;
    void turboq_mark_store_surface(int32_t il, bool is_r) const override;

private:
    struct scratch_layer {
        ggml_tensor * r = nullptr;
        ggml_tensor * s = nullptr;
    };

    void allocate_scratch();
    bool decode_current();

    llama_memory_turboq * mem = nullptr;
    const bool full = false;
    const uint32_t capacity_rows = 0;

    uint32_t current_head = 0;
    uint32_t current_n = 0;
    bool dirty = false;

    std::vector<scratch_layer> scratch_layers;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
};

llama_memory_turboq_recurrent_context::llama_memory_turboq_recurrent_context(
        llama_memory_turboq * mem,
        bool full) :
    mem(mem),
    full(full),
    capacity_rows(std::max(1u, mem->recurrent_size)) {
    allocate_scratch();
    if (full) {
        current_n = capacity_rows;
    }
}

void llama_memory_turboq_recurrent_context::allocate_scratch() {
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*mem->hparams.n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context for TurboQ recurrent scratch");
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }

        return it->second.get();
    };

    scratch_layers.resize(mem->hparams.n_layer);

    for (const auto & layer : mem->recurrent_layers) {
        if (layer.il < 0) {
            continue;
        }

        ggml_context * ctx = ctx_for_buft(layer.buft);
        ggml_tensor * r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(layer.r_dim) * capacity_rows);
        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(layer.s_dim) * capacity_rows);
        ggml_format_name(r, "turboq_r_l%d", layer.il);
        ggml_format_name(s, "turboq_s_l%d", layer.il);
        scratch_layers[layer.il].r = r;
        scratch_layers[layer.il].s = s;
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for TurboQ recurrent scratch");
        }
        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

bool llama_memory_turboq_recurrent_context::decode_current() {
    const uint32_t n_rows = current_n;
    std::vector<uint8_t> decode_rows(capacity_rows, 0);

    for (uint32_t i = 0; i < n_rows; ++i) {
        const auto & cell = mem->recurrent_cells[current_head + i];
        const int32_t src_cell = cell.src0;
        if (src_cell >= 0 && src_cell != mem->recurrent_rs_z) {
            decode_rows[src_cell] = 1;
        }
    }

    for (const auto & layer : mem->recurrent_layers) {
        if (layer.il < 0) {
            continue;
        }

        const auto & r_codebook = llama_turboq::get_codebook(layer.r_bits);
        const auto & s_codebook = llama_turboq::get_codebook(layer.s_bits);
        std::vector<float> r_row(layer.r_dim, 0.0f);
        std::vector<float> s_row(layer.s_dim, 0.0f);
        const uint8_t * r_codes = turboq_tensor_host_data<uint8_t>(layer.r_codes);
        const uint8_t * s_codes = turboq_tensor_host_data<uint8_t>(layer.s_codes);
        const auto * r_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.r_norms);
        const auto * s_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.s_norms);

        for (uint32_t src_cell = 0; src_cell < capacity_rows; ++src_cell) {
            if (!decode_rows[src_cell]) {
                continue;
            }

            const uint8_t * r_code = r_codes + src_cell * layer.r_code_row_bytes;
            const uint8_t * s_code = s_codes + src_cell * layer.s_code_row_bytes;

            llama_turboq::decode_stage1_normed(
                    layer.rot_r,
                    r_codebook,
                    r_code,
                    r_norms[src_cell],
                    r_row.data());

            llama_turboq::decode_stage1_normed(
                    layer.rot_s,
                    s_codebook,
                    s_code,
                    s_norms[src_cell],
                    s_row.data());

            const size_t r_offset = size_t(src_cell) * layer.r_dim * sizeof(float);
            const size_t s_offset = size_t(src_cell) * layer.s_dim * sizeof(float);
            ggml_backend_tensor_set(scratch_layers[layer.il].r, r_row.data(), r_offset, r_row.size() * sizeof(float));
            ggml_backend_tensor_set(scratch_layers[layer.il].s, s_row.data(), s_offset, s_row.size() * sizeof(float));
        }
    }

    return true;
}

bool llama_memory_turboq_recurrent_context::apply(const llama_ubatch & ubatch) {
    if (full) {
        return true;
    }

    std::fill(mem->recurrent_packed_store_r.begin(), mem->recurrent_packed_store_r.end(), 0);
    std::fill(mem->recurrent_packed_store_s.begin(), mem->recurrent_packed_store_s.end(), 0);

    if (!mem->recurrent_find_slot(ubatch)) {
        return false;
    }

    current_head = mem->recurrent_head;
    current_n    = mem->recurrent_n;
    dirty = true;

    return decode_current();
}

bool llama_memory_turboq_recurrent_context::flush() {
    if (full || !dirty) {
        return true;
    }

    const uint32_t n_rows = current_n;

    for (auto & layer : mem->recurrent_layers) {
        if (layer.il < 0) {
            continue;
        }

        const bool skip_r = mem->recurrent_packed_store_r[layer.il] != 0;
        const bool skip_s = mem->recurrent_packed_store_s[layer.il] != 0;
        if (skip_r && skip_s) {
            continue;
        }

        std::vector<float> r_rows;
        std::vector<float> s_rows;

        if (!skip_r) {
            r_rows.assign(size_t(layer.r_dim) * n_rows, 0.0f);
            const size_t r_offset = size_t(current_head) * layer.r_dim * sizeof(float);
            ggml_backend_tensor_get(scratch_layers[layer.il].r, r_rows.data(), r_offset, r_rows.size() * sizeof(float));
        }
        if (!skip_s) {
            s_rows.assign(size_t(layer.s_dim) * n_rows, 0.0f);
            const size_t s_offset = size_t(current_head) * layer.s_dim * sizeof(float);
            ggml_backend_tensor_get(scratch_layers[layer.il].s, s_rows.data(), s_offset, s_rows.size() * sizeof(float));
        }

        const auto & r_codebook = llama_turboq::get_codebook(layer.r_bits);
        const auto & s_codebook = llama_turboq::get_codebook(layer.s_bits);
        uint8_t * r_codes = turboq_tensor_host_data<uint8_t>(layer.r_codes);
        uint8_t * s_codes = turboq_tensor_host_data<uint8_t>(layer.s_codes);
        auto * r_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.r_norms);
        auto * s_norms = turboq_tensor_host_data<ggml_fp16_t>(layer.s_norms);

        for (uint32_t i = 0; i < n_rows; ++i) {
            const uint32_t dst_cell = current_head + i;

            if (!skip_r) {
                llama_turboq::encode_stage1_normed(
                        layer.rot_r,
                        r_codebook,
                        r_rows.data() + size_t(i) * layer.r_dim,
                        r_codes + dst_cell * layer.r_code_row_bytes,
                        &r_norms[dst_cell]);
            }

            if (!skip_s) {
                llama_turboq::encode_stage1_normed(
                        layer.rot_s,
                        s_codebook,
                        s_rows.data() + size_t(i) * layer.s_dim,
                        s_codes + dst_cell * layer.s_code_row_bytes,
                        &s_norms[dst_cell]);
            }
        }
    }

    dirty = false;
    return true;
}

uint32_t llama_memory_turboq_recurrent_context::get_n_rs() const {
    return full ? capacity_rows : current_n;
}

uint32_t llama_memory_turboq_recurrent_context::get_head() const {
    return full ? 0 : current_head;
}

int32_t llama_memory_turboq_recurrent_context::get_rs_z() const {
    return full ? 0 : mem->recurrent_rs_z;
}

uint32_t llama_memory_turboq_recurrent_context::get_size() const {
    return capacity_rows;
}

ggml_tensor * llama_memory_turboq_recurrent_context::get_r_l(int32_t il) const {
    return scratch_layers[il].r;
}

ggml_tensor * llama_memory_turboq_recurrent_context::get_s_l(int32_t il) const {
    return scratch_layers[il].s;
}

int32_t llama_memory_turboq_recurrent_context::s_copy(int i) const {
    return full ? i : mem->recurrent_cells[i + current_head].src0;
}

const void * llama_memory_turboq_recurrent_context::graph_reuse_key() const {
    return static_cast<const void *>(mem);
}

bool llama_memory_turboq_recurrent_context::turboq_get_surface(int32_t il, bool is_r, turboq_surface & out) const {
    if (il < 0 || (size_t) il >= mem->recurrent_layers.size()) {
        return false;
    }

    const auto & layer = mem->recurrent_layers[il];
    if (layer.il < 0) {
        return false;
    }

    out.codes = is_r ? layer.r_codes : layer.s_codes;
    out.norms = is_r ? layer.r_norms : layer.s_norms;
    out.surface_kind = int32_t(is_r ? llama_turboq::surface_kind::recurrent_r : llama_turboq::surface_kind::recurrent_s);
    out.seed = int32_t(layer.seed);
    out.layer_index = layer.il;
    out.bits = int32_t(is_r ? layer.r_bits : layer.s_bits);
    out.dim = int32_t(is_r ? layer.r_dim : layer.s_dim);

    return out.codes != nullptr && out.norms != nullptr;
}

void llama_memory_turboq_recurrent_context::turboq_mark_store_surface(int32_t il, bool is_r) const {
    if (il < 0) {
        return;
    }

    auto & marks = is_r ? mem->recurrent_packed_store_r : mem->recurrent_packed_store_s;
    if ((size_t) il >= marks.size()) {
        return;
    }

    marks[il] = 1;
}

llama_memory_turboq::llama_memory_turboq(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
        const llama_turboq_memory_params & params,
                     bool   enable_recurrent,
                ggml_type   recurrent_type_r,
                ggml_type   recurrent_type_s,
                 uint32_t   recurrent_size,
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_reuse_cb  & reuse) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_turboq(
        model,
        type_k,
        type_v,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        params,
        filter_attn == nullptr ?
            (enable_recurrent ? layer_filter_cb([&](int32_t il) { return !hparams.is_recurrent(il); }) : layer_filter_cb(nullptr))
            : filter_attn,
        reuse
    )),
    recurrent_enabled(enable_recurrent),
    recurrent_offload(offload),
    recurrent_size(recurrent_size),
    recurrent_n_seq_max(n_seq_max),
    recurrent_filter(filter_recr == nullptr ?
        (enable_recurrent ? layer_filter_cb([&](int32_t il) { return hparams.is_recurrent(il); }) : layer_filter_cb(nullptr))
        : filter_recr) {
    GGML_UNUSED(recurrent_type_r);
    GGML_UNUSED(recurrent_type_s);

    if (!recurrent_enabled) {
        return;
    }

    recurrent_cells.resize(recurrent_size);
    recurrent_layers.resize(hparams.n_layer);
    recurrent_packed_store_r.assign(hparams.n_layer, 0);
    recurrent_packed_store_s.assign(hparams.n_layer, 0);

    ggml_init_params ggml_params = {
        /*.mem_size   =*/ size_t(4u*recurrent_layers.size()*ggml_tensor_overhead()),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr recurrent_ctx(ggml_init(ggml_params));
    if (!recurrent_ctx) {
        throw std::runtime_error("failed to create TurboQ recurrent packed storage context");
    }

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (recurrent_filter && !recurrent_filter(il)) {
            continue;
        }

        auto & layer = recurrent_layers[il];
        layer.il = int32_t(il);
        layer.buft = recurrent_offload ? ggml_backend_dev_buffer_type(model.dev_layer(il)) : ggml_backend_cpu_buffer_type();
        layer.seed = params.seed;
        layer.r_bits = params.recurrent_r_bits;
        layer.s_bits = params.recurrent_s_bits;
        layer.r_dim = hparams.n_embd_r();
        layer.s_dim = hparams.n_embd_s();
        layer.rot_r = llama_turboq::make_rotation_plan(params.seed, llama_turboq::surface_kind::recurrent_r, il, layer.r_dim);
        layer.rot_s = llama_turboq::make_rotation_plan(params.seed, llama_turboq::surface_kind::recurrent_s, il, layer.s_dim);
        layer.r_code_row_bytes = llama_turboq::bitpacked_bytes(layer.rot_r.padded_dim, layer.r_bits);
        layer.s_code_row_bytes = llama_turboq::bitpacked_bytes(layer.rot_s.padded_dim, layer.s_bits);
        layer.r_codes = ggml_new_tensor_2d(recurrent_ctx.get(), GGML_TYPE_I8, layer.r_code_row_bytes, recurrent_size);
        layer.r_norms = ggml_new_tensor_2d(recurrent_ctx.get(), GGML_TYPE_F16, 1, recurrent_size);
        layer.s_codes = ggml_new_tensor_2d(recurrent_ctx.get(), GGML_TYPE_I8, layer.s_code_row_bytes, recurrent_size);
        layer.s_norms = ggml_new_tensor_2d(recurrent_ctx.get(), GGML_TYPE_F16, 1, recurrent_size);

        ggml_format_name(layer.r_codes, "turboq_r_codes_l%d", layer.il);
        ggml_format_name(layer.r_norms, "turboq_r_norms_l%d", layer.il);
        ggml_format_name(layer.s_codes, "turboq_s_codes_l%d", layer.il);
        ggml_format_name(layer.s_norms, "turboq_s_norms_l%d", layer.il);
    }

    ggml_backend_buffer_t recurrent_buf = ggml_backend_alloc_ctx_tensors_from_buft(recurrent_ctx.get(), ggml_backend_cpu_buffer_type());
    if (!recurrent_buf) {
        throw std::runtime_error("failed to allocate TurboQ recurrent packed storage buffer");
    }

    ggml_backend_buffer_clear(recurrent_buf, 0);
    recurrent_ctxs_bufs.emplace_back(std::move(recurrent_ctx), recurrent_buf);
}

llama_memory_context_ptr llama_memory_turboq::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    if (!recurrent_enabled) {
        return mem_attn->init_batch(balloc, n_ubatch, embd_all);
    }

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch = embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, true);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (!recurrent_prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare TurboQ recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_turboq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_attn = mem_attn->prepare(ubatches);
        if (sinfos_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare TurboQ attention ubatches\n", __func__);
            return std::make_unique<llama_memory_turboq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_turboq_context>(this, std::move(sinfos_attn), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_turboq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_turboq::init_full() {
    if (!recurrent_enabled) {
        return mem_attn->init_full();
    }
    return std::make_unique<llama_memory_turboq_context>(this);
}

llama_memory_context_ptr llama_memory_turboq::init_update(llama_context * lctx, bool optimize) {
    if (!recurrent_enabled) {
        return mem_attn->init_update(lctx, optimize);
    }
    return std::make_unique<llama_memory_turboq_context>(this, lctx, optimize);
}

bool llama_memory_turboq::get_can_shift() const {
    return mem_attn->get_can_shift();
}

void llama_memory_turboq::clear(bool data) {
    mem_attn->clear(data);

    clear_recurrent(data);
}

void llama_memory_turboq::clear_recurrent(bool data) {
    if (!recurrent_enabled) {
        return;
    }

    std::fill(recurrent_packed_store_r.begin(), recurrent_packed_store_r.end(), 0);
    std::fill(recurrent_packed_store_s.begin(), recurrent_packed_store_s.end(), 0);

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        recurrent_cells[i].pos = -1;
        recurrent_cells[i].seq_id.clear();
        recurrent_cells[i].src = -1;
        recurrent_cells[i].src0 = -1;
        recurrent_cells[i].tail = -1;
    }

    recurrent_head = 0;
    recurrent_used = 0;
    recurrent_n = 0;
    recurrent_rs_z = -1;

    if (data) {
        for (auto & [_, buf] : recurrent_ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_memory_turboq::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (recurrent_enabled && !recurrent_seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_turboq::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (recurrent_enabled) {
        recurrent_seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_turboq::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    if (recurrent_enabled) {
        recurrent_seq_keep(seq_id);
    }
}

void llama_memory_turboq::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    if (recurrent_enabled) {
        recurrent_seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_turboq::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    if (recurrent_enabled) {
        recurrent_seq_div(seq_id, p0, p1, d);
    }
}

llama_pos llama_memory_turboq::seq_pos_min(llama_seq_id seq_id) const {
    if (!recurrent_enabled) {
        return mem_attn->seq_pos_min(seq_id);
    }
    return std::max(mem_attn->seq_pos_min(seq_id), recurrent_seq_pos_min(seq_id));
}

llama_pos llama_memory_turboq::seq_pos_max(llama_seq_id seq_id) const {
    if (!recurrent_enabled) {
        return mem_attn->seq_pos_max(seq_id);
    }
    return std::min(mem_attn->seq_pos_max(seq_id), recurrent_seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_turboq::memory_breakdown() const {
    auto mb = mem_attn->memory_breakdown();
    if (recurrent_enabled) {
        mb[ggml_backend_cpu_buffer_type()] += recurrent_total_packed_size();
    }
    return mb;
}

void llama_memory_turboq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const uint32_t magic = LLAMA_TURBOQ_STATE_MAGIC;
    const uint32_t version = LLAMA_TURBOQ_STATE_VERSION;
    const uint32_t surfaces = recurrent_enabled ? LLAMA_TURBOQ_STATE_HAS_RECURRENT : 0u;

    io.write(&magic, sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&surfaces, sizeof(surfaces));

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    if (recurrent_enabled) {
        recurrent_state_write(io, seq_id, flags);
    }
}

void llama_memory_turboq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t surfaces = 0;

    io.read_to(&magic, sizeof(magic));
    io.read_to(&version, sizeof(version));
    io.read_to(&surfaces, sizeof(surfaces));

    if (magic != LLAMA_TURBOQ_STATE_MAGIC || version != LLAMA_TURBOQ_STATE_VERSION) {
        throw std::runtime_error("TurboQ v2 state format mismatch; TurboQ v1 states are not supported");
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }

    if (surfaces & LLAMA_TURBOQ_STATE_HAS_RECURRENT) {
        if (!recurrent_enabled) {
            throw std::runtime_error("TurboQ v2 state contains recurrent surfaces, but the active model does not");
        }
        recurrent_state_read(io, seq_id, flags);
    }
}

bool llama_memory_turboq::has_recurrent() const {
    return recurrent_enabled;
}

llama_kv_cache_turboq * llama_memory_turboq::get_mem_attn() const {
    return mem_attn.get();
}

bool llama_memory_turboq::recurrent_prepare(const std::vector<llama_ubatch> & ubatches) {
    auto org_cells = recurrent_cells;
    auto org_used = recurrent_used;
    auto org_head = recurrent_head;
    auto org_n = recurrent_n;
    auto org_rs_z = recurrent_rs_z;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!recurrent_find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    recurrent_cells = std::move(org_cells);
    recurrent_used = org_used;
    recurrent_head = org_head;
    recurrent_n = org_n;
    recurrent_rs_z = org_rs_z;

    return success;
}

int32_t llama_memory_turboq::recurrent_find_empty_cell() {
    if (recurrent_size == 0) {
        return -1;
    }

    uint32_t cell = recurrent_head;
    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if (cell >= recurrent_size) {
            cell -= recurrent_size;
        }
        if (recurrent_cells[cell].is_empty()) {
            return int32_t(cell);
        }
        cell += 1;
    }

    return -1;
}

int32_t llama_memory_turboq::recurrent_detach_seq_tail(llama_seq_id seq_id) {
    if (seq_id < 0 || (uint32_t) seq_id >= recurrent_size) {
        return -1;
    }

    auto & seq_meta = recurrent_cells[seq_id];
    const int32_t tail_id = seq_meta.tail;
    if (tail_id < 0) {
        return -1;
    }

    auto & tail_cell = recurrent_cells[tail_id];
    if (tail_cell.seq_id.size() <= 1) {
        return tail_id;
    }

    const int32_t new_cell_id = recurrent_find_empty_cell();
    GGML_ASSERT(new_cell_id >= 0 && "TurboQ recurrent copy-on-write ran out of cells");

    auto & new_cell = recurrent_cells[new_cell_id];
    GGML_ASSERT(new_cell.is_empty());

    new_cell.pos = tail_cell.pos;
    new_cell.src = tail_cell.src >= 0 ? tail_cell.src : tail_id;
    new_cell.src0 = new_cell.src;
    new_cell.seq_id.insert(seq_id);

    tail_cell.seq_id.erase(seq_id);
    seq_meta.tail = new_cell_id;
    recurrent_used += 1;

    if ((uint32_t) new_cell_id < recurrent_head) {
        recurrent_head = new_cell_id;
    }

    return new_cell_id;
}

bool llama_memory_turboq::recurrent_find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    if (recurrent_head > recurrent_used + 2*n_seqs) {
        recurrent_head = 0;
    }

    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = recurrent_size - 1;
    int32_t max = 0;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= recurrent_size) {
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, recurrent_n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = recurrent_cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = recurrent_cells[seq.tail];
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        cell.src0 = -1;
                        recurrent_used -= 1;
                    }
                }
            }
        }
    }

    uint32_t next_empty_cell = recurrent_head;

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if (next_empty_cell >= recurrent_size) {
            next_empty_cell -= recurrent_size;
        }
        auto & cell = recurrent_cells[next_empty_cell];
        if (cell.is_empty()) {
            break;
        }
        next_empty_cell += 1;
    }

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = recurrent_cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = recurrent_cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            if (cell.seq_id.size() == 1) {
                has_cell = true;
            }
        }
        if (!has_cell) {
            auto & empty_cell = recurrent_cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            if (seq_meta.tail >= 0) {
                auto & orig_cell = recurrent_cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id);
                GGML_ASSERT(!orig_cell.is_empty());
            }
            seq_meta.tail = next_empty_cell;
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < recurrent_size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= recurrent_size) {
                        next_empty_cell -= recurrent_size;
                    }
                    auto & cell = recurrent_cells[next_empty_cell];
                    if (cell.is_empty()) {
                        break;
                    }
                }
            }
        }
        if (min > seq_meta.tail) {
            min = seq_meta.tail;
        }
        if (max < seq_meta.tail) {
            max = seq_meta.tail;
        }
    }

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = recurrent_cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = recurrent_cells[dst_id];
            auto & src_cell = recurrent_cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.src0, src_cell.src0);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            for (uint32_t j = 0; j < recurrent_size; ++j) {
                int32_t & tail = recurrent_cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = recurrent_cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            recurrent_cells[seq_id].tail = cell_id;
        }
    }

    {
        std::vector<int32_t> refcounts(recurrent_size, 0);
        for (size_t i = 0; i < recurrent_size; ++i) {
            const int32_t src = recurrent_cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        recurrent_rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                recurrent_rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (recurrent_cells[i].src < 0) {
                recurrent_cells[i].src0 = recurrent_rs_z;
            } else {
                recurrent_cells[i].src0 = recurrent_cells[i].src;
            }
            recurrent_cells[i].src = i;
        }
    }

    recurrent_head = min;
    recurrent_n    = max - min + 1;
    recurrent_used = std::count_if(recurrent_cells.begin(), recurrent_cells.end(),
            [](const recurrent_cell & cell) { return !cell.is_empty(); });

    return recurrent_n >= n_seqs;
}

bool llama_memory_turboq::recurrent_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = recurrent_size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= (int64_t) recurrent_size) {
        return false;
    }

    if (0 <= seq_id) {
        int32_t & tail_id = recurrent_cells[seq_id].tail;
        if (tail_id >= 0) {
            const auto & cell = recurrent_cells[tail_id];
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                return false;
            }
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
        return false;
    }

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if (recurrent_cells[i].pos >= p0 && recurrent_cells[i].pos < p1) {
            if (seq_id < 0) {
                recurrent_cells[i].seq_id.clear();
            } else if (recurrent_cells[i].has_seq_id(seq_id)) {
                recurrent_cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (recurrent_cells[i].is_empty()) {
                if (recurrent_cells[i].pos >= 0) {
                    recurrent_used--;
                }
                recurrent_cells[i].pos = -1;
                recurrent_cells[i].src = -1;
                recurrent_cells[i].src0 = -1;
                if (new_head == recurrent_size) {
                    new_head = i;
                }
            }
        }
    }

    if (new_head != recurrent_size && new_head < recurrent_head) {
        recurrent_head = new_head;
    }

    return true;
}

void llama_memory_turboq::recurrent_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);

    if (seq_id_src == seq_id_dst) {
        return;
    }

    if ((uint32_t) seq_id_dst < recurrent_size && (uint32_t) seq_id_src < recurrent_size) {
        auto & tail_src = recurrent_cells[seq_id_src];
        auto & tail_dst = recurrent_cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            auto & cell_dst = recurrent_cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                cell_dst.src0 = -1;
                recurrent_used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = recurrent_cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_turboq::recurrent_seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = recurrent_size;

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            recurrent_cells[i].tail = -1;
        }

        if (!recurrent_cells[i].has_seq_id(seq_id)) {
            if (recurrent_cells[i].pos >= 0) {
                recurrent_used--;
            }

            recurrent_cells[i].pos = -1;
            recurrent_cells[i].src = -1;
            recurrent_cells[i].src0 = -1;
            recurrent_cells[i].seq_id.clear();

            if (new_head == recurrent_size) {
                new_head = i;
            }
        } else {
            recurrent_cells[i].seq_id.clear();
            recurrent_cells[i].seq_id.insert(seq_id);
        }
    }

    if (new_head != recurrent_size && new_head < recurrent_head) {
        recurrent_head = new_head;
    }
}

void llama_memory_turboq::recurrent_seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
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

    if (0 <= seq_id && seq_id < (int64_t) recurrent_size) {
        const int32_t tail_id = recurrent_detach_seq_tail(seq_id);
        if (tail_id >= 0) {
            auto & cell = recurrent_cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_turboq::recurrent_seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
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

    if (0 <= seq_id && seq_id < (int64_t) recurrent_size) {
        const int32_t tail_id = recurrent_detach_seq_tail(seq_id);
        if (tail_id >= 0) {
            auto & cell = recurrent_cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_turboq::recurrent_seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if (recurrent_cells[i].has_seq_id(seq_id)) {
            result = std::min(result, recurrent_cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_turboq::recurrent_seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < recurrent_size; ++i) {
        if (recurrent_cells[i].has_seq_id(seq_id)) {
            result = std::max(result, recurrent_cells[i].pos);
        }
    }

    return result;
}

size_t llama_memory_turboq::recurrent_total_packed_size() const {
    size_t total = 0;
    for (const auto & layer : recurrent_layers) {
        if (layer.il < 0) {
            continue;
        }
        total += ggml_nbytes(layer.r_codes);
        total += ggml_nbytes(layer.r_norms);
        total += ggml_nbytes(layer.s_codes);
        total += ggml_nbytes(layer.s_norms);
    }
    return total;
}

void llama_memory_turboq::recurrent_state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges;
    uint32_t cell_count = 0;

    uint32_t cell_range_begin = recurrent_size;
    for (uint32_t i = 0; i < recurrent_size; ++i) {
        const auto & cell = recurrent_cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            if (cell_range_begin == recurrent_size) {
                cell_range_begin = i;
            }
        } else if (cell_range_begin != recurrent_size) {
            cell_ranges.emplace_back(cell_range_begin, i);
            cell_range_begin = recurrent_size;
        }
    }
    if (cell_range_begin != recurrent_size) {
        cell_ranges.emplace_back(cell_range_begin, recurrent_size);
    }

    io.write(&cell_count, sizeof(cell_count));
    recurrent_state_write_meta(io, cell_ranges, seq_id);
    recurrent_state_write_data(io, cell_ranges);
}

void llama_memory_turboq::recurrent_state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count = 0;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;
    res = res && recurrent_state_read_meta(io, cell_count, seq_id);
    res = res && recurrent_state_read_data(io, cell_count);

    if (!res) {
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore TurboQ recurrent state");
    }
}

void llama_memory_turboq::recurrent_state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = recurrent_cells[i];
            const llama_pos pos = cell.pos;
            const uint32_t n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos, sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto id : cell.seq_id) {
                    io.write(&id, sizeof(id));
                }
            }
        }
    }
}

void llama_memory_turboq::recurrent_state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t n_layer = hparams.n_layer;
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & layer = recurrent_layers[il];
        const uint32_t present = layer.il >= 0 ? 1u : 0u;
        io.write(&present, sizeof(present));
        if (!present) {
            continue;
        }

        io.write(&layer.r_bits, sizeof(layer.r_bits));
        io.write(&layer.s_bits, sizeof(layer.s_bits));
        io.write(&layer.r_dim, sizeof(layer.r_dim));
        io.write(&layer.s_dim, sizeof(layer.s_dim));

        const uint64_t r_code_row_bytes = layer.r_code_row_bytes;
        const uint64_t s_code_row_bytes = layer.s_code_row_bytes;
        io.write(&r_code_row_bytes, sizeof(r_code_row_bytes));
        io.write(&s_code_row_bytes, sizeof(s_code_row_bytes));

        turboq_write_rows<uint8_t>(io, layer.r_codes, layer.r_code_row_bytes, cell_ranges);
        turboq_write_rows<ggml_fp16_t>(io, layer.r_norms, sizeof(ggml_fp16_t), cell_ranges);
        turboq_write_rows<uint8_t>(io, layer.s_codes, layer.s_code_row_bytes, cell_ranges);
        turboq_write_rows<ggml_fp16_t>(io, layer.s_norms, sizeof(ggml_fp16_t), cell_ranges);
    }
}

bool llama_memory_turboq::recurrent_state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        recurrent_seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());
        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos, sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic TurboQ recurrent cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!recurrent_find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available TurboQ recurrent cells\n", __func__);
            return false;
        }

        GGML_ASSERT(recurrent_head + cell_count <= recurrent_size);
    } else {
        if (cell_count > recurrent_size) {
            LLAMA_LOG_ERROR("%s: not enough TurboQ recurrent cells\n", __func__);
            return false;
        }

        clear_recurrent(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = recurrent_cells[i];

            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos, sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= recurrent_n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, recurrent_n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = recurrent_cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        recurrent_head = 0;
        recurrent_used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        const uint32_t cell_id = recurrent_head + i;
        recurrent_cells[cell_id].src = cell_id;
        recurrent_cells[cell_id].src0 = cell_id;
    }

    recurrent_n = cell_count;
    recurrent_rs_z = -1;

    return true;
}

bool llama_memory_turboq::recurrent_state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t n_layer = 0;
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer) {
        LLAMA_LOG_ERROR("%s: mismatched recurrent layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
        return false;
    }
    if (cell_count > recurrent_size) {
        LLAMA_LOG_ERROR("%s: not enough TurboQ recurrent cells to restore state (%u > %u)\n", __func__, cell_count, recurrent_size);
        return false;
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        uint32_t present = 0;
        io.read_to(&present, sizeof(present));

        const auto & layer = recurrent_layers[il];
        if (!present) {
            if (layer.il >= 0) {
                LLAMA_LOG_ERROR("%s: missing TurboQ recurrent layer %u in state\n", __func__, il);
                return false;
            }
            continue;
        }

        if (layer.il < 0) {
            LLAMA_LOG_ERROR("%s: state contains TurboQ recurrent layer %u that the active model does not expose\n", __func__, il);
            return false;
        }

        uint32_t r_bits = 0, s_bits = 0, r_dim = 0, s_dim = 0;
        uint64_t r_code_row_bytes = 0, s_code_row_bytes = 0;
        io.read_to(&r_bits, sizeof(r_bits));
        io.read_to(&s_bits, sizeof(s_bits));
        io.read_to(&r_dim, sizeof(r_dim));
        io.read_to(&s_dim, sizeof(s_dim));
        io.read_to(&r_code_row_bytes, sizeof(r_code_row_bytes));
        io.read_to(&s_code_row_bytes, sizeof(s_code_row_bytes));

        if (r_bits != layer.r_bits || s_bits != layer.s_bits || r_dim != layer.r_dim || s_dim != layer.s_dim ||
                r_code_row_bytes != layer.r_code_row_bytes || s_code_row_bytes != layer.s_code_row_bytes) {
            LLAMA_LOG_ERROR("%s: mismatched TurboQ recurrent layer metadata at layer %u\n", __func__, il);
            return false;
        }

        turboq_read_rows<uint8_t>(io, recurrent_layers[il].r_codes, layer.r_code_row_bytes, cell_count, recurrent_head);
        turboq_read_rows<ggml_fp16_t>(io, recurrent_layers[il].r_norms, sizeof(ggml_fp16_t), cell_count, recurrent_head);
        turboq_read_rows<uint8_t>(io, recurrent_layers[il].s_codes, layer.s_code_row_bytes, cell_count, recurrent_head);
        turboq_read_rows<ggml_fp16_t>(io, recurrent_layers[il].s_norms, sizeof(ggml_fp16_t), cell_count, recurrent_head);
    }

    return true;
}

llama_memory_turboq_context::llama_memory_turboq_context(llama_memory_status status) : status(status) {
}

llama_memory_turboq_context::llama_memory_turboq_context(llama_memory_turboq * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->has_recurrent() ? std::make_unique<llama_memory_turboq_recurrent_context>(mem, true) : nullptr),
    status(ctx_attn->get_status()) {
}

llama_memory_turboq_context::llama_memory_turboq_context(
        llama_memory_turboq * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    status(ctx_attn->get_status()) {
}

llama_memory_turboq_context::llama_memory_turboq_context(
         llama_memory_turboq * mem,
              slot_info_vec_t   sinfos_attn,
    std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(mem->has_recurrent() ? std::make_unique<llama_memory_turboq_recurrent_context>(mem, false) : nullptr),
    status(ctx_attn->get_status()) {
}

llama_memory_turboq_context::~llama_memory_turboq_context() = default;

bool llama_memory_turboq_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (ctx_recr && !ctx_recr->flush()) {
        return false;
    }

    ctx_attn->next();
    ++i_next;
    return i_next < ubatches.size();
}

bool llama_memory_turboq_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = ctx_attn->apply();
    if (ctx_recr && !ubatches.empty()) {
        res = res && ctx_recr->apply(ubatches[i_next]);
    }
    return res;
}

llama_memory_status llama_memory_turboq_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_turboq_context::get_ubatch() const {
    GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_turboq_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context_i * llama_memory_turboq_context::get_recr() const {
    return ctx_recr.get();
}
