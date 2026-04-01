#pragma once

#include "llama-batch.h"
#include "llama-kv-cache-turboq.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-turboq-codec.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

struct llama_context;
class llama_memory_turboq_recurrent_context;

class llama_memory_turboq : public llama_memory_i {
public:
    llama_memory_turboq(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
        const llama_turboq_memory_params & params,
                     bool   enable_recurrent = false,
                ggml_type   recurrent_type_r = GGML_TYPE_F32,
                ggml_type   recurrent_type_s = GGML_TYPE_F32,
                 uint32_t   recurrent_size = 0,
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
    const layer_reuse_cb  & reuse = nullptr);

    ~llama_memory_turboq() override = default;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    bool has_recurrent() const;
    llama_kv_cache_turboq * get_mem_attn() const;

private:
    friend class llama_memory_turboq_recurrent_context;

    struct recurrent_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1;
        int32_t   src0 = -1;
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }
    };

    struct recurrent_layer_storage {
        int32_t il = -1;
        ggml_backend_buffer_type_t buft = nullptr;
        uint32_t seed = 0;
        uint32_t r_bits = 0;
        uint32_t s_bits = 0;
        uint32_t r_dim = 0;
        uint32_t s_dim = 0;
        size_t r_code_row_bytes = 0;
        size_t s_code_row_bytes = 0;

        llama_turboq::rotation_plan rot_r;
        llama_turboq::rotation_plan rot_s;

        ggml_tensor * r_codes = nullptr;
        ggml_tensor * r_norms = nullptr;
        ggml_tensor * s_codes = nullptr;
        ggml_tensor * s_norms = nullptr;
    };

    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_turboq> mem_attn;
    const bool recurrent_enabled;
    const bool recurrent_offload;
    const uint32_t recurrent_size;
    const uint32_t recurrent_n_seq_max;
    const layer_filter_cb recurrent_filter;

    uint32_t recurrent_head = 0;
    uint32_t recurrent_used = 0;
    uint32_t recurrent_n = 0;
    int32_t recurrent_rs_z = -1;

    std::vector<recurrent_cell> recurrent_cells;
    std::vector<recurrent_layer_storage> recurrent_layers;
    std::vector<uint8_t> recurrent_packed_store_r;
    std::vector<uint8_t> recurrent_packed_store_s;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> recurrent_ctxs_bufs;

    void clear_recurrent(bool data);

    bool recurrent_prepare(const std::vector<llama_ubatch> & ubatches);
    bool recurrent_find_slot(const llama_ubatch & ubatch);
    int32_t recurrent_find_empty_cell();
    int32_t recurrent_detach_seq_tail(llama_seq_id seq_id);

    bool recurrent_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void recurrent_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void recurrent_seq_keep(llama_seq_id seq_id);
    void recurrent_seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);
    void recurrent_seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
    llama_pos recurrent_seq_pos_min(llama_seq_id seq_id) const;
    llama_pos recurrent_seq_pos_max(llama_seq_id seq_id) const;

    size_t recurrent_total_packed_size() const;

    void recurrent_state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const;
    void recurrent_state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0);

    void recurrent_state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void recurrent_state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;
    bool recurrent_state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool recurrent_state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_turboq_context : public llama_memory_context_i, public llama_memory_attn_recurrent_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_turboq_context(llama_memory_status status);

    explicit llama_memory_turboq_context(llama_memory_turboq * mem);

    explicit llama_memory_turboq_context(
        llama_memory_turboq * mem,
              llama_context * lctx,
                       bool   optimize);

    llama_memory_turboq_context(
            llama_memory_turboq * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_turboq_context() override;

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_context * get_attn() const override;
    const llama_memory_recurrent_context_i * get_recr() const override;

private:
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    std::unique_ptr<llama_memory_turboq_recurrent_context> ctx_recr;

    const llama_memory_status status;
};
