#pragma once

#include "llama-kv-cache.h"
#include "llama-turboq-codec.h"

#include <deque>
#include <vector>

class llama_kv_cache_turboq : public llama_kv_cache {
public:
    llama_kv_cache_turboq(
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
        const layer_reuse_cb & reuse);

    ~llama_kv_cache_turboq() override = default;

    bool get_can_shift() const override;

    void clear(bool data) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, ggml_tensor * dep = nullptr) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo, ggml_tensor * dep = nullptr) const override;

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const override;

    ggml_tensor * get_k_corr(
            ggml_context * ctx,
             ggml_tensor * q,
                  int32_t   il,
            const slot_info & sinfo,
             ggml_tensor * dep = nullptr) const override;

    bool prefers_flash_attn() const override;
    bool force_cpu_kqv() const override;

    struct turboq_layer_storage {
        uint32_t il = 0;
        uint32_t n_head_kv = 0;
        uint32_t n_head = 0;
        uint32_t n_gqa = 0;
        uint32_t n_embd_head_k = 0;
        uint32_t n_embd_head_v = 0;
        uint32_t n_embd_k_gqa = 0;
        uint32_t n_embd_v_gqa = 0;
        uint32_t n_rot_k = 0;
        uint32_t n_rot_v = 0;
        uint32_t k_bits = 0;
        uint32_t v_bits = 0;

        size_t k_code_head_bytes = 0;
        size_t k_sign_head_bytes = 0;
        size_t v_code_head_bytes = 0;
        size_t k_code_row_bytes = 0;
        size_t k_sign_row_bytes = 0;
        size_t v_code_row_bytes = 0;

        llama_turboq::rotation_plan rot_k;
        llama_turboq::rotation_plan rot_v;

        ggml_tensor * k_codes = nullptr;
        ggml_tensor * k_signs = nullptr;
        ggml_tensor * k_norms = nullptr;
        ggml_tensor * v_codes = nullptr;
    };

    struct turboq_op_userdata {
        const llama_kv_cache_turboq * cache = nullptr;
        turboq_layer_storage * layer = nullptr;
        uint32_t n_kv = 0;
    };

    bool cell_is_empty(uint32_t cell) const;

private:
    const llama_turboq_memory_params turboq_params;
    std::deque<turboq_layer_storage> turboq_layers;
    mutable std::deque<turboq_op_userdata> op_userdata;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> turboq_ctxs_bufs;
    ggml_tensor * turboq_row_map = nullptr;

    int32_t find_empty_cell() const;
    int32_t detach_seq_cell(llama_kv_cells & cells, llama_seq_id seq_id, uint32_t src_cell);
    void clone_cell_data(uint32_t src_cell, uint32_t dst_cell);

    const turboq_layer_storage & layer_for(int32_t il) const;
    turboq_layer_storage & layer_for(int32_t il);
    const turboq_op_userdata * make_op_userdata(const turboq_layer_storage & layer, uint32_t n_kv) const;

    size_t total_packed_size() const;

    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};
