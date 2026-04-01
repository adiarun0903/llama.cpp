#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4;
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 64 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(32));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static llama_model_ptr make_model(const llm_arch arch, bool moe, size_t seed) {
    auto gguf = get_gguf_ctx(arch, moe);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = nullptr;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &tmp, model_params));
    if (!model) {
        throw std::runtime_error("failed to create in-memory hybrid test model");
    }

    return model;
}

static llama_context_ptr make_turboq_context(llama_model * model) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 128;
    ctx_params.n_batch = 64;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = 4;
    ctx_params.n_threads = 2;
    ctx_params.n_threads_batch = 2;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.memory_codec = LLAMA_MEMORY_CODEC_TURBOQ;
    ctx_params.kv_unified = true;
    ctx_params.offload_kqv = false;
    ctx_params.turboq.attn_k_bits = 3;
    ctx_params.turboq.attn_v_bits = 3;
    ctx_params.turboq.recurrent_r_bits = 3;
    ctx_params.turboq.recurrent_s_bits = 3;
    ctx_params.turboq.attn_k_residual_bits = 1;
    ctx_params.turboq.seed = 7;
    ctx_params.turboq.rotation = LLAMA_TURBOQ_ROTATION_TYPE_HADAMARD_PERMUTE_SIGN;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        throw std::runtime_error("failed to create TurboQ hybrid test context");
    }

    return ctx;
}

static void require(bool cond, const char * msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

static void decode_prompt(llama_context * ctx, const int n_seqs, const int prompt_len, const llama_token token) {
    llama_batch batch = llama_batch_init(n_seqs * prompt_len, 0, 1);
    for (int pos = 0; pos < prompt_len; ++pos) {
        for (int seq = 0; seq < n_seqs; ++seq) {
            common_batch_add(batch, token, pos, { seq }, false);
        }
    }

    require(llama_decode(ctx, batch) == 0, "failed to decode hybrid prompt batch");
    llama_batch_free(batch);
}

static void decode_step(llama_context * ctx, const std::vector<llama_seq_id> & seqs, llama_pos pos, llama_token token) {
    llama_batch batch = llama_batch_init(seqs.size(), 0, 1);
    for (llama_seq_id seq : seqs) {
        common_batch_add(batch, token, pos, { seq }, false);
    }

    require(llama_decode(ctx, batch) == 0, "failed to decode hybrid continuation batch");
    llama_batch_free(batch);
}

static std::vector<uint8_t> get_seq_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size(ctx, seq_id);
    std::vector<uint8_t> data(size);
    const size_t copied = llama_state_seq_get_data(ctx, data.data(), data.size(), seq_id);
    require(copied == data.size(), "failed to serialize seq state");
    return data;
}

static void set_seq_state(llama_context * ctx, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    const size_t copied = llama_state_seq_set_data(ctx, data.data(), data.size(), seq_id);
    require(copied == data.size(), "failed to restore seq state");
}

static std::vector<uint8_t> get_full_state(llama_context * ctx) {
    const size_t size = llama_state_get_size(ctx);
    std::vector<uint8_t> data(size);
    const size_t copied = llama_state_get_data(ctx, data.data(), data.size());
    require(copied == data.size(), "failed to serialize full TurboQ state");
    return data;
}

static void set_full_state(llama_context * ctx, const std::vector<uint8_t> & data) {
    const size_t copied = llama_state_set_data(ctx, data.data(), data.size());
    require(copied == data.size(), "failed to restore full TurboQ state");
}

static void run_case(const llm_arch arch, bool moe) {
    auto model = make_model(arch, moe, 1234);
    auto ctx = make_turboq_context(model.get());
    auto * mem = llama_get_memory(ctx.get());

    constexpr int prompt_len = 8;
    decode_prompt(ctx.get(), 3, prompt_len, 1);

    const llama_pos baseline_pos = llama_memory_seq_pos_max(mem, 1);
    require(baseline_pos == prompt_len - 1, "unexpected baseline position for hybrid seq");

    llama_memory_seq_cp(mem, 1, 3, -1, -1);
    require(llama_memory_seq_pos_max(mem, 3) == baseline_pos, "seq_cp did not clone hybrid seq tail");

    llama_memory_seq_add(mem, 3, -1, -1, 4);
    require(llama_memory_seq_pos_max(mem, 3) == baseline_pos + 4, "seq_add did not update destination seq");
    require(llama_memory_seq_pos_max(mem, 1) == baseline_pos, "seq_add mutated the source seq after seq_cp");

    llama_memory_seq_div(mem, 3, -1, -1, 2);
    require(llama_memory_seq_pos_max(mem, 3) == (baseline_pos + 4) / 2, "seq_div did not update destination seq");
    require(llama_memory_seq_pos_max(mem, 1) == baseline_pos, "seq_div mutated the source seq after seq_cp");

    llama_memory_seq_keep(mem, 3);
    require(llama_memory_seq_pos_max(mem, 0) == -1, "seq_keep retained seq 0 unexpectedly");
    require(llama_memory_seq_pos_max(mem, 1) == -1, "seq_keep retained seq 1 unexpectedly");
    require(llama_memory_seq_pos_max(mem, 2) == -1, "seq_keep retained seq 2 unexpectedly");

    const auto seq3_state = get_seq_state(ctx.get(), 3);
    set_seq_state(ctx.get(), 2, seq3_state);
    const llama_pos shared_pos = llama_memory_seq_pos_max(mem, 3);
    require(llama_memory_seq_pos_max(mem, 2) == shared_pos, "seq-scoped restore did not rebuild destination seq");

    const auto full_state = get_full_state(ctx.get());

    auto ctx2 = make_turboq_context(model.get());
    auto * mem2 = llama_get_memory(ctx2.get());
    set_full_state(ctx2.get(), full_state);

    require(llama_memory_seq_pos_max(mem2, 2) == shared_pos, "full state restore lost seq 2");
    require(llama_memory_seq_pos_max(mem2, 3) == shared_pos, "full state restore lost seq 3");

    decode_step(ctx2.get(), { 2, 3 }, shared_pos + 1, 2);

    require(llama_memory_seq_rm(mem2, 2, -1, -1), "seq_rm failed for restored hybrid seq");
    require(llama_memory_seq_pos_max(mem2, 2) == -1, "seq_rm did not clear restored hybrid seq");

    decode_step(ctx2.get(), { 3 }, shared_pos + 2, 3);
}

int main() {
    common_init();

    try {
        run_case(LLM_ARCH_QWEN35, false);
        run_case(LLM_ARCH_QWEN35MOE, true);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "test-turboq-hybrid-state: %s\n", err.what());
        return 1;
    }

    std::printf("test-turboq-hybrid-state: all tests OK\n");
    return 0;
}
