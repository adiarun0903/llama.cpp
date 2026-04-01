# TurboQ Memory

This fork adds a TurboQ-style runtime memory codec for `llama.cpp`.

## Scope

The current v2 foundation rebuild includes:

- a new public memory API built around `memory_codec`
- TurboQ attention surfaces for dense decoder-only models
- unified TurboQ memory routing for dense and hybrid Qwen3.5-style models
- multi-sequence save/load state parity for dense TurboQ
- speculative decoding examples enabled on TurboQ memory
- runtime memory compression only; the GGUF model file is unchanged

Still pending for the full rebuild:

- dedicated TurboQ recurrent `r/s` packed surfaces
- dedicated TurboQ Metal attention and recurrent kernels
- MLA and SWA cache variants

Metal is the production backend target. The current CPU path remains the correctness oracle and fallback implementation for validation.

## CLI

Use the new memory codec selector plus the per-surface TurboQ bitwidth flags:

```bash
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  --memory-codec turboq \
  --turboq-attn-k-bits 3 \
  --turboq-attn-v-bits 3 \
  --turboq-recurrent-r-bits 3 \
  --turboq-recurrent-s-bits 3 \
  -fa on \
  -c 2048 \
  -n 128 \
  -no-cnv -st \
  -p "The meaning of life is"
```

TurboQ rejects:

- `--cache-type-k` / `--cache-type-v` values other than legacy `f16`
- unsupported architectures such as MLA, encoder-decoder, pure recurrent, and SWA cache variants

## State Save / Load

The `llama-save-load-state` example supports TurboQ v2 state round-trips, including the seq-scoped copy/restore path:

```bash
./build/bin/llama-save-load-state \
  -m /path/to/model.gguf \
  --memory-codec turboq \
  --turboq-attn-k-bits 3 \
  --turboq-attn-v-bits 3 \
  --turboq-recurrent-r-bits 3 \
  --turboq-recurrent-s-bits 3 \
  -n 8
```

TurboQ v2 state blobs use a new wrapper format and explicitly reject TurboQ v1 state payloads.

## Speculative Decode

The `llama-speculative-simple` example can run on TurboQ memory directly:

```bash
./build/bin/llama-speculative-simple \
  -m /path/to/model.gguf \
  --model-draft /path/to/draft.gguf \
  --memory-codec turboq \
  --turboq-attn-k-bits 3 \
  --turboq-attn-v-bits 3 \
  --turboq-recurrent-r-bits 3 \
  --turboq-recurrent-s-bits 3 \
  -p "Hello" \
  -n 32
```

## Benchmark Workflow

Use [`scripts/bench-turboq-kv.sh`](../../scripts/bench-turboq-kv.sh) to compare:

- legacy `f16/f16`
- legacy `q8_0/q8_0`
- legacy `q4_0/q4_0`
- TurboQ `3b/3b`

The script drives:

- `llama-cli` for prompt/decode throughput and runtime KV memory reporting
- optional `llama-perplexity` if an evaluation corpus is provided

Example:

```bash
./scripts/bench-turboq-kv.sh \
  --model /path/to/model.gguf \
  --prompt "The meaning of life is" \
  --ctx-size 2048 \
  --n-predict 128 \
  --perplexity-file ./wiki.test.raw
```

## Recommended Validation

- Verify dense TurboQ generation on a small GGUF first.
- Verify `llama-save-load-state` and `llama-speculative-simple` on TurboQ.
- Verify hybrid Qwen3.5 initialization separately before benchmarking.
- Expect the process to fail fast on unsupported architectures.
- Compare TurboQ against legacy `q4_0/q4_0` and `f16/f16` using the same seed and prompt.
