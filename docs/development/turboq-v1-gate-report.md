# TurboQ v1 CPU Gate Report (Private Fork)

Date: 2026-03-31
Branch: `turboq/v1-rebase-port`
Head: `2a67da061`

## Scope

Implemented per v1 CPU-first plan:
- TurboQ memory codec public API + CLI contract.
- GGML TurboQ ops + CPU backend integration.
- Runtime memory integration for dense/hybrid recurrent paths.
- State I/O wiring into save/load + speculative example path.
- Test + docs + benchmark harness.

## Commit Stack

1. `7daf4b96f` turboq: add memory codec API and CLI contract
2. `8d263b39e` turboq: add ggml turboq ops and cpu backend kernels
3. `f25b3b34e` turboq: integrate turboq memory paths into runtime and models
4. `8ea5a544d` turboq: wire turboq state i/o into save-load and speculative examples
5. `28c3f4a21` turboq: add tests docs and benchmark harness
6. `2a67da061` turboq: fix recurrent surface paths and benchmark harness compatibility

## Model Availability (Gate Inputs)

- Qwen3.5 GGUF: available
  - `~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf`
- gpt-oss-20b GGUF: **not present locally**
  - only safetensors found at `~/.lmstudio/models/mlx-community/gpt-oss-20b-MXFP4-Q8/`

## Test Status

Passed:
- `test-turboq-codec`
- `test-turboq-backend`
- `test-arg-parser`
- `test-turboq-hybrid-state`

Notes:
- Hybrid/recurrent path fixes were required in `llama-graph.cpp` plus model recurrent call-site updates (`mamba-base.cpp`, `lfm2.cpp`, `plamo2.cpp`).

## Qwen Smoke Metrics (n=1 token, fixed seed, flash-attn on, no warmup)

These are smoke checks only (not full quality/perplexity gate):

| Mode | Prompt eval tokens/s | MTL0 self memory (MiB) |
|---|---:|---:|
| legacy-f16 | 112.11 | 19850 |
| legacy-q8_0 | 113.79 | 19832 |
| legacy-q4_0 | 19.50 | 19822 |
| turboq-3bit | 18.14 | 19748 |

Interpretation:
- TurboQ reduces context-memory footprint vs legacy f16/q8/q4 in this smoke configuration.
- Throughput is currently below f16/q8 in this run shape.

## Blocking Issues for Full Acceptance Gate

1. `gpt-oss` hard gate cannot run without a GGUF artifact.
2. Longer generation runs (`n_predict=16` and above) are unstable/intermittent on this branch for some modes (segfault observed in benchmark flow).
3. Perplexity delta gate (`<= 1%`) has not been completed with the intended benchmark matrix.

## Plan Impact

The integration plan is implemented as code + tests + harness on top of latest upstream master, but **acceptance gate is not yet green** due missing model artifact and runtime instability under longer runs.

## Immediate Next Steps

1. Provide/convert `gpt-oss` GGUF and rerun matrix.
2. Stabilize long-run generation path (repro harness retained in `scripts/bench-turboq-kv.sh`).
3. Complete perplexity sweep and fixed prompt quality sanity set, then finalize release tag.
