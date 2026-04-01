# TurboQ v1 CPU Gate Report (Private Fork)

Date: 2026-03-31
Branch: `turboq/v1-rebase-port`
Head: `turboq/v1-rebase-port` (post-rebase port + stabilization)

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
7. (current branch) hybrid graph reuse guard + gate harness expected-unsupported handling + updated gate report

## Model Availability (Gate Inputs)

- Qwen3.5 GGUF: available
  - `~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf`
- gpt-oss-20b GGUF: available
  - `~/.lmstudio/models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf`
  - source: `ggml-org/gpt-oss-20b-GGUF`

## Test Status

Passed:
- `test-turboq-codec`
- `test-turboq-backend`
- `test-arg-parser`
- `test-turboq-hybrid-state`

Notes:
- Hybrid/recurrent path fixes were required in `llama-graph.cpp` plus model recurrent call-site updates (`mamba-base.cpp`, `lfm2.cpp`, `plamo2.cpp`).
- Additional stabilization fix applied for hybrid graph reuse checks in `llama-graph.cpp` to prevent stale-context dereference in long generation runs.

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

## Long-Run Stability Check (Qwen, n_predict=128)

- Command:
  - `scripts/bench-turboq-kv.sh --model ...Qwen3.5-35B-A3B-UD-Q4_K_L.gguf --n-predict 128 --ctx-size 2048 --flash-attn on`
- Outcome:
  - **No segfault after hybrid can_reuse guard fix**.
  - All four modes completed in one pass (`legacy-f16`, `legacy-q8_0`, `legacy-q4_0`, `turboq-3bit`).
- Artifacts:
  - `/tmp/qwen-gate-run-fix1`

## gpt-oss Gate (SWA / unsupported TurboQ variant)

- Command:
  - `scripts/bench-turboq-kv.sh --model ...gpt-oss-20b-mxfp4.gguf --n-predict 16 --ctx-size 2048 --flash-attn on`
- Outcome:
  - `legacy-f16`, `legacy-q8_0`, `legacy-q4_0` complete.
  - `turboq-3bit` returns explicit fail-fast:
    - `TurboQ v2 does not support sliding-window attention cache variants`
  - Harness now marks this as `EXPECTED_UNSUPPORTED` and keeps logs/reports instead of aborting.
- Artifacts:
  - `/tmp/gptoss-gate-run-fix1`

## Acceptance Status

- Supported-v1 target (dense + hybrid recurrent without SWA/MLA): **green on Qwen long-run path**.
- Unsupported architecture policy: **verified fail-fast on gpt-oss SWA path**.
- Full quality gate remains **partial**:
  - Perplexity delta (`<= 1%`) not yet executed in this report.
  - Fixed prompt quality review still smoke-level only.

## Plan Impact

The rebase-and-port plan is implemented with runtime stability fixes and gate harness updates. CPU-first v1 behavior is now consistent with scope: supported paths run, unsupported SWA path fails fast with clear messaging.

## Immediate Next Steps

1. Run perplexity matrix for Qwen baseline vs TurboQ (`legacy f16/q8_0/q4_0` vs `turboq 3b/3b`) and attach numeric deltas.
2. Execute fixed prompt quality set for final sign-off notes.
3. Keep SWA/MLA TurboQ support in phase-2 backlog (`docs/development/turboq-phase2-backlog.md`).
