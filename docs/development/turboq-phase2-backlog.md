# TurboQ Phase-2 Backlog

This file is the fallback issue set because repository issues are disabled on `adiarun0903/llama.cpp`.

## P2-1: Dedicated Metal TurboQ kernels

Owner: TBD  
Priority: High

Scope:
- Implement dedicated Metal kernels for TurboQ:
  - attention decode,
  - attention-K residual correction,
  - recurrent load/store.
- Remove CPU fallback reliance on Apple Silicon for TurboQ hot paths.

Acceptance:
- TurboQ critical ops execute on Metal-native kernels.
- Throughput gain vs CPU fallback is measured and documented.
- No regression in TurboQ correctness tests.

## P2-2: MLA/SWA TurboQ variants

Owner: TBD  
Priority: High

Scope:
- Extend TurboQ memory integration to MLA and SWA cache variants.
- Keep fail-fast diagnostics for unsupported combinations during rollout.

Acceptance:
- MLA/SWA-compatible models run in TurboQ mode without fallback errors.
- Integration + restore tests cover new paths.
- Unsupported combinations fail fast with explicit error messages.

## P2-3: GLM4.7 hard-gate onboarding

Owner: TBD  
Priority: Medium

Scope:
- Add GLM4.7 as a hard acceptance gate model.
- Reuse Qwen/gpt-oss benchmark matrix structure.

Acceptance:
- GLM4.7 legacy vs TurboQ matrix runs end-to-end.
- Throughput, perplexity, and quality deltas are included in the gate report.
- Gate status is included in release checklist criteria.
