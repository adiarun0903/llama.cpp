#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/build/bin"
COMPLETION_BIN="${BIN_DIR}/llama-completion"
CLI_BIN="${COMPLETION_BIN}"
if [[ ! -x "$CLI_BIN" ]]; then
    CLI_BIN="${BIN_DIR}/llama-cli"
fi
PPL_BIN="${BIN_DIR}/llama-perplexity"

MODEL=""
PROMPT="The meaning of life is"
CTX_SIZE=2048
N_PREDICT=128
SEED=123
TEMPERATURE=0.8
FLASH_ATTN="on"
THREADS=""
PERPLEXITY_FILE=""
OUT_DIR="${ROOT_DIR}/out/turboq-bench-$(date +"%Y%m%d-%H%M%S")"

usage() {
    cat <<EOF
Usage: $(basename "$0") --model PATH [options]

Compare legacy memory modes against TurboQ on a causal GGUF.

Options:
  --model PATH             GGUF model path (required)
  --prompt TEXT            Prompt for completion runs (default: ${PROMPT})
  --ctx-size N             Context size (default: ${CTX_SIZE})
  --n-predict N            Number of generated tokens (default: ${N_PREDICT})
  --seed N                 Sampling seed (default: ${SEED})
  --temp V                 Sampling temperature for completion runs (default: ${TEMPERATURE})
  --threads N              CPU threads to pass through
  --flash-attn on|off|auto Flash attention mode for llama-cli/perplexity (default: ${FLASH_ATTN})
  --perplexity-file PATH   Optional text file for llama-perplexity
  --out-dir PATH           Output directory (default: ${OUT_DIR})
  -h, --help               Show this help text

Notes:
  - TurboQ uses --memory-codec turboq and derives compressed surface types from the TurboQ params.
  - MLA and SWA cache variants remain unsupported on the TurboQ v2 path.
EOF
}

require_bin() {
    local path="$1"
    local name="$2"
    if [[ ! -x "$path" ]]; then
        echo "missing ${name}: $path" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --n-predict)
            N_PREDICT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --flash-attn)
            FLASH_ATTN="$2"
            shift 2
            ;;
        --perplexity-file)
            PERPLEXITY_FILE="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    usage >&2
    exit 1
fi

require_bin "$CLI_BIN" "$(basename "$CLI_BIN")"
if [[ -n "$PERPLEXITY_FILE" ]]; then
    require_bin "$PPL_BIN" "llama-perplexity"
fi

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
    -m "$MODEL"
    -c "$CTX_SIZE"
    --seed "$SEED"
    -fa "$FLASH_ATTN"
    --no-warmup
)

if [[ -n "$THREADS" ]]; then
    COMMON_ARGS+=(-t "$THREADS")
fi

run_cli_case() {
    local name="$1"
    shift

    local log_path="${OUT_DIR}/${name}.cli.log"
    local -a completion_mode_args=()
    if [[ "$(basename "$CLI_BIN")" == "llama-completion" ]]; then
        completion_mode_args+=(-no-cnv)
    fi
    echo "==> ${name} ($(basename "$CLI_BIN"))"
    set +e
    "$CLI_BIN" \
        "${COMMON_ARGS[@]}" \
        -p "$PROMPT" \
        -n "$N_PREDICT" \
        "${completion_mode_args[@]}" \
        --temp "$TEMPERATURE" \
        "$@" \
        > "$log_path" 2>&1
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        if rg -q "TurboQ v2 does not support sliding-window attention cache variants" "$log_path"; then
            echo "EXPECTED_UNSUPPORTED: ${name} (TurboQ + SWA model)"
            tail -n 40 "$log_path"
            return 0
        fi
        tail -n 80 "$log_path"
        return $rc
    fi

    tail -n 40 "$log_path"
}

run_ppl_case() {
    local name="$1"
    shift

    if [[ -z "$PERPLEXITY_FILE" ]]; then
        return 0
    fi

    local log_path="${OUT_DIR}/${name}.ppl.log"
    echo "==> ${name} (llama-perplexity)"
    "$PPL_BIN" \
        "${COMMON_ARGS[@]}" \
        -f "$PERPLEXITY_FILE" \
        "$@" \
        > "$log_path" 2>&1
    tail -n 40 "$log_path"
}

run_case() {
    local name="$1"
    shift
    run_cli_case "$name" "$@"
    run_ppl_case "$name" "$@"
}

run_case legacy-f16 --memory-codec legacy --cache-type-k f16 --cache-type-v f16
run_case legacy-q8_0 --memory-codec legacy --cache-type-k q8_0 --cache-type-v q8_0
run_case legacy-q4_0 --memory-codec legacy --cache-type-k q4_0 --cache-type-v q4_0
run_case turboq-3bit --memory-codec turboq --turboq-attn-k-bits 3 --turboq-attn-v-bits 3 --turboq-recurrent-r-bits 3 --turboq-recurrent-s-bits 3

echo
echo "TurboQ benchmark logs written to: ${OUT_DIR}"
