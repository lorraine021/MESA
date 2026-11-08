#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-test_model}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[FATAL] Invalid model path: $MODEL_PATH" >&2; exit 1; }
VLLM_BIN="${VLLM_BIN:-vllm}"
EXTRA_ARGS=()
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-AUTO}"
if [[ "$ATTENTION_BACKEND" != AUTO ]]; then
  EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
else
  unset VLLM_ATTENTION_BACKEND
fi
[[ "${EVAL_ENFORCE_EAGER:-0}" == 1 ]] && EXTRA_ARGS+=(--enforce-eager)
[[ "${EVAL_DISABLE_CUSTOM_ALL_REDUCE:-0}" == 1 ]] && EXTRA_ARGS+=(--disable-custom-all-reduce)
exec "$VLLM_BIN" serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" --tensor-parallel-size "$TP_SIZE" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" "${EXTRA_ARGS[@]}" --port "$VLLM_PORT"
