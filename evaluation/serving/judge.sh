#!/usr/bin/env bash
set -euo pipefail
DEFAULT_OCTOPUS_MODEL_PATH=""
OCTOPUS_MODEL_PATH="${OCTOPUS_MODEL_PATH:-${1:-$DEFAULT_OCTOPUS_MODEL_PATH}}"
OCTOPUS_PORT="${OCTOPUS_PORT:-8002}"
OCTOPUS_TP_SIZE="${OCTOPUS_TP_SIZE:-4}"
[[ -f "$OCTOPUS_MODEL_PATH/config.json" ]] || { echo "[FATAL] Invalid Octopus model path: $OCTOPUS_MODEL_PATH" >&2; exit 1; }
VLLM_BIN="${VLLM_BIN:-vllm}"
EXTRA_ARGS=()
[[ "${EVAL_ENFORCE_EAGER:-0}" == 1 ]] && EXTRA_ARGS+=(--enforce-eager)
[[ "${OCTOPUS_DISABLE_CUSTOM_ALL_REDUCE:-${EVAL_DISABLE_CUSTOM_ALL_REDUCE:-0}}" == 1 ]] && EXTRA_ARGS+=(--disable-custom-all-reduce)
exec "$VLLM_BIN" serve "$OCTOPUS_MODEL_PATH" --served-model-name Octopus-SEval --tensor-parallel-size "$OCTOPUS_TP_SIZE" --max-model-len 32768 --gpu-memory-utilization 0.95 "${EXTRA_ARGS[@]}" --port "$OCTOPUS_PORT"
