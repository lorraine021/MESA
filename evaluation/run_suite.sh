#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_DIR="$SCRIPT_DIR/${EVAL_ARCHITECTURE:?Set EVAL_ARCHITECTURE to deepseek or qwen}"
COMMON_DIR="$SCRIPT_DIR/common"
[[ "$EVAL_ARCHITECTURE" == deepseek || "$EVAL_ARCHITECTURE" == qwen ]] || exit 2
export NO_PROXY="localhost,127.0.0.1,0.0.0.0${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/eval_data}"
PYTHON=${EVAL_PYTHON:-python3}
TARGET_PORT="${TARGET_PORT:-8000}"
OCTOPUS_PORT="${OCTOPUS_PORT:-8002}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3}"
OCTOPUS_GPUS="${OCTOPUS_GPUS:-4,5,6,7}"
TARGET_TP_SIZE="${TARGET_TP_SIZE:-4}"
OCTOPUS_TP_SIZE="${OCTOPUS_TP_SIZE:-4}"
DEFAULT_OCTOPUS_MODEL_PATH=""
SERVICE_WAIT_SECONDS="${SERVICE_WAIT_SECONDS:-3600}"
MATH_JUDGE_MODE="${MATH_JUDGE_MODE:-exact}"
TARGET_PID=""
OCTOPUS_PID=""
RUN_DIR=""

usage() {
  echo "Usage: bash $0 MODEL_PATH RUN_NAME [OCTOPUS_MODEL_PATH]" >&2
  echo "       OCTOPUS_MODEL_PATH=/path/to/Octopus-SEval bash $0 MODEL_PATH RUN_NAME" >&2
  echo "       bash $0 --preflight-only" >&2
  echo "Default Octopus model: $DEFAULT_OCTOPUS_MODEL_PATH" >&2
}

log() { echo "[$(date '+%F %T')] $*"; }

stop_group() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  log "Stopping evaluation services."
  stop_group "$TARGET_PID"
  stop_group "$OCTOPUS_PID"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ -n "${EVAL_ACTIVATE:-}" ]]; then source "$EVAL_ACTIVATE"; fi
cd "$SCRIPT_DIR"

log "Validating local evaluation datasets."
DATA_ROOT="$DATA_ROOT" "$PYTHON" "$COMMON_DIR/validate_data.py"

if [[ "${1:-}" == "--preflight-only" ]]; then
  log "Dataset preflight completed."
  exit 0
fi

[[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
MODEL_PATH=$(readlink -f "$1")
RUN_NAME=$2
OCTOPUS_MODEL_PATH="${3:-${OCTOPUS_MODEL_PATH:-$DEFAULT_OCTOPUS_MODEL_PATH}}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "[FATAL] Invalid RUN_NAME: $RUN_NAME" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[FATAL] Invalid target model: $MODEL_PATH" >&2; exit 2; }
OCTOPUS_MODEL_PATH=$(readlink -f "$OCTOPUS_MODEL_PATH")
[[ -f "$OCTOPUS_MODEL_PATH/config.json" ]] || { echo "[FATAL] Invalid Octopus model: $OCTOPUS_MODEL_PATH" >&2; exit 2; }

RUN_DIR="${RESULTS_ROOT:-$SCRIPT_DIR/results}/$RUN_NAME"
GENERAL_DIR="$RUN_DIR/general"
SAFETY_DIR="$RUN_DIR/safety_generations"
mkdir -p "$GENERAL_DIR" "$SAFETY_DIR"

for port in "$TARGET_PORT" "$OCTOPUS_PORT"; do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "[FATAL] Port $port already has a healthy service. Stop it before running the suite." >&2
    exit 2
  fi
done

wait_service() {
  local name=$1 port=$2 pid=$3 logfile=$4 elapsed=0
  while (( elapsed < SERVICE_WAIT_SECONDS )); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      log "$name is healthy on port $port."
      return 0
    fi
    if grep -Eq 'Failed: Cuda error|Worker proc .* died unexpectedly|Engine core initialization failed' "$logfile"; then
      log "FATAL: $name worker failed during startup."
      tail -50 "$logfile" >&2
      return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[FATAL] $name exited during startup." >&2
      tail -100 "$logfile" >&2 || true
      return 1
    fi
    sleep 5
    elapsed=$((elapsed+5))
  done
  echo "[FATAL] $name startup timed out after ${SERVICE_WAIT_SECONDS}s." >&2
  tail -100 "$logfile" >&2 || true
  return 1
}

log "Starting target model on GPUs $TARGET_GPUS, port $TARGET_PORT."
: > "$RUN_DIR/target_vllm.log"
setsid env CUDA_VISIBLE_DEVICES="$TARGET_GPUS" MODEL_PATH="$MODEL_PATH" SERVED_MODEL_NAME=test_model VLLM_PORT="$TARGET_PORT" TP_SIZE="$TARGET_TP_SIZE"   bash "$SCRIPT_DIR/serving/target.sh" > "$RUN_DIR/target_vllm.log" 2>&1 &
TARGET_PID=$!

log "Starting Octopus on GPUs $OCTOPUS_GPUS, port $OCTOPUS_PORT."
: > "$RUN_DIR/octopus_vllm.log"
setsid env CUDA_VISIBLE_DEVICES="$OCTOPUS_GPUS" OCTOPUS_MODEL_PATH="$OCTOPUS_MODEL_PATH" OCTOPUS_PORT="$OCTOPUS_PORT" OCTOPUS_TP_SIZE="$OCTOPUS_TP_SIZE"   bash "$SCRIPT_DIR/serving/judge.sh" > "$RUN_DIR/octopus_vllm.log" 2>&1 &
OCTOPUS_PID=$!

wait_service "target vLLM" "$TARGET_PORT" "$TARGET_PID" "$RUN_DIR/target_vllm.log"
wait_service "Octopus vLLM" "$OCTOPUS_PORT" "$OCTOPUS_PID" "$RUN_DIR/octopus_vllm.log"

run_task() {
  local name=$1 output=$2
  shift 2
  if [[ -s "$output" ]]; then
    log "SKIP $name: output already exists: $output"
    return 0
  fi
  log "START $name"
  "$@"
  [[ -s "$output" ]] || { echo "[FATAL] $name produced no output: $output" >&2; return 1; }
  log "DONE $name"
}

run_task Math500 "$GENERAL_DIR/math500.json"   "$PYTHON" "$TASK_DIR/eval_math.py" --model-name test_model   --test-file "$DATA_ROOT/math500.jsonl" --dataset-type math500   --port "$TARGET_PORT" --judge-mode "$MATH_JUDGE_MODE" --output-file "$GENERAL_DIR/math500.json"

run_task GSM8K "$GENERAL_DIR/gsm8k.json"   "$PYTHON" "$TASK_DIR/eval_math.py" --model-name test_model   --test-file "$DATA_ROOT/gsm8k.parquet" --dataset-type gsm8k   --port "$TARGET_PORT" --judge-mode "$MATH_JUDGE_MODE" --output-file "$GENERAL_DIR/gsm8k.json"

run_task HumanEval "$GENERAL_DIR/humaneval.json"   "$PYTHON" "$TASK_DIR/eval_humaneval.py" --model-name test_model   --test-file "$DATA_ROOT/humaneval.parquet" --dataset-type humaneval   --port "$TARGET_PORT" --output-file "$GENERAL_DIR/humaneval.json"

run_task MBPP "$GENERAL_DIR/mbpp.json"   "$PYTHON" "$TASK_DIR/eval_humaneval.py" --model-name test_model   --test-file "$DATA_ROOT/mbpp.parquet" --dataset-type mbpp   --port "$TARGET_PORT" --output-file "$GENERAL_DIR/mbpp.json"

run_task GPQA "$GENERAL_DIR/gpqa.json"   "$PYTHON" "$TASK_DIR/eval_knowledge.py" --model-name test_model   --test-file "$DATA_ROOT/gpqa.csv" --port "$TARGET_PORT"   --output-file "$GENERAL_DIR/gpqa.json"

run_task MMLU "$GENERAL_DIR/mmlu.json"   "$PYTHON" "$TASK_DIR/eval_mmlu.py" --model-name test_model   --test-file "$DATA_ROOT/mmlu.parquet" --port "$TARGET_PORT"   --output-file "$GENERAL_DIR/mmlu.json"

run_task "WildJailbreak generation" "$SAFETY_DIR/wildjailbreak.json" \
  "$PYTHON" "$TASK_DIR/generate_harm.py" --model-name test_model --port "$TARGET_PORT" \
  --output-dir "$SAFETY_DIR" --plain-output-names --data-paths \
  "wildjailbreak:$DATA_ROOT/wildjailbreak.parquet"

run_task "StrongReject generation" "$SAFETY_DIR/strongreject.json" \
  "$PYTHON" "$TASK_DIR/generate_harm.py" --model-name test_model --port "$TARGET_PORT" \
  --output-dir "$SAFETY_DIR" --plain-output-names --data-paths \
  "strongreject:$DATA_ROOT/strongreject.parquet"

run_task "Strata SWORD generation" "$SAFETY_DIR/strata_sword_en_level_3.json" \
  "$PYTHON" "$TASK_DIR/generate_harm.py" --model-name test_model --port "$TARGET_PORT" \
  --output-dir "$SAFETY_DIR" --plain-output-names --data-paths \
  "strata_sword_en_level_3:$DATA_ROOT/strata_sword_en_level_3.jsonl"

log "START/RESUME Octopus safety evaluation"
"$PYTHON" "$COMMON_DIR/judge_safety.py" --model-name Octopus-SEval \
  --input-dir "$SAFETY_DIR" --port "$OCTOPUS_PORT" \
  --output-file "$RUN_DIR/safety_octopus.json"
log "DONE Octopus safety evaluation"

log "Building summary."
"$PYTHON" "$COMMON_DIR/build_summary.py" --run-dir "$RUN_DIR" --run-name "$RUN_NAME" --model-path "$MODEL_PATH"
log "Evaluation suite completed: $RUN_DIR"
