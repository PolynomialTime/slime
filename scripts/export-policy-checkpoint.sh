#!/bin/bash
set -euo pipefail

if [ $# -ge 1 ]; then
  ROLLOUT_COUNT="$1"
else
  ROLLOUT_COUNT="${ROLLOUT_COUNT:-}"
fi

if [ -z "${ROLLOUT_COUNT}" ]; then
  echo "Usage: bash scripts/export-policy-checkpoint.sh <rollout_count>" >&2
  exit 1
fi

if ! [[ "$ROLLOUT_COUNT" =~ ^[0-9]+$ ]] || [ "$ROLLOUT_COUNT" -le 0 ]; then
  echo "ERROR: rollout_count must be a positive integer, got: $ROLLOUT_COUNT" >&2
  exit 1
fi

SLIME="${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
cd "$SLIME"
export PYTHONPATH="$SLIME${PYTHONPATH:+:$PYTHONPATH}"

ULTRAFEEDBACK_DIR="${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}"
SAVE_DIR="${SAVE_DIR:-$SLIME/models/save_dir_single_ppo_r1_reward}"
SAVE_TAG="${SAVE_TAG:-$(basename "$SAVE_DIR")}"
EXPORT_TAG="${EXPORT_TAG:-$SAVE_TAG}"
if [[ "$EXPORT_TAG" == save_dir_* ]]; then
  EXPORT_TAG="${EXPORT_TAG#save_dir_}"
fi
CHECKPOINT_TAG="${CHECKPOINT_TAG:-${EXPORT_TAG}_rollout${ROLLOUT_COUNT}}"
POLICY_HF_DIR="${POLICY_HF_DIR:-$SLIME/models/${CHECKPOINT_TAG}_hf}"
OUTPUT_PATH="${OUTPUT_PATH:-$SLIME/eval/outputs_${CHECKPOINT_TAG}.jsonl}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-$SLIME/models/qwen3-8B-base}"
TEST_DATA="${TEST_DATA:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}"
SGLANG_PORT="${SGLANG_PORT:-30010}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-}"
MIN_FREE_DISK_GB_EXPORT="${MIN_FREE_DISK_GB_EXPORT:-${MIN_FREE_DISK_GB:-30}}"
MIN_FREE_DISK_GB_EVAL="${MIN_FREE_DISK_GB_EVAL:-${MIN_FREE_DISK_GB:-10}}"
KEEP_POLICY_HF="${KEEP_POLICY_HF:-0}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

mkdir -p "$SLIME/eval"

run_rm() {
  if command -v sudo >/dev/null 2>&1; then
    sudo rm "$@"
  else
    rm "$@"
  fi
}

check_disk_space() {
  local stage=$1
  local path=${2:-$SLIME/models}
  local min_free_gb=${3:-$MIN_FREE_DISK_GB}
  mkdir -p "$path"
  local avail_gb
  avail_gb=$(df -Pk "$path" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if [ -z "$avail_gb" ]; then
    echo "WARNING: failed to determine free disk space before $stage (path=$path)"
    return 0
  fi
  echo "Disk check [$stage]: available=${avail_gb}GB required>=${min_free_gb}GB path=$path"
  if [ "$avail_gb" -lt "$min_free_gb" ]; then
    echo "ERROR: low disk space before $stage: available=${avail_gb}GB required>=${min_free_gb}GB path=$path" >&2
    exit 1
  fi
}

generate_with_sglang() {
  local model_path=$1
  local output_path=$2

  echo "Starting SGLang server on port $SGLANG_PORT for $model_path"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path "$model_path" \
    --port "$SGLANG_PORT" \
    --tp 4 \
    --host 127.0.0.1 \
    --trust-remote-code &
  local sglang_pid=$!

  cleanup() {
    kill "$sglang_pid" 2>/dev/null || true
    wait "$sglang_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  python3 scripts/eval_generate_sglang.py \
    --model-path "$model_path" \
    --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
    --prompt-data "$TEST_DATA" \
    --output "$output_path" \
    --temperature "$EVAL_TEMPERATURE" \
    --apply-chat-template \
    --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --max-new-tokens 1024 \
    --concurrency 256

  trap - EXIT
  cleanup
}

ITER=$((ROLLOUT_COUNT - 1))

EXPECTED_EVAL_LINES=2000
if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES="$ACTUAL_EVAL_LINES"
  fi
fi

ITER_DIR="$SAVE_DIR/iter_$(printf "%07d" "$ITER")"
if [ -f "$ITER_DIR/common.pt" ]; then
  CKPT_DIR="$ITER_DIR"
elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then
  CKPT_DIR="$ITER_DIR/mp_rank_00"
else
  echo "ERROR: common.pt not found under $ITER_DIR" >&2
  exit 1
fi

output_ready=0
if [ -f "$OUTPUT_PATH" ] && [ "$(awk 'END {print NR}' "$OUTPUT_PATH")" -ge "$EXPECTED_EVAL_LINES" ]; then
  output_ready=1
fi

hf_ready=0
if [ -f "$POLICY_HF_DIR/config.json" ]; then
  hf_ready=1
fi

if [ "$FORCE_EXPORT" -eq 1 ]; then
  run_rm -rf "$POLICY_HF_DIR"
  run_rm -f "$OUTPUT_PATH"
  hf_ready=0
  output_ready=0
fi

if [ "$output_ready" -eq 1 ]; then
  echo "Checkpoint rollout $ROLLOUT_COUNT output already complete at $OUTPUT_PATH"
  if [ "$KEEP_POLICY_HF" -ne 1 ] && [ "$hf_ready" -eq 1 ]; then
    echo "Removing stale HF export at $POLICY_HF_DIR"
    run_rm -rf "$POLICY_HF_DIR"
  fi
  exit 0
fi

if [ "$hf_ready" -ne 1 ]; then
  echo "=== Converting rollout $ROLLOUT_COUNT checkpoint to HF ==="
  check_disk_space "rollout${ROLLOUT_COUNT}-convert-hf" "$SLIME/models" "$MIN_FREE_DISK_GB_EXPORT"
  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir "$CKPT_DIR" \
    --output-dir "$POLICY_HF_DIR" \
    --origin-hf-dir "$ORIGIN_HF_DIR" \
    --force
else
  echo "=== Reusing existing HF export at $POLICY_HF_DIR ==="
fi

echo "=== Generating test outputs for rollout $ROLLOUT_COUNT ==="
check_disk_space "rollout${ROLLOUT_COUNT}-eval-generate" "$SLIME/eval" "$MIN_FREE_DISK_GB_EVAL"
generate_with_sglang "$POLICY_HF_DIR" "$OUTPUT_PATH"

if [ "$KEEP_POLICY_HF" -ne 1 ]; then
  echo "Removing HF export at $POLICY_HF_DIR"
  run_rm -rf "$POLICY_HF_DIR"
fi

echo "Checkpoint export finished"
echo "  export_tag:    $EXPORT_TAG"
echo "  save_dir:      $SAVE_DIR"
echo "  rollout_count: $ROLLOUT_COUNT"
echo "  iter:          $(printf "%07d" "$ITER")"
echo "  checkpoint:    $CKPT_DIR"
echo "  hf_export:     $POLICY_HF_DIR"
echo "  eval_output:   $OUTPUT_PATH"
