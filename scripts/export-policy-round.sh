#!/bin/bash
set -euo pipefail

if [ $# -ge 1 ]; then
  ROUND="$1"
else
  ROUND="${ROUND:-}"
fi

if [ -z "${ROUND}" ]; then
  echo "Usage: bash scripts/export-policy-round.sh <round>" >&2
  exit 1
fi

SLIME="${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
cd "$SLIME"
export PYTHONPATH="$SLIME${PYTHONPATH:+:$PYTHONPATH}"
ULTRAFEEDBACK_DIR="${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}"

TEST_DATA="${TEST_DATA:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}"
EXPECTED_EVAL_LINES="${EXPECTED_EVAL_LINES:-2000}"
if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES="$ACTUAL_EVAL_LINES"
  fi
fi

SFT_HF_DIR="${SFT_HF_DIR:-$SLIME/models/sft_checkpoint_8b_hf}"
ROUND_SAVE_DIR="${ROUND_SAVE_DIR:-$SLIME/models/save_dir_r${ROUND}}"
ROUND_POLICY_HF="${ROUND_POLICY_HF:-$SLIME/models/policy_r${ROUND}_hf}"
ROUND_OUTPUT="${ROUND_OUTPUT:-$SLIME/eval/outputs_policy_r${ROUND}.jsonl}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-$SLIME/models/qwen3-8B-base}"
SGLANG_PORT="${SGLANG_PORT:-30010}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-}"
MIN_FREE_DISK_GB_EXPORT="${MIN_FREE_DISK_GB_EXPORT:-${MIN_FREE_DISK_GB:-30}}"
MIN_FREE_DISK_GB_EVAL="${MIN_FREE_DISK_GB_EVAL:-${MIN_FREE_DISK_GB:-10}}"
KEEP_POLICY_HF="${KEEP_POLICY_HF:-0}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

mkdir -p "$SLIME/eval"

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

if [ ! -f "$ROUND_SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: missing checkpoint marker at $ROUND_SAVE_DIR/latest_checkpointed_iteration.txt" >&2
  exit 1
fi

output_ready=0
if [ -f "$ROUND_OUTPUT" ] && [ "$(awk 'END {print NR}' "$ROUND_OUTPUT")" -ge "$EXPECTED_EVAL_LINES" ]; then
  output_ready=1
fi

hf_ready=0
if [ -f "$ROUND_POLICY_HF/config.json" ]; then
  hf_ready=1
fi

if [ "$FORCE_EXPORT" -ne 1 ] && [ "$output_ready" -eq 1 ]; then
  echo "Round $ROUND eval output already complete at $ROUND_OUTPUT"
  if [ "$KEEP_POLICY_HF" -ne 1 ] && [ "$hf_ready" -eq 1 ]; then
    echo "=== Round $ROUND: removing stale HF export at $ROUND_POLICY_HF ==="
    rm -rf "$ROUND_POLICY_HF"
  fi
  exit 0
fi

ITER=$(cat "$ROUND_SAVE_DIR/latest_checkpointed_iteration.txt")
ITER_DIR="$ROUND_SAVE_DIR/iter_$(printf "%07d" "$ITER")"
if [ -f "$ITER_DIR/common.pt" ]; then
  CKPT_DIR="$ITER_DIR"
elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then
  CKPT_DIR="$ITER_DIR/mp_rank_00"
else
  echo "ERROR: common.pt not found under $ITER_DIR" >&2
  exit 1
fi

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

if [ "$FORCE_EXPORT" -eq 1 ] || [ "$hf_ready" -ne 1 ]; then
  echo "=== Round $ROUND: converting Megatron checkpoint to HF ==="
  check_disk_space "round${ROUND}-convert-hf" "$SLIME/models" "$MIN_FREE_DISK_GB_EXPORT"
  rm -rf "$ROUND_POLICY_HF"
  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir "$CKPT_DIR" \
    --output-dir "$ROUND_POLICY_HF" \
    --origin-hf-dir "$ORIGIN_HF_DIR" \
    --force
else
  echo "=== Round $ROUND: reusing existing HF export at $ROUND_POLICY_HF ==="
fi

if [ "$FORCE_EXPORT" -eq 1 ] || [ "$output_ready" -ne 1 ]; then
  echo "=== Round $ROUND: generating eval outputs ==="
  check_disk_space "round${ROUND}-eval-generate" "$SLIME/eval" "$MIN_FREE_DISK_GB_EVAL"
  generate_with_sglang "$ROUND_POLICY_HF" "$ROUND_OUTPUT"
else
  echo "=== Round $ROUND: eval output already complete at $ROUND_OUTPUT ==="
fi

if [ "$KEEP_POLICY_HF" -ne 1 ]; then
  echo "=== Round $ROUND: removing HF export at $ROUND_POLICY_HF ==="
  rm -rf "$ROUND_POLICY_HF"
fi

echo "Round $ROUND export finished"
echo "  checkpoint: $ROUND_SAVE_DIR"
if [ -d "$ROUND_POLICY_HF" ]; then
  echo "  hf_export:   $ROUND_POLICY_HF"
else
  echo "  hf_export:   <deleted>"
fi
echo "  eval_output: $ROUND_OUTPUT"
