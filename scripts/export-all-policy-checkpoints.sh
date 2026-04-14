#!/bin/bash
set -euo pipefail

SLIME="${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
cd "$SLIME"

SAVE_DIR="${SAVE_DIR:-$SLIME/models/save_dir_single_ppo_r1_reward}"
SAVE_TAG="${SAVE_TAG:-$(basename "$SAVE_DIR")}"
EXPORT_TAG="${EXPORT_TAG:-$SAVE_TAG}"
if [[ "$EXPORT_TAG" == save_dir_* ]]; then
  EXPORT_TAG="${EXPORT_TAG#save_dir_}"
fi
EVAL_DIR="${EVAL_DIR:-$SLIME/eval}"
LOG_DIR="${LOG_DIR:-$SLIME/logs/single_ppo_export}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
RUN_TAG="${RUN_TAG:-${EXPORT_TAG}_export_${RUN_TS}}"
RUN_LOG="$LOG_DIR/${RUN_TAG}.log"
RUN_ENV="$LOG_DIR/${RUN_TAG}.env"
STATUS_TSV="$LOG_DIR/${RUN_TAG}.status.tsv"
MANIFEST_PATH="${MANIFEST_PATH:-$EVAL_DIR/outputs_${EXPORT_TAG}_manifest.json}"
ROLLOUTS_CSV="${ROLLOUTS_CSV:-}"
ULTRAFEEDBACK_DIR="${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}"
TEST_DATA="${TEST_DATA:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}"
EXPECTED_EVAL_LINES=2000

mkdir -p "$EVAL_DIR" "$LOG_DIR"
: > "$STATUS_TSV"

cleanup_dir() {
  if command -v sudo >/dev/null 2>&1; then
    sudo rm -rf "$1"
  else
    rm -rf "$1"
  fi
}

if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES="$ACTUAL_EVAL_LINES"
  fi
fi

{
cat <<EOF
run_tag=$RUN_TAG
run_log=$RUN_LOG
save_dir=$SAVE_DIR
export_tag=$EXPORT_TAG
manifest_path=$MANIFEST_PATH
cwd=$(pwd)
hostname=$(hostname)
start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
rollouts_csv=${ROLLOUTS_CSV}
EOF
} > "$RUN_ENV"

exec > >(tee -a "$RUN_LOG") 2>&1

declare -a AVAILABLE_ROLLOUTS=()
declare -A ROLLOUT_TO_ITER=()

scan_checkpoints() {
  local iter_dir
  while IFS= read -r iter_dir; do
    [ -n "$iter_dir" ] || continue
    local iter_name=${iter_dir##*_}
    local iter_value=$((10#$iter_name))
    local rollout_count=$((iter_value + 1))
    AVAILABLE_ROLLOUTS+=("$rollout_count")
    ROLLOUT_TO_ITER["$rollout_count"]="$iter_value"
  done < <(find "$SAVE_DIR" -maxdepth 1 -type d -name 'iter_*' | sort)
}

parse_requested_rollouts() {
  REQUESTED_ROLLOUTS=()
  if [ -z "$ROLLOUTS_CSV" ]; then
    REQUESTED_ROLLOUTS=("${AVAILABLE_ROLLOUTS[@]}")
    return
  fi

  IFS=',' read -r -a raw_rollouts <<< "$ROLLOUTS_CSV"
  for rollout in "${raw_rollouts[@]}"; do
    rollout=$(echo "$rollout" | xargs)
    if ! [[ "$rollout" =~ ^[0-9]+$ ]] || [ "$rollout" -le 0 ]; then
      echo "ERROR: invalid rollout count in ROLLOUTS_CSV: $rollout" >&2
      exit 1
    fi
    if [ -z "${ROLLOUT_TO_ITER[$rollout]:-}" ]; then
      echo "ERROR: requested rollout $rollout not found under $SAVE_DIR" >&2
      exit 1
    fi
    REQUESTED_ROLLOUTS+=("$rollout")
  done
}

output_is_complete() {
  local path=$1
  [ -f "$path" ] || return 1
  local lines
  lines=$(awk 'END {print NR}' "$path")
  [ "$lines" -ge "$EXPECTED_EVAL_LINES" ]
}

write_manifest() {
  python3 - "$STATUS_TSV" "$MANIFEST_PATH" "$RUN_TAG" "$SAVE_DIR" "$EXPORT_TAG" "$EXPECTED_EVAL_LINES" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
run_tag = sys.argv[3]
save_dir = sys.argv[4]
export_tag = sys.argv[5]
expected_eval_lines = int(sys.argv[6])

entries = []
if status_path.exists():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollout_count, iter_value, output_path, log_path, status, line_count, checkpoint_tag = line.split("\t")
        entries.append(
            {
                "rollout_count": int(rollout_count),
                "iter": int(iter_value),
                "checkpoint_tag": checkpoint_tag,
                "output_path": output_path,
                "log_path": None if log_path == "-" else log_path,
                "status": status,
                "line_count": int(line_count),
            }
        )

payload = {
    "run_tag": run_tag,
    "save_dir": save_dir,
    "export_tag": export_tag,
    "expected_eval_lines": expected_eval_lines,
    "checkpoints": entries,
}
manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
PY
}

trap write_manifest EXIT

scan_checkpoints
if [ "${#AVAILABLE_ROLLOUTS[@]}" -eq 0 ]; then
  echo "ERROR: no iter_* checkpoints found under $SAVE_DIR" >&2
  exit 1
fi
parse_requested_rollouts

echo "[export-all] run_log=$RUN_LOG"
echo "[export-all] manifest=$MANIFEST_PATH"
echo "[export-all] save_dir=$SAVE_DIR"
echo "[export-all] export_tag=$EXPORT_TAG"
echo "[export-all] expected_eval_lines=$EXPECTED_EVAL_LINES"
echo "[export-all] available_rollouts=${AVAILABLE_ROLLOUTS[*]}"
echo "[export-all] requested_rollouts=${REQUESTED_ROLLOUTS[*]}"

for rollout in "${REQUESTED_ROLLOUTS[@]}"; do
  iter_value=${ROLLOUT_TO_ITER[$rollout]}
  checkpoint_tag="${EXPORT_TAG}_rollout${rollout}"
  output_path="$EVAL_DIR/outputs_${checkpoint_tag}.jsonl"
  checkpoint_log="$LOG_DIR/${RUN_TAG}_${checkpoint_tag}.log"
  policy_hf_dir="$SLIME/models/${checkpoint_tag}_hf"

  if output_is_complete "$output_path"; then
    line_count=$(awk 'END {print NR}' "$output_path")
    if [ -d "$policy_hf_dir" ]; then
      echo "Removing stale HF export at $policy_hf_dir"
      cleanup_dir "$policy_hf_dir"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$rollout" "$iter_value" "$output_path" "-" "skipped_existing" "$line_count" "$checkpoint_tag" >> "$STATUS_TSV"
    echo "===== Skip rollout $rollout (iter $(printf "%07d" "$iter_value")): found complete $output_path ====="
    continue
  fi

  echo "===== Export rollout $rollout (iter $(printf "%07d" "$iter_value")) ====="
  if SAVE_DIR="$SAVE_DIR" \
     EXPORT_TAG="$EXPORT_TAG" \
     OUTPUT_PATH="$output_path" \
     POLICY_HF_DIR="$policy_hf_dir" \
     bash scripts/export-policy-checkpoint.sh "$rollout" 2>&1 | tee "$checkpoint_log"; then
    line_count=$(awk 'END {print NR}' "$output_path")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$rollout" "$iter_value" "$output_path" "$checkpoint_log" "completed" "$line_count" "$checkpoint_tag" >> "$STATUS_TSV"
  else
    line_count=0
    if [ -f "$output_path" ]; then
      line_count=$(awk 'END {print NR}' "$output_path")
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$rollout" "$iter_value" "$output_path" "$checkpoint_log" "failed" "$line_count" "$checkpoint_tag" >> "$STATUS_TSV"
    exit 1
  fi
done

echo "[export-all] finished"
echo "[export-all] outputs live under $EVAL_DIR"
echo "[export-all] manifest written to $MANIFEST_PATH"
