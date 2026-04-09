#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
NOTIFY_DIR=${NOTIFY_DIR:-/mnt/shared-storage-gpfs2/wangqianyi2/slime-autoresearch-notify}
MODEL=${MODEL:-gpt-4o}
SMOKE_COUNT=${SMOKE_COUNT:-50}
NUM_WORKERS=${NUM_WORKERS:-4}
PER_WORKER_CONCURRENCY=${PER_WORKER_CONCURRENCY:-4}
PER_WORKER_BATCH_SIZE=${PER_WORKER_BATCH_SIZE:-8}
SMOKE_WORKER_CONCURRENCY=${SMOKE_WORKER_CONCURRENCY:-$PER_WORKER_CONCURRENCY}
SMOKE_WORKER_BATCH_SIZE=${SMOKE_WORKER_BATCH_SIZE:-$PER_WORKER_BATCH_SIZE}
FULL_WORKER_CONCURRENCY=${FULL_WORKER_CONCURRENCY:-$PER_WORKER_CONCURRENCY}
FULL_WORKER_BATCH_SIZE=${FULL_WORKER_BATCH_SIZE:-$PER_WORKER_BATCH_SIZE}
MAX_TOKENS=${MAX_TOKENS:-1024}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
SLEEP_ON_FAILURE=${SLEEP_ON_FAILURE:-15}

cd "$SLIME"
mkdir -p "$ULTRAFEEDBACK_DIR" "$NOTIFY_DIR"

if [ -z "${OPENAI_BASE_URL:-}" ] || [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_BASE_URL / OPENAI_API_KEY must be set" >&2
  exit 1
fi
if [ "$NUM_WORKERS" -lt 1 ]; then
  echo "ERROR: NUM_WORKERS must be >= 1" >&2
  exit 1
fi

TRAIN_INPUT="$ULTRAFEEDBACK_DIR/uf-train-prefs.jsonl"
TRAIN_OUTPUT="$ULTRAFEEDBACK_DIR/uf-train-synth-chosen.jsonl"
TRAIN_SHARD_DIR="$ULTRAFEEDBACK_DIR/uf-train-synth-chosen.shards"
DIRTY_SOURCE="$ULTRAFEEDBACK_DIR/uf-train-prefs-dirty-smoke.jsonl"
SMOKE_INPUT="$ULTRAFEEDBACK_DIR/uf-train-prefs-dirty-smoke${SMOKE_COUNT}.jsonl"
SMOKE_OUTPUT="$ULTRAFEEDBACK_DIR/uf-train-synth-chosen-dirty-smoke${SMOKE_COUNT}.jsonl"
SMOKE_SHARD_DIR="$ULTRAFEEDBACK_DIR/uf-train-synth-chosen-dirty-smoke${SMOKE_COUNT}.shards"
ARCHIVE_DIR="$ULTRAFEEDBACK_DIR/archive"
SMOKE_REPORT="$NOTIFY_DIR/synth_teacher_smoke_${MODEL}.json"
FULL_REPORT="$NOTIFY_DIR/synth_teacher_full_${MODEL}.json"
STATE_FILE="$NOTIFY_DIR/synth_teacher_state_${MODEL}.json"

export ULTRAFEEDBACK_DIR NOTIFY_DIR MODEL SMOKE_COUNT NUM_WORKERS MAX_TOKENS REQUEST_TIMEOUT SLEEP_ON_FAILURE
export TRAIN_INPUT TRAIN_OUTPUT TRAIN_SHARD_DIR DIRTY_SOURCE SMOKE_INPUT SMOKE_OUTPUT SMOKE_SHARD_DIR ARCHIVE_DIR
export SMOKE_REPORT FULL_REPORT STATE_FILE

CURRENT_STAGE=init
CURRENT_INPUT=
CURRENT_SHARD_DIR=
CURRENT_MERGED_OUTPUT=
CHILD_PIDS=()

log() {
  echo "[$(date '+%F %T')] $*"
}

shard_file() {
  local shard_dir=$1
  local shard_index=$2
  local num_shards=$3
  printf '%s/part-%03d-of-%03d.jsonl' "$shard_dir" "$shard_index" "$num_shards"
}

shard_log_file() {
  local stage=$1
  local shard_index=$2
  printf '%s/synth_teacher_%s.%s.shard%02d.log' "$NOTIFY_DIR" "$MODEL" "$stage" "$shard_index"
}

shard_state_file() {
  local stage=$1
  local shard_index=$2
  printf '%s/synth_teacher_state_%s.%s.shard%02d.json' "$NOTIFY_DIR" "$MODEL" "$stage" "$shard_index"
}

compute_shard_total() {
  local total_lines=$1
  local num_shards=$2
  local shard_index=$3
  local base=$((total_lines / num_shards))
  local extra=$((total_lines % num_shards))
  local size=$base
  if [ "$shard_index" -lt "$extra" ]; then
    size=$((size + 1))
  fi
  echo "$size"
}

write_shard_state() {
  local state_path=$1
  local status=$2
  local stage=$3
  local shard_index=$4
  local done=$5
  local total=$6
  local input_path=$7
  local output_path=$8
  python3 - "$state_path" "$status" "$stage" "$shard_index" "$NUM_WORKERS" "$done" "$total" "$input_path" "$output_path" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
state = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "shard_index": int(sys.argv[4]),
    "num_shards": int(sys.argv[5]),
    "done": int(sys.argv[6]),
    "total": int(sys.argv[7]),
    "input": sys.argv[8],
    "output": sys.argv[9],
    "updated_at": datetime.now().strftime("%F %T"),
}
path.write_text(json.dumps(state, indent=2), encoding="utf-8")
PY
}

write_aggregate_state() {
  local stage=$1
  local status=$2
  local input_path=$3
  local shard_dir=$4
  local merged_output=$5
  local total_lines=0
  local done_lines=0
  local completed_shards=0
  local shard_index

  if [ -n "$input_path" ] && [ -f "$input_path" ]; then
    total_lines=$(wc -l < "$input_path")
  fi

  for shard_index in $(seq 0 $((NUM_WORKERS - 1))); do
    local shard_total=0
    local shard_done=0
    local output_path
    output_path=$(shard_file "$shard_dir" "$shard_index" "$NUM_WORKERS")
    if [ "$total_lines" -gt 0 ]; then
      shard_total=$(compute_shard_total "$total_lines" "$NUM_WORKERS" "$shard_index")
    fi
    if [ -f "$output_path" ]; then
      shard_done=$(wc -l < "$output_path")
    fi
    done_lines=$((done_lines + shard_done))
    if [ "$shard_done" -ge "$shard_total" ]; then
      completed_shards=$((completed_shards + 1))
    fi
  done

  python3 - "$STATE_FILE" "$status" "$stage" "$MODEL" "$NUM_WORKERS" "$done_lines" "$total_lines" "$completed_shards" "$input_path" "$shard_dir" "$merged_output" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
state = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "model": sys.argv[4],
    "workers": int(sys.argv[5]),
    "done": int(sys.argv[6]),
    "total": int(sys.argv[7]),
    "completed_shards": int(sys.argv[8]),
    "input": sys.argv[9],
    "shard_dir": sys.argv[10],
    "output": sys.argv[11],
    "updated_at": datetime.now().strftime("%F %T"),
}
path.write_text(json.dumps(state, indent=2), encoding="utf-8")
PY
}

kill_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  CHILD_PIDS=()
}

on_exit() {
  local rc=$?
  local status="exited:$rc"
  kill_children
  if [ "$rc" -eq 0 ] && [ -n "$CURRENT_INPUT" ] && [ -n "$CURRENT_MERGED_OUTPUT" ] && [ -f "$CURRENT_INPUT" ] && [ -f "$CURRENT_MERGED_OUTPUT" ]; then
    local input_lines output_lines
    input_lines=$(wc -l < "$CURRENT_INPUT")
    output_lines=$(wc -l < "$CURRENT_MERGED_OUTPUT")
    if [ "$input_lines" -eq "$output_lines" ]; then
      status=completed
    fi
  fi
  if [ -n "$CURRENT_INPUT" ] && [ -n "$CURRENT_SHARD_DIR" ]; then
    write_aggregate_state "$CURRENT_STAGE" "$status" "$CURRENT_INPUT" "$CURRENT_SHARD_DIR" "$CURRENT_MERGED_OUTPUT"
  fi
  log "runner exit rc=$rc status=$status stage=${CURRENT_STAGE}"
}

on_signal() {
  local sig=$1
  trap - EXIT
  kill_children
  if [ -n "$CURRENT_INPUT" ] && [ -n "$CURRENT_SHARD_DIR" ]; then
    write_aggregate_state "$CURRENT_STAGE" "signal:$sig" "$CURRENT_INPUT" "$CURRENT_SHARD_DIR" "$CURRENT_MERGED_OUTPUT"
  fi
  log "runner received $sig stage=${CURRENT_STAGE}"
  exit 1
}

trap on_exit EXIT
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

ensure_dirty_source() {
  python3 - <<'PY'
import json
import os
import re
from pathlib import Path

ultra = Path(os.environ['ULTRAFEEDBACK_DIR'])
source = ultra / 'uf-train-prefs-dirty-smoke.jsonl'
if source.exists():
    raise SystemExit(0)
full = ultra / 'uf-train-prefs.jsonl'
if not full.exists():
    raise SystemExit(f'missing {full}')
pat = re.compile(
    r"(i['’]?m happy to help|i understand your request|i understand your question|"
    r"thank you for your question|as an ai language model|i must inform you|"
    r"i do not have real[- ]time|i don't have real[- ]time|it would be best to check|"
    r"check (the|their) .* website|contact .* directly)",
    re.I,
)
count = 0
with full.open(encoding='utf-8') as fin, source.open('w', encoding='utf-8') as fout:
    for line in fin:
        row = json.loads(line)
        if pat.search(str(row.get('chosen', ''))[:500]):
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
print(f'created {source} with {count} dirty rows')
PY
}

write_smoke_input() {
  python3 - <<'PY'
from pathlib import Path
import os
count = int(os.environ['SMOKE_COUNT'])
source = Path(os.environ['DIRTY_SOURCE'])
out = Path(os.environ['SMOKE_INPUT'])
with source.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if i >= count:
            break
        fout.write(line)
print(f'wrote smoke input {out} with {count} rows')
PY
}

archive_existing_partial() {
  local output_path=$1
  local shard_dir=$2
  local existing_shards=0
  if [ ! -f "$output_path" ]; then
    return 0
  fi
  if [ -d "$shard_dir" ] && find "$shard_dir" -maxdepth 1 -type f -name 'part-*.jsonl' | grep -q .; then
    return 0
  fi
  mkdir -p "$ARCHIVE_DIR"
  local ts archived
  ts=$(date '+%Y%m%d-%H%M%S')
  archived="$ARCHIVE_DIR/uf-train-synth-chosen.single-worker.partial-${ts}.jsonl"
  mv "$output_path" "$archived"
  log "archived existing single-worker partial to $archived"
}

prepare_fresh_smoke() {
  rm -rf "$SMOKE_SHARD_DIR"
  rm -f "$SMOKE_OUTPUT" "$SMOKE_REPORT"
  local shard_index
  for shard_index in $(seq 0 $((NUM_WORKERS - 1))); do
    rm -f "$(shard_log_file smoke "$shard_index")"
    rm -f "$(shard_state_file smoke "$shard_index")"
  done
}

run_stage_worker() {
  local stage=$1
  local input_path=$2
  local shard_dir=$3
  local shard_index=$4
  local concurrency=$5
  local batch_size=$6
  local total_lines shard_total output_path state_path done

  mkdir -p "$shard_dir"
  total_lines=$(wc -l < "$input_path")
  shard_total=$(compute_shard_total "$total_lines" "$NUM_WORKERS" "$shard_index")
  output_path=$(shard_file "$shard_dir" "$shard_index" "$NUM_WORKERS")
  state_path=$(shard_state_file "$stage" "$shard_index")

  if [ "$shard_total" -eq 0 ]; then
    : > "$output_path"
    write_shard_state "$state_path" completed "$stage" "$shard_index" 0 0 "$input_path" "$output_path"
    log "stage=$stage shard=$shard_index/$NUM_WORKERS empty"
    return 0
  fi

  while true; do
    done=0
    if [ -f "$output_path" ]; then
      done=$(wc -l < "$output_path")
    fi
    write_shard_state "$state_path" running "$stage" "$shard_index" "$done" "$shard_total" "$input_path" "$output_path"
    if [ "$done" -ge "$shard_total" ]; then
      write_shard_state "$state_path" completed "$stage" "$shard_index" "$done" "$shard_total" "$input_path" "$output_path"
      log "stage=$stage shard=$shard_index/$NUM_WORKERS completed $done/$shard_total"
      break
    fi

    log "stage=$stage shard=$shard_index/$NUM_WORKERS progress=$done/$shard_total model=$MODEL concurrency=$concurrency batch=$batch_size"
    if ! python3 scripts/generate_synthetic_chosen.py \
      --input "$input_path" \
      --output "$output_path" \
      --model "$MODEL" \
      --concurrency "$concurrency" \
      --batch-size "$batch_size" \
      --max-tokens "$MAX_TOKENS" \
      --request-timeout "$REQUEST_TIMEOUT" \
      --num-shards "$NUM_WORKERS" \
      --shard-index "$shard_index" \
      --resume; then
      done=0
      if [ -f "$output_path" ]; then
        done=$(wc -l < "$output_path")
      fi
      write_shard_state "$state_path" retrying "$stage" "$shard_index" "$done" "$shard_total" "$input_path" "$output_path"
      log "stage=$stage shard=$shard_index/$NUM_WORKERS generator failed at $done/$shard_total; sleeping ${SLEEP_ON_FAILURE}s then resuming"
      sleep "$SLEEP_ON_FAILURE"
    fi
  done
}

stage_has_alive_workers() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

launch_stage_workers() {
  local stage=$1
  local input_path=$2
  local shard_dir=$3
  local merged_output=$4
  local concurrency=$5
  local batch_size=$6
  local shard_index log_path

  CURRENT_STAGE=$stage
  CURRENT_INPUT=$input_path
  CURRENT_SHARD_DIR=$shard_dir
  CURRENT_MERGED_OUTPUT=$merged_output
  CHILD_PIDS=()

  mkdir -p "$shard_dir"
  for shard_index in $(seq 0 $((NUM_WORKERS - 1))); do
    log_path=$(shard_log_file "$stage" "$shard_index")
    mkdir -p "$(dirname "$log_path")"
    echo "=== $(date '+%F %T') stage=$stage shard=$shard_index/$NUM_WORKERS ===" >> "$log_path"
    (
      run_stage_worker "$stage" "$input_path" "$shard_dir" "$shard_index" "$concurrency" "$batch_size"
    ) >> "$log_path" 2>&1 &
    CHILD_PIDS+=("$!")
  done

  write_aggregate_state "$stage" running "$input_path" "$shard_dir" "$merged_output"
  while stage_has_alive_workers; do
    write_aggregate_state "$stage" running "$input_path" "$shard_dir" "$merged_output"
    sleep 5
  done

  local rc=0 pid
  for pid in "${CHILD_PIDS[@]}"; do
    wait "$pid" || rc=1
  done
  CHILD_PIDS=()
  write_aggregate_state "$stage" joined "$input_path" "$shard_dir" "$merged_output"
  return "$rc"
}

verify_stage_shards() {
  local stage=$1
  local input_path=$2
  local shard_dir=$3
  local total_lines expected actual shard_index shard_path actual_sum=0 expected_sum=0

  total_lines=$(wc -l < "$input_path")
  for shard_index in $(seq 0 $((NUM_WORKERS - 1))); do
    expected=$(compute_shard_total "$total_lines" "$NUM_WORKERS" "$shard_index")
    shard_path=$(shard_file "$shard_dir" "$shard_index" "$NUM_WORKERS")
    if [ ! -f "$shard_path" ]; then
      log "ERROR: missing $stage shard output $shard_path"
      return 1
    fi
    actual=$(wc -l < "$shard_path")
    expected_sum=$((expected_sum + expected))
    actual_sum=$((actual_sum + actual))
    if [ "$actual" -ne "$expected" ]; then
      log "ERROR: $stage shard $shard_index expected $expected rows but found $actual"
      return 1
    fi
  done
  if [ "$expected_sum" -ne "$total_lines" ] || [ "$actual_sum" -ne "$total_lines" ]; then
    log "ERROR: $stage shard totals mismatch expected_sum=$expected_sum actual_sum=$actual_sum total=$total_lines"
    return 1
  fi
  log "$stage shard verification passed rows=$total_lines workers=$NUM_WORKERS"
}

merge_stage_shards() {
  local stage=$1
  local input_path=$2
  local shard_dir=$3
  local merged_output=$4
  local total_lines shard_index shard_path tmp_path merged_lines

  total_lines=$(wc -l < "$input_path")
  tmp_path="${merged_output}.tmp.$$"
  mkdir -p "$(dirname "$merged_output")"
  : > "$tmp_path"
  for shard_index in $(seq 0 $((NUM_WORKERS - 1))); do
    shard_path=$(shard_file "$shard_dir" "$shard_index" "$NUM_WORKERS")
    cat "$shard_path" >> "$tmp_path"
  done
  merged_lines=$(wc -l < "$tmp_path")
  if [ "$merged_lines" -ne "$total_lines" ]; then
    rm -f "$tmp_path"
    log "ERROR: $stage merge produced $merged_lines rows, expected $total_lines"
    return 1
  fi
  mv "$tmp_path" "$merged_output"
  log "$stage merge complete output=$merged_output rows=$merged_lines"
}

quality_report() {
  local input_path=$1
  local output_path=$2
  local report_path=$3
  local expected
  expected=$(wc -l < "$input_path")
  INPUT_PATH="$input_path" OUTPUT_PATH="$output_path" REPORT_PATH="$report_path" EXPECTED_LINES="$expected" python3 - <<'PY'
import json
import os
import re
import statistics
from pathlib import Path

inp = Path(os.environ['INPUT_PATH'])
out = Path(os.environ['OUTPUT_PATH'])
report = Path(os.environ['REPORT_PATH'])
expected = int(os.environ['EXPECTED_LINES'])
pat = re.compile(
    r"(i['’]?m happy to help|i understand your request|i understand your question|"
    r"thank you for your question|as an ai language model|i must inform you|"
    r"i do not have real[- ]time|i don't have real[- ]time|it would be best to check|"
    r"check (the|their) .* website|contact .* directly)",
    re.I,
)
input_rows = [json.loads(line) for line in inp.read_text(encoding='utf-8').splitlines() if line.strip()]
output_rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line.strip()] if out.exists() else []
texts = [str(row.get('chosen', '')) for row in output_rows]
empty = sum(1 for t in texts if not t.strip())
dirty = sum(1 for t in texts if pat.search(t[:500]))
lengths = [len(t.split()) for t in texts if t.strip()]
samples = []
for src, gen in list(zip(input_rows, output_rows))[:10]:
    samples.append({
        'prompt': str(src.get('text', ''))[:200],
        'orig': str(src.get('chosen', ''))[:240],
        'gen': str(gen.get('chosen', ''))[:240],
    })
result = {
    'model': os.environ['MODEL'],
    'input': str(inp),
    'output': str(out),
    'expected_lines': expected,
    'actual_lines': len(output_rows),
    'empty_lines': empty,
    'dirty_lines': dirty,
    'dirty_rate': (dirty / len(output_rows)) if output_rows else 1.0,
    'mean_words': statistics.mean(lengths) if lengths else 0.0,
    'median_words': statistics.median(lengths) if lengths else 0.0,
    'pass': len(output_rows) == expected and empty == 0 and ((dirty / len(output_rows)) if output_rows else 1.0) <= 0.05,
    'samples': samples,
}
report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
PY
}

run_smoke_stage() {
  prepare_fresh_smoke
  CURRENT_STAGE=smoke
  CURRENT_INPUT="$SMOKE_INPUT"
  CURRENT_SHARD_DIR="$SMOKE_SHARD_DIR"
  CURRENT_MERGED_OUTPUT="$SMOKE_OUTPUT"
  log "starting smoke stage workers=$NUM_WORKERS per_worker_concurrency=$SMOKE_WORKER_CONCURRENCY per_worker_batch=$SMOKE_WORKER_BATCH_SIZE"
  launch_stage_workers smoke "$SMOKE_INPUT" "$SMOKE_SHARD_DIR" "$SMOKE_OUTPUT" "$SMOKE_WORKER_CONCURRENCY" "$SMOKE_WORKER_BATCH_SIZE"
  verify_stage_shards smoke "$SMOKE_INPUT" "$SMOKE_SHARD_DIR"
  merge_stage_shards smoke "$SMOKE_INPUT" "$SMOKE_SHARD_DIR" "$SMOKE_OUTPUT"
  quality_report "$SMOKE_INPUT" "$SMOKE_OUTPUT" "$SMOKE_REPORT"

  local smoke_pass
  smoke_pass=$(python3 - <<'PY'
import json
import os
report = json.load(open(os.environ['SMOKE_REPORT'], encoding='utf-8'))
print('1' if report.get('pass') else '0')
PY
)
  if [ "$smoke_pass" != "1" ]; then
    log "smoke failed quality gate; stopping before full run"
    exit 1
  fi
  log "smoke passed; starting full generation"
}

run_full_stage() {
  archive_existing_partial "$TRAIN_OUTPUT" "$TRAIN_SHARD_DIR"
  CURRENT_STAGE=full
  CURRENT_INPUT="$TRAIN_INPUT"
  CURRENT_SHARD_DIR="$TRAIN_SHARD_DIR"
  CURRENT_MERGED_OUTPUT="$TRAIN_OUTPUT"
  log "starting full stage workers=$NUM_WORKERS per_worker_concurrency=$FULL_WORKER_CONCURRENCY per_worker_batch=$FULL_WORKER_BATCH_SIZE"
  launch_stage_workers full "$TRAIN_INPUT" "$TRAIN_SHARD_DIR" "$TRAIN_OUTPUT" "$FULL_WORKER_CONCURRENCY" "$FULL_WORKER_BATCH_SIZE"
  verify_stage_shards full "$TRAIN_INPUT" "$TRAIN_SHARD_DIR"
  merge_stage_shards full "$TRAIN_INPUT" "$TRAIN_SHARD_DIR" "$TRAIN_OUTPUT"
  quality_report "$TRAIN_INPUT" "$TRAIN_OUTPUT" "$FULL_REPORT"
  write_aggregate_state full completed "$TRAIN_INPUT" "$TRAIN_SHARD_DIR" "$TRAIN_OUTPUT"
  log "full generation complete"
}

ensure_dirty_source
write_smoke_input
run_smoke_stage
run_full_stage
