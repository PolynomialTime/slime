#!/bin/bash
set -euo pipefail

SLIME="${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
cd "$SLIME"

EVAL_DIR="${EVAL_DIR:-$SLIME/eval}"
WINRATE_OUTPUT_DIR="${WINRATE_OUTPUT_DIR:-$SLIME/eval_winrate}"
SFT_BASELINE="${SFT_BASELINE:-$EVAL_DIR/outputs_sft_baseline.jsonl}"
PATTERN="${PATTERN:-outputs_single_ppo_r1_reward_rollout*.jsonl}"
ROLLOUTS_CSV="${ROLLOUTS_CSV:-}"
FORCE_EVAL="${FORCE_EVAL:-0}"
CONCURRENCY="${CONCURRENCY:-16}"

export WINRATE_BASE_URL="${WINRATE_BASE_URL:-http://35.220.164.252:3888/v1}"
export WINRATE_API_KEY="${WINRATE_API_KEY:-sk-nGTCaRKDqGnda5kkfM49x2QegKpYqAesj5kUHWTaQ9YSLkYH}"
export WINRATE_MODEL="${WINRATE_MODEL:-gpt-4o}"
export WINRATE_FALLBACK_BASE_URL="${WINRATE_FALLBACK_BASE_URL:-http://35.220.164.252:3888/v1}"
export WINRATE_FALLBACK_API_KEY="${WINRATE_FALLBACK_API_KEY:-sk-Yv3u6E0fiuuIqqLgPiQWTuInrSuTKUL31ui2hqtHZSy8yWVt}"
export WINRATE_FALLBACK_MODEL="${WINRATE_FALLBACK_MODEL:-claude-sonnet-4-6}"

mkdir -p "$WINRATE_OUTPUT_DIR"

if [ ! -f "$SFT_BASELINE" ]; then
  echo "ERROR: missing SFT baseline at $SFT_BASELINE" >&2
  exit 1
fi

is_complete_winrate_json() {
  local path=$1
  [ -f "$path" ] || return 1
  python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ok = isinstance(data, dict) and data.get("winrate_a") is not None and int(data.get("total", 0)) > 0
raise SystemExit(0 if ok else 1)
PY
}

is_current_winrate_json() {
  local output_json=$1
  local outputs_a=$2
  local outputs_b=$3
  is_complete_winrate_json "$output_json" || return 1
  python3 - "$output_json" "$outputs_a" "$outputs_b" <<'PY'
from pathlib import Path
import sys

output_path = Path(sys.argv[1])
deps = [Path(sys.argv[2]), Path(sys.argv[3])]
try:
    output_mtime = output_path.stat().st_mtime
    dep_mtime = max(dep.stat().st_mtime for dep in deps)
except FileNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if output_mtime >= dep_mtime else 1)
PY
}

collect_rollouts() {
  python3 - "$EVAL_DIR" "$PATTERN" "$ROLLOUTS_CSV" <<'PY'
from pathlib import Path
import re
import sys

eval_dir = Path(sys.argv[1])
pattern = sys.argv[2]
requested_csv = sys.argv[3].strip()

files = {}
for path in sorted(eval_dir.glob(pattern)):
    match = re.search(r"rollout(\d+)\.jsonl$", path.name)
    if not match:
        continue
    rollout = int(match.group(1))
    files[rollout] = path

if requested_csv:
    requested = []
    for item in requested_csv.split(","):
        item = item.strip()
        if not item:
            continue
        rollout = int(item)
        if rollout not in files:
            raise SystemExit(f"missing requested rollout output: {rollout}")
        requested.append(rollout)
else:
    requested = sorted(files)

for rollout in requested:
    print(f"{rollout}\t{files[rollout]}")
PY
}

mapfile -t rollout_rows < <(collect_rollouts)

if [ "${#rollout_rows[@]}" -eq 0 ]; then
  echo "ERROR: no checkpoint outputs found in $EVAL_DIR matching $PATTERN" >&2
  exit 1
fi

echo "WINRATE_OUTPUT_DIR=$WINRATE_OUTPUT_DIR"
echo "SFT_BASELINE=$SFT_BASELINE"
echo "WINRATE_MODEL=$WINRATE_MODEL"
echo "WINRATE_BASE_URL=$WINRATE_BASE_URL"
echo "Found rollout outputs:"
printf '  %s\n' "${rollout_rows[@]}"

for row in "${rollout_rows[@]}"; do
  rollout=${row%%$'\t'*}
  outputs_a=${row#*$'\t'}
  output_json="$WINRATE_OUTPUT_DIR/winrate_single_ppo_r1_reward_rollout${rollout}_vs_sft_blind.json"
  output_log="${output_json%.json}.log"

  if [ "$FORCE_EVAL" != "1" ] && is_current_winrate_json "$output_json" "$outputs_a" "$SFT_BASELINE"; then
    echo "Skipping rollout $rollout: found current $output_json"
    continue
  fi

  echo "===== Blind winrate rollout $rollout vs SFT ====="
  python3 "$SLIME/scripts/eval_winrate.py" \
    --outputs-a "$outputs_a" \
    --outputs-b "$SFT_BASELINE" \
    --output "$output_json" \
    --mode blind \
    --api-key "$WINRATE_API_KEY" \
    --model "$WINRATE_MODEL" \
    --base-url "$WINRATE_BASE_URL" \
    --concurrency "$CONCURRENCY" \
    --fallback-model "$WINRATE_FALLBACK_MODEL" \
    --fallback-api-key "$WINRATE_FALLBACK_API_KEY" \
    --fallback-base-url "$WINRATE_FALLBACK_BASE_URL" \
    2>&1 | tee "$output_log"

  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: blind winrate failed for rollout $rollout" >&2
    exit 1
  fi
done

echo "Blind winrate finished. Outputs are in $WINRATE_OUTPUT_DIR"
