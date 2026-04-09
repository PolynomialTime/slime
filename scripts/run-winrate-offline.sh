#!/bin/bash
# Run winrate evaluation on a networked machine against outputs produced by the offline cluster pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT=${SLIME_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
EVAL_DIR=${EVAL_DIR:-$SLIME_ROOT/eval}
WINRATE_OUTPUT_DIR=${WINRATE_OUTPUT_DIR:-$SLIME_ROOT/eval_winrate}
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME_ROOT/ultrafeedback}
TEST_DATA=${TEST_DATA:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}
SFT_BASELINE=${SFT_BASELINE:-$EVAL_DIR/outputs_sft_baseline.jsonl}
WINRATE_MODEL=${WINRATE_MODEL:-gpt-4o}
WINRATE_API_KEY=${WINRATE_API_KEY:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}}}
WINRATE_BASE_URL=${WINRATE_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-}}}
WINRATE_FALLBACK_MODEL=${WINRATE_FALLBACK_MODEL:-}
WINRATE_FALLBACK_API_KEY=${WINRATE_FALLBACK_API_KEY:-}
WINRATE_FALLBACK_BASE_URL=${WINRATE_FALLBACK_BASE_URL:-}
MAX_ROUND=7
ROUNDS_CSV=""
WATCH_INTERVAL=0
MAX_SAMPLES=""
CONCURRENCY=16
EXPECTED_EVAL_LINES=2000
API_ENV_CHECKED=0

if [ -z "$WINRATE_BASE_URL" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
  WINRATE_BASE_URL="https://openrouter.ai/api/v1"
fi
if [ -z "$WINRATE_BASE_URL" ]; then
  WINRATE_BASE_URL="http://35.220.164.252:3888"
fi
case "$WINRATE_BASE_URL" in
  */v1|*/v1/) ;;
  *)
    WINRATE_BASE_URL="${WINRATE_BASE_URL%/}/v1"
    ;;
esac
if [ -z "$WINRATE_FALLBACK_BASE_URL" ] && [ -n "$WINRATE_FALLBACK_MODEL" ]; then
  WINRATE_FALLBACK_BASE_URL="$WINRATE_BASE_URL"
fi
case "$WINRATE_FALLBACK_BASE_URL" in
  "") ;;
  */v1|*/v1/) ;;
  *)
    WINRATE_FALLBACK_BASE_URL="${WINRATE_FALLBACK_BASE_URL%/}/v1"
    ;;
esac

if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES=$ACTUAL_EVAL_LINES
  fi
fi

usage() {
  cat <<USAGE
Usage: bash scripts/run-winrate-offline.sh [options]

Options:
  --rounds 1,2,3        Evaluate only the listed rounds.
  --max-round N         Evaluate rounds 1..N when --rounds is not set. Default: 7.
  --watch-interval SEC  Re-scan every SEC seconds for newly finished rounds.
  --model MODEL         Judge model for scripts/eval_winrate.py. Default: gpt-4o.
  --base-url URL        Optional API base URL.
  --fallback-model M    Optional fallback judge model for content-filtered prompts.
  --fallback-base-url U Optional fallback API base URL.
  --max-samples N       Optional max sample count passed through to eval_winrate.py.
  --concurrency N       Optional request concurrency passed through to eval_winrate.py. Default: 16.
  --help                Show this message.

Environment overrides:
  SLIME_ROOT, EVAL_DIR, WINRATE_OUTPUT_DIR, TEST_DATA, SFT_BASELINE,
  WINRATE_MODEL, WINRATE_API_KEY, WINRATE_BASE_URL, OPENAI_API_KEY,
  WINRATE_FALLBACK_MODEL, WINRATE_FALLBACK_API_KEY, WINRATE_FALLBACK_BASE_URL,
  OPENROUTER_API_KEY, ANTHROPIC_AUTH_TOKEN, OPENAI_BASE_URL, ANTHROPIC_BASE_URL.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rounds)
      ROUNDS_CSV=${2:?missing value for --rounds}
      shift 2
      ;;
    --max-round)
      MAX_ROUND=${2:?missing value for --max-round}
      shift 2
      ;;
    --watch-interval)
      WATCH_INTERVAL=${2:?missing value for --watch-interval}
      shift 2
      ;;
    --model)
      WINRATE_MODEL=${2:?missing value for --model}
      shift 2
      ;;
    --base-url)
      WINRATE_BASE_URL=${2:?missing value for --base-url}
      shift 2
      ;;
    --fallback-model)
      WINRATE_FALLBACK_MODEL=${2:?missing value for --fallback-model}
      shift 2
      ;;
    --fallback-base-url)
      WINRATE_FALLBACK_BASE_URL=${2:?missing value for --fallback-base-url}
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES=${2:?missing value for --max-samples}
      shift 2
      ;;
    --concurrency)
      CONCURRENCY=${2:?missing value for --concurrency}
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

is_positive_int() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) [ "$1" -gt 0 ] ;;
  esac
}

if ! is_positive_int "$MAX_ROUND"; then
  echo "ERROR: --max-round must be a positive integer" >&2
  exit 1
fi
if [ "$WATCH_INTERVAL" != "0" ] && ! is_positive_int "$WATCH_INTERVAL"; then
  echo "ERROR: --watch-interval must be 0 or a positive integer" >&2
  exit 1
fi
if ! is_positive_int "$CONCURRENCY"; then
  echo "ERROR: --concurrency must be a positive integer" >&2
  exit 1
fi
if [ -n "$MAX_SAMPLES" ] && ! is_positive_int "$MAX_SAMPLES"; then
  echo "ERROR: --max-samples must be a positive integer" >&2
  exit 1
fi

build_rounds() {
  ROUNDS=()
  if [ -n "$ROUNDS_CSV" ]; then
    IFS=',' read -r -a raw_rounds <<< "$ROUNDS_CSV"
    for round in "${raw_rounds[@]}"; do
      round=$(echo "$round" | xargs)
      if ! is_positive_int "$round"; then
        echo "ERROR: invalid round in --rounds: $round" >&2
        exit 1
      fi
      ROUNDS+=("$round")
    done
  else
    for ((round = 1; round <= MAX_ROUND; round++)); do
      ROUNDS+=("$round")
    done
  fi
}

outputs_jsonl_is_complete() {
  local path=$1
  [ -f "$path" ] || return 1
  local lines
  lines=$(awk 'END {print NR}' "$path")
  [ "$lines" -ge "$EXPECTED_EVAL_LINES" ]
}

winrate_json_is_complete() {
  local path=$1
  [ -f "$path" ] || return 1
  python3 - "$path" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ok = (
    isinstance(data, dict)
    and data.get("winrate_a") is not None
    and int(data.get("total", 0)) > 0
)
raise SystemExit(0 if ok else 1)
PY
}

winrate_json_is_current() {
  local path=$1
  local outputs_a=$2
  local outputs_b=$3
  winrate_json_is_complete "$path" || return 1
  python3 - "$path" "$outputs_a" "$outputs_b" <<'PY'
from pathlib import Path
import sys

winrate_path = Path(sys.argv[1])
deps = [Path(sys.argv[2]), Path(sys.argv[3])]
try:
    winrate_mtime = winrate_path.stat().st_mtime
    dep_mtime = max(dep.stat().st_mtime for dep in deps)
except FileNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if winrate_mtime >= dep_mtime else 1)
PY
}

print_winrate_summary() {
  local path=$1
  local label=$2
  python3 - "$path" "$label" <<'PY'
import json, sys
path, label = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
print(
    f"{label}: winrate_a={data['winrate_a']:.4f} "
    f"a_wins={data['a_wins']} b_wins={data['b_wins']} ties={data['ties']} total={data['total']}"
)
PY
}

ensure_api_env() {
  if [ "$API_ENV_CHECKED" -eq 1 ]; then
    return 0
  fi
  if [ -z "$WINRATE_API_KEY" ]; then
    echo "ERROR: winrate evaluation requires WINRATE_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or ANTHROPIC_AUTH_TOKEN" >&2
    exit 1
  fi
  if [ -n "$WINRATE_FALLBACK_MODEL" ] && [ -z "$WINRATE_FALLBACK_API_KEY" ]; then
    echo "ERROR: WINRATE_FALLBACK_MODEL is set but WINRATE_FALLBACK_API_KEY is empty" >&2
    exit 1
  fi
  API_ENV_CHECKED=1
}

run_winrate_eval() {
  local outputs_a=$1
  local outputs_b=$2
  local output_json=$3
  local label=$4
  local output_log="${output_json%.json}.log"
  local -a cmd

  if winrate_json_is_current "$output_json" "$outputs_a" "$outputs_b"; then
    echo "Skipping ${label}: found current $output_json"
    print_winrate_summary "$output_json" "$label"
    return 0
  fi

  ensure_api_env

  echo "===== ${label} ====="
  rm -f "$output_log"
  cmd=(
    python3 "$SCRIPT_DIR/eval_winrate.py"
    --outputs-a "$outputs_a"
    --outputs-b "$outputs_b"
    --output "$output_json"
    --api-key "$WINRATE_API_KEY"
    --model "$WINRATE_MODEL"
    --concurrency "$CONCURRENCY"
  )
  if [ -n "$WINRATE_BASE_URL" ]; then
    cmd+=(--base-url "$WINRATE_BASE_URL")
  fi
  if [ -n "$WINRATE_FALLBACK_MODEL" ]; then
    cmd+=(--fallback-model "$WINRATE_FALLBACK_MODEL")
  fi
  if [ -n "$WINRATE_FALLBACK_API_KEY" ]; then
    cmd+=(--fallback-api-key "$WINRATE_FALLBACK_API_KEY")
  fi
  if [ -n "$WINRATE_FALLBACK_BASE_URL" ]; then
    cmd+=(--fallback-base-url "$WINRATE_FALLBACK_BASE_URL")
  fi
  if [ -n "$MAX_SAMPLES" ]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi
  "${cmd[@]}" 2>&1 | tee "$output_log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: winrate evaluation failed for ${label}" >&2
    exit 1
  fi
  if ! winrate_json_is_complete "$output_json"; then
    echo "ERROR: incomplete winrate output at $output_json" >&2
    exit 1
  fi
  print_winrate_summary "$output_json" "$label"
}

scan_once() {
  local evaluated=0
  local pending=0
  local baseline_lines

  echo "===== Winrate scan $(date '+%F %T') ====="

  if ! outputs_jsonl_is_complete "$SFT_BASELINE"; then
    if [ -f "$SFT_BASELINE" ]; then
      baseline_lines=$(awk 'END {print NR}' "$SFT_BASELINE")
      echo "Pending: SFT baseline not ready yet ($baseline_lines/$EXPECTED_EVAL_LINES lines)"
    else
      echo "Pending: SFT baseline not ready yet ($SFT_BASELINE missing)"
    fi
    pending=1
    echo "Scan summary: evaluated=0 pending=1"
    return 0
  fi

  for round in "${ROUNDS[@]}"; do
    local round_output="$EVAL_DIR/outputs_policy_r${round}.jsonl"
    local round_winrate_sft="$WINRATE_OUTPUT_DIR/winrate_r${round}_vs_sft.json"
    local line_count=0

    if [ -f "$round_output" ]; then
      line_count=$(awk 'END {print NR}' "$round_output")
    fi

    if [ ! -f "$round_output" ]; then
      echo "Round $round pending: missing $round_output"
      pending=1
      continue
    fi
    if [ "$line_count" -lt "$EXPECTED_EVAL_LINES" ]; then
      echo "Round $round pending: only $line_count/$EXPECTED_EVAL_LINES lines in $round_output"
      pending=1
      continue
    fi

    run_winrate_eval \
      "$round_output" \
      "$SFT_BASELINE" \
      "$round_winrate_sft" \
      "Round $round winrate vs SFT"

    evaluated=$((evaluated + 1))
  done

  echo "Scan summary: evaluated=$evaluated pending=$pending"
}

build_rounds
mkdir -p "$WINRATE_OUTPUT_DIR"
find "$WINRATE_OUTPUT_DIR" -maxdepth 1 -type f \( -name "winrate_r*_vs_gpt4o.json" -o -name "winrate_r*_vs_gpt4o.log" \) -delete 2>/dev/null || true

echo "SLIME_ROOT=$SLIME_ROOT"
echo "EVAL_DIR=$EVAL_DIR"
echo "WINRATE_OUTPUT_DIR=$WINRATE_OUTPUT_DIR"
echo "SFT_BASELINE=$SFT_BASELINE"
echo "ROUNDS=${ROUNDS[*]}"
echo "EXPECTED_EVAL_LINES=$EXPECTED_EVAL_LINES"
echo "WINRATE_MODEL=$WINRATE_MODEL"
if [ -n "$WINRATE_BASE_URL" ]; then
  echo "WINRATE_BASE_URL=$WINRATE_BASE_URL"
fi
if [ -n "$WINRATE_FALLBACK_MODEL" ]; then
  echo "WINRATE_FALLBACK_MODEL=$WINRATE_FALLBACK_MODEL"
fi
if [ -n "$WINRATE_FALLBACK_BASE_URL" ]; then
  echo "WINRATE_FALLBACK_BASE_URL=$WINRATE_FALLBACK_BASE_URL"
fi

if [ "$WATCH_INTERVAL" -gt 0 ]; then
  while true; do
    scan_once
    echo "Sleeping ${WATCH_INTERVAL}s before next scan..."
    sleep "$WATCH_INTERVAL"
  done
else
  scan_once
fi
