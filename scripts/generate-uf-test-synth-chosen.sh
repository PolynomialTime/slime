#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"

ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
INPUT=${INPUT:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}
OUTPUT=${OUTPUT:-$ULTRAFEEDBACK_DIR/uf-test-synth-chosen.jsonl}
MODEL=${MODEL:-gpt-4o}
CONCURRENCY=${CONCURRENCY:-32}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-1024}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${WINRATE_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  echo "ERROR: set OPENAI_API_KEY (or WINRATE_API_KEY / ANTHROPIC_AUTH_TOKEN) before generating uf-test synthetic chosen." >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "ERROR: missing uf-test data at $INPUT" >&2
  exit 1
fi

python3 scripts/generate_synthetic_chosen.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --resume

python3 - "$INPUT" "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid jsonl: path={path} line={line_no} error={exc}") from exc
    return rows


input_rows = load_jsonl(input_path)
output_rows = load_jsonl(output_path)
if len(input_rows) != len(output_rows):
    raise SystemExit(
        f"line mismatch: input={len(input_rows)} output={len(output_rows)} path={output_path}"
    )

missing = sum(1 for row in output_rows if not str(row.get("chosen", "")).strip())
if missing:
    raise SystemExit(f"generated uf-test synthetic chosen contains {missing} empty chosen responses")

print(f"uf-test synthetic chosen ready: path={output_path} rows={len(output_rows)}")
PY

echo "Generated $OUTPUT"
