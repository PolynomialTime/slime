#!/bin/bash
set -euo pipefail

echo "ERROR: scripts/run-eval-job.sh is deprecated and intentionally disabled." >&2
echo "ERROR: It still points at the old HH-RLHF / Qwen3-1.7B eval path and must not be used." >&2
echo "Use the supported modern export/eval entrypoints instead:" >&2
echo "  - bash scripts/export-policy-round.sh <round>" >&2
echo "  - bash scripts/export-policy-checkpoint.sh <rollout_count>" >&2
echo "  - bash scripts/run-winrate-offline-local.sh" >&2
exit 1
