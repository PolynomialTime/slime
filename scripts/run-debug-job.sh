#!/bin/bash
set -euo pipefail

echo "ERROR: scripts/run-debug-job.sh is deprecated and intentionally disabled." >&2
echo "ERROR: It still points at the old HH-RLHF / Qwen3-1.7B debug path and must not be used." >&2
echo "Use the supported mainline scripts or a dedicated modern debug wrapper instead." >&2
echo "  - bash scripts/run-full-pipeline-job.sh" >&2
echo "  - bash scripts/run-full-pipeline-logged.sh" >&2
exit 1
