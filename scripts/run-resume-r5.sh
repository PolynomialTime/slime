#!/bin/bash
set -euo pipefail

echo "ERROR: scripts/run-resume-r5.sh is deprecated and intentionally disabled." >&2
echo "ERROR: It still points at the old HH-RLHF / Qwen3-1.7B resume path and must not be used." >&2
echo "Use one of the supported mainline entrypoints instead:" >&2
echo "  - bash scripts/run-full-pipeline-job.sh" >&2
echo "  - bash scripts/run-full-pipeline-logged.sh" >&2
exit 1
