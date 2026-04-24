#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

export WINRATE_ENABLE_REFERENCE="${WINRATE_ENABLE_REFERENCE:-0}"
export WINRATE_MODEL="deepseek-v3.2"
export WINRATE_API_KEY="sk-tNiMdAwz4E5BAwFZcCreiuiA7RZ0uAiprmt6aGBuYlvVWD1r"

exec bash "$SLIME_ROOT/scripts/run-winrate-offline.sh" "$@"
