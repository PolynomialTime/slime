#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

export WINRATE_ENABLE_REFERENCE="${WINRATE_ENABLE_REFERENCE:-0}"

exec bash "$SLIME_ROOT/scripts/run-winrate-offline.sh" "$@"
