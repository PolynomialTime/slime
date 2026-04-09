#!/bin/bash
set -euo pipefail

SLIME="${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
cd "$SLIME"

ROUNDS="${ROUNDS:-1 2 3 4 5 6 7}"

for ROUND in $ROUNDS; do
  ROUND_SAVE_DIR="$SLIME/models/save_dir_r${ROUND}"
  if [ ! -f "$ROUND_SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "Skipping round $ROUND: missing $ROUND_SAVE_DIR/latest_checkpointed_iteration.txt"
    continue
  fi
  bash scripts/export-policy-round.sh "$ROUND"
done

echo "Finished exporting requested rounds: $ROUNDS"
