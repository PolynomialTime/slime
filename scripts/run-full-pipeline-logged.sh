#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"

PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR:-$SLIME/logs/pipeline}
PIPELINE_RUN_TS=${PIPELINE_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
PIPELINE_RUN_TAG=${PIPELINE_RUN_TAG:-pipeline_${PIPELINE_RUN_TS}}
RUN_LOG=$PIPELINE_LOG_DIR/${PIPELINE_RUN_TAG}.log
RUN_ENV=$PIPELINE_LOG_DIR/${PIPELINE_RUN_TAG}.env

mkdir -p "$PIPELINE_LOG_DIR"

cat > "$RUN_ENV" <<EOF
run_tag=$PIPELINE_RUN_TAG
run_log=$RUN_LOG
cwd=$(pwd)
git_head=$(git rev-parse --short HEAD 2>/dev/null || true)
hostname=$(hostname)
start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
command=bash scripts/run-full-pipeline-job.sh $*
EOF

echo "[pipeline-log] writing to $RUN_LOG"
echo "[pipeline-log] env snapshot $RUN_ENV"

{
  echo "[pipeline-log] start $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[pipeline-log] command bash scripts/run-full-pipeline-job.sh $*"
  bash scripts/run-full-pipeline-job.sh "$@"
} 2>&1 | tee -a "$RUN_LOG"

exit ${PIPESTATUS[0]}
