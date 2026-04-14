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

TRACKED_ENV_KEYS=(
  NUM_ROUNDS
  NUM_ROLLOUT_PER_ROUND
  BOOTSTRAP_NUM_ROLLOUT
  KEEP_ALL_ROUND_REWARD_SNAPSHOTS
  PPO_START_FROM_SFT
  PPO_REF_FIXED_TO_SFT
  ROLLOUT_TEMPERATURE
  ROLLOUT_MAX_RESPONSE_LEN
  ACTOR_LR
  CRITIC_LR
  KL_LOSS_COEF
  REWARD_EVAL_MAX_SAMPLES
  REWARD_EVAL_SHUFFLE_SEED
  REWARD_UPDATE_EPOCHS
  REWARD_UPDATE_BATCH_SIZE
  REWARD_ONLINE_PREF_WEIGHT
  EVAL_TEMPERATURE
  WINRATE_GATE_ENABLED
  WINRATE_GATE_MAX_ROUND
  WINRATE_GATE_MAX_SAMPLES
  WINRATE_GATE_CONCURRENCY
  WINRATE_GATE_MIN
  INTERROUND_WINRATE_GATE_MIN
)

{
cat <<EOF
run_tag=$PIPELINE_RUN_TAG
run_log=$RUN_LOG
cwd=$(pwd)
git_head=$(git rev-parse --short HEAD 2>/dev/null || true)
hostname=$(hostname)
start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
command=bash scripts/run-full-pipeline-job.sh $*
EOF
for key in "${TRACKED_ENV_KEYS[@]}"; do
  if [ -n "${!key:-}" ]; then
    printf '%s=%s\n' "$key" "${!key}"
  fi
done
} > "$RUN_ENV"

echo "[pipeline-log] writing to $RUN_LOG"
echo "[pipeline-log] env snapshot $RUN_ENV"

{
  echo "[pipeline-log] start $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[pipeline-log] command bash scripts/run-full-pipeline-job.sh $*"
  bash scripts/run-full-pipeline-job.sh "$@"
} 2>&1 | tee -a "$RUN_LOG"

exit ${PIPESTATUS[0]}
