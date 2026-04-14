#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"

SINGLE_PPO_LOG_DIR=${SINGLE_PPO_LOG_DIR:-$SLIME/logs/single_ppo}
SINGLE_PPO_RUN_TS=${SINGLE_PPO_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
SINGLE_PPO_RUN_TAG=${SINGLE_PPO_RUN_TAG:-single_ppo_r1_reward_${SINGLE_PPO_RUN_TS}}
RUN_LOG=$SINGLE_PPO_LOG_DIR/${SINGLE_PPO_RUN_TAG}.log
RUN_ENV=$SINGLE_PPO_LOG_DIR/${SINGLE_PPO_RUN_TAG}.env

mkdir -p "$SINGLE_PPO_LOG_DIR"

TRACKED_ENV_KEYS=(
  NUM_ROLLOUT
  PPO_SAVE_INTERVAL
  ROLLOUT_BATCH_SIZE
  ROLLOUT_MAX_RESPONSE_LEN
  ROLLOUT_TEMPERATURE
  ACTOR_LR
  CRITIC_LR
  KL_LOSS_COEF
  TB_EXP_NAME
  SAVE_DIR
  CRITIC_SAVE_DIR
  ROLLOUT_DEBUG_DIR
  ROUND1_REWARD_DIR
  REWARD_MODEL_DIR
  REWARD_SNAPSHOT_DIR
  PROMPT_DATA
  DEMO_DATA
  ALIGN_ROLLOUT_WITH_SFT
  MODEL_SH
  ALLOW_EXISTING_SAVE_DIR
)

{
cat <<EOF
run_tag=$SINGLE_PPO_RUN_TAG
run_log=$RUN_LOG
cwd=$(pwd)
git_head=$(git rev-parse --short HEAD 2>/dev/null || true)
hostname=$(hostname)
start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
command=bash scripts/run-single-ppo-fixed-reward.sh $*
EOF
for key in "${TRACKED_ENV_KEYS[@]}"; do
  if [ -n "${!key:-}" ]; then
    printf '%s=%s\n' "$key" "${!key}"
  fi
done
} > "$RUN_ENV"

echo "[single-ppo-log] writing to $RUN_LOG"
echo "[single-ppo-log] env snapshot $RUN_ENV"

{
  echo "[single-ppo-log] start $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[single-ppo-log] command bash scripts/run-single-ppo-fixed-reward.sh $*"
  bash scripts/run-single-ppo-fixed-reward.sh "$@"
} 2>&1 | tee -a "$RUN_LOG"

exit ${PIPESTATUS[0]}
