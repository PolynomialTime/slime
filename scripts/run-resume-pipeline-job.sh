#!/bin/bash
# Resume Pipeline: skip SFT + Round 1 PPO (already done), start from Round 1 Reward Update
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

NUM_ROUNDS=7
NUM_ROLLOUT_PER_ROUND=200
START_ROUND=1  # resume from this round's reward update

# Round 1: only reward update (PPO already done with 336 rollouts)
ROUND=1
ROLLOUT_END=335  # Original Round 1 had 336 rollouts (0-335)

echo "===== Round $ROUND/$NUM_ROUNDS: Reward Update (resume) ====="

# Clean old bad reward model (from previous failed run)
rm -rf $SLIME/models/reward_model

pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 5

SLIME=$SLIME \
HF_CKPT=$SLIME/models/sft_checkpoint_hf \
ROUND_ID=$ROUND \
ROLLOUT_END=$ROLLOUT_END \
NUM_ROLLOUT_PER_ROUND=$NUM_ROLLOUT_PER_ROUND \
bash scripts/run-reward-update.sh

echo "===== Round $ROUND/$NUM_ROUNDS completed ====="

# Rounds 2-7: full PPO → Reward Update
for ROUND in $(seq 2 $NUM_ROUNDS); do
  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="

  MODEL_SH=scripts/models/qwen3-1.7B.sh \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  REF_CKPT=$SLIME/models/sft_checkpoint \
  SAVE_DIR=$SLIME/models/save_dir \
  PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND \
  bash scripts/run-irl-prod.sh

  echo "===== Killing PPO processes ====="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  echo "===== Round $ROUND/$NUM_ROUNDS: Reward Update ====="

  # Rollout IDs reset each round (0 to NUM_ROLLOUT_PER_ROUND-1)
  SLIME=$SLIME \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  ROUND_ID=$ROUND \
  ROLLOUT_END=$(( NUM_ROLLOUT_PER_ROUND - 1 )) \
  NUM_ROLLOUT_PER_ROUND=$NUM_ROLLOUT_PER_ROUND \
  bash scripts/run-reward-update.sh

  echo "===== Round $ROUND/$NUM_ROUNDS completed ====="
done

echo "===== All $NUM_ROUNDS rounds completed ====="
