#!/bin/bash
# Resume from Round 5 (Rounds 1-4 completed, Round 5 PPO save crashed)
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

NUM_ROUNDS=7
NUM_ROLLOUT_PER_ROUND=100

# Clean failed Round 5 checkpoint
rm -rf $SLIME/models/save_dir_round5 $SLIME/models/save_dir_round5_critic

for ROUND in $(seq 5 $NUM_ROUNDS); do
  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="

  MODEL_SH=scripts/models/qwen3-1.7B.sh \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  REF_CKPT=$SLIME/models/sft_checkpoint \
  SAVE_DIR=$SLIME/models/save_dir_round${ROUND} \
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

  SLIME=$SLIME \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  ROUND_ID=$ROUND \
  ROLLOUT_END=$(( NUM_ROLLOUT_PER_ROUND - 1 )) \
  NUM_ROLLOUT_PER_ROUND=$NUM_ROLLOUT_PER_ROUND \
  bash scripts/run-reward-update.sh

  echo "===== Round $ROUND/$NUM_ROUNDS completed ====="
done

echo "===== All rounds completed ====="
