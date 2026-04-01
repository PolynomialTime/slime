#!/bin/bash
# Full Pipeline v4: SFT → 7 rounds of (PPO → Eval Generate → Reward Update)
# Real-time winrate eval after each round. Per-round tensorboard.
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

NUM_ROUNDS=7
NUM_ROLLOUT_PER_ROUND=100

# ============ Clean ALL artifacts ============
rm -rf $SLIME/models/sft_checkpoint
rm -rf $SLIME/models/sft_checkpoint_hf
rm -rf $SLIME/models/reward_model
rm -rf $SLIME/models/save_dir*
rm -rf $SLIME/rollout
rm -rf $SLIME/tensorboard_log
rm -rf $SLIME/eval

# ============ Phase 0: SFT ============
echo "===== Phase 0: SFT ====="

MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$SLIME/models/qwen3-1.7b-base \
ACTOR_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
SAVE_DIR=$SLIME/models/sft_checkpoint \
SFT_DATA=$SLIME/hh-rlhf-processed/sft_10k.jsonl \
bash scripts/run-sft-prod.sh

# ============ Weight conversion ============
echo "===== Weight Conversion ====="
LATEST_ITER=$(cat $SLIME/models/sft_checkpoint/latest_checkpointed_iteration.txt)
ITER_DIR=$SLIME/models/sft_checkpoint/iter_$(printf "%07d" $LATEST_ITER)
if [ -f "$ITER_DIR/common.pt" ]; then CKPT_DIR=$ITER_DIR
elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then CKPT_DIR=$ITER_DIR/mp_rank_00
else echo "ERROR: common.pt not found"; exit 1; fi

python3 tools/convert_torch_dist_to_hf.py \
  --input-dir $CKPT_DIR \
  --output-dir $SLIME/models/sft_checkpoint_hf \
  --origin-hf-dir $SLIME/models/qwen3-1.7b-base \
  --force

# ============ Generate SFT baseline outputs (for winrate comparison) ============
echo "===== Generating SFT baseline outputs ====="
mkdir -p $SLIME/eval
python3 scripts/eval_generate.py \
  --model-path $SLIME/models/sft_checkpoint_hf \
  --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $SLIME/eval/outputs_sft_baseline.jsonl \
  --apply-chat-template --batch-size 8 --max-new-tokens 256

# ============ Rounds 1-7 ============
for ROUND in $(seq 1 $NUM_ROUNDS); do
  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="

  rm -rf /tmp/save_dir_tmp /tmp/critic_ckpt 2>/dev/null || true

  MODEL_SH=scripts/models/qwen3-1.7B.sh \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  REF_CKPT=$SLIME/models/sft_checkpoint \
  SAVE_DIR=/tmp/save_dir_tmp \
  PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND \
  TB_EXP_NAME=round${ROUND} \
  bash scripts/run-irl-prod.sh

  # ===== Convert checkpoint to HF and generate outputs (for winrate) =====
  echo "===== Round $ROUND: Generating policy outputs ====="
  ITER=$(cat /tmp/save_dir_tmp/latest_checkpointed_iteration.txt)
  ITER_DIR=/tmp/save_dir_tmp/iter_$(printf "%07d" $ITER)

  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir $ITER_DIR \
    --output-dir /tmp/policy_r${ROUND}_hf \
    --origin-hf-dir $SLIME/models/qwen3-1.7b-base \
    --force

  python3 scripts/eval_generate.py \
    --model-path /tmp/policy_r${ROUND}_hf \
    --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
    --output $SLIME/eval/outputs_policy_r${ROUND}.jsonl \
    --apply-chat-template --batch-size 8 --max-new-tokens 256

  rm -rf /tmp/policy_r${ROUND}_hf /tmp/save_dir_tmp

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

echo "===== All $NUM_ROUNDS rounds completed ====="
echo "Eval outputs:"
ls -la $SLIME/eval/outputs_*.jsonl
