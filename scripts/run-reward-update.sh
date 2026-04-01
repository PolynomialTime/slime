#!/bin/bash

# Reward Update Phase — 4×GPU via accelerate
# Called by run-full-pipeline-job.sh after PPO phase
# All GPUs are free (PPO processes killed before this runs)

# kill any leftover processes
pkill -9 sglang || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

set -ex

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd $SLIME

ROUND_ID=${ROUND_ID:-0}
ROLLOUT_END=${ROLLOUT_END:-335}
NUM_ROLLOUT_PER_ROUND=${NUM_ROLLOUT_PER_ROUND:-336}

# Build reward update args JSON
REWARD_DIR=$SLIME/models/reward_model
mkdir -p $REWARD_DIR

ARGS_JSON=$REWARD_DIR/reward_update_args.json
cat > $ARGS_JSON <<EOF
{
  "hf_checkpoint": "${HF_CKPT:-$SLIME/models/sft_checkpoint_hf}",
  "reward_model_dir": "$REWARD_DIR",
  "reward_model_init": null,
  "reward_demo_path": "$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl",
  "reward_demo_prompt_key": "text",
  "reward_demo_answer_key": "label",
  "reward_update_epochs": 1,
  "reward_update_batch_size": 8,
  "reward_update_lr": 5e-6,
  "c_coef_init": 2.0,
  "c_coef_min": 1.0,
  "c_coef_max": 10.0,
  "coef_scale_up": 1.2,
  "coef_scale_down": 0.8,
  "target_reward_l2_norm": 3.0,
  "apply_chat_template": true,
  "apply_chat_template_kwargs": {},
  "save_debug_rollout_data": "$SLIME/rollout/rollout_{rollout_id}.pt",
  "reward_update_rollout_window": 50,
  "reward_eval_path": "$SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl",
  "reward_eval_prompt_key": "text",
  "reward_eval_chosen_key": "chosen",
  "reward_eval_rejected_key": "rejected",
  "reward_eval_max_samples": 200,
  "reward_eval_batch_size": 32
}
EOF

echo "=== Reward Update Phase (round $ROUND_ID, rollout_end=$ROLLOUT_END) ==="

# 4 GPUs via accelerate
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m slime.local_rm.update_reward_accel \
  --args-json $ARGS_JSON \
  --rollout-id $ROLLOUT_END \
  --rollout-path $SLIME/rollout/rollout_${ROLLOUT_END}.pt

echo "=== Reward Update Phase completed ==="
