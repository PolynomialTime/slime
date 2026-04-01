#!/bin/bash
# Debug job: 跳过 SFT，直接用 base 权重跑 IRL 3 轮 + reward update
# 目的：验证 reward update 在 GPU 3 能跑通
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

rm -rf $SLIME/models/reward_model
rm -rf $SLIME/models/save_dir_debug
rm -rf $SLIME/rollout
rm -rf $SLIME/tensorboard_log

# 不设 CUDA_VISIBLE_DEVICES，让 Ray 自然分配 GPU 0-2
MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$SLIME/models/qwen3-1.7b-base \
REF_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
SAVE_DIR=$SLIME/models/save_dir_debug \
PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
REWARD_UPDATE_LAUNCHER=accelerate \
bash scripts/run-irl-debug.sh
