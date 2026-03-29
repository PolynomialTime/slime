#!/bin/bash
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

# 清理上一次运行的中间产物
rm -rf $SLIME/models/reward_model
rm -rf $SLIME/models/save_dir
rm -rf $SLIME/rollout
rm -rf $SLIME/tensorboard_log
rm -rf $SLIME/eval

MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$SLIME/models/qwen3-1.7b-base \
REF_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
ACTOR_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
SAVE_DIR=$SLIME/models/save_dir \
PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
REWARD_UPDATE_LAUNCHER=accelerate \
bash scripts/run-irl-prod.sh
