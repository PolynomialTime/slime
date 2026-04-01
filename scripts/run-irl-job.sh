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

# SFT checkpoint 路径（需先跑 run-sft-job.sh 生成）
SFT_CKPT=$SLIME/models/sft_checkpoint
SFT_HF=$SLIME/models/sft_checkpoint_hf

MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$SFT_HF \
REF_CKPT=$SFT_CKPT \
ACTOR_CKPT=$SFT_CKPT \
SAVE_DIR=$SLIME/models/save_dir \
PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
REWARD_UPDATE_LAUNCHER=accelerate \
bash scripts/run-irl-prod.sh
