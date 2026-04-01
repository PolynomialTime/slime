#!/bin/bash
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

# 清理上次 SFT 产物
rm -rf $SLIME/models/sft_checkpoint
rm -rf $SLIME/tensorboard_log

MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$SLIME/models/qwen3-1.7b-base \
ACTOR_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
SAVE_DIR=$SLIME/models/sft_checkpoint \
SFT_DATA=$SLIME/hh-rlhf-processed/sft_10k.jsonl \
bash scripts/run-sft-prod.sh
