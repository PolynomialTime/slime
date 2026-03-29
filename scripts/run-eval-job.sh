#!/bin/bash
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

ITER=${ITER:-iter_0050}
EVAL_DIR=$SLIME/eval

mkdir -p $EVAL_DIR

# 生成 IRL 模型回复
python scripts/eval_generate.py \
  --model-path $SLIME/models/save_dir/$ITER \
  --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $EVAL_DIR/outputs_irl_${ITER}.jsonl \
  --apply-chat-template \
  --max-new-tokens 256 \
  --batch-size 8

# 生成 base 模型回复（只跑一次）
if [ ! -f $EVAL_DIR/outputs_base.jsonl ]; then
  python scripts/eval_generate.py \
    --model-path $SLIME/models/qwen3-1.7b-base \
    --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
    --output $EVAL_DIR/outputs_base.jsonl \
    --apply-chat-template \
    --max-new-tokens 256 \
    --batch-size 8
fi
