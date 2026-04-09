#!/bin/bash
# One-time conversion: Qwen3-8B HF format → Megatron torch_dist format (TP=1, no sharding)
# Megatron can reshard TP=1 checkpoint to higher TP at load time.

set -ex
SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

export PYTHONPATH=/root/Megatron-LM/

source scripts/models/qwen3-8B.sh

torchrun --nproc_per_node 1 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint $SLIME/models/qwen3-8b-base \
  --save $SLIME/models/qwen3-8b-base_torch_dist \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --no-load-optim \
  --no-load-rng \
  ${MODEL_ARGS[@]}

echo "Conversion complete: $SLIME/models/qwen3-8b-base_torch_dist"
