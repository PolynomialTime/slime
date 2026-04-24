#!/bin/bash
# One-time conversion: Qwen3-8B HF format → Megatron torch_dist format (TP=1, no sharding)
# Megatron can reshard TP=1 checkpoint to higher TP at load time.

set -ex
SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME
BASE_HF_DIR=${BASE_HF_DIR:-$SLIME/models/qwen3-8B-base}
BASE_TORCH_DIST_DIR=${BASE_TORCH_DIST_DIR:-$SLIME/models/qwen3-8b-base_torch_dist}

export PYTHONPATH=/root/Megatron-LM/

source scripts/models/qwen3-8B.sh

if [ ! -d "$BASE_HF_DIR" ]; then
  echo "ERROR: missing base HF checkpoint at $BASE_HF_DIR" >&2
  exit 1
fi

torchrun --nproc_per_node 1 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint $BASE_HF_DIR \
  --save $BASE_TORCH_DIST_DIR \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --no-load-optim \
  --no-load-rng \
  ${MODEL_ARGS[@]}

echo "Conversion complete: $BASE_TORCH_DIST_DIR"
