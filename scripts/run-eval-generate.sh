#!/bin/bash
# Evaluate Round 1 policy: convert checkpoint + generate outputs
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

EVAL_DIR=$SLIME/eval
mkdir -p $EVAL_DIR

# Step 1: Convert Megatron checkpoint to HF
CKPT_DIR=$SLIME/models/save_dir/iter_0000335
HF_POLICY=$EVAL_DIR/policy_round1_hf

if [ ! -d "$HF_POLICY" ]; then
  echo "=== Converting checkpoint to HF ==="
  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir $CKPT_DIR \
    --output-dir $HF_POLICY \
    --origin-hf-dir $SLIME/models/qwen3-1.7b-base \
    --force
fi

# Step 2: Generate outputs for policy model
echo "=== Generating policy outputs ==="
python3 scripts/eval_generate.py \
  --model-path $HF_POLICY \
  --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $EVAL_DIR/outputs_policy_round1.jsonl \
  --apply-chat-template \
  --batch-size 8 \
  --max-new-tokens 256

# Step 3: Generate outputs for SFT baseline
echo "=== Generating SFT baseline outputs ==="
python3 scripts/eval_generate.py \
  --model-path $SLIME/models/sft_checkpoint_hf \
  --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $EVAL_DIR/outputs_sft_baseline.jsonl \
  --apply-chat-template \
  --batch-size 8 \
  --max-new-tokens 256

echo "=== Eval generation complete ==="
