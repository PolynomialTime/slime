#!/bin/bash
# Generate policy outputs for ALL rounds (for winrate evaluation)
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME
mkdir -p eval

# SFT baseline (if not already generated)
if [ ! -f eval/outputs_sft_baseline.jsonl ]; then
  echo "=== Generating SFT baseline ==="
  python3 scripts/eval_generate.py \
    --model-path models/sft_checkpoint_hf \
    --prompt-data hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
    --output eval/outputs_sft_baseline.jsonl \
    --apply-chat-template --batch-size 8 --max-new-tokens 256
fi

# Generate outputs for each round
for ROUND in 1 2 3 4 5 6 7; do
  CKPT_DIR=models/save_dir_round${ROUND}
  if [ ! -d "$CKPT_DIR" ]; then
    echo "Skipping round $ROUND (no checkpoint)"
    continue
  fi
  if [ -f "eval/outputs_policy_r${ROUND}.jsonl" ]; then
    echo "Skipping round $ROUND (already generated)"
    continue
  fi

  echo "=== Round $ROUND: Converting checkpoint ==="
  ITER=$(cat $CKPT_DIR/latest_checkpointed_iteration.txt)
  ITER_DIR=$CKPT_DIR/iter_$(printf "%07d" $ITER)

  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir $ITER_DIR \
    --output-dir /tmp/policy_r${ROUND}_hf \
    --origin-hf-dir models/qwen3-1.7b-base \
    --force

  echo "=== Round $ROUND: Generating outputs ==="
  python3 scripts/eval_generate.py \
    --model-path /tmp/policy_r${ROUND}_hf \
    --prompt-data hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
    --output eval/outputs_policy_r${ROUND}.jsonl \
    --apply-chat-template --batch-size 8 --max-new-tokens 256

  rm -rf /tmp/policy_r${ROUND}_hf
done

echo "=== All generations complete ==="
ls -la eval/outputs_*.jsonl
