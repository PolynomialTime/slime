#!/bin/bash
# Full Pipeline v4: SFT → 7 rounds of (PPO → Eval Generate → Reward Update)
# Real-time winrate eval after each round. Per-round tensorboard.
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME

NUM_ROUNDS=7
NUM_ROLLOUT_PER_ROUND=150

SFT_HF_DIR=$SLIME/models/sft_checkpoint_hf
SFT_BASELINE=$SLIME/eval/outputs_sft_baseline.jsonl

SKIP_SFT=0
if [ -d "$SFT_HF_DIR" ] && [ -f "$SFT_HF_DIR/config.json" ]; then
  SKIP_SFT=1
fi

SKIP_SFT_BASELINE=0
if [ -f "$SFT_BASELINE" ] && [ "$(awk 'END {print NR}' "$SFT_BASELINE")" -ge 4000 ]; then
  # Verify: responses must be non-empty AND not contaminated with Human: continuation
  CLEAN=$(python3 -c "
import json
clean = sum(1 for l in open('$SFT_BASELINE')
           if (r := json.loads(l).get('response','').strip()) and '\nHuman:' not in r and '\nDo you know' not in r[:50])
print(clean)
" 2>/dev/null || echo 0)
  if [ "$CLEAN" -ge 4000 ]; then
    SKIP_SFT_BASELINE=1
  fi
fi

# ============ Clean ALL artifacts ============
if [ "$SKIP_SFT" -ne 1 ]; then
  rm -rf $SLIME/models/sft_checkpoint
  rm -rf $SFT_HF_DIR
fi
rm -rf $SLIME/models/reward_model
rm -rf $SLIME/models/save_dir*
rm -rf $SLIME/models/policy_r*_hf
rm -rf $SLIME/rollout
rm -rf $SLIME/tensorboard_log
if [ "$SKIP_SFT_BASELINE" -ne 1 ]; then
  rm -rf $SLIME/eval
fi

# ============ Phase 0: SFT ============
echo "===== Phase 0: SFT ====="

if [ "$SKIP_SFT" -eq 1 ]; then
  echo "Skipping SFT: found $SFT_HF_DIR/config.json"
else
  MODEL_SH=scripts/models/qwen3-1.7B.sh \
  HF_CKPT=$SLIME/models/qwen3-1.7b-base \
  ACTOR_CKPT=$SLIME/models/qwen3-1.7b-base_torch_dist \
  SAVE_DIR=$SLIME/models/sft_checkpoint \
  SFT_DATA=$SLIME/hh-rlhf-processed/sft_10k.jsonl \
  bash scripts/run-sft-prod.sh
fi

# ============ Weight conversion ============
echo "===== Weight Conversion ====="
if [ "$SKIP_SFT" -eq 1 ]; then
  echo "Skipping weight conversion: found $SFT_HF_DIR/config.json"
else
  LATEST_ITER=$(cat $SLIME/models/sft_checkpoint/latest_checkpointed_iteration.txt)
  ITER_DIR=$SLIME/models/sft_checkpoint/iter_$(printf "%07d" $LATEST_ITER)
  if [ -f "$ITER_DIR/common.pt" ]; then CKPT_DIR=$ITER_DIR
  elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then CKPT_DIR=$ITER_DIR/mp_rank_00
  else echo "ERROR: common.pt not found"; exit 1; fi

  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir $CKPT_DIR \
    --output-dir $SFT_HF_DIR \
    --origin-hf-dir $SLIME/models/qwen3-1.7b-base \
    --force
fi

# Helper: generate outputs via SGLang (fast multi-GPU inference)
generate_with_sglang() {
  local MODEL_PATH=$1
  local OUTPUT=$2
  local PORT=30010  # avoid conflict with PPO's SGLang port 15000
  echo "Starting SGLang server on port $PORT for $MODEL_PATH"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --tp 4 \
    --host 127.0.0.1 \
    --trust-remote-code &
  SGLANG_PID=$!
  # Wait for server ready
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
      echo "SGLang ready after ${i}s"
      break
    fi
    sleep 2
  done
  python3 scripts/eval_generate_sglang.py \
    --model-path $MODEL_PATH \
    --sglang-url http://127.0.0.1:${PORT} \
    --prompt-data $SLIME/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
    --output $OUTPUT \
    --apply-chat-template \
    --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --max-new-tokens 512 \
    --concurrency 256
  kill $SGLANG_PID 2>/dev/null || true
  wait $SGLANG_PID 2>/dev/null || true
}

# ============ Generate SFT baseline outputs (for winrate comparison) ============
echo "===== Generating SFT baseline outputs ====="
mkdir -p $SLIME/eval
if [ "$SKIP_SFT_BASELINE" -eq 1 ]; then
  echo "Skipping SFT baseline generation: found $SFT_BASELINE"
else
  generate_with_sglang $SFT_HF_DIR $SFT_BASELINE
fi

GPT4O_REF=$SLIME/reference/outputs_gpt4o_full.jsonl  # permanent, not deleted by cleanup


# ============ Rounds 1-7 ============
CURRENT_HF_CKPT=$SFT_HF_DIR
for ROUND in $(seq 1 $NUM_ROUNDS); do
  ROUND_POLICY_HF=$SLIME/models/policy_r${ROUND}_hf
  ROUND_OUTPUT=$SLIME/eval/outputs_policy_r${ROUND}.jsonl
  if [ -f "$ROUND_POLICY_HF/config.json" ] && [ -f "$ROUND_OUTPUT" ] && [ "$(awk 'END {print NR}' "$ROUND_OUTPUT")" -ge 4000 ]; then
    echo "Skipping Round $ROUND: found $ROUND_POLICY_HF/config.json and complete $ROUND_OUTPUT"
    CURRENT_HF_CKPT=$ROUND_POLICY_HF
    continue
  fi

  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="

  rm -rf /tmp/save_dir_tmp /tmp/critic_ckpt 2>/dev/null || true

  MODEL_SH=scripts/models/qwen3-1.7B.sh \
  HF_CKPT=$CURRENT_HF_CKPT \
  REF_CKPT=$SLIME/models/sft_checkpoint \
  SAVE_DIR=/tmp/save_dir_tmp \
  PROMPT_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  DEMO_DATA=$SLIME/hh-rlhf-processed/hh-rlhf-merged-train.jsonl \
  NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND \
  TB_EXP_NAME=round${ROUND} \
  bash scripts/run-irl-prod.sh

  # ===== Convert checkpoint to HF and generate outputs (for winrate) =====
  echo "===== Round $ROUND: Generating policy outputs ====="
  ITER=$(cat /tmp/save_dir_tmp/latest_checkpointed_iteration.txt)
  ITER_DIR=/tmp/save_dir_tmp/iter_$(printf "%07d" $ITER)
  if [ -f "$ITER_DIR/common.pt" ]; then CKPT_DIR=$ITER_DIR
  elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then CKPT_DIR=$ITER_DIR/mp_rank_00
  else echo "ERROR: common.pt not found in $ITER_DIR"; exit 1; fi

  rm -rf $ROUND_POLICY_HF
  python3 tools/convert_torch_dist_to_hf.py \
    --input-dir $CKPT_DIR \
    --output-dir $ROUND_POLICY_HF \
    --origin-hf-dir $SLIME/models/qwen3-1.7b-base \
    --force

  echo "=== Generating Round $ROUND policy outputs (SGLang) ==="
  generate_with_sglang $ROUND_POLICY_HF $ROUND_OUTPUT

  CURRENT_HF_CKPT=$ROUND_POLICY_HF
  rm -rf /tmp/save_dir_tmp

  # ===== Analyze tensorboard metrics =====
  echo "===== Round $ROUND: Analyzing tensorboard ====="
  python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
log_dir = '/mnt/shared-storage-gpfs2/wangqianyi2/slime/tensorboard_log/slime-irl/round${ROUND}'
try:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    issues = []
    metrics = {}
    for tag in ['train/loss','train/pg_clipfrac','train/ppo_kl','train/grad_norm','rollout/truncated','rollout/rewards']:
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            if vals:
                metrics[tag] = {'first': vals[0], 'last': vals[-1], 'max': max(vals)}
    truncated = metrics.get('rollout/truncated', {}).get('last', 0)
    kl = metrics.get('train/ppo_kl', {}).get('last', 0)
    grad = metrics.get('train/grad_norm', {}).get('max', 0)
    clipfrac = metrics.get('train/pg_clipfrac', {}).get('last', 0)
    if truncated > 0.5: issues.append(f'HIGH TRUNCATION {truncated:.2f} (increase max_response_len)')
    if abs(kl) < 0.001: issues.append(f'KL~0 ({kl:.4f}) policy barely changed (reduce kl_coef?)')
    if kl > 0.1: issues.append(f'HIGH KL {kl:.4f} (policy drifting)')
    if grad > 10: issues.append(f'GRAD EXPLODING {grad:.2f}')
    if clipfrac > 0.3: issues.append(f'HIGH CLIP {clipfrac:.2f} (lr too high?)')
    print(f'Round ${ROUND} TB: truncated={truncated:.2f} kl={kl:.4f} grad_max={grad:.2f} clipfrac={clipfrac:.3f}')
    if issues: print('ISSUES: ' + '; '.join(issues))
    else: print('Metrics OK')
except Exception as e:
    print(f'TB analysis failed: {e}')
" 2>&1

  echo "===== Killing PPO processes ====="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  echo "===== Round $ROUND/$NUM_ROUNDS: Reward Update ====="

  SLIME=$SLIME \
  HF_CKPT=$SLIME/models/sft_checkpoint_hf \
  ROUND_ID=$ROUND \
  ROLLOUT_END=$(( NUM_ROLLOUT_PER_ROUND - 1 )) \
  NUM_ROLLOUT_PER_ROUND=$NUM_ROLLOUT_PER_ROUND \
  bash scripts/run-reward-update.sh

  echo "===== Round $ROUND/$NUM_ROUNDS completed ====="
done

echo "===== All $NUM_ROUNDS rounds completed ====="
echo "Eval outputs:"
ls -la $SLIME/eval/outputs_*.jsonl
