#!/bin/bash

# Reward Update Phase -- single-GPU via accelerate
# Called by run-full-pipeline-job.sh after PPO phase
# All GPUs are free (PPO processes killed before this runs)

pkill -9 sglang || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

REWARD_DIR=${REWARD_DIR:-/mnt/shared-storage-gpfs2/wangqianyi2/slime/models/reward_model}
REWARD_LOG_DIR=${REWARD_LOG_DIR:-$REWARD_DIR/logs}
REWARD_RUN_TS=${REWARD_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
REWARD_RUN_TAG=${REWARD_RUN_TAG:-round${ROUND_ID:-0}_rollout${ROLLOUT_END:-335}_${REWARD_RUN_TS}}
RUN_LOG=$REWARD_LOG_DIR/reward_update_${REWARD_RUN_TAG}.log
mkdir -p "$REWARD_LOG_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

set -ex

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd $SLIME
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}

ROUND_ID=${ROUND_ID:-0}
ROLLOUT_END=${ROLLOUT_END:-335}
NUM_ROLLOUT_PER_ROUND=${NUM_ROLLOUT_PER_ROUND:-336}
REWARD_TRAIN_DATA_PATH=${REWARD_TRAIN_DATA_PATH:-${REWARD_TRAIN_SYNTH_PATH:-$ULTRAFEEDBACK_DIR/uf-train-prefs.jsonl}}
REWARD_EVAL_PATH=${REWARD_EVAL_PATH:-}
REWARD_EVAL_TARGET_PATH=${REWARD_EVAL_TARGET_PATH:-}
REWARD_EVAL_REJECTED_KEY=${REWARD_EVAL_REJECTED_KEY:-}
REWARD_EVAL_HOLDOUT_RATIO=${REWARD_EVAL_HOLDOUT_RATIO:-0.1}
REWARD_EVAL_MAX_SAMPLES=${REWARD_EVAL_MAX_SAMPLES:-0}
REWARD_EVAL_SHUFFLE_SEED=${REWARD_EVAL_SHUFFLE_SEED:-42}
REWARD_EVAL_BATCH_SIZE=${REWARD_EVAL_BATCH_SIZE:-32}
REWARD_UPDATE_BATCH_SIZE=${REWARD_UPDATE_BATCH_SIZE:-4}
REWARD_UPDATE_EPOCHS=${REWARD_UPDATE_EPOCHS:-1}
REWARD_ONLINE_PREF_WEIGHT=${REWARD_ONLINE_PREF_WEIGHT:-1.0}

if [ ! -f "$REWARD_TRAIN_DATA_PATH" ]; then
  echo "ERROR: missing reward train data at $REWARD_TRAIN_DATA_PATH" >&2
  exit 1
fi

if [ -n "$REWARD_EVAL_PATH" ] && [ ! -f "$REWARD_EVAL_PATH" ]; then
  echo "ERROR: missing external reward eval data at $REWARD_EVAL_PATH" >&2
  exit 1
fi

if [ -n "$REWARD_EVAL_TARGET_PATH" ] && [ ! -f "$REWARD_EVAL_TARGET_PATH" ]; then
  echo "ERROR: missing external reward eval target data at $REWARD_EVAL_TARGET_PATH" >&2
  exit 1
fi

REWARD_EVAL_MAX_SAMPLES_JSON=$REWARD_EVAL_MAX_SAMPLES
if [ -z "$REWARD_EVAL_MAX_SAMPLES_JSON" ] || [ "$REWARD_EVAL_MAX_SAMPLES_JSON" -le 0 ]; then
  REWARD_EVAL_MAX_SAMPLES_JSON=null
fi

REWARD_DIR=${REWARD_DIR:-$SLIME/models/reward_model}
mkdir -p $REWARD_DIR

ARGS_JSON=$REWARD_DIR/reward_update_args.json
cat > $ARGS_JSON <<EOF
{
  "hf_checkpoint": "${HF_CKPT:-$SLIME/models/sft_checkpoint_8b_hf}",
  "reward_model_dir": "$REWARD_DIR",
  "reward_model_init": null,
  "reward_demo_path": "$REWARD_TRAIN_DATA_PATH",
  "reward_demo_prompt_key": "text",
  "reward_demo_answer_key": "chosen",
  "reward_online_pref_weight": $REWARD_ONLINE_PREF_WEIGHT,
  "reward_update_epochs": $REWARD_UPDATE_EPOCHS,
  "reward_update_batch_size": $REWARD_UPDATE_BATCH_SIZE,
  "reward_update_lr": 1e-6,
  "c_coef_init": 0.5,
  "c_coef_min": 0.01,
  "c_coef_max": 10.0,
  "coef_scale_up": 1.2,
  "coef_scale_down": 0.95,
  "target_reward_l2_norm": 1.5,
  "apply_chat_template": true,
  "apply_chat_template_kwargs": {"enable_thinking": false},
  "save_debug_rollout_data": "$SLIME/rollout/rollout_{rollout_id}.pt",
  "reward_update_rollout_window": 150,
  "reward_eval_path": "$REWARD_EVAL_PATH",
  "reward_eval_prompt_key": "text",
  "reward_eval_chosen_key": "chosen",
  "reward_eval_rejected_key": "$REWARD_EVAL_REJECTED_KEY",
  "reward_eval_target_path": "$REWARD_EVAL_TARGET_PATH",
  "reward_eval_target_prompt_key": "prompt",
  "reward_eval_target_answer_key": "response",
  "reward_eval_holdout_ratio": $REWARD_EVAL_HOLDOUT_RATIO,
  "reward_eval_max_samples": $REWARD_EVAL_MAX_SAMPLES_JSON,
  "reward_eval_shuffle_seed": $REWARD_EVAL_SHUFFLE_SEED,
  "reward_eval_batch_size": $REWARD_EVAL_BATCH_SIZE
}
EOF

echo "=== Reward Update Phase (round $ROUND_ID, rollout_end=$ROLLOUT_END, batch=$REWARD_UPDATE_BATCH_SIZE, log=$RUN_LOG) ==="

TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL} \
PYTHONFAULTHANDLER=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
ROUND_ID=${ROUND_ID:-0} accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  -m slime.local_rm.update_reward_accel \
  --args-json $ARGS_JSON \
  --rollout-id $ROLLOUT_END \
  --rollout-path $SLIME/rollout/rollout_${ROLLOUT_END}.pt

echo "=== Reward Update Phase completed ==="
