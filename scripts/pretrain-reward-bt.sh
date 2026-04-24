#!/bin/bash
# Standalone Bradley-Terry reward-model pretraining.
# Produces a checkpoint compatible with REWARD_MODEL_INIT for run-full-pipeline-job.sh.

pkill -9 sglang || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}

REWARD_BT_BASE_MODEL=${REWARD_BT_BASE_MODEL:-$SLIME/models/qwen3-1.7B-base}
REWARD_BT_ROOT=${REWARD_BT_ROOT:-$SLIME/models/reward_model_bt_pretrain}
REWARD_BT_OUTPUT_DIR=${REWARD_BT_OUTPUT_DIR:-$REWARD_BT_ROOT/latest}
REWARD_BT_LOG_DIR=${REWARD_BT_LOG_DIR:-$REWARD_BT_ROOT/logs}
REWARD_BT_ARGS_JSON=${REWARD_BT_ARGS_JSON:-$REWARD_BT_ROOT/pretrain_reward_bt_args.json}
REWARD_BT_RUN_TS=${REWARD_BT_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
REWARD_BT_RUN_TAG=${REWARD_BT_RUN_TAG:-bt_${REWARD_BT_RUN_TS}}
RUN_LOG=$REWARD_BT_LOG_DIR/pretrain_${REWARD_BT_RUN_TAG}.log
mkdir -p "$REWARD_BT_LOG_DIR" "$REWARD_BT_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

set -ex

REWARD_BT_PREF_PATH=${REWARD_BT_PREF_PATH:-$ULTRAFEEDBACK_DIR/uf-train-synth-prefs-clean.jsonl}
REWARD_BT_EVAL_PATH=${REWARD_BT_EVAL_PATH:-$ULTRAFEEDBACK_DIR/uf-test-synth-prefs-clean.jsonl}
REWARD_BT_PROMPT_KEY=${REWARD_BT_PROMPT_KEY:-text}
REWARD_BT_CHOSEN_KEY=${REWARD_BT_CHOSEN_KEY:-chosen}
REWARD_BT_REJECTED_KEY=${REWARD_BT_REJECTED_KEY:-rejected}
REWARD_BT_EPOCHS=${REWARD_BT_EPOCHS:-1}
REWARD_BT_BATCH_SIZE=${REWARD_BT_BATCH_SIZE:-4}
REWARD_BT_EVAL_BATCH_SIZE=${REWARD_BT_EVAL_BATCH_SIZE:-8}
REWARD_BT_LR=${REWARD_BT_LR:-1e-6}
REWARD_BT_HOLDOUT_RATIO=${REWARD_BT_HOLDOUT_RATIO:-0.1}
REWARD_BT_EVAL_INTERVAL=${REWARD_BT_EVAL_INTERVAL:-200}
REWARD_BT_GRAD_CLIP=${REWARD_BT_GRAD_CLIP:-1.0}
REWARD_BT_WEIGHT_DECAY=${REWARD_BT_WEIGHT_DECAY:-0.0}
REWARD_BT_GRADIENT_CHECKPOINTING=${REWARD_BT_GRADIENT_CHECKPOINTING:-true}
REWARD_BT_C_COEF_INIT=${REWARD_BT_C_COEF_INIT:-0.5}
REWARD_BT_TB_DIR=${REWARD_BT_TB_DIR:-$SLIME/tensorboard_log/slime-reward/bt_pretrain}
REWARD_BT_SEED=${REWARD_BT_SEED:-42}
REWARD_BT_NUM_PROCESSES=${REWARD_BT_NUM_PROCESSES:-8}

if [ ! -d "$REWARD_BT_BASE_MODEL" ]; then
  echo "ERROR: missing base model at $REWARD_BT_BASE_MODEL" >&2
  exit 1
fi
if [ ! -f "$REWARD_BT_PREF_PATH" ]; then
  echo "ERROR: missing BT pretrain data at $REWARD_BT_PREF_PATH" >&2
  exit 1
fi
if [ -n "$REWARD_BT_EVAL_PATH" ] && [ ! -f "$REWARD_BT_EVAL_PATH" ]; then
  echo "ERROR: missing BT eval data at $REWARD_BT_EVAL_PATH" >&2
  exit 1
fi

mkdir -p "$REWARD_BT_ROOT"

ARGS_JSON=$REWARD_BT_ARGS_JSON
cat > "$ARGS_JSON" <<EOF
{
  "base_model": "$REWARD_BT_BASE_MODEL",
  "pref_path": "$REWARD_BT_PREF_PATH",
  "eval_path": "$REWARD_BT_EVAL_PATH",
  "output_dir": "$REWARD_BT_OUTPUT_DIR",
  "prompt_key": "$REWARD_BT_PROMPT_KEY",
  "chosen_key": "$REWARD_BT_CHOSEN_KEY",
  "rejected_key": "$REWARD_BT_REJECTED_KEY",
  "eval_prompt_key": "$REWARD_BT_PROMPT_KEY",
  "eval_chosen_key": "$REWARD_BT_CHOSEN_KEY",
  "eval_rejected_key": "$REWARD_BT_REJECTED_KEY",
  "epochs": $REWARD_BT_EPOCHS,
  "batch_size": $REWARD_BT_BATCH_SIZE,
  "eval_batch_size": $REWARD_BT_EVAL_BATCH_SIZE,
  "lr": $REWARD_BT_LR,
  "holdout_ratio": $REWARD_BT_HOLDOUT_RATIO,
  "eval_interval": $REWARD_BT_EVAL_INTERVAL,
  "grad_clip_norm": $REWARD_BT_GRAD_CLIP,
  "weight_decay": $REWARD_BT_WEIGHT_DECAY,
  "gradient_checkpointing": $REWARD_BT_GRADIENT_CHECKPOINTING,
  "c_coef_init": $REWARD_BT_C_COEF_INIT,
  "tb_dir": "$REWARD_BT_TB_DIR",
  "seed": $REWARD_BT_SEED,
  "apply_chat_template": true,
  "apply_chat_template_kwargs": {"enable_thinking": false}
}
EOF

echo "=== BT reward pretrain (base=$REWARD_BT_BASE_MODEL, output=$REWARD_BT_OUTPUT_DIR, log=$RUN_LOG) ==="

TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL} \
PYTHONFAULTHANDLER=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
accelerate launch \
  --num_processes "$REWARD_BT_NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m slime.local_rm.pretrain_reward_bt \
  --args-json "$ARGS_JSON"

echo "=== BT reward pretrain completed ==="
echo "Checkpoint saved to: $REWARD_BT_OUTPUT_DIR"
echo "Full pipeline auto-detects this path on next run-full-pipeline-job.sh start."
echo "Manual IRL handoff: cp -r $REWARD_BT_OUTPUT_DIR \$REWARD_DIR/latest && export REWARD_MODEL_INIT=$REWARD_BT_BASE_MODEL"
echo "Note: REWARD_MODEL_INIT must point to the raw LLM backbone ($REWARD_BT_BASE_MODEL), not the scalar checkpoint."
