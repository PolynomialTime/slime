#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"

export PATH=/usr/local/nvidia/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

MODEL_ROOT=${MODEL_ROOT:-/mnt/shared-storage-user/ma4agi-gpu/wangqianyi}
QWEN35_HUB_ROOT=${QWEN35_HUB_ROOT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-9B-Base}
if [ -n "${QWEN35_SNAPSHOT:-}" ]; then
  QWEN35_HF_DIR="$QWEN35_HUB_ROOT/snapshots/$QWEN35_SNAPSHOT"
elif [ -s "$QWEN35_HUB_ROOT/refs/main" ]; then
  QWEN35_HF_DIR="$QWEN35_HUB_ROOT/snapshots/$(cat "$QWEN35_HUB_ROOT/refs/main")"
else
  QWEN35_HF_DIR=$(find "$QWEN35_HUB_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')
fi
if [ -z "$QWEN35_HF_DIR" ] || [ ! -f "$QWEN35_HF_DIR/config.json" ]; then
  echo "ERROR: cannot resolve Qwen3.5 HF snapshot under $QWEN35_HUB_ROOT" >&2
  exit 1
fi

mkdir -p "$MODEL_ROOT"
ln -sfn "$QWEN35_HF_DIR" "$MODEL_ROOT/qwen3.5-9B-base"

REWARD_BASE_SOURCE=${REWARD_BASE_SOURCE:-$SLIME/models/qwen3-1.7B-base}
if [ -d "$REWARD_BASE_SOURCE" ]; then
  ln -sfn "$REWARD_BASE_SOURCE" "$MODEL_ROOT/qwen3-1.7B-base"
fi

export MODEL_ROOT
export DATA_MODE=${DATA_MODE:-math}
export MODEL_SH=${MODEL_SH:-scripts/models/qwen3.5-9B.sh}
export MODEL_NAME_FOR_EXPORT=${MODEL_NAME_FOR_EXPORT:-qwen3_5}
export BASE_HF_DIR=${BASE_HF_DIR:-$MODEL_ROOT/qwen3.5-9B-base}
export BASE_TORCH_DIST_DIR=${BASE_TORCH_DIST_DIR:-$MODEL_ROOT/qwen3.5-9b-base_torch_dist}
export SFT_HF_DIR=${SFT_HF_DIR:-$MODEL_ROOT/qwen3.5-9b-math-sft_hf}
export SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$MODEL_ROOT/qwen3.5-9b-math-sft_megatron}
export REWARD_DIR=${REWARD_DIR:-$MODEL_ROOT/qwen3.5-9b-math-reward_model}
export REWARD_MODEL_INIT=${REWARD_MODEL_INIT:-$MODEL_ROOT/qwen3-1.7B-base}
export BT_PRETRAIN_DIR=${BT_PRETRAIN_DIR:-$MODEL_ROOT/qwen3.5-9b-reward_model_bt_pretrain/latest}
export BT_PRETRAIN_BASE_MODEL=${BT_PRETRAIN_BASE_MODEL:-$REWARD_MODEL_INIT}

# Keep Qwen3.5 math runtime artifacts out of $SLIME/eval, $SLIME/rollout, and
# $SLIME/tensorboard_log. Model outputs and generated artifacts stay under
# MODEL_ROOT by default.
export PIPELINE_ARTIFACT_ROOT=${PIPELINE_ARTIFACT_ROOT:-$MODEL_ROOT/qwen3.5-9b-math-artifacts}
export EVAL_DIR=${EVAL_DIR:-$PIPELINE_ARTIFACT_ROOT/eval}
export EVAL_WINRATE_DIR=${EVAL_WINRATE_DIR:-$PIPELINE_ARTIFACT_ROOT/eval_winrate}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$PIPELINE_ARTIFACT_ROOT/rollout}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-$PIPELINE_ARTIFACT_ROOT/tensorboard_log}
export CODEX_RULES_PATH=${CODEX_RULES_PATH:-$PIPELINE_ARTIFACT_ROOT/CODEX_HARD_RULES.md}
export REWARD_LOG_DIR=${REWARD_LOG_DIR:-$PIPELINE_ARTIFACT_ROOT/reward_logs}
export PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR:-$PIPELINE_ARTIFACT_ROOT/logs/pipeline}

# Four-H200 layout: full model replicas fit on one H200; keep TP=1 so actor,
# critic, and rollout can coexist on 4 GPUs.
export NUM_GPUS=${NUM_GPUS:-4}
export TOTAL_GPUS=${TOTAL_GPUS:-4}
export ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-1}
export CRITIC_NUM_GPUS=${CRITIC_NUM_GPUS:-1}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export COLLOCATE=${COLLOCATE:-0}
export REWARD_UPDATE_NUM_PROCESSES=${REWARD_UPDATE_NUM_PROCESSES:-4}
export REWARD_UPDATE_ROLLOUT_WINDOW=${REWARD_UPDATE_ROLLOUT_WINDOW:-150}

# Qwen3.5 tokenizer ids: <|endoftext|>=248044, <|im_start|>=248045,
# <|im_end|>=248046. Do not inherit Qwen3-8B's 151643/151644/151645.
export ROLLOUT_STOP_TOKEN_IDS_OVERRIDE=${ROLLOUT_STOP_TOKEN_IDS_OVERRIDE:-"248044 248045 248046"}
export SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_TP=${SGLANG_TP:-4}
export SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-}
export PIPELINE_RUN_TAG=${PIPELINE_RUN_TAG:-qwen35_math_$(date +%Y%m%d-%H%M%S)}

echo "Qwen3.5 math config:"
echo "  snapshot=$QWEN35_HF_DIR"
echo "  model_root=$MODEL_ROOT"
echo "  artifact_root=$PIPELINE_ARTIFACT_ROOT"
echo "  base_hf=$BASE_HF_DIR"
echo "  base_torch_dist=$BASE_TORCH_DIST_DIR"
echo "  sft_hf=$SFT_HF_DIR"
echo "  sft_megatron=$SFT_MEGATRON_DIR"
echo "  reward_dir=$REWARD_DIR"
echo "  reward_init=$REWARD_MODEL_INIT"
  echo "  gpu_layout total=$TOTAL_GPUS actor=$ACTOR_NUM_GPUS critic=$CRITIC_NUM_GPUS rollout=$ROLLOUT_NUM_GPUS rollout_per_engine=$ROLLOUT_NUM_GPUS_PER_ENGINE tp=$TENSOR_MODEL_PARALLEL_SIZE colocate=$COLLOCATE"

if [ "${QWEN35_DRY_RUN:-0}" = "1" ]; then
  echo "QWEN35_DRY_RUN=1: resolved configuration only; not launching pipeline."
  exit 0
fi

bash scripts/run-full-pipeline-logged.sh "$@"
