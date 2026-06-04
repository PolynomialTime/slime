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
export DATA_MODE=${DATA_MODE:-ultrafeedback}
export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
export MODEL_SH=${MODEL_SH:-scripts/models/qwen3.5-9B.sh}
export MODEL_NAME_FOR_EXPORT=${MODEL_NAME_FOR_EXPORT:-qwen3_5}
export BASE_HF_DIR=${BASE_HF_DIR:-$MODEL_ROOT/qwen3.5-9B-base}
export BASE_TORCH_DIST_DIR=${BASE_TORCH_DIST_DIR:-$MODEL_ROOT/qwen3.5-9b-base_torch_dist}
SHARED_QWEN35_SFT_HF=${SHARED_QWEN35_SFT_HF:-$SLIME/mtbench_hf/qwen35_uf_sft_slime_20260519_111351_hf}
SHARED_QWEN35_SFT_MEGATRON=${SHARED_QWEN35_SFT_MEGATRON:-$SLIME/qwen35_uf_ppo_worker7zdh8/qwen3.5-9b-uf-sft_megatron_from_exact_hf}
REQUIRE_SHARED_QWEN35_SFT=${REQUIRE_SHARED_QWEN35_SFT:-1}
if [ "$REQUIRE_SHARED_QWEN35_SFT" = "1" ]; then
  test -f "$SHARED_QWEN35_SFT_HF/config.json"
  test -f "$SHARED_QWEN35_SFT_HF/SLIME_SFT_MANIFEST.txt"
  grep -q '^run_tag=qwen35_uf_sft_slime_20260519_111351$' "$SHARED_QWEN35_SFT_HF/SLIME_SFT_MANIFEST.txt"
  grep -q '^sft_data_rows=10000$' "$SHARED_QWEN35_SFT_HF/SLIME_SFT_MANIFEST.txt"
  test -f "$SHARED_QWEN35_SFT_MEGATRON/latest_checkpointed_iteration.txt"
  test "$(cat "$SHARED_QWEN35_SFT_MEGATRON/latest_checkpointed_iteration.txt")" = release
  if [ -n "${SFT_HF_DIR:-}" ] && [ "$SFT_HF_DIR" != "$SHARED_QWEN35_SFT_HF" ]; then
    echo "ERROR: SFT_HF_DIR must be the shared qwen35 SFT: $SHARED_QWEN35_SFT_HF" >&2
    exit 1
  fi
  if [ -n "${SFT_MEGATRON_DIR:-}" ] && [ "$SFT_MEGATRON_DIR" != "$SHARED_QWEN35_SFT_MEGATRON" ]; then
    echo "ERROR: SFT_MEGATRON_DIR must be the shared qwen35 SFT: $SHARED_QWEN35_SFT_MEGATRON" >&2
    exit 1
  fi
  export SFT_HF_DIR=$SHARED_QWEN35_SFT_HF
  export SFT_MEGATRON_DIR=$SHARED_QWEN35_SFT_MEGATRON
  export SFT_ENABLED=0
elif [ -f "$SHARED_QWEN35_SFT_HF/SLIME_SFT_MANIFEST.txt" ] && [ -f "$SHARED_QWEN35_SFT_MEGATRON/latest_checkpointed_iteration.txt" ]; then
  export SFT_HF_DIR=${SFT_HF_DIR:-$SHARED_QWEN35_SFT_HF}
  export SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$SHARED_QWEN35_SFT_MEGATRON}
  export SFT_ENABLED=${SFT_ENABLED:-0}
else
  export SFT_HF_DIR=${SFT_HF_DIR:-$MODEL_ROOT/qwen3.5-9b-uf-sft_hf}
  export SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$MODEL_ROOT/qwen3.5-9b-uf-sft_megatron}
fi
export REWARD_DIR=${REWARD_DIR:-$MODEL_ROOT/qwen3.5-9b-uf-reward_model}
export REWARD_MODEL_INIT=${REWARD_MODEL_INIT:-$MODEL_ROOT/qwen3-1.7B-base}
export BT_PRETRAIN_ROOT=${BT_PRETRAIN_ROOT:-$MODEL_ROOT/qwen3.5-9b-uf-reward_model_bt_pretrain}
export BT_PRETRAIN_DIR=${BT_PRETRAIN_DIR:-$BT_PRETRAIN_ROOT/latest}
export BT_PRETRAIN_BASE_MODEL=${BT_PRETRAIN_BASE_MODEL:-$REWARD_MODEL_INIT}

# Keep UltraFeedback runtime artifacts out of $SLIME/eval, $SLIME/rollout, and
# $SLIME/tensorboard_log so a concurrent math run in another worker is not touched.
export PIPELINE_ARTIFACT_ROOT=${PIPELINE_ARTIFACT_ROOT:-$MODEL_ROOT/qwen3.5-9b-uf-artifacts}
export EVAL_DIR=${EVAL_DIR:-$PIPELINE_ARTIFACT_ROOT/eval}
export EVAL_WINRATE_DIR=${EVAL_WINRATE_DIR:-$PIPELINE_ARTIFACT_ROOT/eval_winrate}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$PIPELINE_ARTIFACT_ROOT/rollout}
export TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-$PIPELINE_ARTIFACT_ROOT/tensorboard_log}
export CODEX_RULES_PATH=${CODEX_RULES_PATH:-$PIPELINE_ARTIFACT_ROOT/CODEX_HARD_RULES.md}
export REWARD_LOG_DIR=${REWARD_LOG_DIR:-$PIPELINE_ARTIFACT_ROOT/reward_logs}
export PIPELINE_LOG_DIR=${PIPELINE_LOG_DIR:-$PIPELINE_ARTIFACT_ROOT/logs/pipeline}

# Four-H200 layout: full Qwen3.5-9B replicas fit on one H200; keep TP=1 for
# train actors and use two one-GPU rollout engines.
export NUM_GPUS=${NUM_GPUS:-4}
export TOTAL_GPUS=${TOTAL_GPUS:-4}
export ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-1}
export CRITIC_NUM_GPUS=${CRITIC_NUM_GPUS:-1}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-2}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export REWARD_UPDATE_NUM_PROCESSES=${REWARD_UPDATE_NUM_PROCESSES:-4}

# Qwen3.5 tokenizer ids: <|endoftext|>=248044, <|im_start|>=248045,
# <|im_end|>=248046. Do not inherit Qwen3-8B's 151643/151644/151645.
export ROLLOUT_STOP_TOKEN_IDS_OVERRIDE=${ROLLOUT_STOP_TOKEN_IDS_OVERRIDE:-"248044 248045 248046"}
export SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_TP=${SGLANG_TP:-4}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
export PIPELINE_RUN_TAG=${PIPELINE_RUN_TAG:-qwen35_uf_$(date +%Y%m%d-%H%M%S)}
if [ "${ADVANTAGE_ESTIMATOR}" = "grpo" ] || [ "${ADVANTAGE_ESTIMATOR}" = "gspo" ]; then
  export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
  export NORMALIZE_ADVANTAGES=${NORMALIZE_ADVANTAGES:-0}
fi
export SLIME_RUBRIC_ENABLED=${SLIME_RUBRIC_ENABLED:-0}
export SLIME_RUBRIC_PLUGINS=${SLIME_RUBRIC_PLUGINS:-none}
export SLIME_WARMUP_RM_ENABLED=${SLIME_WARMUP_RM_ENABLED:-0}
export SLIME_WARMUP_RM_DIR=${SLIME_WARMUP_RM_DIR:-}
export SLIME_WARMUP_RM_COEF=${SLIME_WARMUP_RM_COEF:-0.0}

echo "Qwen3.5 UltraFeedback config:"
echo "  snapshot=$QWEN35_HF_DIR"
echo "  model_root=$MODEL_ROOT"
echo "  artifact_root=$PIPELINE_ARTIFACT_ROOT"
echo "  base_hf=$BASE_HF_DIR"
echo "  base_torch_dist=$BASE_TORCH_DIST_DIR"
echo "  sft_hf=$SFT_HF_DIR"
echo "  sft_megatron=$SFT_MEGATRON_DIR"
echo "  reward_dir=$REWARD_DIR"
echo "  reward_init=$REWARD_MODEL_INIT"
echo "  advantage_estimator=$ADVANTAGE_ESTIMATOR n_samples_per_prompt=${N_SAMPLES_PER_PROMPT:-<pipeline-default>} normalize_advantages=${NORMALIZE_ADVANTAGES:-<pipeline-default>}"
echo "  rollout_batch_size=$ROLLOUT_BATCH_SIZE"
echo "  gpu_layout total=$TOTAL_GPUS actor=$ACTOR_NUM_GPUS critic=$CRITIC_NUM_GPUS rollout=$ROLLOUT_NUM_GPUS tp=$TENSOR_MODEL_PARALLEL_SIZE"

if [ "${QWEN35_DRY_RUN:-0}" = "1" ]; then
  echo "QWEN35_DRY_RUN=1: resolved configuration only; not launching pipeline."
  exit 0
fi

bash scripts/run-full-pipeline-logged.sh "$@"
