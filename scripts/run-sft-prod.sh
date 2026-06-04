#!/bin/bash

# SFT training script — 8×H200, Qwen3-8B, UltraFeedback SFT

# for rerun the task
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONUNBUFFERED=1
NUM_GPUS=${NUM_GPUS:-8}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT=${SLIME_ROOT:-$(dirname "$SCRIPT_DIR")}

# Model args
if [ -n "${MODEL_SH}" ]; then
  source "${MODEL_SH}"
fi
if [ -z "${MODEL_ARGS+x}" ]; then
  echo "MODEL_ARGS not set. Provide MODEL_SH=... to source a model config."
  exit 1
fi

# Required paths
HF_CKPT=${HF_CKPT:-"/path/to/hf_ckpt"}
ACTOR_CKPT=${ACTOR_CKPT:-"/path/to/actor_ckpt"}
SAVE_DIR=${SAVE_DIR:-"/path/to/save_dir"}
SFT_DATA=${SFT_DATA:-"/path/to/sft_data.jsonl"}
SFT_INPUT_KEY=${SFT_INPUT_KEY:-messages}
SFT_LABEL_KEY=${SFT_LABEL_KEY:-}

if [[ "$HF_CKPT" == "/path/to/"* ]]; then
  echo "Please set HF_CKPT/ACTOR_CKPT/SAVE_DIR/SFT_DATA."
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${ACTOR_CKPT}
   --load ${SAVE_DIR}
   --save ${SAVE_DIR}
   # Only save at end-of-training. slime/utils/misc.py:should_run_periodic_action
   # always triggers on the final rollout_id regardless of save_interval, so
   # setting this larger than total iters effectively disables mid-training saves.
   # An 8B Megatron checkpoint with optimizer state is ~107GB; storing only the
   # final one saves one full copy per SFT run. Trade-off: no resume from a mid-
   # training crash (acceptable for one-shot SFT warmup).
   --save-interval 999999
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${SFT_DATA}
   --input-key ${SFT_INPUT_KEY}
   --rollout-shuffle
   --num-epoch ${SFT_NUM_EPOCHS:-2}
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

if [ -n "${SFT_LABEL_KEY}" ]; then
  SFT_ARGS+=(--label-key ${SFT_LABEL_KEY})
fi

SFT_APPLY_CHAT_TEMPLATE=${SFT_APPLY_CHAT_TEMPLATE:-0}
SFT_APPLY_CHAT_TEMPLATE_KWARGS=${SFT_APPLY_CHAT_TEMPLATE_KWARGS:-'{"enable_thinking":false}'}
if [ "${SFT_APPLY_CHAT_TEMPLATE}" = "1" ]; then
  SFT_ARGS+=(--apply-chat-template)
  if [ -n "${SFT_APPLY_CHAT_TEMPLATE_KWARGS}" ]; then
    SFT_ARGS+=(--apply-chat-template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS}")
  fi
fi

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)
if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name slime-sft
   --tb-experiment-name qwen3-8b-sft
)
TENSORBOARD_DIR=${TENSORBOARD_DIR:-tensorboard_log/slime-sft/qwen3-8b-sft}

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}:/root/Megatron-LM/\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
