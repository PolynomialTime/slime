#!/bin/bash

# Generic IRL training script (PPO -> reward update -> PPO ...)
# Usage:
#   MODEL_SH=scripts/models/qwen3-1.7B.sh \
#   HF_CKPT=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/models/qwen3-1.7b-base \
#   REF_CKPT=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/models/qwen3-1.7b-base_torch_dist \
#   ACTOR_CKPT=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/models/qwen3-1.7b-base_torch_dist \
#   SAVE_DIR=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/models/save_dir \
#   PROMPT_DATA=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/hh-rlhf-processed/hh-rlhf-merged-train-debug.jsonl \
#   DEMO_DATA=/mnt/shared-storage-user/ma4agi-gpu/wangqianyi/slime/hh-rlhf-processed/hh-rlhf-merged-train-debug.jsonl \
#   REWARD_UPDATE_LAUNCHER=accelerate
#   bash scripts/run-irl.sh

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

export PYTHONBUFFERED=1

# Reserve GPU 3 for reward update (accelerate), leave training on 0,1,2.
# Adjust if your hardware indices differ.
export CUDA_VISIBLE_DEVICES=0,1,2
export REWARD_UPDATE_ACCELERATE_NUM_PROC=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Model args (optional)
if [ -n "${MODEL_SH}" ]; then
  source "${MODEL_SH}"
fi
if [ -z "${MODEL_ARGS+x}" ]; then
  echo "MODEL_ARGS not set. Provide MODEL_SH=... to source a model config."
  exit 1
fi

# Required paths (set via env)
HF_CKPT=${HF_CKPT:-"/path/to/hf_ckpt"}
REF_CKPT=${REF_CKPT:-"/path/to/ref_ckpt"}
ACTOR_CKPT=${ACTOR_CKPT:-"/path/to/actor_ckpt"}
SAVE_DIR=${SAVE_DIR:-"/path/to/save_dir"}
PROMPT_DATA=${PROMPT_DATA:-"/path/to/prompt.jsonl"}
DEMO_DATA=${DEMO_DATA:-"/path/to/demo.jsonl"}

if [[ "$HF_CKPT" == "/path/to/"* ]]; then
  echo "Please set HF_CKPT/REF_CKPT/ACTOR_CKPT/SAVE_DIR/PROMPT_DATA/DEMO_DATA."
  exit 1
fi

# Resource config
ACTOR_GPUS=${ACTOR_GPUS:-1}
CRITIC_GPUS=${CRITIC_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
USE_COLOCATE=${USE_COLOCATE:-0}

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_CKPT}
   --load ${ACTOR_CKPT}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key text
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --num-rollout 7
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 96
   --rollout-temperature 0.8

   --global-batch-size 4
   --balance-data
)

PPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

IRL_ARGS=(
   --custom-rm-path slime.local_rm.custom_rm.custom_rm
   --reward-demo-path ${DEMO_DATA}
   --reward-demo-prompt-key text
   --reward-demo-answer-key label
   --reward-eval-path /mnt/shared-storage-user/wangqianyi/slime/hh-rlhf-processed/hh-rlhf-merged-test.jsonl
   --reward-eval-prompt-key text
   --reward-eval-chosen-key chosen
   --reward-eval-rejected-key rejected
   --reward-model-dir /mnt/shared-storage-user/wangqianyi/slime/models/reward_model
   --reward-update-interval 1
   --reward-update-epochs 1
   --reward-update-batch-size 8
   --reward-update-lr 1e-5
   --reward-update-cuda-visible-devices 3
   --save-debug-rollout-data /mnt/shared-storage-user/wangqianyi/slime/rollout/rollout_{rollout_id}.pt
)

IRL_ARGS+=(--reward-update-launcher ${REWARD_UPDATE_LAUNCHER:-direct})
if [ -n "${REWARD_UPDATE_ACCELERATE_CONFIG}" ]; then
  IRL_ARGS+=(--reward-update-accelerate-config ${REWARD_UPDATE_ACCELERATE_CONFIG})
fi
if [ -n "${REWARD_UPDATE_ACCELERATE_NUM_PROC}" ]; then
  IRL_ARGS+=(--reward-update-accelerate-num-proc ${REWARD_UPDATE_ACCELERATE_NUM_PROC})
fi

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime /path/to/aime.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group irl-run
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [ "$USE_COLOCATE" -eq 1 ]; then
  COLOCATE_ARGS=(--colocate)
  ROLLOUT_RESOURCE_ARGS=()
else
  COLOCATE_ARGS=()
  ROLLOUT_RESOURCE_ARGS=(--rollout-num-gpus ${ROLLOUT_GPUS})
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_irl.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --critic-num-nodes 1 \
   --critic-num-gpus-per-node ${CRITIC_GPUS} \
   --num-gpus-per-node 4 \
   ${COLOCATE_ARGS[@]} \
   ${ROLLOUT_RESOURCE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${IRL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
