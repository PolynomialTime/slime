#!/bin/bash

# PPO Phase — 4×GPU, no reward update
# Called by run-full-pipeline-job.sh for each round

# kill previous processes
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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -n "${MODEL_SH}" ]; then
  source "${MODEL_SH}"
fi
if [ -z "${MODEL_ARGS+x}" ]; then
  echo "MODEL_ARGS not set."
  exit 1
fi

HF_CKPT=${HF_CKPT:-"/path/to/hf_ckpt"}
REF_CKPT=${REF_CKPT:-"/path/to/ref_ckpt"}
SAVE_DIR=${SAVE_DIR:-"/path/to/save_dir"}
PROMPT_DATA=${PROMPT_DATA:-"/path/to/prompt.jsonl"}
DEMO_DATA=${DEMO_DATA:-"/path/to/demo.jsonl"}
SLIME_ROOT=${SLIME_ROOT:-$(dirname "$SCRIPT_DIR")}
NUM_ROLLOUT=${NUM_ROLLOUT:-336}

if [[ "$HF_CKPT" == "/path/to/"* ]]; then
  echo "Please set required env vars."
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_CKPT}
   --no-load-optim
   --no-load-rng
   --save ${SAVE_DIR}
   --critic-save /tmp/critic_ckpt
   --save-interval 999
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key text
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-stop-token-ids 151645
   --rollout-shuffle

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --rollout-max-response-len 512
   --rollout-temperature 1.0

   --global-batch-size 64
   --balance-data
)

PPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.05
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

IRL_ARGS=(
   --custom-rm-path slime.local_rm.custom_rm.custom_rm
   --reward-model-dir ${SLIME_ROOT}/models/reward_model
   --reward-update-interval 999999
   --save-debug-rollout-data ${SLIME_ROOT}/rollout/rollout_{rollout_id}.pt
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
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

TB_EXP_NAME=${TB_EXP_NAME:-prod}

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name slime-irl
   --tb-experiment-name ${TB_EXP_NAME}
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

# 4 GPUs for PPO
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Save real CUDA devices before Ray overrides them (for custom_rm to use GPU 3)
export _REAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"_REAL_CUDA_VISIBLE_DEVICES\": \"${_REAL_CUDA_VISIBLE_DEVICES}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_irl.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --critic-num-nodes 1 \
   --critic-num-gpus-per-node 1 \
   --num-gpus-per-node 4 \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${IRL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
