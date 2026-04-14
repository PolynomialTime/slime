#!/bin/bash

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
ACTOR_LOAD=${ACTOR_LOAD:-""}
PROMPT_DATA=${PROMPT_DATA:-"/path/to/prompt.jsonl"}
DEMO_DATA=${DEMO_DATA:-"/path/to/demo.jsonl"}
SLIME_ROOT=${SLIME_ROOT:-$(dirname "$SCRIPT_DIR")}
NUM_ROLLOUT=${NUM_ROLLOUT:-336}
PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-$NUM_ROLLOUT}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
ALIGN_ROLLOUT_WITH_SFT=${ALIGN_ROLLOUT_WITH_SFT:-0}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
ACTOR_LR=${ACTOR_LR:-5e-7}
CRITIC_LR=${CRITIC_LR:-1e-6}
CRITIC_LR_WARMUP_ITERS=${CRITIC_LR_WARMUP_ITERS:-10}
CLIP_GRAD=${CLIP_GRAD:-0.5}
CRITIC_CLIP_GRAD=${CRITIC_CLIP_GRAD:-10.0}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144}
KL_LOSS_COEF=${KL_LOSS_COEF:-1.0}
REWARD_MODEL_DIR=${REWARD_MODEL_DIR:-${SLIME_ROOT}/models/reward_model}
CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-/tmp/critic_ckpt}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SLIME_ROOT}/rollout}
ROLLOUT_DEBUG_PATH_TEMPLATE=${ROLLOUT_DEBUG_PATH_TEMPLATE:-}
if [ -z "${ROLLOUT_DEBUG_PATH_TEMPLATE}" ]; then
  ROLLOUT_DEBUG_PATH_TEMPLATE="${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
fi

if [ "${ALIGN_ROLLOUT_WITH_SFT}" = "1" ]; then
  # Qwen chat-template boundaries: stop on end-of-turn, next-turn start, or end-of-text.
  IFS=' ' read -r -a ROLLOUT_STOP_TOKEN_IDS <<< "${ROLLOUT_STOP_TOKEN_IDS_OVERRIDE:-151643 151644 151645}"
else
  IFS=' ' read -r -a ROLLOUT_STOP_TOKEN_IDS <<< "${ROLLOUT_STOP_TOKEN_IDS_OVERRIDE:-151645}"
fi

if [[ "$HF_CKPT" == "/path/to/"* ]]; then
  echo "Please set required env vars."
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_CKPT}
   --no-load-optim
   --no-load-rng
   --finetune
   --no-save-optim
   --save ${SAVE_DIR}
   --critic-save ${CRITIC_SAVE_DIR}
   --save-interval ${PPO_SAVE_INTERVAL}
)
if [ -n "$ACTOR_LOAD" ] && [ -d "$ACTOR_LOAD" ]; then
  CKPT_ARGS+=(--load ${ACTOR_LOAD})
fi

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key text
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-stop-token-ids "${ROLLOUT_STOP_TOKEN_IDS[@]}"
   --rollout-shuffle

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 1
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature ${ROLLOUT_TEMPERATURE}

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

if [ "${ALIGN_ROLLOUT_WITH_SFT}" = "1" ]; then
  ROLLOUT_ARGS+=(
     --rollout-skip-special-tokens
  )
fi

PPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF}
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --value-clip 5.0
   --normalize-advantages
)

IRL_ARGS=(
   --custom-rm-path slime.local_rm.custom_rm.custom_rm
   --reward-model-dir ${REWARD_MODEL_DIR}
   --reward-update-interval 999999
   --save-debug-rollout-data ${ROLLOUT_DEBUG_PATH_TEMPLATE}
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${ACTOR_LR}
   --critic-lr ${CRITIC_LR}
   --critic-lr-warmup-iters ${CRITIC_LR_WARMUP_ITERS}
   --critic-clip-grad ${CRITIC_CLIP_GRAD}
   --clip-grad ${CLIP_GRAD}
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
   --sglang-mem-fraction-static 0.9
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

EXTRA_RUN_ARGS=()
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
  EXTRA_RUN_ARGS+=(--debug-rollout-only)
fi

RUN_PPO_ARGS=("${PPO_ARGS[@]}")
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
  # Pure bootstrap collection should not instantiate PPO/critic-specific training state.
  RUN_PPO_ARGS=()
fi

mkdir -p "${SAVE_DIR}" "${CRITIC_SAVE_DIR}" "$(dirname "${ROLLOUT_DEBUG_PATH_TEMPLATE}")"

echo "Effective rollout config: num_rollout=${NUM_ROLLOUT} batch=${ROLLOUT_BATCH_SIZE} max_new_tokens=${ROLLOUT_MAX_RESPONSE_LEN} temperature=${ROLLOUT_TEMPERATURE} align_with_sft=${ALIGN_ROLLOUT_WITH_SFT} stop_token_ids=${ROLLOUT_STOP_TOKEN_IDS[*]} debug_rollout_only=${DEBUG_ROLLOUT_ONLY} use_ppo_args=$([ \"${DEBUG_ROLLOUT_ONLY}\" = \"1\" ] && echo 0 || echo 1)"
echo "Effective checkpoint config: hf_ckpt=${HF_CKPT} actor_load=${ACTOR_LOAD:-<none>} ref_ckpt=${REF_CKPT} reward_model_dir=${REWARD_MODEL_DIR} save_dir=${SAVE_DIR} critic_save_dir=${CRITIC_SAVE_DIR} rollout_debug_path=${ROLLOUT_DEBUG_PATH_TEMPLATE} save_interval=${PPO_SAVE_INTERVAL} tb_experiment=${TB_EXP_NAME:-prod}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export _REAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Let custom_rm pick the least-occupied GPU from the visible set instead of pinning GPU 0.
export SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES="${SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES:-auto}"
export SLIME_CUSTOM_RM_MAX_RESPONSE_LEN="${SLIME_CUSTOM_RM_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"_REAL_CUDA_VISIBLE_DEVICES\": \"${_REAL_CUDA_VISIBLE_DEVICES}\",
    \"SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES\": \"${SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES}\",
    \"SLIME_CUSTOM_RM_MAX_RESPONSE_LEN\": \"${SLIME_CUSTOM_RM_MAX_RESPONSE_LEN}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_irl.py \
   ${EXTRA_RUN_ARGS[@]} \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --critic-num-nodes 1 \
   --critic-num-gpus-per-node 2 \
   --num-gpus-per-node 8 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${RUN_PPO_ARGS[@]} \
   ${IRL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
