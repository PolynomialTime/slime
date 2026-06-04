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

TOTAL_GPUS=${TOTAL_GPUS:-8}
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-2}
CRITIC_NUM_GPUS=${CRITIC_NUM_GPUS:-2}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
COLLOCATE=${COLLOCATE:-0}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.9}
TRAIN_MEMORY_MARGIN_BYTES=${TRAIN_MEMORY_MARGIN_BYTES:-}
DISABLE_WEIGHTS_BACKUPER=${DISABLE_WEIGHTS_BACKUPER:-0}
PPO_FINETUNE=${PPO_FINETUNE:-1}

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
PROMPT_INPUT_KEY=${PROMPT_INPUT_KEY:-text}
PROMPT_LABEL_KEY=${PROMPT_LABEL_KEY:-}
SLIME_ROOT=${SLIME_ROOT:-$(dirname "$SCRIPT_DIR")}
NUM_ROLLOUT=${NUM_ROLLOUT:-336}
PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-$NUM_ROLLOUT}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-256}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.4}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-ppo}
NORMALIZE_ADVANTAGES=${NORMALIZE_ADVANTAGES:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
ALIGN_ROLLOUT_WITH_SFT=${ALIGN_ROLLOUT_WITH_SFT:-0}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
ACTOR_LR=${ACTOR_LR:-3e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}
CRITIC_LR_WARMUP_ITERS=${CRITIC_LR_WARMUP_ITERS:-10}
CLIP_GRAD=${CLIP_GRAD:-0.5}
CRITIC_CLIP_GRAD=${CRITIC_CLIP_GRAD:-10.0}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-6144}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:--1}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.10}
REWARD_MODEL_DIR=${REWARD_MODEL_DIR:-${SLIME_ROOT}/models/reward_model}
REWARD_MODEL_INIT=${REWARD_MODEL_INIT:-""}
CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-${SAVE_DIR%/}_critic}
CRITIC_LOAD_DIR=${CRITIC_LOAD_DIR:-""}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SLIME_ROOT}/rollout}
ROLLOUT_DEBUG_PATH_TEMPLATE=${ROLLOUT_DEBUG_PATH_TEMPLATE:-}
SLIME_CUSTOM_RM_TRUNCATION_PENALTY=${SLIME_CUSTOM_RM_TRUNCATION_PENALTY:-2.5}
SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC=${SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC:-0.8}
SLIME_WARMUP_RM_DIR=${SLIME_WARMUP_RM_DIR:-}
SLIME_WARMUP_RM_ENABLED=${SLIME_WARMUP_RM_ENABLED:-0}
SLIME_WARMUP_RM_COEF=${SLIME_WARMUP_RM_COEF:-0.0}
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

case "${ADVANTAGE_ESTIMATOR}" in
  ppo|grpo|gspo|reinforce_plus_plus|reinforce_plus_plus_baseline)
    ;;
  *)
    echo "ERROR: unsupported ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR}" >&2
    exit 1
    ;;
esac
if [ "${DEBUG_ROLLOUT_ONLY}" != "1" ] && [ "${ADVANTAGE_ESTIMATOR}" = "grpo" ] && [ "${N_SAMPLES_PER_PROMPT}" -lt 2 ]; then
  echo "ERROR: GRPO needs N_SAMPLES_PER_PROMPT>=2, got ${N_SAMPLES_PER_PROMPT}" >&2
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_CKPT}
   --no-load-optim
   --no-load-rng
   --no-save-optim
   --save ${SAVE_DIR}
   --critic-save ${CRITIC_SAVE_DIR}
   --save-interval ${PPO_SAVE_INTERVAL}
)
if [ "${PPO_FINETUNE}" = "1" ]; then
  CKPT_ARGS+=(--finetune)
fi
if [ -n "$ACTOR_LOAD" ] && [ -d "$ACTOR_LOAD" ]; then
  CKPT_ARGS+=(--load ${ACTOR_LOAD})
fi
if [ -n "$CRITIC_LOAD_DIR" ] && [ -d "$CRITIC_LOAD_DIR" ] && [ -f "$CRITIC_LOAD_DIR/latest_checkpointed_iteration.txt" ]; then
  CKPT_ARGS+=(--critic-load ${CRITIC_LOAD_DIR})
fi

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key ${PROMPT_INPUT_KEY}
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-stop-token-ids "${ROLLOUT_STOP_TOKEN_IDS[@]}"
   --rollout-shuffle

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature ${ROLLOUT_TEMPERATURE}

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

if [ -n "${PROMPT_LABEL_KEY}" ]; then
  ROLLOUT_ARGS+=(--label-key ${PROMPT_LABEL_KEY})
fi

if [ "${ALIGN_ROLLOUT_WITH_SFT}" = "1" ]; then
  ROLLOUT_ARGS+=(
     --rollout-skip-special-tokens
  )
fi

PPO_ARGS=(
   --advantage-estimator ${ADVANTAGE_ESTIMATOR}
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF}
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
if [ "${ADVANTAGE_ESTIMATOR}" = "ppo" ]; then
   PPO_ARGS+=(--value-clip 5.0)
fi
if [ "${NORMALIZE_ADVANTAGES}" = "1" ]; then
   PPO_ARGS+=(--normalize-advantages)
fi

IRL_ARGS=(
   --custom-rm-path slime.local_rm.custom_rm.custom_rm
   --reward-model-dir ${REWARD_MODEL_DIR}
   --reward-update-interval 999999
   --save-debug-rollout-data ${ROLLOUT_DEBUG_PATH_TEMPLATE}
)
if [ -n "$REWARD_MODEL_INIT" ]; then
   IRL_ARGS+=(--reward-model-init ${REWARD_MODEL_INIT})
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
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)
if [ "${LOG_PROBS_CHUNK_SIZE}" -gt 0 ]; then
   PERF_ARGS+=(--log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE})
fi
if [ -n "${TRAIN_MEMORY_MARGIN_BYTES}" ]; then
   PERF_ARGS+=(--train-memory-margin-bytes ${TRAIN_MEMORY_MARGIN_BYTES})
fi
if [ "${DISABLE_WEIGHTS_BACKUPER}" = "1" ]; then
   PERF_ARGS+=(--disable-weights-backuper)
fi
if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

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
TENSORBOARD_DIR=${TENSORBOARD_DIR:-tensorboard_log/slime-irl/${TB_EXP_NAME}}

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}
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

echo "Effective rollout config: num_rollout=${NUM_ROLLOUT} batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE} max_new_tokens=${ROLLOUT_MAX_RESPONSE_LEN} temperature=${ROLLOUT_TEMPERATURE} align_with_sft=${ALIGN_ROLLOUT_WITH_SFT} stop_token_ids=${ROLLOUT_STOP_TOKEN_IDS[*]} debug_rollout_only=${DEBUG_ROLLOUT_ONLY} use_ppo_args=$([ \"${DEBUG_ROLLOUT_ONLY}\" = \"1\" ] && echo 0 || echo 1)"
echo "Effective checkpoint config: hf_ckpt=${HF_CKPT} actor_load=${ACTOR_LOAD:-<none>} critic_load_dir=${CRITIC_LOAD_DIR:-<none>} ref_ckpt=${REF_CKPT} reward_model_dir=${REWARD_MODEL_DIR} reward_model_init=${REWARD_MODEL_INIT:-<default>} save_dir=${SAVE_DIR} critic_save_dir=${CRITIC_SAVE_DIR} rollout_debug_path=${ROLLOUT_DEBUG_PATH_TEMPLATE} save_interval=${PPO_SAVE_INTERVAL} finetune=${PPO_FINETUNE} tb_experiment=${TB_EXP_NAME:-prod}"
echo "Effective PPO/RM config: advantage_estimator=${ADVANTAGE_ESTIMATOR} actor_lr=${ACTOR_LR} critic_lr=${CRITIC_LR} kl_loss_coef=${KL_LOSS_COEF} log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE} train_memory_margin_bytes=${TRAIN_MEMORY_MARGIN_BYTES:-<default>} disable_weights_backuper=${DISABLE_WEIGHTS_BACKUPER} truncation_penalty=${SLIME_CUSTOM_RM_TRUNCATION_PENALTY} truncation_threshold_frac=${SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC} rubric_plugins=${SLIME_RUBRIC_PLUGINS:-<default>} warmup_rm_enabled=${SLIME_WARMUP_RM_ENABLED} warmup_rm_dir=${SLIME_WARMUP_RM_DIR:-<disabled>} warmup_rm_coef=${SLIME_WARMUP_RM_COEF}"
echo "Effective GPU layout: total=${TOTAL_GPUS} actor=${ACTOR_NUM_GPUS} critic=${CRITIC_NUM_GPUS} rollout=${ROLLOUT_NUM_GPUS} rollout_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE} tensor_mp=${TENSOR_MODEL_PARALLEL_SIZE} colocate=${COLLOCATE} sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export _REAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Let custom_rm pick the least-occupied GPU from the visible set instead of pinning GPU 0.
export SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES="${SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES:-auto}"
export SLIME_CUSTOM_RM_MAX_RESPONSE_LEN="${SLIME_CUSTOM_RM_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
export SLIME_CUSTOM_RM_TRUNCATION_PENALTY="${SLIME_CUSTOM_RM_TRUNCATION_PENALTY}"
export SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC="${SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC}"
export SLIME_WARMUP_RM_DIR="${SLIME_WARMUP_RM_DIR}"
export SLIME_WARMUP_RM_ENABLED="${SLIME_WARMUP_RM_ENABLED}"
export SLIME_WARMUP_RM_COEF="${SLIME_WARMUP_RM_COEF}"
export SLIME_RUBRIC_ENABLED="${SLIME_RUBRIC_ENABLED:-0}"
export SLIME_RUBRIC_PLUGINS="${SLIME_RUBRIC_PLUGINS:-}"
export SLIME_RUBRIC_W_IRL="${SLIME_RUBRIC_W_IRL:-1.0}"
export SLIME_RUBRIC_W_FORMAT="${SLIME_RUBRIC_W_FORMAT:-0.5}"
export SLIME_RUBRIC_W_ANSWER="${SLIME_RUBRIC_W_ANSWER:-1.0}"
PYTORCH_CUDA_ALLOC_CONF_EFFECTIVE=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${TOTAL_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}:/root/Megatron-LM/\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF_EFFECTIVE}\",
    \"_REAL_CUDA_VISIBLE_DEVICES\": \"${_REAL_CUDA_VISIBLE_DEVICES}\",
    \"SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES\": \"${SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES}\",
    \"SLIME_CUSTOM_RM_MAX_RESPONSE_LEN\": \"${SLIME_CUSTOM_RM_MAX_RESPONSE_LEN}\",
    \"SLIME_CUSTOM_RM_TRUNCATION_PENALTY\": \"${SLIME_CUSTOM_RM_TRUNCATION_PENALTY}\",
    \"SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC\": \"${SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC}\",
    \"SLIME_WARMUP_RM_DIR\": \"${SLIME_WARMUP_RM_DIR}\",
    \"SLIME_WARMUP_RM_ENABLED\": \"${SLIME_WARMUP_RM_ENABLED}\",
    \"SLIME_WARMUP_RM_COEF\": \"${SLIME_WARMUP_RM_COEF}\",
    \"SLIME_RUBRIC_ENABLED\": \"${SLIME_RUBRIC_ENABLED}\",
    \"SLIME_RUBRIC_PLUGINS\": \"${SLIME_RUBRIC_PLUGINS}\",
    \"SLIME_RUBRIC_W_IRL\": \"${SLIME_RUBRIC_W_IRL}\",
    \"SLIME_RUBRIC_W_FORMAT\": \"${SLIME_RUBRIC_W_FORMAT}\",
    \"SLIME_RUBRIC_W_ANSWER\": \"${SLIME_RUBRIC_W_ANSWER}\"
  }
}"

TRAIN_CMD=(
   python3 train_irl.py
   "${EXTRA_RUN_ARGS[@]}"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
   --critic-num-nodes 1
   --critic-num-gpus-per-node "${CRITIC_NUM_GPUS}"
   --num-gpus-per-node "${TOTAL_GPUS}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${RUN_PPO_ARGS[@]}"
   "${IRL_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
)
if [ "${COLLOCATE}" = "1" ]; then
   TRAIN_CMD+=(--colocate)
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_CMD[@]}"
