#!/bin/bash

# HH-RLHF 数据集训练脚本
# 这是一个示例脚本，需要根据你的实际环境进行调整

# 清理之前的进程
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# 防止 ray 缓冲 stdout/stderr
export PYTHONBUFFERED=16

# 检测 NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ============================================
# 模型配置 - 请根据你要使用的模型进行调整
# ============================================
# 示例：使用 Qwen3-4B 模型
# 你需要先下载模型并转换为 Megatron 格式
# source "${SCRIPT_DIR}/slime/scripts/models/qwen3-4B.sh"

# 或者使用 GLM4-9B
# source "${SCRIPT_DIR}/slime/scripts/models/glm4-9B.sh"

# ============================================
# 检查点和路径配置
# ============================================
CKPT_ARGS=(
   # HuggingFace 模型路径（用于加载 tokenizer）
   --hf-checkpoint /path/to/your/model
   
   # 参考模型的 Megatron 格式检查点
   --ref-load /path/to/your/model_torch_dist
   
   # Actor 模型加载路径
   --load /path/to/your/model_slime/
   
   # 训练过程中模型保存路径
   --save /path/to/your/model_slime/
   
   # 模型保存间隔
   --save-interval 20
)

# ============================================
# Rollout 参数 - 数据生成配置
# ============================================
ROLLOUT_ARGS=(
   # HH-RLHF 训练数据路径
   --prompt-data ${SCRIPT_DIR}/hh-rlhf-processed/helpful-base-train.jsonl
   
   # 使用 slime 默认字段名（text 和 label）
   # --input-key text  # 默认值，可以省略
   # --label-key label # 可选，用于参考答案
   
   # 应用 chat template
   --apply-chat-template
   
   # 打乱数据
   --rollout-shuffle

   # Reward Model 类型
   --rm-type deepscaler

   # 训练轮次和批次配置
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   # 全局批次大小 (rollout-batch-size * n-samples-per-prompt)
   --global-batch-size 64
   --balance-data
)

# ============================================
# 评估参数
# ============================================
EVAL_ARGS=(
   # 评估间隔
   --eval-interval 50
   
   # 评估数据集
   --eval-prompt-data hh-rlhf-test ${SCRIPT_DIR}/hh-rlhf-processed/helpful-base-test.jsonl
   
   # 每个评估 prompt 的采样数
   --n-samples-per-eval-prompt 4
   
   # 评估时最大响应长度
   --eval-max-response-len 2048
   
   # 评估采样参数
   --eval-top-p 0.9
)

# ============================================
# 性能和并行参数
# ============================================
PERF_ARGS=(
   # 张量并行
   --tensor-model-parallel-size 1
   
   # 流水线并行
   --pipeline-model-parallel-size 1
   
   # 上下文并行
   --context-parallel-size 1
   
   # 专家并行（MoE 模型）
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # 重计算配置
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # 动态批处理
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

# ============================================
# GRPO 算法参数
# ============================================
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ============================================
# 优化器参数
# ============================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ============================================
# Weights & Biases 配置（可选）
# ============================================
WANDB_ARGS=(
   #--use-wandb
   #--wandb-project hh-rlhf-training
   #--wandb-group my-experiment
   #--wandb-key ${WANDB_KEY}
)

# ============================================
# SGLang 推理服务参数
# ============================================
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
)

# ============================================
# 其他参数
# ============================================
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================
# 启动 Ray 集群
# ============================================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# 构建运行时环境
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# ============================================
# 提交训练任务
# ============================================
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
