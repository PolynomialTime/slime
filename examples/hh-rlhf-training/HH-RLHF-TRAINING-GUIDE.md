# HH-RLHF 数据集训练指南

本指南说明如何使用 slime 框架在 HH-RLHF 数据集上进行强化学习训练。

## 📁 文件说明

- `slime/` - slime 框架主目录
- `hh-rlhf/` - 原始 HH-RLHF 数据集
- `hh-rlhf-processed/` - 处理后的训练数据
- `prepare_hh_rlhf.py` - 数据预处理脚本
- `run-hh-rlhf-training.sh` - 训练启动脚本

## 📊 数据集统计

已成功下载并处理 HH-RLHF 数据集：

| 数据集 | 训练集 | 测试集 |
|--------|--------|--------|
| helpful-base | 43,834 | 2,354 |
| harmless-base | 42,491 | 2,308 |

## 🚀 快速开始

### 1. 环境准备

#### 方式一：使用 Docker（推荐）

```bash
# 拉取 slime 官方镜像
docker pull slimerl/slime:latest

# 启动容器（将当前目录挂载到容器）
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -it slimerl/slime:latest /bin/bash

# 在容器内更新 slime
cd /workspace/slime
git pull
pip install -e . --no-deps
```

#### 方式二：使用 Conda

```bash
# 参考 slime 的 build_conda.sh 脚本
cd slime
bash build_conda.sh
```

### 2. 下载和准备模型

选择一个基础模型进行训练，例如 Qwen3-4B 或 GLM4-9B：

```bash
# 示例：下载 Qwen3-4B 模型
hf download Qwen/Qwen3-4B --local-dir /path/to/Qwen3-4B

# 转换为 Megatron 格式
cd slime
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /path/to/Qwen3-4B \
  --save /path/to/Qwen3-4B_torch_dist
```

### 3. 配置训练脚本

编辑 `run-hh-rlhf-training.sh`，修改以下关键配置：

```bash
# 1. 加载模型配置（取消注释对应的模型）
source "${SCRIPT_DIR}/slime/scripts/models/qwen3-4B.sh"

# 2. 设置模型路径
CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen3-4B
   --ref-load /path/to/Qwen3-4B_torch_dist
   --load /path/to/Qwen3-4B_slime/
   --save /path/to/Qwen3-4B_slime/
   --save-interval 20
)

# 3. 根据你的 GPU 数量调整并行配置
--actor-num-gpus-per-node 4  # 训练使用的 GPU 数
--rollout-num-gpus 4          # 推理使用的 GPU 数
```

### 4. 启动训练

```bash
# 给脚本添加执行权限
chmod +x run-hh-rlhf-training.sh

# 启动训练
bash run-hh-rlhf-training.sh
```

## 🔧 关键参数说明

### 数据相关
- `--prompt-data`: 训练数据路径
- `--input-key`: 输入字段名（默认为 `text`，可省略）
- `--label-key`: 标签字段名（默认为 `label`，可选）
- `--apply-chat-template`: 应用对话模板

**注意**：本项目使用 slime 框架的标准字段名 `text` 和 `label`，因此不需要显式指定 `--input-key` 参数。

### 训练控制
- `--num-rollout`: 总训练轮次
- `--rollout-batch-size`: 每轮采样的 prompt 数量
- `--n-samples-per-prompt`: 每个 prompt 生成的回复数量
- `--global-batch-size`: 参数更新的批次大小

**重要约束**：
```
rollout-batch-size × n-samples-per-prompt = global-batch-size × num-steps-per-rollout
```

### 并行配置
- `--tensor-model-parallel-size`: 张量并行度
- `--pipeline-model-parallel-size`: 流水线并行度
- `--context-parallel-size`: 上下文并行度

### GRPO 算法
- `--advantage-estimator`: 优势估计器（grpo/gspo/ppo）
- `--kl-loss-coef`: KL 散度损失系数
- `--eps-clip`: PPO 裁剪参数

## 📈 监控训练

### 使用 Ray Dashboard
训练启动后，可以通过浏览器访问：
```
http://localhost:8265
```

### 使用 Weights & Biases（可选）
在 `run-hh-rlhf-training.sh` 中启用 wandb：
```bash
WANDB_ARGS=(
   --use-wandb
   --wandb-project hh-rlhf-training
   --wandb-group my-experiment
   --wandb-key ${WANDB_KEY}
)
```

## 🎯 训练后操作

### 转换模型回 HuggingFace 格式

```bash
cd slime

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/model_slime/iter_xxx/ \
  --output-dir /path/to/model_hf_iter_xxx \
  --origin-hf-dir /path/to/original_model
```

## 💡 常见问题

### 1. 显存不足
- 减小 `--max-tokens-per-gpu`
- 增加 `--tensor-model-parallel-size`
- 启用 `--recompute-granularity full`

### 2. 训练速度慢
- 启用 `--use-dynamic-batch-size`
- 调整 `--rollout-batch-size` 和 `--global-batch-size`
- 检查 `--balance-data` 是否启用

### 3. 训推一体化模式
如果 GPU 数量有限，可以使用 colocated 模式：
```bash
ray job submit ... \
  -- python3 train.py \
  --actor-num-gpus-per-node 8 \
  --colocate \
  --sglang-mem-fraction-static 0.8 \
  ...
```

## 📚 参考资源

- [slime 官方文档](https://github.com/THUDM/slime)
- [HH-RLHF 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GRPO 论文](https://arxiv.org/abs/2402.03300)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SGLang](https://github.com/sgl-project/sglang)

## 🔄 数据集变体

如果想使用其他 HH-RLHF 子集，修改训练脚本中的数据路径：

```bash
# 使用 harmless-base
--prompt-data ${SCRIPT_DIR}/hh-rlhf-processed/harmless-base-train.jsonl

# 使用 helpful-online
--prompt-data ${SCRIPT_DIR}/hh-rlhf-processed/helpful-online-train.jsonl
```

## ⚠️ 注意事项

1. **硬件要求**：建议使用 H100/H200 或 B200 系列 GPU
2. **模型配置**：确保模型配置文件中的参数与实际模型匹配
3. **路径设置**：所有路径需要根据实际环境调整
4. **Docker 环境**：推荐使用官方 Docker 镜像以避免依赖问题

---

**祝训练顺利！** 🎉
