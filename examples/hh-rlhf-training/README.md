# HH-RLHF + Slime 训练环境

基于 [slime](https://github.com/THUDM/slime) 框架的 HH-RLHF 数据集训练环境配置。

## 📋 项目简介

本项目提供了使用 slime 框架在 HH-RLHF 数据集上进行强化学习训练的完整配置和脚本。

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. 下载 slime 框架

```bash
git clone https://github.com/THUDM/slime.git
```

### 3. 下载 HH-RLHF 数据集

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Anthropic/hh-rlhf', repo_type='dataset', local_dir='./hh-rlhf', local_dir_use_symlinks=False)"
```

### 4. 处理数据集

```bash
python prepare_hh_rlhf.py
```

这将生成符合 **slime 框架标准格式**的训练数据（使用 `text` 和 `label` 字段）。

### 5. 验证数据

```bash
python verify_data_format.py
```

## 📊 数据集统计

- **helpful-base**: 训练 43,834 条，测试 2,354 条
- **harmless-base**: 训练 42,487 条，测试 2,308 条
- **总计**: 90,983 条数据

## 📝 数据格式

**符合 slime 框架标准格式**：

```json
{
  "text": "Human: 问题...\n\nAssistant: 回答...\n\nHuman: 继续问题...",
  "label": "最后的回复内容"
}
```

- `text`: slime 框架默认的 prompt 字段名
- `label`: 可选的参考答案字段

详细说明请参考 [DATA-FORMAT.md](DATA-FORMAT.md) 和 [SLIME-FORMAT-COMPLIANCE.md](SLIME-FORMAT-COMPLIANCE.md)

## 🔧 训练配置

详细的训练步骤请参考：
- [HH-RLHF-TRAINING-GUIDE.md](HH-RLHF-TRAINING-GUIDE.md) - 完整训练指南
- [DATA-FORMAT.md](DATA-FORMAT.md) - 数据格式说明
- [FINAL-SUMMARY.md](FINAL-SUMMARY.md) - 环境配置总结

## 📂 项目结构

```
.
├── README.md                       # 项目说明
├── prepare_hh_rlhf.py             # 数据预处理脚本
├── verify_data_format.py          # 数据格式验证脚本
├── run-hh-rlhf-training.sh        # 训练启动脚本
├── HH-RLHF-TRAINING-GUIDE.md      # 训练指南
├── DATA-FORMAT.md                 # 数据格式说明
├── SLIME-FORMAT-COMPLIANCE.md     # 格式合规性报告
├── SETUP-SUMMARY.md               # 环境搭建总结
├── hh-rlhf/                       # 原始数据集（需下载）
├── hh-rlhf-processed/             # 处理后的数据
└── slime/                         # slime 框架（需下载）
```

## 🛠️ 环境要求

- Python 3.8+
- CUDA 支持的 NVIDIA GPU
- Docker（推荐）或 Conda
- huggingface_hub

## 📚 参考资源

- [slime 官方仓库](https://github.com/THUDM/slime)
- [HH-RLHF 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GRPO 论文](https://arxiv.org/abs/2402.03300)

## ⚠️ 注意事项

1. 原始数据集和处理后的数据文件较大，已在 `.gitignore` 中排除
2. 需要自行下载 slime 框架和 HH-RLHF 数据集
3. 训练前需要下载并转换基础模型（如 Qwen3-4B 或 GLM4-9B）

## 📄 许可证

本项目遵循 MIT 许可证。slime 框架和 HH-RLHF 数据集请遵循其各自的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Happy Training!** 🎉
