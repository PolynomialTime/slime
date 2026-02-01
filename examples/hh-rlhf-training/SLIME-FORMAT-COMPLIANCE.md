# Slime 框架格式合规性报告

## ✅ 合规性状态

**状态**: 已完全符合 slime 框架标准

**更新时间**: 2026-01-31

## 📋 变更摘要

### 之前的格式（不符合标准）
```json
{
  "prompt": "对话历史",
  "answer": "参考答案"
}
```

### 当前格式（符合 slime 标准）
```json
{
  "text": "对话历史",
  "label": "参考答案"
}
```

## 🔍 为什么要更改？

根据 slime 框架源码分析（`slime/slime/utils/data.py`）：

1. **默认字段名**: slime 框架默认使用 `"text"` 作为 prompt 字段
   ```python
   def __init__(self, path, tokenizer, processor, max_length, *, 
                prompt_key="text",  # 默认值是 "text"
                ...):
   ```

2. **减少配置**: 使用标准字段名可以省略 `--input-key` 参数
   - ❌ 之前: `--input-key prompt --label-key answer`
   - ✅ 现在: 不需要指定，使用默认值

3. **框架一致性**: 与 slime 官方示例和文档保持一致

## 📊 数据验证结果

所有数据文件已通过格式验证：

| 文件 | 数据条数 | 验证状态 |
|------|---------|---------|
| helpful-base-train.jsonl | 43,834 | ✅ 通过 |
| helpful-base-test.jsonl | 2,354 | ✅ 通过 |
| harmless-base-train.jsonl | 42,487 | ✅ 通过 |
| harmless-base-test.jsonl | 2,308 | ✅ 通过 |

**总计**: 90,983 条数据，100% 符合标准

## 🔧 更新的文件

### 1. 数据处理脚本
- **文件**: `prepare_hh_rlhf.py`
- **更改**: 输出字段从 `prompt/answer` 改为 `text/label`

### 2. 训练脚本
- **文件**: `run-hh-rlhf-training.sh`
- **更改**: 移除 `--input-key prompt --label-key answer` 参数

### 3. 文档更新
- `DATA-FORMAT.md` - 数据格式说明
- `HH-RLHF-TRAINING-GUIDE.md` - 训练指南
- `SLIME-FORMAT-COMPLIANCE.md` - 本文档

### 4. 验证工具
- **新增**: `verify_data_format.py` - 数据格式验证脚本

## 📝 字段说明

### `text` 字段（必需）
- **类型**: 字符串
- **内容**: 完整的对话历史（除最后一个 Assistant 回复）
- **格式**: `Human: ... \n\nAssistant: ... \n\nHuman: ...`
- **用途**: 作为模型的输入 prompt

### `label` 字段（可选）
- **类型**: 字符串
- **内容**: 最后一个 Assistant 的回复
- **用途**: 
  - 在 RLHF 训练中作为参考答案
  - 用于评估和对比
  - 实际训练时模型会生成新的响应

### `metadata` 字段（可选）
- **类型**: 对象
- **内容**: 额外的元数据信息
- **用途**: 存储工具调用、上下文等信息

## 🎯 使用方法

### 训练脚本配置（简化版）

```bash
ROLLOUT_ARGS=(
   # 数据路径
   --prompt-data ./hh-rlhf-processed/helpful-base-train.jsonl
   
   # 不需要指定 --input-key 和 --label-key
   # 框架会自动使用默认值 "text" 和 "label"
   
   --apply-chat-template
   --rollout-shuffle
   ...
)
```

### 如果需要自定义字段名

如果你的数据使用其他字段名，仍然可以通过参数指定：

```bash
--input-key custom_prompt_field
--label-key custom_label_field
```

## ✨ 优势

1. **标准化**: 完全符合 slime 框架规范
2. **简化配置**: 减少训练脚本参数
3. **减少错误**: 避免字段名配置错误
4. **可维护性**: 更容易理解和维护
5. **兼容性**: 与 slime 官方示例保持一致

## 🔄 迁移指南

如果你有使用旧格式的数据：

### 方法 1: 重新生成数据（推荐）
```bash
python prepare_hh_rlhf.py
```

### 方法 2: 使用参数适配
在训练脚本中添加：
```bash
--input-key prompt
--label-key answer
```

## 📚 参考资料

- Slime 数据加载源码: `slime/slime/utils/data.py`
- Slime 参数定义: `slime/slime/utils/arguments.py`
- Slime 官方文档: https://github.com/THUDM/slime

## ✅ 验证方法

运行验证脚本确认数据格式：

```bash
python verify_data_format.py
```

预期输出：
```
🎉 所有数据文件格式验证通过！
✅ 数据已符合 slime 框架标准，可以开始训练
```

## 📞 问题反馈

如果遇到格式相关问题：

1. 运行 `python verify_data_format.py` 检查数据
2. 查看 `DATA-FORMAT.md` 了解格式详情
3. 参考 `slime/slime/utils/data.py` 源码

---

**结论**: 所有数据已完全符合 slime 框架标准，可以安全地用于训练。
