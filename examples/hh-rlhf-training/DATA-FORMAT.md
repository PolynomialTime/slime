# HH-RLHF 数据格式说明

## 数据集概览

HH-RLHF 数据集已转换为 **slime 框架标准格式**，使用 `text` 和 `label` 字段。

### 数据统计
- helpful-base: 训练 43,834 条，测试 2,354 条
- harmless-base: 训练 42,487 条，测试 2,308 条
- 总计: 训练 86,321 条，测试 4,662 条

## 数据格式规范

### Slime 标准格式
```json
{
  "text": "Human: 问题1\n\nAssistant: 回答1\n\nHuman: 问题2",
  "label": "回答2"
}
```

### 字段说明

- **`text`**: slime 框架的默认 prompt 字段名
  - 包含完整的对话历史（除了最后一个 Assistant 回复）
  - 格式：`Human: ... \n\nAssistant: ... \n\nHuman: ...`

- **`label`**: 可选的参考答案字段
  - 包含最后一个 Assistant 的回复
  - 在 RLHF 训练中主要用于评估和参考
  - 实际训练时，模型会通过 rollout 生成新的响应

## 为什么使用 `text` 而不是 `prompt`？

slime 框架默认使用 `"text"` 作为 prompt 字段名（见 `slime/slime/utils/data.py`）：
- ✅ 使用默认字段名可以省略 `--input-key` 参数
- ✅ 保持与框架标准一致，减少配置错误
- ✅ 更好的代码可维护性

## 转换逻辑

1. 从原始 HH-RLHF 数据中提取 `chosen` 对话
2. 解析对话，分离 Human 和 Assistant 轮次
3. 提取对话历史（除最后一个 Assistant 回复）作为 `text`
4. 提取最后一个 Assistant 的回复作为 `label`
5. 保留完整的多轮对话上下文

## 使用方法

### 在训练脚本中配置

使用标准字段名后，配置更简洁：

```bash
ROLLOUT_ARGS=(
   --prompt-data ./hh-rlhf-processed/helpful-base-train.jsonl
   # 不需要指定 --input-key，使用默认的 "text"
   # --label-key 也可以省略（可选）
   --apply-chat-template
   ...
)
```

### 如果需要自定义字段名

如果你的数据使用其他字段名，可以通过参数指定：

```bash
--input-key custom_prompt_field
--label-key custom_label_field
```

## 数据示例

### 单轮对话
```json
{
  "text": "Human: Do you know why turkeys became the official food of thanksgiving?",
  "label": "To be honest, I don't know anything about that..."
}
```

### 多轮对话
```json
{
  "text": "Human: How can I find out what types of butterflies are in my area?\n\nAssistant: Which location are you in?\n\nHuman: I am in Oregon.\n\nAssistant: There are about 175 species of butterflies in Oregon...\n\nHuman: Great. What are some common species then?",
  "label": "About 150 species of butterflies live in Oregon..."
}
```

## 参考资料

- Slime 数据加载代码: `slime/slime/utils/data.py`
- Slime 参数定义: `slime/slime/utils/arguments.py`
- 数据转换脚本: `prepare_hh_rlhf.py`
