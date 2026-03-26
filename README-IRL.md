# IRL 训练使用手册

## 1. 环境变量准备

训练前导出以下路径变量（建议写入 `.env` 或直接 `export`）：

```bash
export SLIME_ROOT=/mnt/shared-storage-user/wangqianyi/slime

export HF_CKPT=$SLIME_ROOT/models/qwen3-1.7b-base          # HF 格式原始权重
export REF_CKPT=$SLIME_ROOT/models/qwen3-1.7b-base_torch_dist  # 参考模型（KL loss）
export ACTOR_CKPT=$SLIME_ROOT/models/qwen3-1.7b-base_torch_dist # 初始 actor 权重
export SAVE_DIR=$SLIME_ROOT/models/save_dir                 # policy checkpoint 保存路径
export PROMPT_DATA=$SLIME_ROOT/hh-rlhf-processed/hh-rlhf-merged-train-debug.jsonl
export DEMO_DATA=$SLIME_ROOT/hh-rlhf-processed/hh-rlhf-merged-train-debug.jsonl
```

---

## 2. 数据格式

### 训练数据（PROMPT_DATA / DEMO_DATA）

每行一个 JSON 对象，必须包含：

| 字段 | 说明 |
|------|------|
| `text` | prompt 文本（可以是多轮对话，最后一句留空由模型补全） |
| `label` | 专家回复（用于 reward model 的 demo） |

```jsonl
{"text": "Human: How are you?\nAssistant:", "label": "I'm doing well, thank you!"}
```

### 评测数据（reward_eval_path / win_rate_eval_path）

每行一个 JSON 对象，用于评估 reward model 准确率：

| 字段 | 说明 |
|------|------|
| `text` | prompt |
| `chosen` | 更好的回复 |
| `rejected` | 较差的回复 |

```jsonl
{"text": "Human: ...\nAssistant:", "chosen": "good response", "rejected": "bad response"}
```

---

## 3. IRL 训练

### 快速启动

```bash
MODEL_SH=scripts/models/qwen3-1.7B.sh \
HF_CKPT=$HF_CKPT \
REF_CKPT=$REF_CKPT \
ACTOR_CKPT=$ACTOR_CKPT \
SAVE_DIR=$SAVE_DIR \
PROMPT_DATA=$PROMPT_DATA \
DEMO_DATA=$DEMO_DATA \
REWARD_UPDATE_LAUNCHER=accelerate \
bash scripts/run-irl.sh
```

### GPU 分配

| GPU | 用途 |
|-----|------|
| 0, 1, 2 | Policy 训练 + SGLang rollout（Ray 管理） |
| 3 | Reward model 更新 + 评估（子进程独立运行） |

### 可选环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ACTOR_GPUS` | 1 | Actor 训练 GPU 数 |
| `CRITIC_GPUS` | 1 | Critic 训练 GPU 数 |
| `ROLLOUT_GPUS` | 1 | SGLang rollout GPU 数 |
| `USE_COLOCATE` | 0 | 1 = 把 rollout 和训练 colocate 到同一 GPU |
| `REWARD_UPDATE_LAUNCHER` | `direct` | `direct`（单卡）或 `accelerate`（多卡） |
| `REWARD_UPDATE_ACCELERATE_NUM_PROC` | 1 | Accelerate 进程数 |
| `WIN_RATE_OPENROUTER_KEY` | — | OpenRouter key；设置后每轮自动跑 online winrate |
| `MASTER_ADDR` | `127.0.0.1` | Ray head 地址 |

### 关键 IRL 参数（在 run-irl.sh 中修改）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--reward-update-interval` | 1 | 每隔 N 轮 rollout 更新一次 reward model |
| `--reward-update-epochs` | 1 | 每次 reward 更新的 epoch 数 |
| `--reward-update-lr` | 1e-5 | Reward 更新学习率 |
| `--reward-update-batch-size` | 8 | Reward 更新 batch size |
| `--target-reward-l2-norm` | 5.0 | epsilon 的目标值（控制 reward 变化幅度） |
| `--c-coef-init` | 1.0 | epsilon 惩罚系数初始值 |
| `--c-coef-min` / `--c-coef-max` | 0.1 / 10.0 | 系数范围 |
| `--reward-update-rollout-window` | 1 | 聚合最近 N 轮 rollout 数据训练 reward |

### 训练产物

```
$SAVE_DIR/
  iter_0010/          # policy checkpoint（每 20 轮保存一次）
  iter_0020/
  ...

$SLIME_ROOT/models/reward_model/
  latest/             # 最新 reward model（原子替换）
  step_0/             # 每轮 reward update 的快照
  step_1/
  ...
  reward_eval_rollout_N.json    # reward model 准确率评估结果
  win_rate_rollout_N.json       # online winrate 结果（需要 OpenRouter key）

$SLIME_ROOT/rollout/
  rollout_0.pt        # rollout 数据（用于 reward model 更新，--save-debug-rollout-data）
  rollout_1.pt
  ...
```

---

## 4. Winrate 评测（离线，GPT-4o 评判）

集群无法联网时，用以下两步流程做 GPT-4o winrate 评测。

### Step 1：集群上生成 responses

对两个模型各跑一次，输出保存到共享存储：

```bash
# IRL 训练后的模型
python scripts/eval_generate.py \
  --model-path $SAVE_DIR/iter_0010 \
  --prompt-data $SLIME_ROOT/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $SLIME_ROOT/eval/outputs_irl.jsonl \
  --apply-chat-template \
  --max-new-tokens 256 \
  --batch-size 4

# Base 模型（对照组）
python scripts/eval_generate.py \
  --model-path $HF_CKPT \
  --prompt-data $SLIME_ROOT/hh-rlhf-processed/hh-rlhf-merged-test.jsonl \
  --output $SLIME_ROOT/eval/outputs_base.jsonl \
  --apply-chat-template \
  --max-new-tokens 256 \
  --batch-size 4
```

`eval_generate.py` 参数说明：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | 必填 | HF 格式 checkpoint 路径 |
| `--prompt-data` | 必填 | 测试集 JSONL |
| `--output` | 必填 | 输出 JSONL 路径 |
| `--prompt-key` | `text` | JSONL 中 prompt 的字段名 |
| `--max-new-tokens` | 256 | 最大生成 token 数 |
| `--batch-size` | 4 | 推理 batch size |
| `--apply-chat-template` | — | 与训练 pipeline 保持一致 |

### Step 2：开发机上跑 GPT-4o 评判

集群和开发机共用同一文件存储，直接读取上一步的输出：

```bash
# 在开发机上执行（需要 OpenAI API key）
python scripts/eval_winrate.py \
  --outputs-a $SLIME_ROOT/eval/outputs_irl.jsonl \
  --outputs-b $SLIME_ROOT/eval/outputs_base.jsonl \
  --output $SLIME_ROOT/eval/winrate.json \
  --api-key sk-... \
  --model gpt-4o \
  --concurrency 16
```

也可以通过环境变量传入 key：

```bash
export OPENAI_API_KEY=sk-...
python scripts/eval_winrate.py \
  --outputs-a $SLIME_ROOT/eval/outputs_irl.jsonl \
  --outputs-b $SLIME_ROOT/eval/outputs_base.jsonl \
  --output $SLIME_ROOT/eval/winrate.json
```

`eval_winrate.py` 参数说明：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--outputs-a` | 必填 | 模型 A 的 JSONL（通常是 IRL 模型） |
| `--outputs-b` | 必填 | 模型 B 的 JSONL（通常是 base 模型） |
| `--output` | 必填 | 结果 JSON 路径 |
| `--api-key` | 环境变量 | OpenAI API key（或 `OPENAI_API_KEY`） |
| `--model` | `gpt-4o` | 评判使用的模型 |
| `--max-samples` | 全部 | 限制评测样本数 |
| `--concurrency` | 16 | 并发 API 请求数 |

### 输出结果格式

```json
{
  "total": 100,
  "a_wins": 55,
  "b_wins": 32,
  "ties": 13,
  "winrate_a": 0.615,
  "model": "gpt-4o",
  "samples": [...]
}
```

`winrate_a = (a_wins + 0.5 × ties) / total`

> **注：** 每对评测随机打乱 A/B 顺序（50% 概率互换），消除 position bias。

---

## 5. 完整评测流程一览

```
训练阶段（集群）
  run-irl.sh
    → 每轮 rollout + PPO 更新
    → 每轮 reward model 更新（GPU 3）
    → 内置 reward accuracy eval（GPU 3）
    → 若设置 WIN_RATE_OPENROUTER_KEY：在线 winrate eval（需网络）

离线 Winrate 评测（集群生成 + 开发机评判）
  eval_generate.py (集群，GPU)
    → 生成 outputs_irl.jsonl
    → 生成 outputs_base.jsonl
  eval_winrate.py (开发机，仅 CPU + 网络)
    → 读取两份 JSONL
    → GPT-4o 评判
    → 输出 winrate.json
```
