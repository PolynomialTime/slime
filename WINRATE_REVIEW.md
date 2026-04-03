# Winrate 排查总报告

## 1. 结论摘要

当前仓库里和 `winrate` 相关的主要问题，不是单点 bug，而是四条链路同时失真：

1. `winrate` 评测脚本本身大概率坏了，现有结果基本不能信。
2. 7 个 round 的 PPO 没有累计上一轮 policy，而是每轮从 SFT 重新开始。
3. `HH-RLHF` 数据被保存成字符串后，又在训练和评测时被当成单轮 user message 套 chat template，多轮对话结构被破坏。
4. reward update 的目标、采样方式和长度惩罚共同把策略往“礼貌废话、无意义追问、重复句、拖长度”方向推。

如果以 `icml.pdf` 的 bi-level IRL 目标为准，真正优先要修的不是“换方法”，而是把实现重新拉回这三个条件：

- `pi_old` 必须真的是上一轮 policy，而不是每轮回到 SFT
- `r_theta` 必须连续 warm-start，才能让 `epsilon` 保持小步变化
- reward / policy 必须真正交替，而不是 reward 永远滞后一轮

基于现有 `eval/outputs_*.jsonl` 的静态统计，policy 输出也没有体现“逐轮变好”：

- `I'm not sure` 出现率：SFT baseline `4.2%`，`r3` `10.8%`，`r6` `11.0%`
- 重复句样本占比：SFT baseline `9.7%`，`r3` `25.0%`
- `eval/winrate_r1_vs_sft.json`、`eval/winrate_r1_vs_gpt4o.json` 等结果全部是 `200/200 ties`

因此当前最重要的工作顺序不是调 PPO 超参，而是：

1. 修评测
2. 修 round 累积训练
3. 修数据格式
4. 修 reward update
5. 最后再调参

### 1.1 以 `icml.pdf` 为约束的重审结论

`icml.pdf` 的核心不是“再找一个更强的 reward 学法”，而是优化下面这个 surrogate：

- 原目标：`L(theta)=J(pi_E,r_theta)-J(pi_theta,r_theta)`
- surrogate：`L_old(theta)=J(pi_E,r_theta)-J(pi_old,r_theta)`
- 保证条件：`r_theta_new` 和 `r_theta_old` 的差异 `epsilon` 要足够小

所以之前所有建议里，真正符合你原始思路的是：

1. 修 `winrate` evaluator
2. 让 policy round 逐轮累计
3. 让 reward model 逐轮 warm-start
4. 先有 reward，再做下一轮 PPO
5. 统一 conversation schema，确保 `(x, y)` 表示的就是论文里的 prompt-response 对

不该作为主方案的建议是：

- 把 reward 主目标改成 pairwise preference / DPO 风格训练
- 用另一套完全不同的 reward learning 替换当前 bi-level IRL 主干

### 1.2 当前 HEAD 复查结论

按当前仓库代码复查，以下几项已经有明确改动：

1. `scripts/run-full-pipeline-job.sh` 已经引入 `CURRENT_HF_CKPT`，并按 round 保存 `policy_r{N}_hf`
2. `scripts/run-reward-update.sh` 已经把 `apply_chat_template_kwargs` 对齐到 `{"enable_thinking": false}`
3. `scripts/eval_generate_sglang.py` 已经支持透传 `apply_chat_template_kwargs`
4. `slime/local_rm/update_reward_accel.py` 已经加入 `random.shuffle(...)`，并把 reward 产物改成按 round 命名

但当前最关键的几个问题仍然没有真正解决：

1. policy 连续训练表面上修了，实际上还没修透。`run-full-pipeline-job.sh` 虽然逐轮更新了 `HF_CKPT`，但 `run-irl-prod.sh` 仍然没有传 `--load`；而在这个项目里，当 `--load` 缺失时，训练会从 `--ref-load` 初始化，见 `slime/utils/arguments.py:683`。你现在仍把 `REF_CKPT` 固定在 SFT，所以 actor 还是每轮从 SFT 起步。
2. `eval_winrate.py` 不再是“只看首词”，但新的 fallback 仍会系统性误判。普通英文里的冠词 `a` 会被误判成选择 A，而 `"Between A and B, A is better."` 这类明确判决又会被吞成 `Tie`。
3. 多轮对话 schema 在仓库代码里仍没有真正切到 `messages`。本地仓库也没有 `hh-rlhf-processed/` 产物，无法验证你线上数据是否已经另行转换；从代码本身看，默认链路仍是字符串 `text -> 单条 user message`。
4. round 1 前仍没有 reward bootstrap。`custom_rm` 在 `reward_model/latest` 不存在时仍返回 `0.0`，而 full pipeline 还是先 PPO、后 reward update。
5. generic IRL 链路还停留在旧语义：`scripts/run-irl.sh` 仍默认 `direct` launcher，而 `slime/local_rm/update_reward.py` 依旧每轮从 base model 重置 RM。

当前 `eval/` 里的输出也没有表现出“逐轮变好”。现有静态指标只说明：

- `r1` 比之前那版健康一些，`I'm not sure` 从 `4.2%` 降到 `2.2%`，`Human:` continuation 降到 `0`
- 但 `r2` 开始又明显变差，`r3/r5` 仍有显著 `Human:` continuation 和重复句
- 所以“逐轮提升”这个目标到当前 HEAD 还没有形成

---

## 2. 本次审查范围

本次主要阅读了以下内容：

- 根目录文档：`README.md`、`README_zh.md`、`README-IRL.md`、`CLAUDE.md`
- `eval/` 目录下的所有输出与 winrate 结果
- 训练与评测脚本：
  - `scripts/run-full-pipeline-job.sh`
  - `scripts/run-irl-prod.sh`
  - `scripts/run-irl.sh`
  - `scripts/run-reward-update.sh`
  - `scripts/eval_generate.py`
  - `scripts/eval_generate_sglang.py`
  - `scripts/eval_winrate.py`
- 数据准备与 reward 相关实现：
  - `prepare_hh_rlhf.py`
  - `scripts/prepare_sft_data.py`
  - `slime/utils/data.py`
  - `slime/local_rm/custom_rm.py`
  - `slime/local_rm/data.py`
  - `slime/local_rm/model.py`
  - `slime/local_rm/update_reward.py`
  - `slime/local_rm/update_reward_accel.py`
  - `slime/local_rm/win_rate_eval.py`

---

## 3. 项目真实链路

### 3.1 数据准备链路

`prepare_hh_rlhf.py` 会把 Anthropic HH-RLHF 转成两类文件：

- 训练集：`{"text": "对话历史", "label": "最后一个 assistant 回复"}`
- 测试/偏好集：`{"text": "对话历史", "chosen": "...", "rejected": "..."}`

关键位置：

- `prepare_hh_rlhf.py:79`
- `prepare_hh_rlhf.py:115`

这一步输出的是字符串格式的 `text`，不是 `messages`。

仓库里另外有一个 `scripts/prepare_sft_data.py`，它会把同样的数据转成真正的 `messages` 列表：

- `scripts/prepare_sft_data.py:14`
- `scripts/prepare_sft_data.py:59`

这个脚本是正确方向，但目前 IRL 主链路没用它。

### 3.2 训练主链路

生产 round pipeline 在 `scripts/run-full-pipeline-job.sh`：

1. 先做 SFT
2. 生成 SFT baseline 输出
3. 做 7 个 round：
   - 跑 PPO：`scripts/run-irl-prod.sh`
   - 转 HF checkpoint
   - 生成 policy 输出：`scripts/eval_generate_sglang.py`
   - 再单独做 reward update：`scripts/run-reward-update.sh`

关键位置：

- `scripts/run-full-pipeline-job.sh:124`
- `scripts/run-full-pipeline-job.sh:130`
- `scripts/run-full-pipeline-job.sh:196`

### 3.3 PPO round 内链路

`scripts/run-irl-prod.sh` 的说明非常关键：它实际上是 “PPO Phase — no reward update”。

关键位置：

- `scripts/run-irl-prod.sh:3`
- `scripts/run-irl-prod.sh:89`

它做的事情是：

- 训练 actor / critic / rollout
- 通过 `--custom-rm-path slime.local_rm.custom_rm.custom_rm` 使用当前 `reward_model/latest`
- 但 `--reward-update-interval 999999`，即本 round 内不更新 RM

### 3.4 Reward update 链路

reward update 由 `scripts/run-reward-update.sh` 单独完成：

1. 读取 `hh-rlhf-merged-train.jsonl` 作为 demo
2. 读取最近 `reward_update_rollout_window=100` 个 rollout 文件
3. 调 `accelerate launch -m slime.local_rm.update_reward_accel`
4. 将新 reward model 写回 `models/reward_model/latest`

关键位置：

- `scripts/run-reward-update.sh:27`
- `scripts/run-reward-update.sh:46`
- `scripts/run-reward-update.sh:60`

### 3.5 离线评测链路

当前离线评测链路是：

1. `scripts/eval_generate.py` 或 `scripts/eval_generate_sglang.py` 生成两个模型的 JSONL
2. `scripts/eval_winrate.py` 调 OpenAI / OpenRouter 做 judge

关键位置：

- `scripts/eval_generate.py:49`
- `scripts/eval_generate_sglang.py:27`
- `scripts/eval_winrate.py:44`

---

## 4. 现有证据

### 4.1 `winrate` 结果整体失真

以下文件全部给出：

- `a_wins = 0`
- `b_wins = 0`
- `ties = 200`

示例：

- `eval/winrate_r1_vs_sft.json:1`
- `eval/winrate_r1_vs_gpt4o.json:1`
- `eval/winrate_r2_vs_sft.json`
- `eval/winrate_r3_vs_gpt4o.json`

这在输出差异巨大的情况下几乎不可能成立。

### 4.2 输出不是“没变化”，而是“真的变差了”

我比对了 `outputs_policy_r*` 和 `outputs_sft_baseline` 的逐条完全相等比例：

| 文件 | 与 SFT 完全相同 |
|---|---:|
| `r1` | `2.13%` |
| `r2` | `0.32%` |
| `r3` | `0.30%` |
| `r4` | `0.37%` |
| `r5` | `0.15%` |
| `r6` | `0.19%` |
| `r7` | `0.30%` |

这说明：

- 不是“policy 没变化”
- 是“policy 变了，但大概率朝坏方向变”

### 4.3 典型退化模式已经非常明显

按当前 `eval/` 中已有产物做简单静态统计如下：

| 文件 | `I'm not sure` | `Human:` continuation | 重复句 | `<50` 词短回复 |
|---|---:|---:|---:|---:|
| SFT baseline | `4.2%` | `0.1%` | `12.7%` | `72.5%` |
| `r1` | `2.2%` | `0.0%` | `10.1%` | `82.9%` |
| `r2` | `7.4%` | `0.0%` | `14.4%` | `33.6%` |
| `r3` | `10.8%` | `4.9%` | `21.4%` | `39.8%` |
| `r4` | `9.1%` | `0.1%` | `8.5%` | `45.2%` |
| `r5` | `8.0%` | `4.2%` | `17.7%` | `33.1%` |
| `r6` | `11.0%` | `0.0%` | `14.3%` | `30.8%` |
| `r7` | `10.4%` | `0.0%` | `10.6%` | `45.1%` |

可以直接在输出里看到这些模式：

- 长段重复废话：`eval/outputs_policy_r7.jsonl:1`
- 语义回退到“我不懂你在说什么”：`eval/outputs_policy_r7.jsonl:4`
- 生成 `Human:` continuation / 对话角色错乱：`eval/outputs_policy_r3.jsonl:4629`
- SFT baseline 本身也很差，但 round 后更差：`eval/outputs_sft_baseline.jsonl:5`

---

## 5. 主要问题清单

## P0-1. `eval_winrate.py` 已改过一轮，但当前 parser 仍会系统性误判

位置：

- `scripts/eval_winrate.py:67`
- `scripts/eval_winrate.py:77`
- `scripts/eval_winrate.py:153`

现状：

- 现在已经支持 `A.`、`**A**`、`Response A` 这类格式
- 但 fallback 里仍会在整句搜索 `\\bA\\b` / `\\bB\\b`
- judge 原始输出在保存前就被压成了 `A/B/Tie`，`raw_verdict` 实际不是 raw

为什么严重：

- 普通英文里的冠词 `a` 会被误判成选择 A
- 例如 `"I cannot determine a clear winner."`、`"There is not a clear winner here."` 当前都会被解析成 `A`
- `"Between A and B, A is better."`、`"Between A and B, B is better."` 这类明确判决又会被落成 `Tie`
- 后续没有办法区分“真 Tie”和“解析失败后被当 Tie”

建议：

1. 同时保存 `verdict_raw` 与 `verdict_parsed`
2. 去掉 `\\bA\\b` / `\\bB\\b` 这种会误吃普通英文冠词的宽松 fallback
3. 解析失败单独记成 `parse_error`
4. 修完后再拿现有 `outputs_policy_r*` 全量重跑一遍

## P0-2. full pipeline 表面上传递了 `CURRENT_HF_CKPT`，但 actor 仍没有真正连续训练

位置：

- `scripts/run-full-pipeline-job.sh:127`
- `scripts/run-full-pipeline-job.sh:141`
- `scripts/run-irl-prod.sh:50`
- `slime/utils/arguments.py:683`

现状：

- `run-full-pipeline-job.sh` 现在会把 `CURRENT_HF_CKPT` 逐轮更新到 `policy_r{N}_hf`
- 但 `run-irl-prod.sh` 依旧没有传 `--load`
- 在这个项目里，`--load` 缺失时，会用 `--ref-load` 作为训练初始 checkpoint
- 你当前仍把 `REF_CKPT` 固定为 SFT checkpoint

为什么严重：

- 这意味着 actor 还是会从 SFT 起步，而不是从上一轮 policy 接着学
- `CURRENT_HF_CKPT` 目前没有真正接管 actor state
- 这直接破坏了 `icml.pdf` 里 surrogate 的 `pi_old` 语义

建议：

1. 下一轮 actor 必须 `--load` 上一轮 policy checkpoint
2. `REF_CKPT` 可以继续固定为 SFT，用作 KL anchor
3. 当前 `HF_CKPT=$CURRENT_HF_CKPT` 可以保留，但不能替代 `--load`
4. 修完后再谈“逐轮提升”，否则 `r1..r7` 仍不是真正的连续优化

## P0-3. 多轮对话被错误地当成单轮 user message 套 chat template

位置：

- `prepare_hh_rlhf.py:79`
- `slime/utils/data.py:118`
- `slime/utils/data.py:123`
- `slime/local_rm/model.py:203`
- `scripts/eval_generate.py:49`
- `scripts/eval_generate_sglang.py:27`

现状：

- `prepare_hh_rlhf.py` 输出 `text` 字符串：
  - `"Human: ...\n\nAssistant: ...\n\nHuman: ..."`
- 训练和 reward/tokenize/eval 一旦开 `apply_chat_template`
- 这段字符串会被统一包装成：
  - `[{"role": "user", "content": prompt}]`
- 当前本地仓库没有 `hh-rlhf-processed/` 目录，无法验证你线上运行时是否已经把数据产物改成 `messages`

为什么严重：

- 原本的多轮对话结构消失
- 模型看到的是“一条很长的 user 文本”，里面混着 `Human:` / `Assistant:`
- 这会直接诱发：
  - 回声式回答
  - 角色延续错误
  - 无端追问
  - 输出 `Human:` continuation

建议：

1. IRL 主链路统一改为 `messages` 格式
2. 用 `scripts/prepare_sft_data.py` 的解析方式，把 `text` 提前转成真正的消息列表
3. 训练、reward update、reward eval、eval generation 必须统一同一份 conversation schema
4. 如果你线上数据已经切成 `messages`，需要把这一步的数据准备也落库；否则下次重跑仍会退回旧行为
5. 如果必须继续兼容 `text`，就在 `slime/utils/data.py` 和 `slime/local_rm/model.py` 里显式解析 `Human:` / `Assistant:` 标记，而不是直接包成单轮 user

---

## 6. 次高优先级问题

## P1-1. 第一轮 PPO 没有有效 reward，reward 信号还滞后一轮

位置：

- `scripts/run-full-pipeline-job.sh:126`
- `scripts/run-full-pipeline-job.sh:212`
- `slime/local_rm/custom_rm.py:157`

现状：

- PPO 在前，reward update 在后
- `custom_rm` 发现 `reward_model/latest` 不存在时直接返回 `0.0`

影响：

- Round 1 基本是无 reward / 极弱 reward 的 PPO
- 真正的 RM 要从 round 1 结束后才产生
- 整个闭环天然滞后 1 round
- 这和 `icml.pdf` 想要的“给定 `r_t` 优化 `pi_t`，再用 `pi_t` 更新 `r_{t+1}`”不一致

建议：

1. 在 round 1 开始前先做一次 reward warm-start
2. 或者把流程改成：
   - SFT
   - reward init/update
   - PPO round 1

## P1-2. prod 路径已经统一到 warm-start `accelerate`，但 generic 路径仍保留 `direct` 旧语义

位置：

- `slime/local_rm/update_reward.py:58`
- `slime/local_rm/update_reward_accel.py:129`
- `scripts/run-irl.sh:133`

现状：

- `scripts/run-full-pipeline-job.sh` + `scripts/run-reward-update.sh` 这条 prod 外链路现在固定走 `accelerate`
- 但 `scripts/run-irl.sh` 仍默认 `REWARD_UPDATE_LAUNCHER=direct`
- `slime/local_rm/update_reward.py` 依旧每轮从 base model 重新初始化 RM

影响：

- 相同配置，不同 launcher，学习动态完全不同
- debug 和 prod 结果会非常难对齐
- 如果按 `icml.pdf` 里的理论看，这还会主动放大 successive reward 的变化，和控制 `epsilon` 的目标相冲突

建议：

1. 如果 `run-irl.sh` 还要保留，就把默认 launcher 改成 `accelerate`
2. 或者把 `direct` 路径同步改成 warm-start + shuffle
3. 文档里明确区分 prod 路径已修、generic 路径未修

## P1-3. reward 采样偏差在 `accelerate` 路径上已部分修复，但 `direct` 路径仍是旧实现

位置：

- `slime/local_rm/data.py:18`
- `slime/local_rm/update_reward.py:101`
- `slime/local_rm/update_reward_accel.py:207`
- `slime/local_rm/update_reward_accel.py:238`

现状：

- `slime/local_rm/update_reward_accel.py` 现在已经在 epoch 前和每个 epoch 开始时对 demo / rollout 做 `random.shuffle(...)`
- 但 `slime/local_rm/update_reward.py` 仍按原顺序读入，并直接 `zip(...)` 截断
- 两条路径的训练分布仍不一致

影响：

- reward model 会反复看到 demo 文件头部和 rollout 文件头部
- 训练数据分布偏得很厉害
- 对大窗口 `reward_update_rollout_window=100` 尤其不稳定

建议：

1. 继续保留 `accelerate` 路径里的 shuffle
2. 把同样的逻辑补到 `direct` 路径，或者直接废弃 `direct`
3. `zip(...)` 截断仍然存在，后续最好改成重采样或显式平衡采样

## P1-4. reward 目标和长度惩罚会鼓励“拉长废话”

位置：

- `slime/local_rm/custom_rm.py:18`
- `slime/local_rm/custom_rm.py:129`
- `slime/local_rm/model.py:182`
- `slime/local_rm/update_reward_accel.py:257`

现状：

- `response_length < 50` 直接扣 `3.0`
- `response_length >= 0.9 * 512` 才扣截断惩罚
- reward model 只取最后一个 token 的 scalar
- 更新目标是 `mean(reward_demo) - mean(reward_rollout)`，没有显式偏好对比

影响：

- 模型会倾向：
  - 先凑长度
  - 用模板化语言拖时间
  - 用重复句把回答抻长
- 这与真实 `winrate` 目标并不一致
- 但这里不需要改掉你的 bi-level IRL 主目标，只需要把 reward 参数化和 shaping 从“可刷分”修回“更接近响应质量”

建议：

1. 去掉硬编码短回复扣分，至少先降权
2. 增加针对坏模式的惩罚：
   - 重复 n-gram / 重复句惩罚
   - `Human:` continuation 惩罚
   - 无意义追问惩罚
3. 保持当前 `l_old - c * epsilon` 这条主目标，不要先切到另一套 preference learning
4. 只用最后 token 打分太脆，`response span pooling` 可以作为 reward 头部改造候选，但这属于次级优化，不是第一优先级

## P1-5. reward 产物按 round 命名的问题已基本修复，但 round / rollout 语义仍然混杂

位置：

- `scripts/run-full-pipeline-job.sh:217`
- `scripts/run-reward-update.sh:65`
- `slime/local_rm/update_reward_accel.py:354`
- `slime/local_rm/update_reward_accel.py:361`

现状：

- `update_reward_accel.py` 现在已经把产物改成 `step_round{ROUND_ID}` 和 `reward_eval_round_{ROUND_ID}.json`
- 因此“每轮互相覆盖”这个问题在 prod 路径上已经基本修复
- 但 `run-reward-update.sh` 仍把 `--rollout-id` 固定成 `149`
- 最终 JSON 内部的 `rollout_id`、日志里的 `rollout=149` 仍会和 round 语义混在一起

影响：

- 现在可以回看 round 级 reward 产物了
- 但日志和 JSON 字段仍不够直观，排查时容易把“本轮 round id”和“本轮最后一个 rollout id”混淆

建议：

1. 文件名继续保留按 `ROUND_ID` 命名
2. JSON 内容里额外写入 `round_id`
3. 日志打印也尽量统一用 `round=...`，减少混淆

## P1-6. prod 路径的模板参数已对齐，但 generic HF eval / generic 训练路径仍未跟进

位置：

- `scripts/run-irl-prod.sh:64`
- `scripts/run-reward-update.sh:44`

现状：

- `scripts/run-reward-update.sh` 已经把 `apply_chat_template_kwargs` 对齐到 `{"enable_thinking": false}`
- `scripts/eval_generate_sglang.py` 也已支持透传模板参数
- 但 `scripts/eval_generate.py` 仍没有 `apply_chat_template_kwargs`
- `scripts/run-irl.sh` 这条 generic 训练链路也没有把同样的 kwargs 明确传下去

影响：

- prod 主链路里这项已经明显改善
- 但 generic HF eval / generic 训练链路仍可能和 prod tokenization 不一致

建议：

1. `eval_generate.py` 补齐 `--apply-chat-template-kwargs`
2. `run-irl.sh` 也显式传同一份 kwargs
3. 把“prod 已对齐、generic 未对齐”写进文档，避免误以为全项目都一致了

---

## 7. 低优先级但值得注意的问题

## P2-1. SFT baseline 本身质量也不高

位置：

- `eval/outputs_sft_baseline.jsonl:1`
- `eval/outputs_sft_baseline.jsonl:5`

现状：

- baseline 自己就已经有明显重复、回声、低信息量回复

影响：

- 这会降低“对 SFT 提升”的门槛
- 但不改变当前 round 之后继续退化的判断

建议：

1. baseline 要么换更干净的 SFT
2. 要么至少把 SFT 阶段也改成 `messages` 统一格式

## P2-2. 安全相关样本有明显异常

位置：

- `eval/outputs_policy_r1.jsonl:4631`
- `eval/outputs_policy_r1.jsonl:4641`

现状：

- 在部分危险问题上输出明显不稳

影响：

- 如果后续 `winrate` judge 不把安全性纳入权重，很容易出现“有害但冗长”的错误正反馈

建议：

1. 在 judge prompt 里明确加入 safety / appropriateness
2. 单独做安全集回归，不要只看通用 `winrate`

---

## 8. 提升 winrate 的执行顺序

## 阶段 A：先把评测修准

目标：

- 先拿到可信的 `winrate`

动作：

1. 修 `scripts/eval_winrate.py` 的 verdict 解析
2. 保存 raw judge 输出
3. 对现有 `outputs_policy_r1..r7` 全量重评
4. 把样本分成：
   - 真 Tie
   - 解析失败
   - A 胜
   - B 胜

判定完成标准：

- 不再出现“policy 对 SFT、对 gpt-4o 全部 200/200 tie”这种结果

## 阶段 B：修训练闭环

目标：

- 让 round 之间真正积累 policy

动作：

1. 下一轮 actor 从上一轮 policy checkpoint 加载
2. round 1 之前先初始化 reward model
3. 保持 `REF_CKPT` 固定为 SFT 作为 KL anchor

判定完成标准：

- `r2` 应该是“在 `r1` 基础上继续优化”，而不是重新采样一个独立实验

## 阶段 C：统一对话数据格式

目标：

- 让模型真正看到多轮对话，而不是看一大块伪 user 文本

动作：

1. 训练数据改为 `messages`
2. reward demo / reward eval / winrate eval prompt 全部统一为 `messages`
3. `eval_generate.py` / `eval_generate_sglang.py` 不再把字符串硬包成单轮 user

判定完成标准：

- 输出里 `Human:` continuation、角色错乱、无意义追问显著下降

## 阶段 D：重做 reward update

目标：

- 把 reward 从“可被长度刷分”改成“贴近 winrate”

动作：

1. 统一使用 `accelerate` warm-start
2. 把 `accelerate` 路径里已经加上的 shuffle 逻辑同步到 generic 路径，或直接废弃 `direct`
3. 去掉硬编码短回复大惩罚
4. 增加重复惩罚
5. 保持 `l_old - c * epsilon` 主目标不变，只修 reward shaping 和估计偏差

判定完成标准：

- `I'm not sure`、重复句、废话拉长这几类坏模式下降

## 阶段 E：最后才调 PPO

目标：

- 在前四步都稳定后，再做超参优化

优先尝试：

1. `kl-loss-coef`
2. `lr`
3. `rollout-max-response-len`
4. `global-batch-size`
5. `num_rollout_per_round`

不建议当前阶段优先做的事：

- 直接大调 PPO 超参
- 只盯 reward accuracy 不看真实输出
- 在评测脚本没修好前解读 `winrate`

---

## 9. 最短闭环建议

如果只做最小必要改动，并且明确不改变 `icml.pdf` 的 IRL 主思路，我建议按下面顺序推进：

1. 修 `scripts/eval_winrate.py`
2. 把 round 改成累计 actor checkpoint
3. 把 HH-RLHF IRL 主链路从 `text` 切到 `messages`
4. 统一 reward update 到 warm-start `accelerate`
5. 先做一次 RM bootstrap，再去掉短回复硬惩罚并加入重复惩罚
6. 把 reward JSON / logs 里的字段也统一到 round 语义，和现在的 round 级文件名对齐

这六步做完，再去看新的 `winrate` 曲线才有意义。

---

## 10. 一句话版本

现在的核心问题不是“你的 IRL 思路不对”，而是“当前实现把这套 IRL 思路跑歪了”。先把 `pi_old` 连续性、`epsilon` 小步更新、reward-policy 交替、评测可信度这几件事修正，逐轮提升才有可能真实出现。
