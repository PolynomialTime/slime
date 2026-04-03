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
| `r1` | `14.47%` |
| `r2` | `0.39%` |
| `r3` | `0.30%` |
| `r4` | `0.37%` |
| `r5` | `0.15%` |
| `r6` | `0.19%` |
| `r7` | `0.30%` |

这说明：

- 不是“policy 没变化”
- 是“policy 变了，但大概率朝坏方向变”

### 4.3 典型退化模式已经非常明显

静态统计如下：

| 文件 | `I'm not sure` | 重复句 | `<50` 词短回复 |
|---|---:|---:|---:|
| SFT baseline | `4.2%` | `9.7%` | `72.5%` |
| `r1` | `4.1%` | `15.1%` | `65.0%` |
| `r2` | `9.0%` | `20.0%` | `39.7%` |
| `r3` | `10.8%` | `25.0%` | `39.8%` |
| `r4` | `9.1%` | `15.0%` | `45.2%` |
| `r5` | `8.0%` | `22.7%` | `33.1%` |
| `r6` | `11.0%` | `20.0%` | `30.8%` |
| `r7` | `10.4%` | `15.4%` | `45.1%` |

可以直接在输出里看到这些模式：

- 长段重复废话：`eval/outputs_policy_r7.jsonl:1`
- 语义回退到“我不懂你在说什么”：`eval/outputs_policy_r7.jsonl:4`
- 生成 `Human:` continuation / 对话角色错乱：`eval/outputs_policy_r1.jsonl:4629`
- SFT baseline 本身也很差，但 round 后更差：`eval/outputs_sft_baseline.jsonl:5`

---

## 5. 主要问题清单

## P0-1. `winrate` judge 解析过于脆弱，导致几乎全 Tie

位置：

- `scripts/eval_winrate.py:66`
- `scripts/eval_winrate.py:72`

现状：

- judge 返回文本后，只看第一个词
- 第一词不是严格的 `A` / `B` / `Tie`，就直接按 `Tie` 处理

为什么严重：

- judge 模型经常返回 `A.`、`A - ...`、`**A**`、`Response A` 这类格式
- 这会把大量本来有胜负的样本硬吞成 `Tie`
- 现有 `200/200 ties` 几乎可以直接判定为评测脚本失真

建议：

1. 把原始 judge 输出完整保存
2. 解析规则改成正则匹配：
   - `^A[\\W_]*$`
   - `^B[\\W_]*$`
   - `\\bA\\b`
   - `\\bB\\b`
   - `Tie|Same`
3. 对解析失败样本单独计数，不要混入 `Tie`
4. 先拿现有 `outputs_policy_r*` 全量重跑一遍，得到真实曲线

## P0-2. 每个 round 都从 SFT 重新开跑，而不是累计上一轮 policy

位置：

- `scripts/run-full-pipeline-job.sh:130`
- `scripts/run-full-pipeline-job.sh:145`
- `scripts/run-irl-prod.sh:50`

现状：

- 每轮 PPO 都固定：
  - `HF_CKPT=$SLIME/models/sft_checkpoint_hf`
  - `REF_CKPT=$SLIME/models/sft_checkpoint`
- 没有把上一轮 actor checkpoint 作为下一轮 `--load`
- 每轮输出 `r1..r7` 更像“7 次独立实验”，不是“7 次连续优化”

为什么严重：

- reward update 在 round 结束后才生效
- 但下一轮 policy 没继承上一轮 policy，前一轮 policy 学到的行为会直接丢失
- 这会严重削弱 reward learning 对 policy 的累计影响
- 更关键的是，这直接破坏了 `icml.pdf` 里 surrogate 的 `pi_old` 语义；当前 `pi_old` 不是“上一轮 policy”，而是每轮重新从 SFT 开出来的独立 policy

建议：

1. 下一轮 actor 必须 `--load` 上一轮 policy checkpoint
2. `REF_CKPT` 可以继续固定为 SFT，用作 KL anchor
3. 把 round 定义成：
   - round 0: SFT
   - round 1: SFT + RM1 -> policy1
   - round 2: policy1 + RM2 -> policy2
4. 重新定义 `r1..r7` 的含义，不要再让它们互相独立

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
4. 如果必须继续兼容 `text`，就在 `slime/utils/data.py` 和 `slime/local_rm/model.py` 里显式解析 `Human:` / `Assistant:` 标记，而不是直接包成单轮 user

---

## 6. 次高优先级问题

## P1-1. 第一轮 PPO 没有有效 reward，reward 信号还滞后一轮

位置：

- `scripts/run-full-pipeline-job.sh:124`
- `scripts/run-full-pipeline-job.sh:196`
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

## P1-2. reward update 的 `direct` 和 `accelerate` 路径语义不一致

位置：

- `slime/local_rm/update_reward.py:58`
- `slime/local_rm/update_reward_accel.py:129`
- `scripts/run-irl.sh:133`

现状：

- `direct` 路径每轮都从 base model 重新初始化 RM
- `accelerate` 路径会 warm-start 上一版 RM

影响：

- 相同配置，不同 launcher，学习动态完全不同
- debug 和 prod 结果会非常难对齐
- 如果按 `icml.pdf` 里的理论看，这还会主动放大 successive reward 的变化，和控制 `epsilon` 的目标相冲突

建议：

1. 统一只保留 `accelerate` warm-start 语义
2. 或者两条路径都改成 warm-start
3. 明确在文档中写清楚 reward update 的真实行为

## P1-3. reward update 的采样方式有系统性偏差

位置：

- `slime/local_rm/data.py:18`
- `slime/local_rm/update_reward.py:101`
- `slime/local_rm/update_reward_accel.py:207`
- `slime/local_rm/update_reward_accel.py:238`

现状：

- demo 样本按文件顺序加载
- rollout 样本按文件顺序加载
- batch 数不匹配时直接 `zip(...)` 截断短边
- 没有全局 shuffle

影响：

- reward model 会反复看到 demo 文件头部和 rollout 文件头部
- 训练数据分布偏得很厉害
- 对大窗口 `reward_update_rollout_window=100` 尤其不稳定

建议：

1. demo 与 rollout 在每轮 reward update 前都做独立 shuffle
2. 不要简单 `zip` 截断，改成：
   - 随机采样到同样批次数
   - 或者按较短一侧做重采样平衡
3. 把使用率打印出来，并持久化到日志

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

## P1-5. full pipeline 目前无法稳定观测“逐轮 reward 是否提升”

位置：

- `scripts/run-full-pipeline-job.sh:201`
- `scripts/run-reward-update.sh:65`
- `slime/local_rm/update_reward_accel.py:347`
- `slime/local_rm/update_reward_accel.py:353`

现状：

- full pipeline 每轮都把 `ROLLOUT_END` 设成 `NUM_ROLLOUT_PER_ROUND - 1`
- 当前配置下就是固定 `149`
- reward update 存档名和 eval 文件名都绑定 `cli.rollout_id`
- 结果是 `step_149/` 和 `reward_eval_rollout_149.json` 会被每轮覆盖

影响：

- 你没法可靠回看“reward accuracy 是否逐轮提升”
- 即使 reward 真在变好，产物也会把 round 间差异抹掉

建议：

1. 奖励模型快照和 eval 结果按 `ROUND_ID` 命名，而不是按固定 `rollout_id`
2. 或者把 full pipeline 的 rollout id 设计成跨 round 单调递增
3. reward 曲线、policy 曲线、checkpoint 命名都要用同一 round 语义

## P1-6. reward update 的模板参数和 PPO/eval 不一致

位置：

- `scripts/run-irl-prod.sh:64`
- `scripts/run-reward-update.sh:44`

现状：

- PPO 阶段有 `--apply-chat-template-kwargs '{"enable_thinking":false}'`
- 外部 reward update 里 `apply_chat_template_kwargs` 是空字典

影响：

- 同一条样本在 PPO 看到的 tokenization，和 reward update / reward eval 看到的 tokenization 不一定一致

建议：

1. 把 PPO 使用的 chat template kwargs 原样传给 reward update
2. 训练、reward、eval 三条链路统一模板参数

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
2. demo / rollout 全局 shuffle
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
6. 把 reward 产物按 round 正确命名，保证能看到逐轮曲线

这六步做完，再去看新的 `winrate` 曲线才有意义。

---

## 10. 一句话版本

现在的核心问题不是“你的 IRL 思路不对”，而是“当前实现把这套 IRL 思路跑歪了”。先把 `pi_old` 连续性、`epsilon` 小步更新、reward-policy 交替、评测可信度这几件事修正，逐轮提升才有可能真实出现。
