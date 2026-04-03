# Winrate 排查总报告

## 1. 结论摘要

当前仓库里和 `winrate` 相关的主要问题，不是单点 bug，而是四条链路还存在剩余断点：

1. `winrate` parser 已经明显好于上一版，也会保存 `raw_verdict` / `parse_error`，但仍有部分常见裁决句式解析不到，现有 judge 结果还需要重跑确认。
2. full pipeline 的 actor 连续加载在当前 HEAD 已经补上，但仍需要一次实跑验证确认 round 间确实连续。
3. `HH-RLHF` 多轮对话在训练 / RM / SGLang eval 主链路上已经做了运行时解析，但数据产物仍是 `text`，generic HF eval 仍未对齐。
4. reward update 在 prod 路径上已经修了一批问题，但 round 1 无 RM、generic 路径分叉、reward 性能记录不完全可信这些问题还在。

如果以 `icml.pdf` 的 bi-level IRL 目标为准，真正优先要修的不是“换方法”，而是把实现重新拉回这三个条件：

- `pi_old` 必须真的是上一轮 policy，而不是每轮回到 SFT
- `r_theta` 必须连续 warm-start，才能让 `epsilon` 保持小步变化
- reward / policy 必须真正交替，而不是 reward 永远滞后一轮

基于当前 `eval/outputs_*.jsonl` 的静态统计，policy 输出仍没有体现“逐轮变好”：

- `I'm not sure` 出现率：SFT baseline `4.2%`，`r3` `10.8%`，`r6` `11.0%`
- 重复句样本占比：SFT baseline `12.7%`，`r3` `21.4%`
- `eval/winrate_r1_vs_sft.json`、`eval/winrate_r1_vs_gpt4o.json` 等结果全部是 `200/200 ties`

因此当前最重要的工作顺序仍然不是调 PPO 超参，而是：

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

但当前最关键的几个问题里，已经有一项得到实质修复，其余问题仍然存在：

1. actor 连续加载这项在当前 HEAD 已经补上了。`run-full-pipeline-job.sh` 现在传 `ACTOR_LOAD=$PREV_SAVE_DIR`，`run-irl-prod.sh` 也会在目录存在时追加 `--load ${ACTOR_LOAD}`。这解决了之前“每轮从 SFT 起步”的核心断点，但还需要一轮真实训练来确认行为与预期一致。
2. `eval_winrate.py` 不再是“只看首词”，但 parser 现在仍有剩余盲区。像 `"The better response is A."`、`"I prefer B."`、`"Answer: A"`、`"Verdict: B"` 这类句式当前仍会落成 `Tie`。
3. `HH-RLHF text` 在 `slime/utils/data.py`、`slime/local_rm/model.py`、`scripts/eval_generate_sglang.py` 里已经会被即时解析成多轮 `messages`，所以 prod 主链路不再是“整段字符串硬包成单条 user”。但 `prepare_hh_rlhf.py` 仍产出 `text`，`scripts/eval_generate.py` 这条 generic HF eval 仍未跟进，schema 仍然分裂。
4. round 1 前仍没有 reward bootstrap。`custom_rm` 在 `reward_model/latest` 不存在时仍返回 `0.0`，而 full pipeline 还是先 PPO、后 reward update。
5. generic IRL 链路还停留在旧语义：`scripts/run-irl.sh` 仍默认 `direct` launcher，而 `slime/local_rm/update_reward.py` 依旧每轮从 base model 重置 RM。
6. reward checkpoint 虽然会恢复 inline eval 的最佳权重，但 `reward_eval_round_{N}.json` 当前仍写入“最后一次 eval”的准确率，不一定等于最终保存下来的 best checkpoint 表现。

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

## P0-1. `eval_winrate.py` 已补上 `raw_verdict` / `parse_error`，但 parser 还存在剩余鲁棒性缺口

位置：

- `scripts/eval_winrate.py:67`
- `scripts/eval_winrate.py:69`
- `scripts/eval_winrate.py:84`
- `scripts/eval_winrate.py:155`
- `scripts/eval_winrate.py:156`

现状：

- 当前 parser 已经支持 `A.`、`**A**`、`Response A`、`My answer: B`、`I choose B`、`Between A and B, A is better.` 这类常见格式
- `raw_verdict` 现在会保存原始 judge 输出，`parse_error` 也会单独统计
- 之前“普通英文里的冠词 `a` 被误判成 A”这个问题已经消失
- 但仍有一些合理句式会被吞成 `Tie`，例如 `"The better response is A."`、`"I prefer B."`、`"Answer: A"`、`"Verdict: B"`

为什么重要：

- 这版 parser 已经不是灾难性的，但还不能直接当作完全可信
- 只要 parse fallback 还会吞掉明确裁决，`winrate` 仍可能被系统性拉向 `Tie`
- 现有 `eval/winrate_*.json` 很可能混有旧 parser 产物；在没有重跑之前，不能把这些数字当作当前 HEAD 的真实结论

建议：

1. 继续补齐常见裁决句式，优先覆盖 `THE BETTER RESPONSE IS X`、`I PREFER X`、`ANSWER:`、`VERDICT:`
2. 保留当前 `raw_verdict` 与 `parse_error` 输出
3. 修完后拿现有 `outputs_policy_r*` 全量重跑一遍，并直接检查 `parse_errors / total`

## P0-2. actor 连续加载在当前 HEAD 已补上，但需要一次实跑确认 round 间确实连续

位置：

- `scripts/run-full-pipeline-job.sh:129`
- `scripts/run-full-pipeline-job.sh:150`
- `scripts/run-irl-prod.sh:40`
- `scripts/run-irl-prod.sh:61`

现状：

- `run-full-pipeline-job.sh` 现在会把上一轮 torch-dist checkpoint 通过 `ACTOR_LOAD=$PREV_SAVE_DIR` 传给下一轮
- `run-irl-prod.sh` 也会在目录存在时显式追加 `--load ${ACTOR_LOAD}`
- 因此 actor state 已经不再只依赖 `--ref-load` / SFT 初始化
- 当前剩下的问题不是“没接上”，而是还没有一次实跑证据证明 round 间真的按预期连续

为什么重要：

- 这是 `icml.pdf` 里 `pi_old` 语义成立的必要条件
- 这项如果不通，逐轮提升就无从谈起
- 现在代码路径已经打通，优先级从“实现缺失”降到了“需要验证”

建议：

1. 保留当前 `ACTOR_LOAD -> --load` 方案
2. `REF_CKPT` 继续固定为 SFT，作为 KL anchor
3. 下一次实跑时重点验证 round2 是否真正从 round1 actor 继续训练
4. 这项验证一旦通过，就可以从主问题清单里移除

## P0-3. HH-RLHF 多轮对话在 prod 主链路上已做兼容解析，但数据产物和 generic eval 仍未统一

位置：

- `prepare_hh_rlhf.py:79`
- `slime/utils/data.py:118`
- `slime/utils/data.py:123`
- `slime/local_rm/model.py:203`
- `scripts/eval_generate.py:49`
- `scripts/eval_generate_sglang.py:27`

现状：

- `prepare_hh_rlhf.py` 仍输出 `text` 字符串，而不是持久化的 `messages`
- 但 `slime/utils/data.py`、`slime/local_rm/model.py`、`scripts/eval_generate_sglang.py` 现在已经会把 `Human: ... / Assistant: ...` 字符串即时解析成多轮消息
- 因此 prod 训练 / reward tokenize / SGLang eval 这条主链路，不再是简单的 `[{"role":"user","content": prompt}]`
- 剩余未对齐点在于：
  - `scripts/eval_generate.py` 这条 generic HF eval 仍然把字符串直接包成单轮 user
  - 仓库里的数据产物仍是 `text-at-rest, messages-at-runtime`
  - 当前本地仓库没有 `hh-rlhf-processed/` 产物，无法验证你线上是否已经把数据物理迁移成 `messages`

为什么重要：

- 主链路的角色错乱风险比之前小了很多，这是实质性修复
- 但 schema 分裂仍会带来两类问题：
  - generic eval / generic 训练链路和 prod 行为不一致
  - debug 时同一份数据在“落盘格式”和“运行时格式”之间来回切换，复现问题会很痛苦

建议：

1. 如果追求最稳妥，还是把 `prepare_hh_rlhf.py` 产物直接切成持久化 `messages`
2. 如果短期不改数据格式，至少把 `scripts/eval_generate.py` 也补上同样的 HH 文本解析逻辑
3. 文档里明确记录当前状态是 `text-at-rest, messages-at-runtime`
4. 如果你线上数据已经切成 `messages`，把转换链路也落到仓库，避免下次重跑退回兼容路径

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

实现建议：

1. 把 `pi_0` 明确定义成 SFT policy
2. 在进入 round loop 之前，先用 SFT policy 生成一批 bootstrap rollout，并保存成和 PPO 一样的 `rollout_*.pt`
3. 用这批 `pi_0` rollout 配合 demo 先训练出第一个 reward，记作 `r_1`
4. 再进入 PPO loop，用 `r_1` 从 `pi_0` 训练出 `pi_1`
5. 后续 round 统一成：
   - 用 `pi_{t-1}` 的 rollout 更新 `r_t`
   - 用 `r_t` 从 `pi_{t-1}` 训练 `pi_t`

落到当前仓库，最小改法应该是：

1. 在 `scripts/run-full-pipeline-job.sh` 的 round loop 前插一个 bootstrap phase
2. 这一步不要只生成 `outputs_sft_baseline.jsonl`，还要额外落一份 reward update 可直接读取的 `rollout_*.pt`
3. 这份 bootstrap rollout 最稳妥的来源是复用现有 rollout engine / `--save-debug-rollout-data`，而不是单独发明一套 JSONL 格式
4. 然后在进入 round 1 之前先调用一次 `scripts/run-reward-update.sh`
5. `ROUND_ID` / 日志里最好显式标成 `bootstrap` 或 `round0_reward`，避免和正式 round 混淆

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

## P1-4. reward shaping 已经比上一版合理很多，但长度偏置和序列打分方式仍可能扭曲 winrate

位置：

- `slime/local_rm/custom_rm.py:18`
- `slime/local_rm/custom_rm.py:129`
- `slime/local_rm/model.py:182`
- `slime/local_rm/update_reward_accel.py:257`

现状：

- `custom_rm.py` 已经不再是上一版那种粗暴 shaping：
  - 短回复惩罚从 `3.0` 降到了 `1.0`
  - 新增了 `Human:` continuation、`Assistant:` 前缀、重复 4-gram 惩罚
- 但 `response_length < 50` 仍然会被硬扣分
- reward model 仍只取最后一个 token 的 scalar
- 更新目标仍是 `mean(reward_demo) - mean(reward_rollout)`，这点本身符合你当前 IRL 主思路

影响：

- 这版 reward 已经明显比之前更不容易被“纯长度”刷分
- 但它仍然天然不利于简短但高质量的回答
- 对“reward 性能逐轮提高”的观测也会有干扰，因为 reward 分数里还混着较强的人工 shaping 偏置

建议：

1. 保留现有的重复惩罚和角色混淆惩罚
2. 继续弱化硬长度惩罚，或者把它改成只针对明显不完整回答的软约束
3. 保持当前 `l_old - c * epsilon` 主目标，不要切走 IRL 主思路
4. 如果后续还有明显 reward / winrate 脱节，再考虑 reward head 的更稳健聚合方式

## P1-5. reward 产物按 round 命名已基本修复，但性能记录仍和最终保存模型不完全一致

位置：

- `scripts/run-full-pipeline-job.sh:217`
- `scripts/run-reward-update.sh:65`
- `slime/local_rm/update_reward_accel.py:354`
- `slime/local_rm/update_reward_accel.py:361`
- `slime/local_rm/update_reward_accel.py:374`

现状：

- `update_reward_accel.py` 现在已经把产物改成 `step_round{ROUND_ID}` 和 `reward_eval_round_{ROUND_ID}.json`
- 因此“每轮互相覆盖”这个问题在 prod 路径上已经基本修复
- 但当前实现会先做 final eval，再恢复 inline eval 期间的 best checkpoint
- `reward_eval_round_{ROUND_ID}.json` 里的顶层 `accuracy` 用的是“最后一次 eval”的结果，不一定等于最终保存下来的 best checkpoint 表现
- 同时 JSON 里仍只写 `rollout_id`，没有显式 `round_id`

影响：

- 现在可以回看 round 级 reward 产物了
- 但如果你要判断“reward 性能是否逐轮提高”，当前 JSON 可能会低估最终保存模型的真实表现
- 日志和 JSON 字段也仍然容易把“本轮 round id”和“本轮最后一个 rollout id”混淆

建议：

1. 文件名继续保留按 `ROUND_ID` 命名
2. JSON 里同时写入 `round_id`、`final_acc`、`best_acc`
3. 如果恢复了 best checkpoint，最好在恢复后再跑一次 final eval，保证落盘指标和最终模型一致

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

## 阶段 A：把评测修到可重跑、可解释

目标：

- 先拿到可信的 `winrate`

动作：

1. 继续补齐 `scripts/eval_winrate.py` 的 verdict 解析
2. 保留并检查 `raw_verdict` / `parse_error`
3. 对现有 `outputs_policy_r1..r7` 全量重评
4. 把样本分成：
   - 真 Tie
   - 解析失败
   - A 胜
   - B 胜

判定完成标准：

- 不再出现“policy 对 SFT、对 gpt-4o 全部 200/200 tie”这种结果

## 阶段 B：验证并补齐训练闭环

目标：

- 让 round 之间真正积累 policy

动作：

1. 实跑确认下一轮 actor 确实从上一轮 policy checkpoint 继续
2. round 1 之前先初始化 reward model
3. 保持 `REF_CKPT` 固定为 SFT 作为 KL anchor

判定完成标准：

- `r2` 应该是“在 `r1` 基础上继续优化”，而不是重新采样一个独立实验

## 阶段 C：把对话 schema 真正统一

目标：

- 让模型真正看到多轮对话，而不是看一大块伪 user 文本

动作：

1. 要么直接把训练数据改为持久化 `messages`
2. 要么至少把 generic `eval_generate.py` 补到和 prod 主链路一致
3. reward demo / reward eval / winrate eval prompt 使用同一套 conversation 语义

判定完成标准：

- 输出里 `Human:` continuation、角色错乱、无意义追问显著下降

## 阶段 D：重做 reward update

目标：

- 把 reward 从“可被长度刷分”改成“贴近 winrate”

动作：

1. 统一使用 `accelerate` warm-start
2. 把 `accelerate` 路径里已经加上的 shuffle 逻辑同步到 generic 路径，或直接废弃 `direct`
3. 继续弱化硬编码短回复惩罚
4. 保留并监控重复惩罚 / 角色混淆惩罚
5. 保证 reward 落盘指标和最终保存模型一致
6. 保持 `l_old - c * epsilon` 主目标不变，只修 reward shaping 和估计偏差

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

1. 补齐 `scripts/eval_winrate.py` 剩余 parser 漏洞，并重跑现有 `winrate`
2. 实跑验证累计 actor checkpoint 已经真的生效
3. 先做一次 RM bootstrap：用 `pi_0 = SFT` 的 rollout 配 demo 训练出 `r_1`
4. 把 generic eval / generic IRL 路径继续往 prod 语义靠齐
5. 继续弱化硬长度惩罚，同时保留重复 / 角色混淆惩罚
6. 把 reward JSON / logs 的指标改成和最终保存 checkpoint 一致，并补 `round_id`

这六步做完，再去看新的 `winrate` 曲线才有意义。

如果按“完整 round = 先 reward，再 PPO”来定义，推荐记账方式是：

- bootstrap：`pi_0 -> r_1`
- round 1：`(pi_0, r_1) -> pi_1`
- round 2 前 reward：`pi_1 -> r_2`
- round 2：`(pi_1, r_2) -> pi_2`

这样 round 的语义会比现在清楚得多，也更贴近 `icml.pdf` 里的 `pi_old -> r_t -> pi_t`。

---

## 10. 一句话版本

现在的核心问题不是“你的 IRL 思路不对”，而是“当前实现把这套 IRL 思路跑歪了”。先把 `pi_old` 连续性、`epsilon` 小步更新、reward-policy 交替、评测可信度这几件事修正，逐轮提升才有可能真实出现。
