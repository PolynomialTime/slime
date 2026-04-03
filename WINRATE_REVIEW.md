# Winrate 排查总报告

## 1. 结论摘要

当前仓库里和 `winrate` 相关的主要问题，不是单点 bug，而是四条链路还存在剩余断点：

1. `winrate` parser 现在已经能覆盖大多数常见 judge 输出格式，主要剩余工作是重跑全量结果并审计 `parse_error`。
2. full pipeline 的 actor 连续加载在正常顺序执行下已经补上，但断点续跑仍依赖上一轮 `save_dir` 没被清掉。
3. `HH-RLHF` 多轮对话在训练 / RM / SGLang eval / generic HF eval 主链路上都已经做了运行时解析，但数据产物仍是 `text-at-rest`，schema 还没有真正统一。
4. reward bootstrap 已经接进 pipeline，但当前 bootstrap rollout 是通过 PPO runner 生成的，不是纯粹固定 `pi_0` 的 rollout-only phase。

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
3. `scripts/eval_generate_sglang.py` 和 `scripts/eval_generate.py` 都已经支持 HH 文本解析和模板参数透传
4. `slime/local_rm/update_reward_accel.py` 已经加入 `random.shuffle(...)`，并把 reward 产物改成按 round 命名，同时落盘 `best_acc/final_acc/restored_acc`
5. `scripts/run-irl.sh` 已经把默认 `reward-update-launcher` 切到了 `accelerate`
6. `scripts/run-full-pipeline-job.sh` 已经接入 bootstrap reward phase

但当前最关键的几个问题里，剩下的是这些实现细节：

1. actor 连续加载这项在正常顺序执行下已经补上了。`run-full-pipeline-job.sh` 现在传 `ACTOR_LOAD=$PREV_SAVE_DIR`，`run-irl-prod.sh` 也会在目录存在时追加 `--load ${ACTOR_LOAD}`。但如果某轮被 skip 且对应 `save_dir_r{N}` 已被清掉，后续 round 仍有机会回退到 `ref_load` / SFT 初始化。
2. `eval_winrate.py` 的 parser 常见句式已经基本补齐，当前首要工作已经从“修 parser”变成“重跑现有 `winrate` 并检查 `parse_errors / total`”。
3. `HH-RLHF text` 在训练 / RM / SGLang eval / generic HF eval 里都已经会被即时解析成多轮 `messages`，但 `prepare_hh_rlhf.py` 仍产出 `text`，仓库状态仍然是 `text-at-rest, messages-at-runtime`。
4. reward bootstrap 已经接进了 full pipeline，但当前 bootstrap phase 还是通过 `run-irl-prod.sh` 这条 PPO runner 产生 rollout；这意味着 `r_1` 吃到的是“零 reward PPO 过程中采到的 rollout”，而不是理论上最干净的固定 `pi_0` rollout。
5. `direct` reward update 路径虽然不再是默认值，但旧实现仍在，显式切回去时依旧会每轮从 base model 重置 RM。

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

## P0-1. bootstrap reward 已接入，但当前 bootstrap rollout 不是严格固定 `pi_0` 的 rollout-only 数据

位置：

- `scripts/run-full-pipeline-job.sh:125`
- `scripts/run-full-pipeline-job.sh:140`
- `scripts/run-irl-prod.sh:3`
- `scripts/run-irl-prod.sh:95`

现状：

- `run-full-pipeline-job.sh` 现在会在 round loop 前跑一个 bootstrap phase
- 但这一步调用的仍是 `run-irl-prod.sh`，而 `run-irl-prod.sh` 是 PPO phase，不是纯 rollout-only 脚本
- 因此 bootstrap reward update 吃到的是零 reward PPO 过程中采出来的 rollout 数据，而不是理论上最干净的固定 `pi_0 = SFT` 样本

为什么重要：

- 你的目标是先用 `pi_0` 对比 demo 训练出 `r_1`
- 当前做法在工程上是可运行的，但在语义上并不等于“固定 `pi_0` 采样再训 reward”
- 这会让 bootstrap reward 的定义比论文设定更模糊，也会额外引入一次没必要的 PPO 训练开销

建议：

1. 最稳妥的修法是单独做一个 rollout-only bootstrap phase
2. 直接复用现有 rollout engine 和 `--save-debug-rollout-data`，但不要在 bootstrap 时真的进入 PPO 训练
3. 如果短期先保留当前实现，至少把它在文档里明确记成“bootstrap rollout via PPO runner”，不要把它当成严格的 `pi_0` rollout

## P0-2. actor 连续加载在正常顺序执行下已补上，但断点续跑仍依赖上一轮 `save_dir` 存在

位置：

- `scripts/run-full-pipeline-job.sh:162`
- `scripts/run-full-pipeline-job.sh:186`
- `scripts/run-irl-prod.sh:40`
- `scripts/run-irl-prod.sh:61`
- `slime/utils/arguments.py:1525`

现状：

- `run-full-pipeline-job.sh` 现在会把上一轮 torch-dist checkpoint 通过 `ACTOR_LOAD=$PREV_SAVE_DIR` 传给下一轮
- `run-irl-prod.sh` 也会在目录存在时显式追加 `--load ${ACTOR_LOAD}`
- 这保证了正常顺序执行时，下一轮 actor 会从上一轮 Megatron save_dir 连续训练
- 但 round 被 skip 时，脚本只在 `save_dir_r{N}` 目录真实存在时才回填 `PREV_SAVE_DIR`
- 一旦某轮 policy / eval / reward_eval 都在，但对应 `save_dir_r{N}` 已经被删掉，后续 round 可能会因为没有有效 `--load` 而回退到 `ref_load`

为什么重要：

- 这是 `icml.pdf` 里 `pi_old` 语义成立的必要条件
- 正常连续跑已经没问题，但一旦涉及断点续跑或人工清理 save_dir，这个坑就会重新出现
- 你现在的脚本本来就会自动删除较早轮次的 save_dir，所以这个场景不是纯理论风险

建议：

1. 保留当前 `ACTOR_LOAD -> --load` 方案
2. `REF_CKPT` 继续固定为 SFT，作为 KL anchor
3. 如果某轮被 skip 但 `save_dir` 已不存在，就不要默认继续往后跑；应该显式报错或要求从最近仍有 save_dir 的 round 恢复
4. 下一次实跑时除了验证正常 round 连续，还要专门验证一次断点续跑场景

## P0-3. HH-RLHF 多轮对话主链路已经对齐，但数据仍停留在 `text-at-rest, messages-at-runtime`

位置：

- `prepare_hh_rlhf.py:79`
- `slime/utils/data.py:118`
- `slime/utils/data.py:123`
- `slime/local_rm/model.py:203`
- `scripts/eval_generate_sglang.py:27`
- `scripts/eval_generate.py:53`

现状：

- `prepare_hh_rlhf.py` 仍输出 `text` 字符串，而不是持久化的 `messages`
- 但 `slime/utils/data.py`、`slime/local_rm/model.py`、`scripts/eval_generate_sglang.py` 现在已经会把 `Human: ... / Assistant: ...` 字符串即时解析成多轮消息
- 因此 prod 训练 / reward tokenize / SGLang eval 这条主链路，不再是简单的 `[{"role":"user","content": prompt}]`
- `scripts/eval_generate.py` 这条 generic HF eval 现在也已经补上同样的 HH 文本解析逻辑
- 当前真正剩下的未统一点是：
  - 仓库里的数据产物仍是 `text-at-rest, messages-at-runtime`
  - 当前本地仓库没有 `hh-rlhf-processed/` 产物，无法验证你线上是否已经把数据物理迁移成 `messages`

为什么重要：

- 主链路的角色错乱风险比之前小了很多，这是实质性修复
- 但 schema 仍然没有真正收敛到单一数据格式
- debug 时同一份数据在“落盘格式”和“运行时格式”之间来回切换，复现问题仍然很痛苦

建议：

1. 如果追求最稳妥，还是把 `prepare_hh_rlhf.py` 产物直接切成持久化 `messages`
2. 文档里明确记录当前状态是 `text-at-rest, messages-at-runtime`
3. 如果你线上数据已经切成 `messages`，把转换链路也落到仓库，避免下次重跑退回兼容路径

---

## 6. 次高优先级问题

## P1-1. round 1 前的 reward bootstrap 已补上，但 bootstrap 的跳过条件仍然过于宽松

位置：

- `scripts/run-full-pipeline-job.sh:128`
- `scripts/run-full-pipeline-job.sh:149`
- `scripts/run-full-pipeline-job.sh:158`

现状：

- full pipeline 现在已经在 round loop 前加入了 RM bootstrap phase
- 但 bootstrap 是否跳过，只看 `models/reward_model/latest/config.json` 是否存在
- 也就是说，只要目录里留着一个旧 reward model，脚本就会直接跳过 bootstrap

影响：

- 同一个工程目录里如果残留了上一次实验的 RM，新的 pipeline 可能会直接复用它
- 这样 round 1 的 reward 起点就不再对应当前这次 SFT / 当前这次 bootstrap rollout
- 对想观察“reward 和 policy 是否逐轮提高”的实验来说，这会污染起点

建议：

1. bootstrap 跳过条件至少要同时校验当前实验的标识或 bootstrap 产物
2. 最低限度也应该检查 `reward_eval_round_0.json` 或类似 bootstrap 标记文件
3. 如果实验切换了 SFT / 数据 / 超参，最好强制重新 bootstrap

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

## P1-2. `accelerate` 已成为默认路径，但 `direct` 旧实现仍然保留为可选分支

位置：

- `slime/local_rm/update_reward.py:58`
- `slime/local_rm/update_reward_accel.py:129`
- `scripts/run-irl.sh:133`

现状：

- `scripts/run-full-pipeline-job.sh` + `scripts/run-reward-update.sh` 这条 prod 外链路现在固定走 `accelerate`
- `scripts/run-irl.sh` 也已经把默认 `REWARD_UPDATE_LAUNCHER` 切到了 `accelerate`
- 但 `slime/local_rm/update_reward.py` 这条 `direct` 实现还在，且依旧每轮从 base model 重新初始化 RM

影响：

- 默认行为已经比之前一致很多
- 但只要有人显式切回 `direct`，学习动态仍会和 prod 差很大
- 如果按 `icml.pdf` 的理论看，这仍会主动放大 successive reward 的变化，和控制 `epsilon` 的目标相冲突

建议：

1. 既然默认值已经切了，下一步要么补齐 `direct`，要么明确把它标成 deprecated
2. 如果继续保留 `direct`，至少要同步 warm-start + shuffle
3. 文档里明确区分“默认路径已修”和“备用路径未修”

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

## P1-5. reward 产物和性能记录已经基本修复，剩下的是选定一套对外展示的 canonical metric

位置：

- `scripts/run-reward-update.sh:65`
- `slime/local_rm/update_reward_accel.py:361`
- `slime/local_rm/update_reward_accel.py:382`

现状：

- `update_reward_accel.py` 现在已经把产物改成 `step_round{ROUND_ID}` 和 `reward_eval_round_{ROUND_ID}.json`
- JSON 里已经写入了 `round_id`、`best_acc`、`final_acc`、`restored_acc`
- 并且在恢复 best checkpoint 之后又补跑了一次 restored eval

影响：

- 现在可以比较稳定地回看 round 级 reward 产物了
- 当前真正剩下的问题不是“记错了”，而是后续图表 / 报告里到底应该统一展示 `best_acc`、`final_acc` 还是 `restored_acc`

建议：

1. 文件名继续保留按 `ROUND_ID` 命名
2. 对外汇报时优先统一使用 `restored_acc`，因为它最接近最终落盘模型
3. 如果后续画曲线，也最好明确注明展示的是 `best_acc` 还是 `restored_acc`

## P1-6. prod / generic 路径的模板参数和 HH 解析已经基本对齐，剩余问题主要回到数据落盘格式

位置：

- `scripts/run-irl-prod.sh:64`
- `scripts/run-reward-update.sh:44`
- `scripts/run-irl.sh:82`
- `scripts/eval_generate.py:53`

现状：

- `scripts/run-reward-update.sh` 已经把 `apply_chat_template_kwargs` 对齐到 `{"enable_thinking": false}`
- `scripts/eval_generate_sglang.py` 和 `scripts/eval_generate.py` 都已支持透传模板参数
- `scripts/run-irl.sh` 也已经显式传同一份 kwargs

影响：

- prod / generic 路径的 tokenization 一致性已经明显改善
- 剩余不一致更多来自“数据落盘是 text，运行时再解析成 messages”这层 schema 折中

建议：

1. 这一项从主问题里基本可以降级
2. 后续重点应转到持久化 `messages`，而不是继续修模板参数

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

1. 直接对现有 `outputs_policy_r1..r7` 全量重评
2. 检查 `raw_verdict` / `parse_error`
3. 必要时再继续补 verdict 解析
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

1. 把 bootstrap rollout 从 PPO runner 里拆出来，做成更干净的 rollout-only phase
2. 实跑确认下一轮 actor 确实从上一轮 policy checkpoint 继续
3. 保持 `REF_CKPT` 固定为 SFT 作为 KL anchor

判定完成标准：

- `r2` 应该是“在 `r1` 基础上继续优化”，而不是重新采样一个独立实验

## 阶段 C：把对话 schema 真正统一

目标：

- 让模型真正看到多轮对话，而不是看一大块伪 user 文本

动作：

1. 要么直接把训练数据改为持久化 `messages`
2. reward demo / reward eval / winrate eval prompt 使用同一套 conversation 语义
3. 避免长期停留在 `text-at-rest, messages-at-runtime`

判定完成标准：

- 输出里 `Human:` continuation、角色错乱、无意义追问显著下降

## 阶段 D：重做 reward update

目标：

- 把 reward 从“可被长度刷分”改成“贴近 winrate”

动作：

1. 统一使用 `accelerate` warm-start
2. 把 `accelerate` 路径里已经加上的 shuffle 逻辑同步到 `direct`，或直接废弃 `direct`
3. 继续弱化硬编码短回复惩罚
4. 保留并监控重复惩罚 / 角色混淆惩罚
5. 对外统一使用一套 reward 指标，比如 `restored_acc`
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

1. 直接重跑现有 `winrate`，并审计 `parse_errors`
2. 把 bootstrap rollout 从 PPO runner 里拆出来，做成更干净的 `pi_0` rollout
3. 实跑验证累计 actor checkpoint 在断点续跑场景下也真的生效
4. 继续弱化硬长度惩罚，同时保留重复 / 角色混淆惩罚
5. 对外统一 reward 指标口径，优先用 `restored_acc`
6. 再决定是否把 `text-at-rest` 正式迁移成持久化 `messages`

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
