# Inverse Reinforcement Learning (IRL) with Bi-Level Optimization for LLMs and SGLang

This module implements a bi-level optimization framework for jointly training an LLM policy and a trainable reward model, enabling Inverse Reinforcement Learning (IRL) within the SLIME RL framework. It integrates with **SGLang** for high-performance LLM serving.

## Overview

### What is Bi-Level IRL for LLMs?

Traditional LLM RL uses a fixed reward function. This module enables **learning reward functions from expert demonstrations** using bi-level optimization that alternates between:

1. **Policy Update (Level 1)**: Train LLM policy using PPO with **sequence-level rewards** from a trainable reward model
   - LLM generation powered by **SGLang** for efficient inference
2. **Reward Update (Level 2)**: Train reward model using IRL objectives to match expert sequences

Key features:
- **Sequence-level rewards**: Single scalar per entire generated text
- **SGLang integration**: Fast batch inference with prefix caching
- **Expert demonstrations**: Learn reward functions from high-quality examples

### Architecture

```
┌──────────────────────────────────────────────────────┐
│ Bi-Level Optimization for LLM + IRL                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │ LEVEL 1: LLM Policy Update (PPO)            │    │
│  │ ├─ Generate: Sample LLM sequences           │    │
│  │ ├─ Last Hidden: Extract final token state   │    │
│  │ ├─ Reward: R(τ) = Reward_Model(last_hidden)│    │
│  │ ├─ Advantage: Compute A_t via critic        │    │
│  │ └─ Update: PPO loss on policy               │    │
│  └─────────────────────────────────────────────┘    │
│                  ↕                                   │
│                (alternate)                          │
│                  ↕                                   │
│  ┌─────────────────────────────────────────────┐    │
│  │ LEVEL 2: Reward Model Update (IRL)          │    │
│  │ ├─ Expert: Load expert LLM sequences        │    │
│  │ ├─ Policy: Policy-generated sequences       │    │
│  │ ├─ Loss: -E[R(τ_expert)] + E[R(τ_policy)]  │    │
│  │ └─ Update: Gradient descent on reward model │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Components

### 1. Reward Model (`reward_model.py`)

**Class: `RewardModel`**
- MLP that maps LLM hidden states → scalar reward
- **Input**: Final token hidden state [hidden_size] OR aggregated sequence [hidden_size]
- **Output**: Single scalar reward per sequence (not per-token)
- **For LLMs**: Takes last token's hidden state, outputs single float
- Output: Scalar rewards for advantage estimation

**Class: `TrajectoryRewardModel`**
- Transformer-based reward model
- Processes entire trajectory at once
- More expressive but slower

**Factory: `create_reward_model(args, model_type)`**
- Creates reward model based on configuration
- Supports "mlp" (default) and "trajectory" types

### 2. IRL Training (`irl_trainer.py`)

**Class: `MaxEntropyIRLObjective`**

Implements the maximum entropy IRL objective:

```
L = -E_expert[R(τ)] + E_policy[R(τ)] - β·H(π)
```

Where:
- First term: Maximize reward on expert data
- Second term: Minimize reward under policy
- Third term: Encourage exploration via entropy

**Class: `IRLTrainer`**
- Orchestrates reward model training
- Handles batch processing of expert and policy data
- Computes IRL losses and updates reward model

### 3. Bi-Level Optimizer (`bi_level_optimizer.py`)

**Class: `BiLevelOptimizer`**
- Manages alternation between policy and reward updates
- **New**: Integrates with SGLang via `generate_rollouts_with_sglang()`
- Provides reward function for policy training
- Orchestrates IRL training steps

**SGLang Integration**:
- Method: `generate_rollouts_with_sglang(sglang_runtime, prompts, ...)`
- Features: Batch generation, hidden state extraction, efficient serving
- Returns: Rollout dictionary with hidden states, logits, log probabilities

**Class: `BiLevelTrainingLoop`**
- Helper for integrating into standard training loops
- Tracks update ratios and iteration counts

## SGLang Integration

SGLang provides high-performance LLM inference with:
- **Prefix caching**: Reuse cached prompts across generations
- **Batch inference**: Efficient processing of variable-length sequences  
- **GPU kernels**: Optimized attention and sampling operations

### Setup SGLang

```bash
# Install SGLang
pip install sglang

# Or install from source for latest features
git clone https://github.com/hpcaitech/sglang.git
cd sglang
pip install -e .
```

### Initialize SGLang Runtime

```python
import sglang as sgl

# Single GPU
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=1,           # Tensor parallelism
    max_total_tokens=4096,
)

# Multi-GPU (tensor parallelism)
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-70b-hf",
    tp_size=4,           # 4-way tensor parallelism
    max_total_tokens=4096,
)
```

## Usage

### Step 1: Prepare Expert Demonstrations

Expert demonstrations should be stored as a list of trajectory dictionaries with **SEQUENCE-LEVEL REWARDS**:

```python
expert_trajectories = [
    {
        "hidden_states": torch.Tensor([seq_len, hidden_size]),  # From policy model
        "actions": torch.Tensor([seq_len]),                      # Action indices
        "rewards": 0.8,  # SCALAR - reward for entire sequence
        "returns": 1.5,  # SCALAR - return for entire sequence
        "length": 128,   # Optional: actual sequence length
    },
    # ... more trajectories
]

torch.save(expert_trajectories, "expert_data.pt")
```

**Important**: Unlike traditional RL, rewards are **per-sequence not per-token**. This makes sense for LLMs where you want to evaluate the quality of the entire generated text.

### Step 2: Setup Bi-Level Optimizer

```python
from slime.irl import BiLevelOptimizer, create_reward_model
from slime.irl.example_integration import load_expert_demonstrations, setup_bi_level_irl

# Load expert data
expert_dataloader = load_expert_demonstrations("expert_data.pt", batch_size=32)

# Setup arguments
args = Namespace(
    irl_objective="max_entropy",
    irl_update_ratio=1,  # Alternate: 1 policy, 1 reward
    irl_num_epochs=1,
    reward_model_hidden_size=512,
    reward_model_num_layers=2,
    reward_model_dropout=0.1,
    reward_model_lr=1e-4,
    hidden_size=4096,  # From your policy model
)

# Create bi-level optimizer
bi_level_optimizer = setup_bi_level_irl(
    policy_model=actor_model,
    critic_model=critic_model,
    expert_dataloader=expert_dataloader,
    args=args,
    device="cuda",
)
```

### Step 3: Integrate with SGLang for LLM Generation

```python
import sglang as sgl
from slime.irl.example_integration import generate_rollouts_with_sglang

# Initialize SGLang runtime
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=1,
)

# Generate LLM rollouts with SGLang
prompts = [
    "Summarize: The quick brown fox...",
    "Translate to French: Hello world",
]

rollout_data = generate_rollouts_with_sglang(
    bi_level_optimizer=bi_level_optimizer,
    sglang_runtime=runtime,
    prompts=prompts,
    max_new_tokens=128,
    temperature=1.0,
    top_p=0.9,
)
# Returns: {
#   "completions": [...],
#   "hidden_states": [batch, seq_len, hidden_size],
#   "log_probs": [batch, seq_len],
#   "logits": [batch, seq_len, vocab_size],
# }
```

### Step 4: Integrate with Training Loop

In your main training loop (similar to `train.py`):

```python
from slime.irl.example_integration import integrate_with_rollout, generate_rollouts_with_sglang

for rollout_id in range(args.num_rollout):
    # 1. Generate LLM sequences with SGLang (instead of fixed rollout_manager)
    prompts = prepare_prompts_for_batch(rollout_id)
    rollout_data = generate_rollouts_with_sglang(
        bi_level_optimizer,
        sglang_runtime,
        prompts,
        max_new_tokens=128,
        temperature=1.0,
    )
    
    # 2. Extract final token hidden states from LLM generation
    hidden_states = rollout_data["hidden_states"]  # [batch, seq_len, hidden_size]
    final_hidden = hidden_states[:, -1, :]         # [batch, hidden_size]
    
    # 3. Compute sequence-level rewards from trainable model
    rollout_data = integrate_with_rollout(
        bi_level_optimizer,
        rollout_data,
        hidden_states=final_hidden,
    )
    # Now rollout_data["rewards"] = trainable model outputs (not fixed)
    
    # 4. Train policy via PPO (standard SLIME with trainable rewards)
    critic_model.async_train(rollout_id, rollout_data)
    actor_model.async_train(rollout_id, rollout_data)
    
    # 5. Update reward model via IRL (bi-level alternation)
    should_update_reward = bi_level_optimizer.step()
    if should_update_reward:
        metrics = bi_level_optimizer.update_reward_model(
            policy_rollouts=rollout_data,
            num_epochs=args.irl_num_epochs,
        )
        logger.info(f"Reward IRL metrics: {metrics}")

runtime.shutdown()  # Clean up SGLang
```

## Key Parameters

### SGLang Configuration
```python
# Runtime initialization
tp_size = 1                    # Tensor parallelism degree
pp_size = 1                    # Pipeline parallelism degree
max_total_tokens = 4096        # Max tokens in memory

# Generation settings
max_new_tokens = 128           # Tokens to generate per prompt
temperature = 1.0              # Sampling temperature
top_p = 0.9                    # Nucleus sampling parameter
```

### Reward Model Configuration (for LLMs)
```python
--reward-model-hidden-size 512      # Should match or be smaller than LLM hidden size
--reward-model-num-layers 2         # 2-3 layers typically sufficient
--reward-model-dropout 0.1          # Prevent overfitting to expert data
--reward-model-lr 1e-4              # Usually smaller than policy LR
--reward-model-type mlp             # MLP for speed, "trajectory" for more capacity
```

### IRL Configuration
```python
--irl-objective max_entropy          # Only option currently
--irl-update-ratio 2                 # 2:1 means train policy twice, then reward once
--irl-num-epochs 1                   # Reward model training epochs per update
```

### IRL Loss Weights (in MaxEntropyIRLObjective)
```python
reward_weight = 1.0       # Weight on expert reward term
policy_weight = 0.1       # Weight on policy reward term  
entropy_weight = 0.01     # Weight on policy entropy term (often set to 0 for LLMs)
```

## Mathematical Details

### Maximum Entropy IRL for LLM Sequences

Given:
- Expert sequences $\tau_e$ (text from human/expert policy)
- Policy sequences $\tau_\pi$ (text from current LLM policy)
- Trainable reward model $R_\phi(\tau)$ that outputs scalar reward per sequence

The objective is:
$$L = -E_{\tau \sim D_{expert}}[R_\phi(\tau)] + E_{\tau \sim \pi}[R_\phi(\tau)] - \beta H(\pi)$$

**Interpretation for LLMs:**
- First term: Reward model learns to assign **high values** to expert sequences
- Second term: Reward model assigns **low values** to policy-generated sequences  
- Third term: Policy entropy encourages diverse generation (often disabled for LLMs)
- $R_\phi(\tau)$ is a single scalar per sequence (e.g., coherence, correctness, safety score)

### Why Sequence-Level Rewards?

For LLMs, sequence-level (trajectory-level) rewards make more sense than token-level because:
1. **Meaning is global**: A sentence's quality depends on the full text, not individual words
2. **Expert demonstrations**: Usually provided as complete sequences with a single quality score
3. **Simplicity**: One reward per sequence vs. $|seq\_len|$ per-token rewards
4. **Computational efficiency**: Reward model is simpler (one MLP forward pass)

### Bi-Level Optimization for LLMs

At iteration $t$:
1. **Policy Update** (N steps): Train LLM via PPO using $R_\phi(\tau)$ as reward
2. **Reward Update** (M steps): Update $\phi$ to match expert behavior via IRL loss
3. **Alternation**: After $N/M$ iterations, switch to reward update

## Example: Learning LLM Behavior from Expert Demonstrations

### Scenario
Train an LLM to match expert text generation (e.g., summarization, translation) without explicit reward labels.

### Solution with Bi-Level IRL
1. Collect expert demonstrations (high-quality summaries, translations, etc.)
2. Initialize trainable reward model
3. Training loop alternates:
   - **Policy phase**: LLM generates sequences, reward model scores them, PPO trains policy
   - **Reward phase**: Reward model learns to prefer expert sequences over policy sequences
4. Converges to: LLM behaves like expert, reward model captures expert preferences

## Implementation Files

- `reward_model.py`: Trainable reward models (MLP for LLM hidden states)
- `irl_trainer.py`: IRL training with sequence-level rewards
- `bi_level_optimizer.py`: Alternation between policy and reward updates
- `example_integration.py`: Integration examples with SLIME training loop
- `__init__.py`: Module exports

## Advanced Features

### Using Mean Pooling Instead of Last Token

If you want to use average hidden states instead of just the last token:

```python
# In integrate_with_rollout
hidden_states = rollout_data["hidden_states"]  # [batch, seq_len, hidden_size]
aggregated_hidden = hidden_states.mean(dim=1)  # [batch, hidden_size]

rollout_data = integrate_with_rollout(
    bi_level_optimizer,
    rollout_data,
    hidden_states=aggregated_hidden,
)
```

### Custom IRL Objectives

To implement a different IRL objective (e.g., AIRL, GAIL):

```python
from slime.irl.irl_trainer import IRLTrainer

class CustomIRLObjective:
    def compute_loss(self, expert_traj, policy_traj):
        # Your custom loss computation
        return loss, metrics

# Register in IRLTrainer.__init__
if irl_objective == "custom":
    self.objective = CustomIRLObjective(...)
```

### Multi-Reward Heads

For learning multiple reward signals:

```python
class MultiRewardModel(nn.Module):
    def __init__(self, input_size, num_rewards=3):
        super().__init__()
        self.mlp = nn.Sequential(...)
        self.reward_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_rewards)
        ])
    
    def forward(self, features):
        hidden = self.mlp(features)
        rewards = [head(hidden) for head in self.reward_heads]
        return torch.cat(rewards, dim=-1)
```

## Troubleshooting

### Issue: Reward model learning trivial rewards

**Solutions:**
- Increase `reward_weight` in IRL objective
- Decrease `policy_weight` to reduce focus on policy data
- Ensure expert data quality and diversity

### Issue: Policy ignores reward model

**Solutions:**
- Check reward scaling (add normalization if needed)
- Increase reward model learning rate
- Reduce `update_ratio` to update reward model more frequently

### Issue: Divergence during training

**Solutions:**
- Reduce learning rates for both policy and reward model
- Add gradient clipping (already done in IRLTrainer)
- Increase batch size for more stable updates

## References

- Maximum Entropy IRL: Ziebart et al. (2008)
- Bi-level Optimization: Lorraine et al. (2020)
- PPO: Schulman et al. (2017)

## Citation

If you use this IRL implementation, please cite the SLIME framework and the relevant papers above.
