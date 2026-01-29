"""Example integration of bi-level IRL with SLIME PPO training and SGLang.

This module demonstrates how to integrate the IRL pipeline with the existing
SLIME training framework and SGLang for efficient LLM serving. It shows:

1. How to load expert demonstrations
2. How to create the bi-level optimizer
3. How to generate LLM rollouts with SGLang
4. How to integrate with the standard training loop
5. How to use the trainable reward model instead of a fixed one
"""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slime.irl import BiLevelOptimizer, create_reward_model
from slime.irl.irl_trainer import ExpertDataset

logger = logging.getLogger(__name__)


def load_expert_demonstrations(
    expert_path: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> DataLoader:
    """Load expert demonstrations from file.
    
    For LLM-based RL, expert trajectories should contain SEQUENCE-LEVEL rewards.
    
    Expected file format (pickle or torch):
    - List of trajectories, each containing:
        {
            "hidden_states": torch.Tensor [seq_len, hidden_size],
            "actions": torch.Tensor [seq_len],
            "rewards": float (SCALAR - reward for entire sequence, not per-token),
            "returns": float (SCALAR - return for entire sequence),
            "length": int (actual sequence length, optional),
        }
    
    Args:
        expert_path: Path to saved expert demonstrations
        batch_size: Batch size for dataloader
        device: Device to use
    
    Returns:
        DataLoader for expert trajectories (batch size is number of sequences, not tokens)
    """
    logger.info(f"Loading expert demonstrations from {expert_path}")
    
    # Load demonstrations
    if expert_path.endswith(".pt"):
        trajectories = torch.load(expert_path, map_location=device)
    elif expert_path.endswith(".pkl"):
        import pickle
        with open(expert_path, "rb") as f:
            trajectories = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {expert_path}")
    
    logger.info(f"Loaded {len(trajectories)} expert trajectories")
    
    # Create dataset and dataloader
    expert_dataset = ExpertDataset(trajectories)
    expert_dataloader = DataLoader(
        expert_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    
    return expert_dataloader


def setup_bi_level_irl(
    policy_model: nn.Module,
    critic_model: Optional[nn.Module] = None,
    expert_dataloader: Optional[DataLoader] = None,
    args: Optional[Namespace] = None,
    device: str = "cuda",
) -> BiLevelOptimizer:
    """Setup bi-level optimization for IRL.
    
    Args:
        policy_model: Policy model (actor)
        critic_model: Value function (critic)
        expert_dataloader: Expert demonstration loader
        args: Training arguments
        device: Device to use
    
    Returns:
        Initialized BiLevelOptimizer
    """
    logger.info("Setting up bi-level IRL optimizer")
    
    # Create trainable reward model
    reward_model = create_reward_model(args, model_type="mlp")
    logger.info(f"Created reward model: {reward_model}")
    
    # Create bi-level optimizer
    bi_level_optimizer = BiLevelOptimizer(
        reward_model=reward_model,
        policy_model=policy_model,
        critic_model=critic_model,
        expert_dataloader=expert_dataloader,
        args=args,
        device=device,
    )
    
    logger.info("Bi-level IRL optimizer initialized")
    return bi_level_optimizer


def integrate_with_rollout(
    bi_level_optimizer: BiLevelOptimizer,
    rollout_data: Dict[str, torch.Tensor],
    hidden_states_from_policy: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Integrate trainable reward model into rollout processing.
    
    This replaces the fixed reward model with the trainable one,
    and uses the bi-level optimizer to compute rewards.
    
    Args:
        bi_level_optimizer: BiLevelOptimizer instance
        rollout_data: Rollout data dict (may contain old rewards)
        hidden_states_from_policy: Hidden states from policy forward pass
    
    Returns:
        Updated rollout_data with trainable rewards
    """
    # Get reward function from bi-level optimizer
    reward_fn = bi_level_optimizer.get_reward_fn()
    
    # Compute rewards using trainable reward model
    with torch.no_grad():
        computed_rewards = reward_fn(hidden_states_from_policy)
    
    # Update rollout data
    rollout_data["rewards"] = computed_rewards
    rollout_data["reward_model"] = "trainable"  # Mark as from trainable model
    
    return rollout_data


# ============================================================================
# Example usage in training loop
# ============================================================================

def example_training_loop():
    """Example of how to use bi-level IRL in a training loop.
    
    This shows the integration point with the standard SLIME training loop.
    """
    
    # Setup (done once)
    args = Namespace(
        # IRL configuration
        irl_objective="max_entropy",
        irl_update_ratio=1,  # Alternate: 1 policy, 1 reward update
        irl_num_epochs=1,
        reward_model_hidden_size=512,
        reward_model_num_layers=2,
        reward_model_dropout=0.1,
        reward_model_lr=1e-4,
        hidden_size=4096,
        # ... other args
    )
    
    # Initialize models
    policy_model = nn.Linear(4096, 50257)  # Dummy policy
    critic_model = nn.Linear(4096, 1)      # Dummy critic
    
    # Load expert demonstrations
def generate_rollouts_with_sglang(
    bi_level_optimizer: BiLevelOptimizer,
    sglang_runtime: Any,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """Generate LLM rollouts using SGLang for efficient serving.
    
    SGLang provides optimized kernels for LLM inference with:
    - Prefix caching for prompt reuse
    - Batch processing with variable lengths
    - Efficient attention computation
    
    Args:
        bi_level_optimizer: BiLevelOptimizer instance
        sglang_runtime: SGLang Runtime (e.g., from sglang.Runtime)
        prompts: List of prompt strings for generation
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (> 1 for more diversity)
        top_p: Nucleus sampling parameter
    
    Returns:
        Rollout dictionary with hidden_states, logits, log_probs, etc.
    """
    logger.info(f"Generating {len(prompts)} rollouts with SGLang")
    
    rollout_data = bi_level_optimizer.generate_rollouts_with_sglang(
        sglang_runtime=sglang_runtime,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    return rollout_data


def example_training_loop_with_sglang():
    """Example training loop using SGLang for LLM rollout generation.
    
    This shows how to integrate SGLang with bi-level IRL training.
    """
    try:
        import sglang as sgl
    except ImportError:
        logger.error("SGLang not installed. Install with: pip install sglang")
        return
    
    # Load expert demonstrations
    expert_dataloader = load_expert_demonstrations(
        "path/to/expert_data.pt",
        batch_size=32,
    )
    
    # Setup models (would normally be loaded from checkpoints)
    hidden_size = 4096
    vocab_size = 50257
    policy_model = nn.Linear(hidden_size, vocab_size)  # Dummy
    critic_model = nn.Linear(hidden_size, 1)  # Dummy
    
    # Setup bi-level optimizer
    args = Namespace(
        irl_update_ratio=1,
        irl_num_epochs=1,
        reward_model_lr=1e-4,
        irl_objective="max_entropy",
    )
    
    bi_level_optimizer = setup_bi_level_irl(
        policy_model=policy_model,
        critic_model=critic_model,
        expert_dataloader=expert_dataloader,
        args=args,
        device="cuda",
    )
    
    # Initialize SGLang runtime
    # In practice, you'd load an actual LLM
    try:
        runtime = sgl.Runtime(
            model_path="meta-llama/Llama-2-7b-hf",  # Example model
            tp_size=1,  # Tensor parallelism
            max_total_tokens=2048,
        )
    except Exception as e:
        logger.warning(f"Could not initialize SGLang runtime: {e}")
        logger.info("Using dummy runtime for demonstration")
        runtime = None
    
    # Training loop
    prompts_template = [
        "Summarize this text:",
        "Translate to French:",
        "Answer the question:",
    ]
    
    for rollout_id in range(10):
        # Prepare prompts for this batch
        batch_size = 4
        prompts = [prompts_template[rollout_id % len(prompts_template)]] * batch_size
        
        # Generate rollouts with SGLang
        if runtime is not None:
            rollout_data = generate_rollouts_with_sglang(
                bi_level_optimizer=bi_level_optimizer,
                sglang_runtime=runtime,
                prompts=prompts,
                max_new_tokens=128,
                temperature=1.0,
                top_p=0.9,
            )
        else:
            # Dummy rollout for demonstration
            rollout_data = {
                "hidden_states": torch.randn(batch_size, 128, 4096),
                "logits": torch.randn(batch_size, 128, vocab_size),
                "log_probs": torch.randn(batch_size, 128),
            }
        
        # Extract final hidden states for reward model
        hidden_states = rollout_data.get("hidden_states")
        if hidden_states is not None:
            # Use final token's hidden state
            final_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
            
            # Compute rewards from trainable model
            rollout_data = integrate_with_rollout(
                bi_level_optimizer,
                rollout_data,
                hidden_states=final_hidden,
            )
        
        # Update reward model if needed (bi-level alternation)
        should_update_reward = bi_level_optimizer.step()
        
        if should_update_reward:
            logger.info(f"Rollout {rollout_id}: Updating reward model via IRL")
            metrics = bi_level_optimizer.update_reward_model(
                policy_rollouts=rollout_data,
                num_epochs=args.irl_num_epochs,
            )
            logger.info(f"IRL metrics: {metrics}")
        
        if rollout_id % 5 == 0:
            logger.info(f"Rollout {rollout_id} completed")
    
    logger.info("Training completed!")
    
    # Cleanup
    if runtime is not None:
        runtime.shutdown()



if __name__ == "__main__":
    # Example usage
    logger.info("Running example with SGLang...")
    example_training_loop_with_sglang()

