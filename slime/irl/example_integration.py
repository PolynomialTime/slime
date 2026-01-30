"""Example integration of bi-level IRL with SLIME PPO training and SGLang.

This module demonstrates how to integrate the IRL pipeline with the existing
SLIME training framework and SGLang for efficient LLM serving. It shows:

1. How to load expert demonstrations
2. How to create the bi-level optimizer  
3. How to generate LLM rollouts with SGLang using SLIME rollout API
4. How to integrate with the standard training loop
5. How to use the trainable reward model instead of a fixed one
"""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slime.irl import BiLevelOptimizer, create_reward_model
from slime.irl.irl_trainer import ExpertDataset
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def load_expert_demonstrations(
    expert_path: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> DataLoader:
    """Load expert demonstrations from file or Sample objects.
    
    For LLM-based RL, expert trajectories should contain SEQUENCE-LEVEL rewards.
    
    Expected file format (pickle or torch):
    - List of trajectories/Sample objects, each containing:
        For trajectories:
        {
            "hidden_states": torch.Tensor [seq_len, hidden_size],
            "actions": torch.Tensor [seq_len],
            "rewards": float (SCALAR - reward for entire sequence, not per-token),
            "returns": float (SCALAR - return for entire sequence),
            "length": int (actual sequence length, optional),
        }
        
        For Sample objects:
        Sample(
            response="...",
            tokens=[...],
            reward=0.5,
            rollout_log_probs=[...],
        )
    
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
    rollout_data: Dict[str, Any],
    hidden_states: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Integrate trainable reward model into rollout processing.
    
    This replaces the fixed reward model with the trainable one,
    and uses the bi-level optimizer to compute rewards.
    
    Args:
        bi_level_optimizer: BiLevelOptimizer instance
        rollout_data: Rollout data dict (may contain old rewards)
        hidden_states: Optional hidden states to use for reward computation
    
    Returns:
        Updated rollout_data with trainable rewards
    """
    # Get reward function from bi-level optimizer
    reward_fn = bi_level_optimizer.get_reward_fn()
    
    # Use provided hidden states or extract from rollout data
    if hidden_states is None:
        hidden_states = rollout_data.get("hidden_states")
    
    if hidden_states is not None:
        # Compute rewards using trainable reward model
        with torch.no_grad():
            computed_rewards = reward_fn(hidden_states)
        
        # Update rollout data
        rollout_data["rewards"] = computed_rewards
        rollout_data["reward_model"] = "trainable"  # Mark as from trainable model
    
    return rollout_data


def generate_rollouts_with_sglang_api(
    bi_level_optimizer: BiLevelOptimizer,
    args: Namespace,
    prompts: List[str],
    data_source: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate LLM rollouts using SGLang API from SLIME framework.
    
    This uses the high-level generate_rollout API from slime.rollout.sglang_rollout,
    which handles all the complexity of generation, reward model evaluation, and
    metrics collection.
    
    Args:
        bi_level_optimizer: BiLevelOptimizer instance
        args: Arguments with SGLang configuration
        prompts: List of prompt strings for generation
        data_source: Optional data source for rollout management
    
    Returns:
        Rollout dictionary with samples, completions, rewards, and metrics
    """
    logger.info(f"Generating {len(prompts)} rollouts with SGLang API")
    
    # Use the bi-level optimizer's SGLang generation method
    rollout_data = bi_level_optimizer.generate_rollouts_with_sglang(
        args=args,
        prompts=prompts,
        data_source=data_source,
    )
    
    logger.info(f"Generated {len(rollout_data.get('samples', []))} samples")
    return rollout_data


# ============================================================================
# Example usage in training loop
# ============================================================================

def example_training_loop_with_sglang():
    """Example training loop using SGLang for LLM rollout generation.
    
    This shows how to integrate SGLang with bi-level IRL training
    using the SLIME framework's rollout API.
    """
    
    # Load expert demonstrations
    logger.info("Loading expert demonstrations...")
    expert_dataloader = load_expert_demonstrations(
        "path/to/expert_data.pt",
        batch_size=32,
    )
    
    # Setup models (would normally be loaded from checkpoints)
    hidden_size = 4096
    vocab_size = 50257
    policy_model = nn.Linear(hidden_size, vocab_size)
    critic_model = nn.Linear(hidden_size, 1)
    
    # Setup bi-level optimizer with IRL configuration
    args = Namespace(
        # SGLang configuration
        hf_checkpoint="meta-llama/Llama-2-7b-hf",
        sglang_router_ip="localhost",
        sglang_router_port=30000,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=None,
        rollout_temperature=1.0,
        rollout_top_p=0.9,
        rollout_top_k=0,
        rollout_max_response_len=128,
        rollout_stop=[],
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=False,
        n_samples_per_prompt=1,
        rollout_seed=42,
        rollout_global_dataset=True,
        rollout_batch_size=32,
        over_sampling_batch_size=32,
        group_rm=False,
        custom_generate_function_path=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        ci_test=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        sglang_enable_deterministic_inference=False,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        multimodal_keys=[],
        reward_key=None,
        eval_reward_key=None,
        # IRL configuration
        irl_update_ratio=1,
        irl_num_epochs=1,
        reward_model_lr=1e-4,
        reward_model_hidden_size=512,
        reward_model_num_layers=2,
        reward_model_dropout=0.1,
        irl_objective="max_entropy",
        hidden_size=hidden_size,
    )
    
    bi_level_optimizer = setup_bi_level_irl(
        policy_model=policy_model,
        critic_model=critic_model,
        expert_dataloader=expert_dataloader,
        args=args,
        device="cuda",
    )
    
    # Training loop
    prompts = [
        "Summarize the following text in one sentence:",
        "Translate to French:",
        "Answer the question:",
        "Explain the concept:",
    ]
    
    logger.info("Starting training loop...")
    for rollout_id in range(10):
        # Prepare batch of prompts
        batch_size = 4
        batch_prompts = [prompts[rollout_id % len(prompts)]] * batch_size
        
        # Generate rollouts with SGLang (including reward evaluation)
        try:
            rollout_data = generate_rollouts_with_sglang_api(
                bi_level_optimizer=bi_level_optimizer,
                args=args,
                prompts=batch_prompts,
                data_source=None,
            )
        except Exception as e:
            logger.error(f"Error during rollout generation: {e}")
            logger.info("Using dummy rollout for demonstration")
            # Dummy rollout for demonstration
            rollout_data = {
                "samples": [],
                "completions": ["dummy"] * batch_size,
                "rewards": torch.randn(batch_size),
                "log_probs": torch.randn(batch_size, 128),
                "metrics": {},
            }
        
        # Update reward model if needed (bi-level alternation)
        should_update_reward = bi_level_optimizer.step()
        
        if should_update_reward:
            logger.info(f"Rollout {rollout_id}: Updating reward model via IRL")
            try:
                metrics = bi_level_optimizer.update_reward_model(
                    policy_rollouts=rollout_data.get("samples", []),
                    num_epochs=args.irl_num_epochs,
                )
                logger.info(f"IRL metrics: {metrics}")
            except Exception as e:
                logger.warning(f"Error during reward model update: {e}")
        
        if rollout_id % 5 == 0:
            logger.info(
                f"Rollout {rollout_id}: "
                f"Mean reward: {rollout_data.get('rewards', torch.tensor([])).mean():.4f}"
            )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger.info("Running example with SGLang...")
    example_training_loop_with_sglang()
