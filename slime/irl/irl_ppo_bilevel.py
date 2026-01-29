#!/usr/bin/env python3
"""Example script for bi-level IRL training with SLIME.

This script demonstrates how to train a policy and reward model jointly
using bi-level optimization with PPO and maximum entropy IRL.

Usage:
    python examples/irl_ppo_bilevel.py \
        --expert-data-path path/to/expert_data.pt \
        --irl-objective max_entropy \
        --irl-update-ratio 1
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slime.irl import BiLevelOptimizer, create_reward_model
from slime.irl.example_integration import (
    load_expert_demonstrations,
    setup_bi_level_irl,
    integrate_with_rollout,
)
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def add_irl_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add IRL-specific arguments to parser."""
    
    # IRL data
    parser.add_argument(
        "--expert-data-path",
        type=str,
        required=True,
        help="Path to expert demonstrations (.pt or .pkl file)",
    )
    
    # IRL configuration
    parser.add_argument(
        "--irl-objective",
        type=str,
        default="max_entropy",
        choices=["max_entropy"],
        help="IRL objective to use",
    )
    
    parser.add_argument(
        "--irl-update-ratio",
        type=int,
        default=1,
        help="Policy:Reward update ratio (1 means alternate)",
    )
    
    parser.add_argument(
        "--irl-num-epochs",
        type=int,
        default=1,
        help="Number of training epochs per reward model update",
    )
    
    # Reward model architecture
    parser.add_argument(
        "--reward-model-hidden-size",
        type=int,
        default=512,
        help="Reward model hidden layer dimension",
    )
    
    parser.add_argument(
        "--reward-model-num-layers",
        type=int,
        default=2,
        help="Number of reward model layers",
    )
    
    parser.add_argument(
        "--reward-model-dropout",
        type=float,
        default=0.1,
        help="Reward model dropout rate",
    )
    
    parser.add_argument(
        "--reward-model-lr",
        type=float,
        default=1e-4,
        help="Reward model learning rate",
    )
    
    parser.add_argument(
        "--reward-model-type",
        type=str,
        default="mlp",
        choices=["mlp", "trajectory"],
        help="Type of reward model architecture",
    )
    
    # IRL loss weights
    parser.add_argument(
        "--irl-reward-weight",
        type=float,
        default=1.0,
        help="Weight for expert reward term in IRL loss",
    )
    
    parser.add_argument(
        "--irl-policy-weight",
        type=float,
        default=0.1,
        help="Weight for policy reward term in IRL loss",
    )
    
    parser.add_argument(
        "--irl-entropy-weight",
        type=float,
        default=0.01,
        help="Weight for policy entropy term in IRL loss",
    )
    
    return parser


def create_dummy_models(
    hidden_size: int = 4096,
    vocab_size: int = 50257,
) -> tuple[nn.Module, nn.Module]:
    """Create dummy policy and critic models for demonstration.
    
    In a real scenario, these would be loaded from checkpoints.
    """
    
    # Policy model: outputs logits
    policy_model = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, vocab_size),
    )
    
    # Critic model: outputs values
    critic_model = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )
    
    return policy_model, critic_model


def create_dummy_rollout_data(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Create dummy rollout data for demonstration.
    
    In a real scenario, this comes from rollout_manager.generate().
    """
    
    return {
        "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
        "actions": torch.randint(0, 50257, (batch_size, seq_len)),
        "log_probs": torch.randn(batch_size, seq_len),
        "response_lengths": torch.randint(100, seq_len, (batch_size,)),
        "loss_masks": torch.ones(batch_size, seq_len),
    }


def main():
    """Main training loop with bi-level IRL."""
    
    # Setup logging
    configure_logger()
    logger.info("Starting bi-level IRL training")
    
    # Parse arguments
    parser = ArgumentParser(description="Bi-level IRL with PPO")
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    parser = add_irl_arguments(parser)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    logger.info(f"Arguments: {args}")
    
    # Check expert data path
    if not Path(args.expert_data_path).exists():
        logger.error(f"Expert data not found: {args.expert_data_path}")
        logger.info("Creating dummy expert data for demonstration...")
        
        # Create dummy expert data
        expert_trajectories = [
            {
                "hidden_states": torch.randn(512, args.hidden_size),
                "actions": torch.randint(0, args.vocab_size, (512,)),
                "rewards": torch.randn(512),
                "returns": torch.randn(512),
            }
            for _ in range(10)
        ]
        
        # Save dummy data
        Path(args.expert_data_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(expert_trajectories, args.expert_data_path)
        logger.info(f"Saved dummy expert data to {args.expert_data_path}")
    
    # Load expert demonstrations
    expert_dataloader = load_expert_demonstrations(
        args.expert_data_path,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Create models
    logger.info("Creating policy and critic models...")
    policy_model, critic_model = create_dummy_models(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )
    policy_model.to(args.device)
    critic_model.to(args.device)
    
    # Setup bi-level IRL
    logger.info("Setting up bi-level IRL optimizer...")
    bi_level_optimizer = setup_bi_level_irl(
        policy_model=policy_model,
        critic_model=critic_model,
        expert_dataloader=expert_dataloader,
        args=args,
        device=args.device,
    )
    
    # Training loop
    logger.info("Starting training loop...")
    
    for rollout_id in range(args.num_rollouts):
        # 1. Generate rollout data
        # (In real code, comes from rollout_manager)
        rollout_data = create_dummy_rollout_data(
            batch_size=args.batch_size,
            seq_len=512,
            hidden_size=args.hidden_size,
        )
        rollout_data = {
            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in rollout_data.items()
        }
        
        # 2. Use trainable reward model
        hidden_states = rollout_data["hidden_states"]
        rollout_data = integrate_with_rollout(
            bi_level_optimizer,
            rollout_data,
            hidden_states,
        )
        
        # 3. (Normally) Train policy here
        # In real code: actor_model.async_train(rollout_id, rollout_data)
        #              critic_model.async_train(rollout_id, rollout_data)
        
        # 4. Update reward model if needed
        should_update_reward = bi_level_optimizer.step()
        
        if should_update_reward:
            logger.info(f"\n{'='*60}")
            logger.info(f"Rollout {rollout_id}: Updating reward model")
            logger.info(f"{'='*60}")
            
            metrics = bi_level_optimizer.update_reward_model(
                policy_rollouts=rollout_data,
                num_epochs=args.irl_num_epochs,
            )
            
            logger.info(f"Reward update metrics:")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.6f}")
        
        # Log progress
        if (rollout_id + 1) % 10 == 0:
            logger.info(f"Rollout {rollout_id + 1}/{args.num_rollouts} completed")
    
    logger.info("\nTraining completed!")
    logger.info(f"Bi-level optimizer info: {bi_level_optimizer.get_info()}")


if __name__ == "__main__":
    main()
