#!/usr/bin/env python3
"""Training script for bi-level IRL optimization.

This script demonstrates how to train an LLM policy and trainable reward model
jointly using bi-level optimization with PPO and maximum entropy IRL.

Usage:
    python -m slime.irl.train \
        --expert-data-path path/to/expert_data.pt \
        --irl-objective max_entropy \
        --irl-update-ratio 1
"""

import logging
from argparse import Namespace

import ray

from slime.irl import BiLevelOptimizer, create_reward_model
from slime.irl.example_integration import load_expert_demonstrations, setup_bi_level_irl
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def train_irl(args):
    """Train policy and reward model using bi-level optimization.
    
    Args:
        args: Training arguments from command line
    """
    configure_logger()
    
    # Allocate GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
    
    # Create rollout manager with SGLang
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    
    # Create policy (actor) and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    
    # Load expert demonstrations
    logger.info(f"Loading expert demonstrations from {args.expert_data_path}")
    expert_dataloader = load_expert_demonstrations(
        args.expert_data_path,
        batch_size=getattr(args, "expert_batch_size", 32),
        device="cuda",
    )
    
    # Setup bi-level optimizer with IRL
    logger.info("Setting up bi-level IRL optimizer")
    bi_level_optimizer = setup_bi_level_irl(
        policy_model=actor_model,
        critic_model=critic_model,
        expert_dataloader=expert_dataloader,
        args=args,
        device="cuda",
    )
    
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    
    # Always update weight first
    actor_model.update_weights()
    
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
    
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
    
    # Special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))
    
    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()
    
    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
    
    # Main training loop
    logger.info(f"Starting bi-level IRL training for {args.num_rollout} rollouts")
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Evaluation before training
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))
        
        # Generate rollouts (with SGLang)
        logger.info(f"Rollout {rollout_id}: Generating sequences")
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        
        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        
        # Train policy with current reward model
        logger.info(f"Rollout {rollout_id}: Training policy")
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        
        # Bi-level alternation: Update reward model via IRL
        should_update_reward = bi_level_optimizer.step()
        if should_update_reward:
            logger.info(f"Rollout {rollout_id}: Updating reward model via IRL")
            try:
                # Extract policy samples from rollout data for IRL training
                # Note: rollout_data_ref is a Ray reference, may need adjustment based on actual structure
                irl_metrics = bi_level_optimizer.update_reward_model(
                    policy_rollouts=rollout_data_ref,
                    num_epochs=getattr(args, "irl_num_epochs", 1),
                )
                logger.info(f"IRL metrics: {irl_metrics}")
            except Exception as e:
                logger.warning(f"Error updating reward model: {e}")
        
        # Periodic saving
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            logger.info(f"Rollout {rollout_id}: Saving checkpoint")
            save(rollout_id)
        
        # Offload and reload weights
        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())
        
        # Periodic evaluation
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            logger.info(f"Rollout {rollout_id}: Running evaluation")
            ray.get(rollout_manager.eval.remote(rollout_id))
        
        # Log bi-level optimizer state
        optimizer_info = bi_level_optimizer.get_info()
        logger.info(
            f"Rollout {rollout_id}: "
            f"Iteration {optimizer_info['iteration']}, "
            f"Update ratio {optimizer_info['update_ratio']}"
        )
    
    # Cleanup
    logger.info("Training completed. Cleaning up resources")
    ray.get(rollout_manager.dispose.remote())


def add_irl_arguments(parser):
    """Add IRL-specific arguments to parser."""
    parser.add_argument(
        "--expert-data-path",
        type=str,
        required=True,
        help="Path to expert demonstrations (.pt or .pkl file)",
    )
    
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
        "--expert-batch-size",
        type=int,
        default=32,
        help="Batch size for expert demonstrations",
    )
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = parse_args(return_parser=True)
    parser = add_irl_arguments(parser)
    args = parser.parse_args()
    
    # Run bi-level IRL training
    train_irl(args)
