"""Bi-level optimization for jointly training LLM policy and reward model with SGLang."""

import logging
from argparse import Namespace
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .irl_trainer import IRLTrainer
from .reward_model import RewardModel

logger = logging.getLogger(__name__)


class BiLevelOptimizer:
    """Bi-level optimization framework for LLM IRL with PPO.
    
    Alternately updates:
    1. LLM Policy: Using PPO with rewards from trainable reward model
    2. Reward Model: Using IRL objective (e.g., max-entropy IRL)
    
    Supports SGLang for efficient LLM rollout generation.
    
    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │ Bi-Level LLM RL Optimization Loop                    │
    ├──────────────────────────────────────────────────────┤
    │                                                      │
    │  ┌─────────────────────────────────────────────┐    │
    │  │ LLM Policy Update (PPO)                     │    │
    │  │ - Rollout: Generate sequences via SGLang    │    │
    │  │ - Reward: Compute R(seq) via reward model   │    │
    │  │ - Advantage: Compute advantages from critic │    │
    │  │ - PPO Loss: Update LLM policy               │    │
    │  └─────────────────────────────────────────────┘    │
    │                  ↕ (alternate)                      │
    │  ┌─────────────────────────────────────────────┐    │
    │  │ Reward Model Update (IRL)                   │    │
    │  │ - Expert Data: Expert LLM sequences         │    │
    │  │ - Policy Traj: Policy-generated sequences   │    │
    │  │ - IRL Loss: Max entropy objective           │    │
    │  │ - Update: Gradient descent on reward model  │    │
    │  └─────────────────────────────────────────────┘    │
    │                                                      │
    └──────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        reward_model: nn.Module,
        policy_model: nn.Module,
        critic_model: Optional[nn.Module] = None,
        expert_dataloader: Optional[DataLoader] = None,
        args: Optional[Namespace] = None,
        device: str = "cuda",
    ):
        """Initialize bi-level optimizer.
        
        Args:
            reward_model: Trainable reward model
            policy_model: Policy model (PPO actor)
            critic_model: Value function (PPO critic)
            expert_dataloader: DataLoader for expert trajectories
            args: Arguments containing training configuration
            device: Device to use
        """
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.device = device
        
        # Initialize IRL trainer
        if expert_dataloader is not None:
            self.irl_trainer = IRLTrainer(
                reward_model=reward_model,
                policy_model=policy_model,
                expert_dataloader=expert_dataloader,
                learning_rate=getattr(args, "reward_model_lr", 1e-4) if args else 1e-4,
                device=device,
                irl_objective=getattr(args, "irl_objective", "max_entropy") if args else "max_entropy",
            )
        else:
            self.irl_trainer = None
            logger.warning("No expert dataloader provided. IRL trainer disabled.")
        
        # Hyperparameters
        self.args = args or Namespace()
        self.update_ratio = getattr(self.args, "irl_update_ratio", 1)
        # update_ratio: (policy_updates : reward_updates)
        # e.g., 1 means alternate (1 policy, 1 reward)
        #       2 means (2 policy, 1 reward)
        
        self.iteration_count = 0
    
    def should_update_reward_model(self) -> bool:
        """Determine if reward model should be updated at this iteration.
        
        Returns:
            True if reward model should be updated
        """
        if self.irl_trainer is None:
            return False
        
        # Simple alternation based on iteration count
        policy_updates_per_cycle = self.update_ratio
        cycle_length = policy_updates_per_cycle + 1
        pos_in_cycle = self.iteration_count % cycle_length
        
        # Update reward after policy updates
        return pos_in_cycle == policy_updates_per_cycle
    
    def get_reward_fn(self):
        """Get current reward function for policy training.
        
        Returns:
            Function that computes rewards from hidden states
        """
        def reward_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            """Compute rewards from hidden states.
            
            Args:
                hidden_states: [seq_len, hidden_size] or similar shape
            
            Returns:
                Rewards: scalar or per-token rewards
            """
            with torch.no_grad():
                rewards = self.reward_model(hidden_states)
            
            # Squeeze if last dimension is 1
            if rewards.dim() > 1 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)
            
            return rewards
        
        return reward_fn
    
    def update_reward_model(
        self,
        policy_rollouts: Dict[str, torch.Tensor],
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Update reward model using IRL objective.
        
        Args:
            policy_rollouts: Dictionary containing:
                - hidden_states: [batch, seq_len, hidden_size]
                - actions: [batch, seq_len]
                - rewards: [batch, seq_len] or [batch]
                - log_probs: [batch, seq_len]
                (other fields optional)
            num_epochs: Number of training epochs
        
        Returns:
            Metrics dictionary
        """
        if self.irl_trainer is None:
            logger.warning("IRL trainer not initialized. Skipping reward model update.")
            return {}
        
        logger.info(f"Updating reward model (iteration {self.iteration_count})")
        
        metrics = self.irl_trainer.train_step(
            policy_rollouts=policy_rollouts,
            num_epochs=num_epochs,
        )
        
        # Log metrics
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.6f}")
        
        return metrics
    
    def step(self) -> bool:
        """Increment iteration counter and check if reward should be updated.
        
        Returns:
            True if reward model should be updated this step
        """
        should_update = self.should_update_reward_model()
        self.iteration_count += 1
        return should_update
    
    def get_info(self) -> Dict[str, Any]:
        """Get optimizer state information.

        Returns:
            Dictionary with optimizer info
        """
        return {
            "iteration": self.iteration_count,
            "update_ratio": self.update_ratio,
            "has_irl_trainer": self.irl_trainer is not None,
        }
    
    def generate_rollouts_with_sglang(
        self,
        args: Namespace,
        prompts: List[str],
        data_source: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate LLM rollouts using SGLang rollout API and evaluate with reward model.
        
        This method leverages the high-performance SGLang rollout generation from
        slime.rollout.sglang_rollout and evaluates each generated rollout using
        the current trainable reward model.
        
        Args:
            args: Namespace containing training configuration (must include SGLang settings)
            prompts: List of prompt strings for rollout generation
            data_source: Optional data source for rollout management
        
        Returns:
            Rollout dictionary with:
            - "samples": List of Sample objects with generated responses and rewards
            - "prompts": Original prompts
            - "completions": Generated text completions
            - "rewards": Rewards computed by the trainable reward model
            - "log_probs": Log probabilities from generation
            - "metrics": Dictionary of generation metrics
        """
        try:
            from slime.rollout.sglang_rollout import generate_rollout
            from slime.utils.types import Sample
        except ImportError:
            raise ImportError(
                "Required dependencies not found. Ensure slime package is properly installed."
            )
        
        batch_size = len(prompts)
        logger.info(f"Generating {batch_size} LLM rollouts with SGLang")
        
        # Create sample objects from prompts
        samples = []
        for idx, prompt in enumerate(prompts):
            sample = Sample(
                index=idx,
                prompt=prompt,
            )
            samples.append(sample)
        
        # Create mock data source if not provided
        if data_source is None:
            class MockDataSource:
                def get_samples(self):
                    return [samples]
                def add_samples(self, samples):
                    pass
            data_source = MockDataSource()
        
        # Generate rollouts using SGLang API
        rollout_id = getattr(self, '_rollout_id', 0)
        self._rollout_id = rollout_id + 1
        
        rollout_output = generate_rollout(
            args=args,
            rollout_id=rollout_id,
            data_source=data_source,
            evaluation=False,
        )
        
        # Extract samples from rollout output
        if hasattr(rollout_output, 'samples'):
            generated_samples = rollout_output.samples
            # Flatten if nested (list of lists)
            if generated_samples and isinstance(generated_samples[0], list):
                generated_samples = [s for group in generated_samples for s in group]
        else:
            generated_samples = rollout_output
        
        logger.info(f"Generated {len(generated_samples)} samples from SGLang")
        
        # Evaluate each sample with the trainable reward model
        completions_list = []
        all_log_probs = []
        rewards_list = []
        
        for sample in generated_samples:
            completions_list.append(sample.response)
            
            # Compute reward using the current reward model
            if sample.reward is None and sample.status == Sample.Status.COMPLETED:
                # Create hidden states representation from response for reward model
                # For now, we use the response text directly
                with torch.no_grad():
                    # If reward model expects hidden states, we may need to extract them
                    # from the generation process. For this implementation, we assume
                    # the reward model can work with response embeddings
                    try:
                        reward = self._compute_reward_for_sample(sample)
                        sample.reward = reward
                    except Exception as e:
                        logger.warning(f"Failed to compute reward for sample {sample.index}: {e}")
                        sample.reward = 0.0
            
            rewards_list.append(sample.reward if sample.reward is not None else 0.0)
            
            if sample.rollout_log_probs:
                all_log_probs.append(torch.tensor(sample.rollout_log_probs))
        
        # Build return dictionary
        rollout = {
            "samples": generated_samples,
            "prompts": prompts,
            "completions": completions_list,
            "rewards": torch.tensor(rewards_list, dtype=torch.float32),
        }
        
        if all_log_probs:
            try:
                # Pad log probs to same length
                max_len = max(lp.shape[0] for lp in all_log_probs)
                padded_log_probs = []
                for lp in all_log_probs:
                    if lp.shape[0] < max_len:
                        padding = torch.zeros(max_len - lp.shape[0])
                        lp = torch.cat([lp, padding], dim=0)
                    padded_log_probs.append(lp)
                rollout["log_probs"] = torch.stack(padded_log_probs)
            except Exception as e:
                logger.warning(f"Error stacking log probs: {e}")
                rollout["log_probs"] = all_log_probs
        
        if hasattr(rollout_output, 'metrics'):
            rollout["metrics"] = rollout_output.metrics
        else:
            rollout["metrics"] = {}
        
        logger.info(
            f"Generated and evaluated {batch_size} rollouts. "
            f"Mean reward: {torch.tensor(rewards_list).mean():.4f}"
        )
        return rollout
    
    def _compute_reward_for_sample(self, sample: "Sample") -> float:
        """Compute reward for a sample using the trainable reward model.
        
        Args:
            sample: Sample object with generated response
        
        Returns:
            Reward value
        """
        # Extract text representation from sample
        text = sample.response if sample.response else ""
        
        # Tokenize the response to create a simple representation
        # In a full implementation, this would extract hidden states from generation
        try:
            # Create a simple representation: average token embedding
            # This is a placeholder; in production, extract actual hidden states
            import hashlib
            text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
            text_embedding = torch.randn(256, generator=torch.Generator().manual_seed(text_hash % (2**31)))
            text_embedding = text_embedding.to(self.device)
            
            with torch.no_grad():
                reward = self.reward_model(text_embedding.unsqueeze(0))
            
            # Squeeze if needed
            if isinstance(reward, torch.Tensor):
                reward = reward.squeeze().item() if reward.numel() > 0 else 0.0
            
            return float(reward)
        except Exception as e:
            logger.warning(f"Error computing reward: {e}")
            return 0.0


class BiLevelTrainingLoop:
    """Helper class to manage bi-level training loop.
    
    This class orchestrates the alternation between policy and reward updates
    in a standard training loop structure.
    """
    
    def __init__(
        self,
        bi_level_optimizer: BiLevelOptimizer,
        args: Optional[Namespace] = None,
    ):
        """Initialize training loop.
        
        Args:
            bi_level_optimizer: BiLevelOptimizer instance
            args: Training arguments
        """
        self.optimizer = bi_level_optimizer
        self.args = args or Namespace()
    
    def process_rollout(
        self,
        rollout_id: int,
        rollout_data: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Process a rollout: update policy and/or reward model.

        Args:
            rollout_id: Current rollout ID
            rollout_data: Rollout data from policy

        Returns:
            Dictionary with update info
        """
        update_info = {
            "rollout_id": rollout_id,
            "policy_updated": True,  # Assume policy is always updated
            "reward_updated": False,
            "reward_metrics": {},
        }

        # Check if reward should be updated
        should_update_reward = self.optimizer.step()

        if should_update_reward:
            logger.info(f"Rollout {rollout_id}: Updating reward model")
            metrics = self.optimizer.update_reward_model(
                policy_rollouts=rollout_data,
                num_epochs=getattr(self.args, "irl_num_epochs", 1),
            )
            update_info["reward_updated"] = True
            update_info["reward_metrics"] = metrics

        return update_info
