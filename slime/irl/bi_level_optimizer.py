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
    
    def get_info(self) -> Dict[str, any]:
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
        sglang_runtime,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """Generate LLM rollouts using SGLang runtime.
        
        SGLang provides high-performance LLM serving with optimized kernels.
        This method generates sequences and returns hidden states for the reward model.
        
        Args:
            sglang_runtime: SGLang Runtime instance (e.g., from sglang.Runtime)
            prompts: List of prompt strings for batch generation
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Rollout dictionary with:
            - "prompts": Original prompts
            - "completions": Generated text completions
            - "hidden_states": [batch, seq_len, hidden_size] hidden states
            - "logits": [batch, seq_len, vocab_size] token logits
            - "log_probs": [batch, seq_len] log probabilities
            - "token_ids": [batch, seq_len] token IDs (including prompt)
        """
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "SGLang not installed. Install with: pip install sglang"
            )
        
        batch_size = len(prompts)
        logger.info(f"Generating {batch_size} LLM rollouts with SGLang")
        
        # Generate completions with SGLang
        # SGLang backend will capture hidden states if available
        completions_list = []
        all_hidden_states = []
        all_logits = []
        all_token_ids = []
        all_log_probs = []
        
        for prompt in prompts:
            # Use SGLang to generate with captured activations
            state = sglang_runtime.forward(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                return_hidden_states=True,  # Request hidden states
                return_logits=True,         # Request logits
            )
            
            completions_list.append(state["text"])
            
            # Extract hidden states from last token
            # Shape: [hidden_size] or [seq_len, hidden_size]
            if "hidden_states" in state:
                hidden_state = state["hidden_states"]
                # Ensure [seq_len, hidden_size] format
                if hidden_state.dim() == 1:
                    hidden_state = hidden_state.unsqueeze(0)
                all_hidden_states.append(hidden_state)
            
            if "logits" in state:
                logits = state["logits"]  # [seq_len, vocab_size]
                all_logits.append(logits)
            
            if "token_ids" in state:
                token_ids = state["token_ids"]  # [seq_len]
                all_token_ids.append(token_ids)
            
            if "log_probs" in state:
                log_probs = state["log_probs"]  # [seq_len]
                all_log_probs.append(log_probs)
        
        # Stack into batches
        rollout = {
            "prompts": prompts,
            "completions": completions_list,
        }
        
        # Handle potentially variable-length sequences
        try:
            # Pad sequences to same length
            if all_hidden_states:
                max_len = max(h.shape[0] for h in all_hidden_states)
                hidden_batch = []
                for h in all_hidden_states:
                    if h.shape[0] < max_len:
                        pad_len = max_len - h.shape[0]
                        h = torch.cat([h, torch.zeros(pad_len, h.shape[1])], dim=0)
                    hidden_batch.append(h)
                rollout["hidden_states"] = torch.stack(hidden_batch)  # [batch, seq_len, hidden_size]
            
            if all_logits:
                max_len = max(l.shape[0] for l in all_logits)
                logit_batch = []
                for l in all_logits:
                    if l.shape[0] < max_len:
                        pad_len = max_len - l.shape[0]
                        l = torch.cat([l, torch.zeros(pad_len, l.shape[1])], dim=0)
                    logit_batch.append(l)
                rollout["logits"] = torch.stack(logit_batch)  # [batch, seq_len, vocab_size]
            
            if all_log_probs:
                max_len = max(lp.shape[0] if lp.dim() == 1 else lp.shape[0] for lp in all_log_probs)
                logprob_batch = []
                for lp in all_log_probs:
                    if isinstance(lp, torch.Tensor):
                        if lp.shape[0] < max_len:
                            pad_len = max_len - lp.shape[0]
                            lp = torch.cat([lp, torch.zeros(pad_len)], dim=0)
                    logprob_batch.append(lp)
                rollout["log_probs"] = torch.stack(logprob_batch)  # [batch, seq_len]
            
            if all_token_ids:
                max_len = max(t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in all_token_ids)
                token_batch = []
                for t in all_token_ids:
                    if isinstance(t, list):
                        t = torch.tensor(t)
                    if t.shape[0] < max_len:
                        pad_len = max_len - t.shape[0]
                        t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype)], dim=0)
                    token_batch.append(t)
                rollout["token_ids"] = torch.stack(token_batch)  # [batch, seq_len]
        
        except Exception as e:
            logger.warning(f"Error stacking sequences: {e}. Returning raw lists.")
            rollout["hidden_states"] = all_hidden_states
            rollout["logits"] = all_logits
            rollout["log_probs"] = all_log_probs
            rollout["token_ids"] = all_token_ids
        
        logger.info(f"Generated {batch_size} LLM sequences with SGLang")
        return rollout


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
    ) -> Dict[str, any]:
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
