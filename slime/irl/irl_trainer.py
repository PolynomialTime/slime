"""IRL Training module for maximum entropy IRL with policy updates."""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    """Dataset for expert demonstrations.
    
    Stores trajectories from expert or high-quality policy rollouts.
    Supports both raw trajectory dicts and Sample objects from SGLang generation.
    """
    
    def __init__(
        self,
        trajectories: List[Dict[str, torch.Tensor] | Sample],
    ):
        """Initialize expert dataset.
        
        Args:
            trajectories: List of trajectory dicts or Sample objects, each containing:
                - hidden_states: [seq_len, hidden_size] from policy model
                - actions: [seq_len] action indices
                - rewards: scalar reward for the entire sequence (NOT per-token)
                - returns: scalar return for the entire sequence
                - length: int, actual sequence length (excluding padding)
                
                Or Sample objects with:
                - response: Generated text
                - tokens: Token sequence
                - reward: Scalar reward for the sample
                - rollout_log_probs: Log probabilities from generation
        """
        self.trajectories = trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Handle Sample objects
        if isinstance(traj, Sample):
            return {
                "response": traj.response,
                "tokens": torch.tensor(traj.tokens) if traj.tokens else torch.tensor([]),
                "reward": torch.tensor(traj.reward if traj.reward is not None else 0.0),
                "log_probs": torch.tensor(traj.rollout_log_probs) if traj.rollout_log_probs else torch.tensor([]),
            }
        
        # Handle dict trajectories
        return traj


class MaxEntropyIRLObjective:
    """Maximum Entropy IRL objective for LLM sequences.
    
    Trains reward model to assign high rewards to expert sequences while
    assigning low rewards to policy-generated sequences, with entropy regularization.
    
    Objective:
    L = -E_{τ~expert}[R(τ)] + E_{τ~policy}[R(τ)] - β*H(π)
    
    where:
    - R(τ) is scalar sequence reward from the reward model
    - H(π) is policy entropy (encourages exploration)
    - τ is a complete sequence/trajectory
    """
    
    def __init__(
        self,
        reward_model: nn.Module,
        policy_model: nn.Module,
        expert_dataloader: DataLoader,
        device: str = "cuda",
        reward_weight: float = 1.0,
        policy_weight: float = 0.1,
        entropy_weight: float = 0.0,
    ):
        """Initialize maximum entropy IRL objective for LLM.
        
        Args:
            reward_model: Trainable reward model
            policy_model: LLM policy model (used for entropy regularization)
            expert_dataloader: DataLoader for expert LLM trajectories
            device: Device to use
            reward_weight: Weight for expert reward term
            policy_weight: Weight for policy reward term
            entropy_weight: Weight for LLM policy entropy term (typically 0 for LLMs)
        """
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.expert_dataloader = expert_dataloader
        self.device = device
        
        self.reward_weight = reward_weight
        self.policy_weight = policy_weight
        self.entropy_weight = entropy_weight
    
    def compute_loss(
        self,
        expert_trajectories: Dict[str, torch.Tensor] | List[Sample],
        policy_trajectories: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute IRL loss for LLM policy learning.
        
        Args:
            expert_trajectories: Expert LLM sequence trajectories with structure:
                - hidden_states: [batch, seq_len, hidden_size] from expert LLM
                - rewards: [batch] scalar sequence rewards
                OR list of Sample objects
            policy_trajectories: Current LLM policy rollout trajectories with structure:
                - hidden_states: [batch, seq_len, hidden_size] from policy LLM
                - rewards: [batch] scalar sequence rewards
                - log_probs: [batch, seq_len] log probabilities (optional, for entropy)
                OR list of Sample objects
        
        Returns:
            (loss tensor, metrics dictionary)
        """
        metrics = {}
        
        # Convert Sample objects to tensor dicts if needed
        if isinstance(expert_trajectories, list) and len(expert_trajectories) > 0:
            if isinstance(expert_trajectories[0], Sample):
                expert_trajectories = self._samples_to_tensor_dict(expert_trajectories)
        
        if isinstance(policy_trajectories, list) and len(policy_trajectories) > 0:
            if isinstance(policy_trajectories[0], Sample):
                policy_trajectories = self._samples_to_tensor_dict(policy_trajectories)
        
        # Extract final hidden states (last token) for sequence-level reward
        expert_hidden = expert_trajectories.get("hidden_states")
        policy_hidden = policy_trajectories.get("hidden_states")
        
        # If no hidden states, try to work with what we have
        if expert_hidden is None or policy_hidden is None:
            logger.warning(
                "No hidden_states in trajectories. "
                "IRL loss computation requires hidden_states or embedded representations."
            )
            # Return dummy loss if we can't compute properly
            return torch.tensor(0.0, device=self.device), {"error": 1.0}
        
        # Move to device
        expert_hidden = expert_hidden.to(self.device)
        policy_hidden = policy_hidden.to(self.device)
        
        # Get last token hidden state for each sequence [batch, hidden_size]
        if expert_hidden.dim() == 3:
            expert_final_hidden = expert_hidden[:, -1, :]
        else:
            expert_final_hidden = expert_hidden
        
        if policy_hidden.dim() == 3:
            policy_final_hidden = policy_hidden[:, -1, :]
        else:
            policy_final_hidden = policy_hidden
        
        # Compute sequence-level rewards from reward model
        expert_rewards = self.reward_model(expert_final_hidden)  # [batch]
        policy_rewards = self.reward_model(policy_final_hidden)
        
        # Ensure scalar outputs
        if expert_rewards.dim() > 1:
            expert_rewards = expert_rewards.squeeze(-1)
        if policy_rewards.dim() > 1:
            policy_rewards = policy_rewards.squeeze(-1)
        
        # Expert term: maximize reward on expert LLM sequences
        expert_loss = -expert_rewards.mean()  # Negative because we minimize
        
        # Policy term: minimize reward under current LLM policy
        policy_loss = policy_rewards.mean()
        
        # Entropy term: encourage LLM exploration (usually 0 for LLMs)
        entropy_loss = 0.0
        if self.entropy_weight > 0:
            # Compute policy entropy from LLM log probabilities
            policy_log_probs = policy_trajectories.get("log_probs")
            if policy_log_probs is not None:
                policy_log_probs = policy_log_probs.to(self.device)
                # For LLMs, log_probs are already log probabilities, not logits
                # Entropy = -E[log p(a|s)] which is the negative average of log probs
                entropy = -policy_log_probs.mean()
                entropy_loss = -self.entropy_weight * entropy  # Maximize entropy
        
        # Combine losses
        total_loss = (
            self.reward_weight * expert_loss +
            self.policy_weight * policy_loss +
            entropy_loss
        )
        
        metrics["expert_loss"] = expert_loss.item()
        metrics["policy_loss"] = policy_loss.item()
        metrics["entropy_loss"] = entropy_loss if isinstance(entropy_loss, (int, float)) else entropy_loss.item()
        metrics["total_loss"] = total_loss.item()
        metrics["expert_reward_mean"] = expert_rewards.mean().item()
        metrics["policy_reward_mean"] = policy_rewards.mean().item()
        
        return total_loss, metrics
    
    def _samples_to_tensor_dict(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Convert Sample objects to tensor dictionary for loss computation.
        
        Args:
            samples: List of Sample objects
        
        Returns:
            Dictionary with tensorized sample data
        """
        rewards = []
        log_probs_list = []
        
        for sample in samples:
            rewards.append(sample.reward if sample.reward is not None else 0.0)
            if sample.rollout_log_probs:
                log_probs_list.append(torch.tensor(sample.rollout_log_probs))
        
        result = {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
        }
        
        if log_probs_list:
            # Pad to same length
            max_len = max(lp.shape[0] for lp in log_probs_list)
            padded = []
            for lp in log_probs_list:
                if lp.shape[0] < max_len:
                    lp = torch.cat([lp, torch.zeros(max_len - lp.shape[0])])
                padded.append(lp)
            result["log_probs"] = torch.stack(padded)
        
        return result


class IRLTrainer:
    """Trainer for reward model in bi-level LLM RL optimization.
    
    Handles reward model updates given LLM policy rollouts and expert demonstrations.
    The reward model learns to assign high scores to expert LLM sequences and low scores
    to policy-generated sequences.
    """
    
    def __init__(
        self,
        reward_model: nn.Module,
        policy_model: nn.Module,
        expert_dataloader: DataLoader,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        irl_objective: str = "max_entropy",
    ):
        """Initialize IRL trainer for LLM policy.
        
        Args:
            reward_model: Trainable reward model for LLM sequences
            policy_model: LLM policy model
            expert_dataloader: DataLoader for expert LLM demonstrations
            learning_rate: Learning rate for reward model optimizer
            device: Device to use (cuda recommended for LLMs)
            irl_objective: IRL objective type (currently only "max_entropy")
        """
        self.reward_model = reward_model.to(device)
        self.policy_model = policy_model
        self.expert_dataloader = expert_dataloader
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=learning_rate,
        )
        
        if irl_objective == "max_entropy":
            self.objective = MaxEntropyIRLObjective(
                reward_model=self.reward_model,
                policy_model=self.policy_model,
                expert_dataloader=self.expert_dataloader,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown IRL objective: {irl_objective}")
    
    def train_step(
        self,
        policy_rollouts: Dict[str, torch.Tensor] | List[Sample],
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Perform reward model training step on LLM sequences.
        
        Args:
            policy_rollouts: LLM policy rollout trajectories or Sample objects with:
                - hidden_states: [batch, seq_len, hidden_size]
                - log_probs: [batch, seq_len] (optional)
                OR list of Sample objects with rewards already computed
            num_epochs: Number of training epochs over expert data
        
        Returns:
            Metrics dictionary with loss values
        """
        accumulated_metrics = {}
        num_batches = 0
        
        for epoch in range(num_epochs):
            for expert_batch in self.expert_dataloader:
                # Prepare expert batch
                expert_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in expert_batch.items()
                }
                
                # Prepare policy batch (convert if needed)
                if isinstance(policy_rollouts, list):
                    policy_batch = policy_rollouts
                else:
                    policy_batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in policy_rollouts.items()
                    }
                
                # Compute IRL loss
                loss, metrics = self.objective.compute_loss(expert_batch, policy_batch)
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()
                
                # Accumulate metrics
                for key, val in metrics.items():
                    if key not in accumulated_metrics:
                        accumulated_metrics[key] = 0.0
                    accumulated_metrics[key] += val
                
                num_batches += 1
        
        # Average metrics over all batches
        for key in accumulated_metrics:
            accumulated_metrics[key] /= max(num_batches, 1)
        
        return accumulated_metrics
    
    def get_reward_fn(self):
        """Get reward function for use during LLM rollout generation.
        
        Returns:
            Function that takes final hidden states and returns scalar sequence rewards
        """
        def reward_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            """Compute sequence-level reward for LLM outputs.
            
            Args:
                hidden_states: [batch, hidden_size] final token hidden states from LLM
            
            Returns:
                [batch] scalar rewards for each sequence
            """
            with torch.no_grad():
                rewards = self.reward_model(hidden_states)
            
            # Ensure scalar outputs
            if rewards.dim() > 1:
                rewards = rewards.squeeze(-1)
            
            return rewards
        
        return reward_fn
