"""Trainable Reward Model for IRL.

For LLM-based RL, the reward model predicts a single scalar reward per sequence.
This is trained to match expert demonstrations using IRL objectives like
maximum entropy IRL.

The reward model takes:
- Last hidden state of the policy model output [hidden_size]
  OR aggregated hidden states from the sequence [hidden_size]

And outputs:
- Scalar sequence-level reward (single float per trajectory)
"""

import logging
from argparse import Namespace
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Trainable reward model for LLM sequences in IRL.
    
    Architecture:
        input_features (hidden_size) → hidden_layers → scalar_reward
    
    For LLMs:
    - Takes the final hidden state [hidden_size] from the last token
    - OR aggregates all token hidden states (mean/max pooling)
    - Outputs single scalar reward for the entire sequence
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        """Initialize reward model.
        
        Args:
            input_size: Dimension of input features (e.g., hidden_size from policy)
            hidden_size: Dimension of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout rate
            output_size: Dimension of reward output (typically 1 for scalar reward)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Build MLP
        layers = []
        prev_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict sequence-level reward.
        
        Args:
            features: Input features from last token hidden state
                     [batch_size, input_size] or just [input_size] for single sequence
        
        Returns:
            Predicted scalar rewards [batch_size] or scalar
        """
        # Ensure proper shape
        if features.dim() == 1:
            features = features.unsqueeze(0)  # [input_size] → [1, input_size]
        
        reward = self.mlp(features)  # [batch, 1]
        return reward.squeeze(-1)  # [batch]
    
    def get_reward_for_sequence(
        self,
        hidden_states: torch.Tensor,
        aggregation: str = "last",
    ) -> torch.Tensor:
        """Compute sequence-level reward from hidden states.
        
        For LLM sequences, we aggregate hidden states to a single feature vector,
        then pass through the reward model to get a scalar reward.
        
        Args:
            hidden_states: [seq_len, hidden_size] or [batch, seq_len, hidden_size]
            aggregation: How to aggregate sequence to single feature vector:
                - "last": Use last token hidden state (default for LLM causal models)
                - "mean": Mean pooling across all tokens
                - "max": Max pooling across all tokens
        
        Returns:
            Scalar reward or batch of rewards [batch]
        """
        # Handle 2D input (single sequence)
        if hidden_states.dim() == 2:
            # [seq_len, hidden_size]
            if aggregation == "last":
                agg_hidden = hidden_states[-1]  # Last token [hidden_size]
            elif aggregation == "mean":
                agg_hidden = hidden_states.mean(dim=0)  # [hidden_size]
            elif aggregation == "max":
                agg_hidden = hidden_states.max(dim=0)[0]  # [hidden_size]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            # Predict reward
            reward = self.forward(agg_hidden)  # scalar
            return reward
        
        # Handle 3D input (batch of sequences)
        elif hidden_states.dim() == 3:
            # [batch, seq_len, hidden_size]
            if aggregation == "last":
                agg_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
            elif aggregation == "mean":
                agg_hidden = hidden_states.mean(dim=1)  # [batch, hidden_size]
            elif aggregation == "max":
                agg_hidden = hidden_states.max(dim=1)[0]  # [batch, hidden_size]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            # Predict rewards
            rewards = self.forward(agg_hidden)  # [batch]
            return rewards
        
        else:
            raise ValueError(f"Expected 2D or 3D hidden states, got {hidden_states.dim()}D")


class TrajectoryRewardModel(nn.Module):
    """Trajectory-level reward model that processes full trajectories.
    
    This variant processes the entire trajectory (or sequence of hidden states)
    and outputs a single trajectory-level reward, which may be more suitable
    for some IRL objectives.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize trajectory reward model.
        
        Args:
            input_size: Dimension of input features per timestep
            hidden_size: Hidden layer dimension
            num_layers: Number of transformer layers for trajectory encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Transformer encoder for trajectory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=8 if input_size >= 512 else 4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # MLP head to output scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to predict trajectory-level reward.
        
        Args:
            hidden_states: [batch_size, seq_len, input_size] trajectory encodings
            mask: [batch_size, seq_len] attention mask (optional)
        
        Returns:
            Trajectory rewards [batch_size, 1]
        """
        # Encode trajectory
        encoded = self.trajectory_encoder(hidden_states, src_key_padding_mask=mask)
        # [batch_size, seq_len, input_size]
        
        # Pool: take mean of all timesteps
        if mask is not None:
            # Average only over non-masked positions
            # mask is [batch_size, seq_len], True for padding positions
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            # Set masked positions to 0, compute sum, then divide by number of valid positions
            encoded_masked = encoded.masked_fill(mask_expanded, 0.0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)  # [batch_size, 1]
            pooled = encoded_masked.sum(dim=1) / valid_counts  # [batch_size, input_size]
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, input_size]
        
        # Get scalar reward
        reward = self.reward_head(pooled)  # [batch_size, 1]
        return reward


def create_reward_model(
    args: Namespace,
    model_type: str = "mlp",
) -> nn.Module:
    """Factory function to create reward model.

    Args:
        args: Arguments containing reward model configuration
            - reward_model_hidden_size: hidden dimension
            - reward_model_num_layers: number of layers
            - reward_model_dropout: dropout rate
        model_type: "mlp" or "trajectory"

    Returns:
        Reward model instance
    """
    hidden_size = getattr(args, "reward_model_hidden_size", 512)
    num_layers = getattr(args, "reward_model_num_layers", 2)
    dropout = getattr(args, "reward_model_dropout", 0.1)
    # Safely get input_size with multiple fallbacks
    input_size = getattr(args, "hf_config_hidden_size", getattr(args, "hidden_size", 4096))

    if model_type == "mlp":
        return RewardModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "trajectory":
        return TrajectoryRewardModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown reward model type: {model_type}")
