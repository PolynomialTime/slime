"""Inverse Reinforcement Learning (IRL) module for SLIME.

This module provides bi-level optimization framework for training a policy
and reward model jointly using maximum entropy IRL and PPO.
"""

from .reward_model import RewardModel, create_reward_model
from .irl_trainer import IRLTrainer
from .bi_level_optimizer import BiLevelOptimizer

__all__ = [
    "RewardModel",
    "create_reward_model",
    "IRLTrainer",
    "BiLevelOptimizer",
]
