"""Inverse Reinforcement Learning (IRL) module for SLIME.

This module provides bi-level optimization framework for training a policy
and reward model jointly using maximum entropy IRL and PPO.
"""

from .reward_model import RewardModel, TrajectoryRewardModel, create_reward_model
from .irl_trainer import IRLTrainer, ExpertDataset
from .bi_level_optimizer import BiLevelOptimizer, BiLevelTrainingLoop

__all__ = [
    "RewardModel",
    "TrajectoryRewardModel",
    "create_reward_model",
    "IRLTrainer",
    "ExpertDataset",
    "BiLevelOptimizer",
    "BiLevelTrainingLoop",
]
