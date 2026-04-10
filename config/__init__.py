# -*- coding: utf-8 -*-
"""
Mô-đun cấu hình cho dự án UAV - Multi-Agent RL

Config chứa các siêu tham số cho huấn luyện các thuật toán:
- PPO (1 UAV)
- IPPO (2 UAV)
- MAPPO (Multi-Agent PPO)
- VDPPO (Value Decomposition PPO)
"""

from .hyperparameters import (
    PPO_CONFIG,
    TRAINING_CONFIG,
    EVALUATION_CONFIG,
    ENVIRONMENT_CONFIG,
)

__all__ = [
    "PPO_CONFIG",
    "TRAINING_CONFIG",
    "EVALUATION_CONFIG",
    "ENVIRONMENT_CONFIG",
]
