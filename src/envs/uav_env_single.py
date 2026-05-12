"""
UAVPatrolEnv — Single-agent environment for PPO baseline.

This environment wraps the UAV patrol problem as a standard
Gymnasium environment compatible with Stable-Baselines3 PPO.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVPatrolEnv(gym.Env):
    """Single-agent UAV Patrol Environment for PPO baseline."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, map_config: dict = None, render_mode: str = None):
        super().__init__()
        self.render_mode = render_mode
        self.map_config = map_config or {}

        # TODO: Define observation_space and action_space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)  # up, down, left, right, hover

        self._state = None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # TODO: Initialize environment state
        self._state = np.zeros(10, dtype=np.float32)
        info = {}
        return self._state, info

    def step(self, action):
        """Execute one step in the environment."""
        # TODO: Implement step logic
        observation = self._state
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        # TODO: Implement rendering
        pass

    def close(self):
        """Clean up resources."""
        pass
