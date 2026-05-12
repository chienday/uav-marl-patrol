"""
Neural network architectures — SharedActor, CentralCritic, AgentValueHead.

All networks use orthogonal initialization as per the MAPPO/VDPPO notebooks.
"""

import numpy as np
import torch
import torch.nn as nn


class SharedActor(nn.Module):
    """
    Shared actor network for MAPPO/VDPPO.

    Architecture: 2-layer MLP with Tanh activations.
    Orthogonal init; small output gain -> logits near 0 -> near-uniform softmax at init.
    Outputs raw logits (NOT probabilities) — use Categorical(logits=...) for sampling.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small output gain -> logits near 0 -> near-uniform softmax at init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    """
    Centralised critic for MAPPO/VDPPO.

    Takes the full global state as input (G*G*3 + 4 features for 10x10 grid).
    Extra hidden layer; orthogonal init.
    """

    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),        nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),   nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class AgentValueHead(nn.Module):
    """
    Per-agent local value head for VDPPO.

    Input : local observation of one UAV (obs_dim)
    Output: scalar value estimate V_agent_i

    V_total = V_team (CentralCritic) + mean(V_agent_i)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)   # (B,)
