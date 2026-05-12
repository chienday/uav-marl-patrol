"""
IPPOTrainer — Independent PPO using SB3 PPO per agent.

Each UAV is wrapped as a single-agent Gymnasium env via SingleAgentWrapper.
Then trained independently with Stable-Baselines3 PPO.

This matches the ippo-final.ipynb notebook implementation.
"""

import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from ..envs.uav_env_multi import UAVPatrolEnvIPPO
from ..envs.reward import RewardConfig, IPPORewardConfig


# ---------------------------------------------------------------------------
# SingleAgentWrapper — wraps multi-agent env as single-agent for SB3 PPO
# ---------------------------------------------------------------------------

class SingleAgentWrapper(gym.Env):
    """
    Wraps UAVPatrolEnvIPPO for one specific agent, making it
    compatible with Stable-Baselines3 single-agent PPO.

    The other agent follows a fixed policy:
      - If `partner_model` is provided, use its deterministic action
      - Otherwise, take a random action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: UAVPatrolEnvIPPO,
        agent_id: int = 0,
        partner_model=None,
    ):
        super().__init__()
        self.env            = env
        self.agent_id       = agent_id
        self.partner_id     = 1 - agent_id
        self.partner_model  = partner_model

        self.observation_space = env.single_observation_space
        self.action_space      = env.single_action_space
        self._last_obs         = None

    def _partner_action(self) -> int:
        if self.partner_model is not None and self._last_obs is not None:
            obs = self._last_obs[self.partner_id]
            action, _ = self.partner_model.predict(obs, deterministic=True)
            return int(action)
        return int(self.env.single_action_space.sample())

    def reset(self, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed)
        self._last_obs = obs_dict
        return obs_dict[self.agent_id], info

    def step(self, action):
        partner_act = self._partner_action()
        actions = [0, 0]
        actions[self.agent_id]  = int(action)
        actions[self.partner_id] = partner_act

        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        self._last_obs = obs_dict
        return (
            obs_dict[self.agent_id],
            float(rewards[self.agent_id]),
            terminated,
            truncated,
            info,
        )

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# IPPOTrainer — Independent PPO Trainer
# ---------------------------------------------------------------------------

class IPPOTrainer:
    """
    Independent PPO Trainer — trains each agent separately using SB3 PPO.

    Training scheme (per round):
        1. Train agent-0 (agent-1 follows current policy / random)
        2. Train agent-1 (agent-0 follows current policy)
    Alternating rounds let each agent adapt to the latest partner.
    """

    def __init__(
        self,
        map_file: str,
        map_paths: dict          = None,
        backup_dir: str          = "./checkpoints_ippo",
        n_envs: int              = 4,
        train_steps_per_round: int = 2_000_000,
        n_rounds: int            = 3,
        learning_rate: float     = 2e-4,
        n_steps: int             = 2048,
        batch_size: int          = 256,
        n_epochs: int            = 10,
        gamma: float             = 0.995,
        gae_lambda: float        = 0.95,
        clip_range: float        = 0.2,
        ent_coef: float          = 0.04,
        obs_radius: int          = 2,
        reward_cfg: Optional[RewardConfig] = None,
        seed: int                = 42,
    ):
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required for IPPOTrainer.")

        self.map_file              = map_file
        self.map_paths             = map_paths or {}
        self.backup_dir            = backup_dir
        self.n_envs                = n_envs
        self.train_steps_per_round = train_steps_per_round
        self.n_rounds              = n_rounds
        self.learning_rate         = learning_rate
        self.n_steps               = n_steps
        self.batch_size            = batch_size
        self.n_epochs              = n_epochs
        self.gamma                 = gamma
        self.gae_lambda            = gae_lambda
        self.clip_range            = clip_range
        self.ent_coef              = ent_coef
        self.obs_radius            = obs_radius
        self.seed                  = seed
        # Default to IPPORewardConfig (larger reward scale for SB3 VecNormalize)
        self.reward_cfg            = reward_cfg or IPPORewardConfig()

        os.makedirs(backup_dir, exist_ok=True)

        self.models: Dict[int, PPO] = {}
        self.eval_history = []

    def _make_env(self, agent_id: int, partner_model=None, env_seed: int = 0):
        """Create a single-agent wrapped env for training."""
        env = UAVPatrolEnvIPPO(
            map_file=self.map_file,
            reward_cfg=self.reward_cfg,
            obs_radius=self.obs_radius,
        )
        return SingleAgentWrapper(env, agent_id=agent_id, partner_model=partner_model)

    def _make_vec_env(self, agent_id: int, partner_model=None):
        """Create vectorized envs for parallel SB3 training."""
        def make_fn(i):
            def _init():
                return self._make_env(agent_id, partner_model, env_seed=self.seed + i)
            return _init
        vec = SubprocVecEnv([make_fn(i) for i in range(self.n_envs)])
        return VecNormalize(vec, norm_obs=True, norm_reward=True)

    def train(self) -> None:
        """
        Alternating-round IPPO training.
        Each round trains agent-0 then agent-1 for `train_steps_per_round` steps.
        """
        print("=" * 70)
        print(f"IPPO Training -- {self.n_rounds} rounds x {self.train_steps_per_round} steps/round")
        print(f"  Map: {self.map_file}")
        print(f"  obs_radius={self.obs_radius}  lr={self.learning_rate}  batch={self.batch_size}  gamma={self.gamma}  ent={self.ent_coef}")
        print("=" * 70)

        model0, model1 = None, None

        for round_idx in range(1, self.n_rounds + 1):
            print(f"\n--- Round {round_idx}/{self.n_rounds} ---")

            # -- Train Agent 0 (partner = current model1 or random) ----------
            print("  Training Agent-0...")
            vec_env0 = self._make_vec_env(agent_id=0, partner_model=model1)
            model0 = PPO(
                "MlpPolicy",
                vec_env0,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                verbose=0,
                seed=self.seed,
            )
            model0.learn(total_timesteps=self.train_steps_per_round)
            vec_env0.close()

            # -- Train Agent 1 (partner = updated model0) --------------------
            print("  Training Agent-1...")
            vec_env1 = self._make_vec_env(agent_id=1, partner_model=model0)
            model1 = PPO(
                "MlpPolicy",
                vec_env1,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                verbose=0,
                seed=self.seed + 1,
            )
            model1.learn(total_timesteps=self.train_steps_per_round)
            vec_env1.close()

            self.models = {0: model0, 1: model1}

            # -- Evaluate after each round -----------------------------------
            eval_res = self.evaluate_all_maps()
            self.eval_history.append({
                "round": round_idx,
                **eval_res,
            })
            print(f"  Round {round_idx} eval: {eval_res}")

        print("\n" + "=" * 70)
        print("IPPO Training complete!")
        print("=" * 70)

    def evaluate_map(self, map_file: str, n_episodes: int = 10) -> dict:
        """Evaluate joint agent performance on a single map."""
        env = UAVPatrolEnvIPPO(map_file=map_file, reward_cfg=self.reward_cfg, obs_radius=self.obs_radius)
        coverages = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=500 + ep)
            done = False
            while not done:
                a0, _ = self.models[0].predict(obs[0], deterministic=True) if 0 in self.models else (env.single_action_space.sample(), None)
                a1, _ = self.models[1].predict(obs[1], deterministic=True) if 1 in self.models else (env.single_action_space.sample(), None)
                obs, _, t, tr, info = env.step([int(a0), int(a1)])
                done = t or tr
            coverages.append(info["coverage_ratio"] * 100.0)
        return {"coverage": float(np.mean(coverages))}

    def evaluate_all_maps(self, n_episodes: int = 10) -> dict:
        results = {}
        for name, mp in self.map_paths.items():
            results[name] = self.evaluate_map(mp, n_episodes)
        return results

    def save(self, prefix: str = "agent") -> None:
        """Save trained SB3 PPO models."""
        for agent_id, model in self.models.items():
            path = os.path.join(self.backup_dir, f"{prefix}{agent_id}_final")
            model.save(path)
            print(f"  Saved -> {path}.zip")

    def load(self, prefix: str = "agent") -> None:
        """Load trained SB3 PPO models."""
        for agent_id in range(2):
            path = os.path.join(self.backup_dir, f"{prefix}{agent_id}_final")
            self.models[agent_id] = PPO.load(path)
            print(f"  Loaded <- {path}.zip")
