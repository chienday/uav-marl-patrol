# -*- coding: utf-8 -*-
"""
PPO Single Agent - 1 UAV với PPO


"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from .base_agent import BaseAgent
from ..environments import UAVPatrolEnv


class PPOSingleAgent(BaseAgent):
    """
    PPO Agent cho 1 UAV
    
    Workflow:
    1. Tạo môi trường
    2. Wrap với Monitor và VecNormalize
    3. Khởi tạo mô hình PPO
    4. Huấn luyện
    5. Đánh giá
    """
    
    def __init__(self, config: Dict[str, Any], map_paths: Optional[List[str]] = None):
        """
        Khởi tạo PPO single agent
        
        Args:
            config (Dict): Configuration từ config/hyperparameters.py
            map_paths (List[str]): Danh sách đường dẫn map files
        """
        super().__init__(name=config["name"], config=config)
        
        self.map_paths = map_paths or []
        self.device = config.get("device", "cuda")
        self.seed = config.get("seed", 42)
        
        # Fix reproducibility
        self._set_seed()
    
    def _set_seed(self) -> None:
        """Fix random seed"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _make_env(self, map_path: str, rank: int = 0):
        """
        Factory function tạo môi trường đơn lẻ
        
        Args:
            map_path (str): Đường dẫn map file
            rank (int): Rank của env (để tạo seed khác nhau)
        
        Returns:
            callable: Hàm khởi tạo env
        """
        def _init():
            env = UAVPatrolEnv(map_file=map_path)
            env.reset(seed=self.seed + rank)
            return Monitor(env)
        return _init
    
    def setup_env(self, num_envs: int = 8) -> None:
        """
        Set up vectorized environment
        
        Args:
            num_envs (int): Số environments
        """
        if not self.map_paths:
            raise ValueError("map_paths chưa được set!")
        
        # Distribute maps across envs
        env_makers = []
        for i in range(num_envs):
            map_idx = i % len(self.map_paths)
            env_makers.append(self._make_env(self.map_paths[map_idx], rank=i))
        
        # Create vectorized env
        self.vec_env = DummyVecEnv(env_makers)
        
        # Normalize rewards
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=self.config.get("normalize_obs", False),
            norm_reward=self.config.get("normalize_reward", True),
            clip_reward=self.config.get("clip_reward", 10),
            gamma=self.config.get("gamma", 0.995),
        )
    
    def build_model(self) -> None:
        """Khởi tạo mô hình PPO"""
        if self.vec_env is None:
            raise RuntimeError("Gọi setup_env() trước!")
        
        self.model = PPO(
            policy=self.config["policy"],
            env=self.vec_env,
            learning_rate=self.config["learning_rate"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            ent_coef=self.config["ent_coef"],
            clip_range=self.config["clip_range"],
            policy_kwargs=dict(net_arch=self.config["net_arch"]),
            device=self.device,
            seed=self.seed,
            verbose=1,
        )
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks=None,
    ) -> None:
        """
        Huấn luyện agent
        
        Args:
            total_timesteps (int): Số timesteps (mặc định: từ config)
            callbacks: List callbacks
        """
        if self.model is None:
            raise RuntimeError("Gọi build_model() trước!")
        
        total_timesteps = total_timesteps or self.config["total_timesteps"]
        
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=callbacks,
        )
    
    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Any]:
        """
        Dự đoán hành động
        
        Args:
            obs (np.ndarray): Observation
            deterministic (bool): Sử dụng deterministic policy
        
        Returns:
            Tuple[int, Any]: (action, logits)
        """
        if self.model is None:
            raise RuntimeError("Model không được tạo!")
        
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return int(action), _states
    
    def evaluate(
        self,
        env: UAVPatrolEnv,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Đánh giá agent trên môi trường
        
        Args:
            env (UAVPatrolEnv): Môi trường test
            n_episodes (int): Số episodes
            deterministic (bool): Sử dụng deterministic
        
        Returns:
            Dict: Metrics (coverage, reward, steps)
        """
        if self.model is None:
            raise RuntimeError("Model không được tạo!")
        
        coverages, rewards, steps_list = [], [], []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            coverage = env.coverage.sum() / env.free_cells * 100
            coverages.append(coverage)
            rewards.append(episode_reward)
            steps_list.append(env.steps)
        
        return {
            "mean_coverage": float(np.mean(coverages)),
            "std_coverage": float(np.std(coverages)),
            "mean_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "success_rate": float(sum(c >= 90 for c in coverages) / n_episodes * 100),
        }
    
    def save(self, path: str) -> None:
        """
        Lưu model
        
        Args:
            path (str): Đường dẫn lưu (không cần .zip)
        """
        if self.model is None:
            raise RuntimeError("Model không được tạo!")
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        self.model.save(path)
        print(f"✓ Model saved: {path}.zip")
        
        # Lưu VecNormalize nếu có
        if self.vec_env is not None:
            vec_path = f"{path}_vecnorm.pkl"
            self.vec_env.save(vec_path)
            print(f"✓ VecNormalize saved: {vec_path}")
    
    def load(self, path: str) -> None:
        """
        Load model
        
        Args:
            path (str): Đường dẫn model (không cần .zip)
        """
        if self.vec_env is None:
            raise RuntimeError("Gọi setup_env() trước!")
        
        self.model = PPO.load(path, env=self.vec_env)
        
        # Load VecNormalize nếu có
        vec_path = f"{path}_vecnorm.pkl"
        if os.path.exists(vec_path):
            self.vec_env = VecNormalize.load(vec_path, self.vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        
        print(f"✓ Model loaded: {path}")
    
    def reset(self) -> None:
        """Reset agent state"""
        if self.vec_env is not None:
            self.vec_env.reset()
    
    def close(self) -> None:
        """Đóng môi trường"""
        if self.vec_env is not None:
            self.vec_env.close()


if __name__ == "__main__":
    # Test
    from config import PPO_CONFIG, ENVIRONMENT_CONFIG
    import json
    
    # Create dummy map files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        map_paths = []
        for name, config in ENVIRONMENT_CONFIG["maps"].items():
            path = f"{tmpdir}/{name}.json"
            with open(path, "w") as f:
                json.dump(config, f)
            map_paths.append(path)
        
        # Create agent
        agent = PPOSingleAgent(PPO_CONFIG, map_paths=map_paths)
        agent.setup_env(num_envs=4)
        agent.build_model()
        
        print(f"Agent: {agent}")
        print(f"Model: {agent.model}")
        print("PPOSingleAgent test passed! ✓")
