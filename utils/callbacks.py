# -*- coding: utf-8 -*-
"""
Callbacks cho huấn luyện

Tác giả: [DATN Project]
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict, List


class CoverageLogCallback(BaseCallback):
    """
    Log coverage progress mỗi N bước
    
    Chạy evaluation trên 3 maps (simple, mixed, bottleneck)
    và in ra kết quả
    """
    
    def __init__(self, map_paths: List[str], check_freq: int = 200_000):
        """
        Args:
            map_paths (List[str]): Danh sách đường dẫn map files
            check_freq (int): Tần suất kiểm tra (mỗi N timesteps)
        """
        super().__init__()
        self.check_freq = check_freq
        self.map_paths = map_paths
        self.n_eval_episodes = 5
    
    def _on_step(self) -> bool:
        """Callback mỗi step"""
        if self.n_calls % self.check_freq == 0:
            results = []
            
            # Đánh giá trên từng map
            from ..environments import UAVPatrolEnv
            
            for mp in self.map_paths:
                coverages = []
                
                for _ in range(self.n_eval_episodes):
                    env = UAVPatrolEnv(map_file=mp)
                    obs, _ = env.reset()
                    done = False
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(int(action))
                        done = terminated or truncated
                    
                    cov_pct = env.coverage.sum() / env.free_cells * 100
                    coverages.append(cov_pct)
                
                results.append(np.mean(coverages))
            
            # In kết quả
            map_names = [
                p.split("/")[-1].replace(".json", "") for p in self.map_paths
            ]
            line = "  |  ".join(
                f"{name:12s}: {r:.1f}%"
                for name, r in zip(map_names, results)
            )
            
            min_cov = min(results)
            goal_str = " ✓ GOAL (90%+)" if min_cov >= 90 else ""
            
            print(
                f"[{self.num_timesteps:>9,d} steps]  {line}{goal_str}"
            )
        
        return True


class AutoSaveCallback(BaseCallback):
    """
    Tự động lưu checkpoint mỗi N bước
    """
    
    def __init__(
        self,
        save_freq: int = 1_000_000,
        save_dir: str = "./checkpoints",
    ):
        """
        Args:
            save_freq (int): Tần suất lưu (mỗi N timesteps)
            save_dir (str): Thư mục lưu
        """
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
    
    def _on_step(self) -> bool:
        """Callback mỗi step"""
        if self.n_calls % self.save_freq == 0:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            
            tag = f"{self.num_timesteps // 1_000_000}M"
            model_path = f"{self.save_dir}/ppo_uav_{tag}"
            
            self.model.save(model_path)
            
            # Lưu VecNormalize nếu có
            if hasattr(self.model, "get_env") and self.model.get_env() is not None:
                env = self.model.get_env()
                if hasattr(env, "save"):
                    vec_path = f"{self.save_dir}/vec_normalize_{tag}.pkl"
                    env.save(vec_path)
            
            print(f"   [{tag} steps] Checkpoint saved → {self.save_dir}")
        
        return True
