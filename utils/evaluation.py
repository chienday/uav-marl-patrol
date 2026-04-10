# -*- coding: utf-8 -*-
"""
Hàm đánh giá agents

"""

from typing import Dict, Any, Optional
import numpy as np
from ..environments import UAVPatrolEnv
from ..agents.base_agent import BaseAgent


def evaluate_agent(
    agent: BaseAgent,
    map_paths: list,
    n_episodes: int = 20,
    deterministic: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    """
    Đánh giá agent trên các maps khác nhau
    
    Args:
        agent (BaseAgent): Agent để đánh giá
        map_paths (list): Danh sách đường dẫn maps
        n_episodes (int): Số episodes mỗi map
        deterministic (bool): Dùng deterministic policy
        render (bool): In render mỗi episode
    
    Returns:
        Dict: Kết quả đánh giá
            {
                "map_name": {
                    "mean_coverage": float,
                    "std_coverage": float,
                    "success_rate": float,
                    "mean_reward": float,
                }
            }
    """
    results = {}
    
    for map_path in map_paths:
        map_name = map_path.split("/")[-1].replace(".json", "")
        
        coverages = []
        rewards = []
        steps_list = []
        
        for _ in range(n_episodes):
            env = UAVPatrolEnv(map_file=map_path)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                episode_reward += reward
                done = terminated or truncated
            
            if render:
                env.render()
            
            cov_pct = env.coverage.sum() / env.free_cells * 100
            coverages.append(cov_pct)
            rewards.append(episode_reward)
            steps_list.append(env.steps)
        
        success_rate = sum(c >= 90 for c in coverages) / n_episodes * 100
        
        results[map_name] = {
            "mean_coverage": float(np.mean(coverages)),
            "std_coverage": float(np.std(coverages)),
            "min_coverage": float(np.min(coverages)),
            "max_coverage": float(np.max(coverages)),
            "success_rate": success_rate,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
        }
    
    # Calculate average across all maps
    avg_coverage = np.mean([r["mean_coverage"] for r in results.values()])
    results["average"] = {"mean_coverage": avg_coverage}
    
    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    In kết quả đánh giá dạng bảng
    
    Args:
        results (Dict): Kết quả từ evaluate_agent()
    """
    print("\n" + "=" * 80)
    print(f"{'Map':<15} {'Coverage':>12} {'Std':>6} {'Success':>10} {'Reward':>10} {'Steps':>8}")
    print("─" * 80)
    
    for map_name, metrics in results.items():
        if map_name == "average":
            continue
        
        cov = metrics["mean_coverage"]
        std = metrics["std_coverage"]
        success = metrics["success_rate"]
        reward = metrics["mean_reward"]
        steps = metrics["mean_steps"]
        
        flag = "✓" if cov >= 90 else "✗"
        print(
            f"{map_name:<15} {cov:>10.1f}% {flag}  {std:>5.1f}  "
            f"{success:>8.0f}%  {reward:>10.0f}  {steps:>8.0f}"
        )
    
    if "average" in results:
        avg_cov = results["average"]["mean_coverage"]
        print("─" * 80)
        print(f"{'Average':<15} {avg_cov:>10.1f}%")
    
    print("=" * 80 + "\n")
