# -*- coding: utf-8 -*-
"""
Visualize agent trajectories

Cách sử dụng:
    python scripts/visualize.py \
        --model_path ./checkpoints/ppo_uav_v3 \
        --map simple


"""

import argparse
import os
import numpy as np
from pathlib import Path

from config import ENVIRONMENT_CONFIG, PPO_CONFIG
from environments import UAVPatrolEnv
from agents import PPOSingleAgent
from utils import plot_trajectory, plot_visit_heatmap


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Đường dẫn model để load"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="simple",
        choices=["simple", "mixed", "bottleneck"],
        help="Map để visualize"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Số episodes để vẽ"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Thư mục lưu ảnh"
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="./environments/maps",
        help="Thư mục chứa map files"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚁 UAV Trajectory Visualization")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Map: {args.map}")
    print(f"Episodes: {args.n_episodes}")
    print()
    
    # ========== Create output dir ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ========== Load agent ==========
    print("🤖 Loading agent...")
    map_paths = []
    for name in ENVIRONMENT_CONFIG["maps"].keys():
        path = os.path.join(args.maps_dir, f"map_{name}.json")
        if os.path.exists(path):
            map_paths.append(path)
    
    agent = PPOSingleAgent(PPO_CONFIG, map_paths=map_paths)
    agent.setup_env(num_envs=1)
    agent.load(args.model_path)
    print("✓ Agent loaded")
    print()
    
    # ========== Get map path ==========
    map_path = os.path.join(args.maps_dir, f"map_{args.map}.json")
    if not os.path.exists(map_path):
        print(f"❌ Map not found: {map_path}")
        return
    
    print(f"📍 Map: {map_path}")
    print()
    
    # ========== Visualize episodes ==========
    print(f"🎬 Running {args.n_episodes} episodes...")
    print("-" * 70)
    
    total_visits = None
    
    for ep_idx in range(args.n_episodes):
        print(f"\nEpisode {ep_idx + 1}/{args.n_episodes}:")
        
        # Reset environment
        env = UAVPatrolEnv(map_file=map_path)
        obs, _ = env.reset()
        
        # Collect trajectory
        trajectory = [env.uav_pos.copy()]
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            trajectory.append(env.uav_pos.copy())
            done = terminated or truncated
        
        trajectory = np.array(trajectory)
        coverage = env.coverage.sum() / env.free_cells * 100
        
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Steps: {env.steps}")
        print(f"  Trajectory length: {len(trajectory)}")
        
        # ========== Plot trajectory ==========
        save_path = os.path.join(
            args.save_dir,
            f"trajectory_{args.map}_ep{ep_idx+1}.png"
        )
        plot_trajectory(env, trajectory, coverage, save_path=save_path)
        
        # ========== Accumulate visit counts ==========
        if total_visits is None:
            total_visits = env.visit_count.copy()
        else:
            total_visits += env.visit_count
    
    # ========== Plot accumulated heatmap ==========
    print("\n📊 Plotting visit heatmap...")
    env = UAVPatrolEnv(map_file=map_path)
    heatmap_path = os.path.join(
        args.save_dir,
        f"heatmap_{args.map}.png"
    )
    plot_visit_heatmap(total_visits, env.obstacles, save_path=heatmap_path)
    
    print()
    print("=" * 70)
    print("✅ Visualization completed!")
    print(f"Saved to: {args.save_dir}")
    print()
    
    agent.close()


if __name__ == "__main__":
    main()
