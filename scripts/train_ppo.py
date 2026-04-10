# -*- coding: utf-8 -*-
"""
Train PPO Agent cho 1 UAV

Cách sử dụng:
    python scripts/train_ppo.py \
        --total_timesteps 10000000 \
        --num_envs 8 \
        --save_dir ./checkpoints

Tác giả: [DATN Project]
"""

import argparse
import json
import os
from pathlib import Path

from config import PPO_CONFIG, ENVIRONMENT_CONFIG
from agents import PPOSingleAgent
from utils import CoverageLogCallback, AutoSaveCallback


def create_map_files(env_maps_dir: str):
    """Tạo map JSON files"""
    os.makedirs(env_maps_dir, exist_ok=True)
    
    map_paths = []
    for name, config in ENVIRONMENT_CONFIG["maps"].items():
        path = os.path.join(env_maps_dir, f"map_{name}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        map_paths.append(path)
        print(f"✓ Created: {path}")
    
    return map_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent cho UAV tuần tra"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=PPO_CONFIG["total_timesteps"],
        help="Tổng timesteps huấn luyện"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Số environments chạy song song"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Thư mục lưu checkpoints"
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="./environments/maps",
        help="Thư mục chứa map JSON files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device để train (cuda hoặc cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚁 UAV PPO Training")
    print("=" * 70)
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Num Environments: {args.num_envs}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print()
    
    # ========== Tạo map files ==========
    print("📍 Creating map files...")
    map_paths = create_map_files(args.maps_dir)
    print()
    
    # ========== Khởi tạo config ==========
    print("⚙️  Setting config...")
    config = dict(PPO_CONFIG)
    config["total_timesteps"] = args.total_timesteps
    config["device"] = args.device
    print(f"   Algorithm: {config['name']}")
    print(f"   Policy: {config['policy']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Net Arch: {config['net_arch']}")
    print()
    
    # ========== Tạo agent ==========
    print("🤖 Creating agent...")
    agent = PPOSingleAgent(config, map_paths=map_paths)
    print(f"   {agent}")
    print()
    
    # ========== Setup environment ==========
    print("🌍 Setting up environment...")
    agent.setup_env(num_envs=args.num_envs)
    print(f"   Vectorized Env: {agent.vec_env}")
    print(f"   Obs Space: {agent.vec_env.observation_space}")
    print(f"   Action Space: {agent.vec_env.action_space}")
    print()
    
    # ========== Build model ==========
    print("🧠 Building model...")
    agent.build_model()
    print(f"   Model: {agent.model}")
    print()
    
    # ========== Callbacks ==========
    print("📋 Setting up callbacks...")
    callbacks = [
        CoverageLogCallback(map_paths=map_paths, check_freq=200_000),
        AutoSaveCallback(save_freq=1_000_000, save_dir=args.save_dir),
    ]
    print(f"   Callbacks: {len(callbacks)}")
    print()
    
    # ========== Training ==========
    print("🚀 Starting training...")
    print("=" * 70)
    
    try:
        agent.train(
            total_timesteps=args.total_timesteps,
            callbacks=callbacks,
        )
        
        print("=" * 70)
        print("✅ Training completed!")
        print()
        
        # ========== Save model ==========
        print("💾 Saving model...")
        save_path = os.path.join(args.save_dir, "ppo_uav_final")
        agent.save(save_path)
        print()
        
        # ========== Close environment ==========
        agent.close()
        
        print("🎉 All done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        agent.close()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        agent.close()
        raise


if __name__ == "__main__":
    main()
