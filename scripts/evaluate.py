# -*- coding: utf-8 -*-
"""
Evaluate trained agent

Cách sử dụng:
    python scripts/evaluate.py \
        --model_path ./checkpoints/ppo_uav_v3 \
        --n_episodes 20


"""

import argparse
import os
from pathlib import Path

from config import ENVIRONMENT_CONFIG
from agents import PPOSingleAgent
from utils import evaluate_agent, print_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Đường dẫn model (.zip) để load"
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="./environments/maps",
        help="Thư mục chứa map JSON files"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Số episodes mỗi map"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Dùng deterministic policy"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚁 UAV Agent Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Episodes per map: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print()
    
    # ========== Prepare map paths ==========
    map_paths = []
    for name in ENVIRONMENT_CONFIG["maps"].keys():
        path = os.path.join(args.maps_dir, f"map_{name}.json")
        if os.path.exists(path):
            map_paths.append(path)
    
    if not map_paths:
        print(f"❌ No maps found in {args.maps_dir}")
        return
    
    print(f"📍 Found {len(map_paths)} maps: {[Path(p).stem for p in map_paths]}")
    print()
    
    # ========== Load agent ==========
    print("🤖 Loading agent...")
    from config import PPO_CONFIG
    agent = PPOSingleAgent(PPO_CONFIG, map_paths=map_paths)
    agent.setup_env(num_envs=1)
    agent.load(args.model_path)
    print("✓ Agent loaded")
    print()
    
    # ========== Evaluate ==========
    print("📊 Evaluating...")
    print("-" * 70)
    results = evaluate_agent(
        agent=agent,
        map_paths=map_paths,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
    )
    print()
    
    # ========== Print results ==========
    print_evaluation_results(results)
    
    # ========== Summary ==========
    print("📈 Summary:")
    print("-" * 70)
    for map_name, metrics in results.items():
        if map_name == "average":
            continue
        cov = metrics["mean_coverage"]
        success = metrics["success_rate"]
        reward = metrics["mean_reward"]
        
        status = "✅ PASS" if cov >= 90 else "❌ FAIL"
        print(f"{map_name:15} | Coverage: {cov:6.1f}% {status} | Success: {success:5.0f}% | Reward: {reward:7.0f}")
    
    print()
    avg_cov = results["average"]["mean_coverage"]
    overall_status = "✅ SUCCESS" if avg_cov >= 90 else "⚠️  WARNING"
    print(f"Average Coverage: {avg_cov:.1f}%  {overall_status}")
    print()
    
    agent.close()


if __name__ == "__main__":
    main()
