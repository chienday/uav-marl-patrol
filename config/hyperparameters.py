# -*- coding: utf-8 -*-
"""
Các siêu tham số cho các thuật toán RL và cấu hình môi trường
"""

# ========== CẤU HÌNH MÔI TRƯỜNG ==========
ENVIRONMENT_CONFIG = {
    # Các map tuần tra
    "maps": {
        "simple": {
            "grid_size": 10,
            "obstacles": [[3, 3], [3, 4], [6, 6], [6, 7]],
            "start_position": [0, 0],
            "max_steps": 600,
        },
        "mixed": {
            "grid_size": 10,
            "obstacles": [[2, 5], [3, 5], [4, 5], [5, 2], [6, 2], [7, 2], [7, 3]],
            "start_position": [0, 0],
            "max_steps": 600,
        },
        "bottleneck": {
            "grid_size": 10,
            "obstacles": [
                [5, 0], [5, 1], [5, 2], [5, 3],
                [5, 5], [5, 6], [5, 7], [5, 8], [5, 9]
            ],
            "start_position": [0, 0],
            "max_steps": 800,
        },
    },
    # Tỷ lệ mục tiêu coverage
    "target_coverage": 0.90,
}

# ========== CẤU HÌNH REWARD VARIANTS ==========
# Biến thể A: Đơn giản (baseline)
REWARD_VARIANT_A = {
    "name": "Simple",
    "EXPLORE_REWARD": 50.0,
    "OBSTACLE_PENALTY": 20.0,
    "COVERAGE_SCALE": 150.0,
    "FRONTIER_SCALE": 0.0,
    "REVISIT_MULT": 0.0,
    "REVISIT_CAP": 0.0,
    "BFS_SCALE": 0.0,
    "STEP_PENALTY": 0.05,
    "COMPLETE_BONUS": 3000.0,
    "PARTIAL_SCALE": 0.0,
    "PASSAGE_BONUS": 0.0,
}

# Biến thể B: Dày đặc (không BFS/passage)
REWARD_VARIANT_B = {
    "name": "Dense",
    "EXPLORE_REWARD": 50.0,
    "OBSTACLE_PENALTY": 20.0,
    "COVERAGE_SCALE": 30.0,
    "FRONTIER_SCALE": 15.0,
    "REVISIT_MULT": 1.0,
    "REVISIT_CAP": 15.0,
    "BFS_SCALE": 0.0,
    "STEP_PENALTY": 0.1,
    "COMPLETE_BONUS": 3000.0,
    "PARTIAL_SCALE": 300.0,
    "PASSAGE_BONUS": 0.0,
}

# Biến thể C: Đầy đủ (hiện tại - đã chọn tốt nhất)
REWARD_VARIANT_C = {
    "name": "Full",
    "EXPLORE_REWARD": 60.0,
    "OBSTACLE_PENALTY": 20.0,
    "COVERAGE_SCALE": 25.0,
    "FRONTIER_SCALE": 12.0,
    "REVISIT_MULT": 1.5,
    "REVISIT_CAP": 25.0,
    "BFS_SCALE": 8.0,
    "STEP_PENALTY": 0.1,
    "COMPLETE_BONUS": 3000.0,
    "PARTIAL_SCALE": 600.0,
    "PASSAGE_BONUS": 80.0,
}

# Chọn biến thể reward mặc định
SELECTED_REWARD_VARIANT = REWARD_VARIANT_C

# ========== CẤU HÌNH PPO (1 UAV) ==========
PPO_CONFIG = {
    "name": "PPO_Single_UAV",
    "algorithm": "PPO",
    "num_agents": 1,
    "policy": "MlpPolicy",
    
    # Siêu tham số chính
    "learning_rate": 2e-4,
    "gamma": 0.995,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "n_steps": 2048,  # Số bước trước khi update
    "batch_size": 256,
    "n_epochs": 10,  # Số epoch mỗi update
    "ent_coef": 0.04,  # Entropy coefficient
    "clip_range": 0.2,  # PPO clip range
    
    # Kiến trúc mạng
    "net_arch": [512, 256, 128],
    
    # Cấu hình training
    "total_timesteps": 10_000_000,
    "eval_freq": 100_000,
    "n_eval_episodes": 8,
    
    # Device
    "device": "cuda",  # "cuda" hoặc "cpu"
    "seed": 42,
    
    # VecNormalize
    "normalize_obs": False,
    "normalize_reward": True,
    "clip_reward": 10,
}

# ========== CẤU HÌNH IPPO (2 UAV) - Chuẩn bị ==========
IPPO_CONFIG = {
    "name": "IPPO_Two_UAVs",
    "algorithm": "IPPO",
    "num_agents": 2,
    "policy": "MlpPolicy",
    
    # Siêu tham số chính
    "learning_rate": 2e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "ent_coef": 0.04,
    "clip_range": 0.2,
    
    # Kiến trúc mạng
    "net_arch": [512, 256, 128],
    
    # Cấu hình training
    "total_timesteps": 10_000_000,
    "eval_freq": 100_000,
    "n_eval_episodes": 8,
    
    # Device
    "device": "cuda",
    "seed": 42,
    
    # Communication setup (sắp tới)
    "communication_range": 3,  # Phạm vi giao tiếp giữa các UAV
    "use_shared_experience": False,
}

# ========== CẤU HÌNH MAPPO - Chuẩn bị ==========
MAPPO_CONFIG = {
    "name": "MAPPO_Multi_Agent",
    "algorithm": "MAPPO",
    "num_agents": 2,  # Có thể mở rộng
    
    # Siêu tham số actor
    "actor_learning_rate": 2e-4,
    "actor_gamma": 0.995,
    "actor_gae_lambda": 0.95,
    "actor_n_steps": 2048,
    
    # Siêu tham số critic
    "critic_learning_rate": 2e-4,
    "critic_gamma": 0.995,
    
    # Kiến trúc mạng
    "actor_net_arch": [512, 256, 128],
    "critic_net_arch": [512, 256, 128],
    
    # Cấu hình training
    "total_timesteps": 10_000_000,
    "device": "cuda",
    "seed": 42,
    
    # Multi-agent specific
    "use_centralized_critic": True,
    "use_value_active_mask": True,
}

# ========== CẤU HÌNH VDPPO - Chuẩn bị ==========
VDPPO_CONFIG = {
    "name": "VDPPO_Value_Decomposition",
    "algorithm": "VDPPO",
    "num_agents": 2,
    
    # Siêu tham số chính
    "learning_rate": 2e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    
    # Kiến trúc mạng
    "net_arch": [512, 256, 128],
    
    # Value decomposition
    "decomposition_type": "qmix",  # "qmix", "qtran", hoặc "vdn"
    
    # Cấu hình training
    "total_timesteps": 10_000_000,
    "device": "cuda",
    "seed": 42,
}

# ========== CẤU HÌNH TRAINING ==========
TRAINING_CONFIG = {
    # Logging
    "log_interval": 1000,
    "save_checkpoint_interval": 1_000_000,
    "backup_dir": "./checkpoints",
    
    # Reproducibility
    "seed": 42,
    "deterministic": True,
    
    # Environment
    "n_env_workers": 8,  # Số envs chạy song song
    "env_names": ["simple", "mixed", "bottleneck"],
    
    # Callbacks
    "use_eval_callback": True,
    "eval_freq": 100_000,
    "n_eval_episodes": 10,
}

# ========== CẤU HÌNH EVALUATION ==========
EVALUATION_CONFIG = {
    "n_episodes": 20,
    "render": False,
    "deterministic": True,
    "measure_coverage": True,
    "measure_efficiency": True,
    "compare_vs_random": True,
}

if __name__ == "__main__":
    # Test config
    print("PPO Config:")
    print(f"  Learning Rate: {PPO_CONFIG['learning_rate']}")
    print(f"  Total Timesteps: {PPO_CONFIG['total_timesteps']:,}")
    print(f"  Net Architecture: {PPO_CONFIG['net_arch']}")
    print(f"\nSelected Reward Variant: {SELECTED_REWARD_VARIANT['name']}")
    print(f"\nEnvironment Maps: {list(ENVIRONMENT_CONFIG['maps'].keys())}")
