# -*- coding: utf-8 -*-
"""
Môi trường UAVPatrolEnv - Tuần tra grid bằng UAV

Mô tả:
    UAVPatrolEnv là môi trường Gymnasium cho nhiệm vụ tuần tra
    một lưới grid với UAV (Unmanned Aerial Vehicle).
    
    Hành động:
    - 0: Dịch chuyển Up (hàng giảm)
    - 1: Dịch chuyển Down (hàng tăng)
    - 2: Dịch chuyển Left (cột giảm)
    - 3: Dịch chuyển Right (cột tăng)
    
    Mục tiêu:
    Đạt coverage >= 90% với số bước tối thiểu

"""

import gymnasium as gym
import numpy as np
import json
from collections import deque
from gymnasium import spaces


class UAVPatrolEnv(gym.Env):
    """
    Môi trường tuần tra UAV trên grid
    
    Reward Components:
        - EXPLORE_REWARD: Thưởng khi khám phá ô mới
        - OBSTACLE_PENALTY: Phạt khi đi vào chướng ngại vật
        - COVERAGE_SCALE: Thưởng tỷ lệ coverage hiện tại
        - FRONTIER_SCALE: Thưởng khi ở gần ranh giới
        - REVISIT_MULT: Phạt khi revisit (nhân với bình phương số lần thăm)
        - BFS_SCALE: Thưởng dựa khoảng cách tới frontier
        - PASSAGE_BONUS: Thưởng khi vượt qua hẹp
        - STEP_PENALTY: Phạt mỗi bước
        - COMPLETE_BONUS: Thưởng hoàn thành (coverage >= 97%)
        - PARTIAL_SCALE: Thưởng cuối episode dựa coverage
    """
    
    metadata = {"render_modes": ["human"]}

    # ========== Siêu tham số Reward ==========
    # Trong v1: Hiện tại sử dụng Variant C (Full)
    EXPLORE_REWARD = 60.0
    OBSTACLE_PENALTY = 20.0
    COVERAGE_SCALE = 25.0
    FRONTIER_SCALE = 12.0
    REVISIT_MULT = 1.5
    REVISIT_CAP = 25.0
    BFS_SCALE = 8.0
    STEP_PENALTY = 0.1
    COMPLETE_BONUS = 3000.0
    PARTIAL_SCALE = 600.0
    PASSAGE_BONUS = 80.0

    def __init__(self, grid_size=10, max_steps=600, map_file=None):
        """
        Khởi tạo môi trường
        
        Args:
            grid_size (int): Kích thước grid (mặc định: 10)
            max_steps (int): Số bước tối đa mỗi episode (mặc định: 600)
            map_file (str): Đường dẫn file JSON map (tùy chọn)
        """
        super().__init__()
        
        # ========== Load config từ map_file nếu có ==========
        if map_file:
            with open(map_file) as f:
                config = json.load(f)
            grid_size = config.get("grid_size", grid_size)
            max_steps = config.get("max_steps", max_steps)
            self.start_position = config.get("start_position", [0, 0])
            self.obstacles = config.get("obstacles", [])
        else:
            self.start_position = [0, 0]
            self.obstacles = []

        # ========== Cấu hình m ôi trường cơ bản ==========
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacles = set(tuple(o) for o in self.obstacles)
        self.free_cells = grid_size * grid_size - len(self.obstacles)
        self.passages = self._detect_passages()

        # ========== Không gian hành động & quan sát ==========
        self.action_space = spaces.Discrete(4)  # 4 hướng
        # Obs: [coverage (100), obstacles (100), visit_norm (100), pos (2), dist (1)] = 303 chiều
        obs_size = grid_size * grid_size * 3 + 3
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # ========== State (sẽ reset sau) ==========
        self.reset()

    def _detect_passages(self):
        """
        Phát hiện các 'hẹp' (passages)
        
        Hẹp: Hàng hoặc cột bị chặn >= 70% bằng chướng ngại vật
        Mục đích: Thưởng lên khi UAV vượt qua hẹp
        
        Returns:
            set: Tập hợp các vị trí là passages
        """
        passages = set()
        g = self.grid_size
        
        # Kiểm tra hàng
        for r in range(g):
            if sum(1 for c in range(g) if (r, c) in self.obstacles) >= max(
                1, int(g * 0.7)
            ):
                for c in range(g):
                    if (r, c) not in self.obstacles:
                        passages.add((r, c))
        
        # Kiểm tra cột
        for c in range(g):
            if sum(1 for r in range(g) if (r, c) in self.obstacles) >= max(
                1, int(g * 0.7)
            ):
                for r in range(g):
                    if (r, c) not in self.obstacles:
                        passages.add((r, c))
        
        return passages

    def reset(self, seed=None, options=None):
        """
        Reset môi trường
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # ========== Reset state ==========
        self.coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visit_count = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self._visited_passages = set()
        
        # ========== Chọn vị trí bắt đầu ngẫu nhiên (tránh obstacles) ==========
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.obstacles:
                break
        
        self.uav_pos = [x, y]
        self.coverage[x, y] = self.visit_count[x, y] = 1.0
        self.steps = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Lấy observation hiện tại
        
        Obs gồm:
        - coverage map (100 chiều)
        - obstacles map (100 chiều)
        - visit count normalized (100 chiều)
        - UAV position normalized (2 chiều)
        - khoảng cách tới frontier (1 chiều)
        
        Returns:
            np.ndarray: Observation vector (303,)
        """
        # Coverage map
        # Obstacles map
        obs_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for o in self.obstacles:
            obs_map[o] = 1.0
        
        # Visit count normalized
        mv = self.visit_count.max()
        visit_norm = (
            (self.visit_count / mv if mv > 0 else self.visit_count).flatten()
        )
        
        # BFS distance tới frontier gần nhất
        dist = self._bfs_nearest_frontier()
        dist_norm = np.array(
            [min(dist, self.grid_size * 2) / (self.grid_size * 2)],
            dtype=np.float32,
        )
        
        # UAV position normalized
        pos = np.array(self.uav_pos, dtype=np.float32) / self.grid_size
        
        # Concatenate
        return np.concatenate(
            [
                self.coverage.flatten(),
                obs_map.flatten(),
                visit_norm,
                pos,
                dist_norm,
            ]
        ).astype(np.float32)

    def _bfs_nearest_frontier(self):
        """
        Tính khoảng cách BFS tới ranh giới (frontier) gần nhất
        
        Frontier: Ô được phát hiện nhưng chưa thăm (coverage == 0)
        nằm cạnh ô đã thăm
        
        Returns:
            int: Khoảng cách BFS
        """
        sx, sy = self.uav_pos
        if self.coverage[sx, sy] == 0:
            return 0
        
        visited = {(sx, sy)}
        queue = deque([(sx, sy, 0)])
        
        while queue:
            cx, cy, d = queue.popleft()
            
            # Kiểm tra 4 hướng
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                
                # Kiểm tra tính hợp lệ
                if (
                    0 <= nx < self.grid_size
                    and 0 <= ny < self.grid_size
                    and (nx, ny) not in visited
                    and (nx, ny) not in self.obstacles
                ):
                    # Tìm frontier
                    if self.coverage[nx, ny] == 0:
                        return d + 1
                    
                    visited.add((nx, ny))
                    queue.append((nx, ny, d + 1))
        
        return self.grid_size * 2  # Tối đa nếu không tìm thấy

    def step(self, action):
        """
        Thực hiện bước trong môi trường
        
        Args:
            action (int): Hành động (0=Up, 1=Down, 2=Left, 3=Right)
        
        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """
        x, y = self.uav_pos
        old_pos = [x, y]
        
        # ========== Dịch chuyển ==========
        if action == 0:
            x -= 1  # Up
        elif action == 1:
            x += 1  # Down
        elif action == 2:
            y -= 1  # Left
        elif action == 3:
            y += 1  # Right
        
        # Clip vào grid boundaries
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        
        reward = 0.0
        
        # ========== Kiểm tra chướng ngại vật ==========
        if (x, y) in self.obstacles:
            reward -= self.OBSTACLE_PENALTY
            x, y = old_pos  # Revert vị trí
        
        # ========== Update vị trí & visit count ==========
        self.uav_pos = [x, y]
        self.visit_count[x, y] += 1
        
        # ========== Thưởng khám phá ==========
        if self.coverage[x, y] == 0:
            self.coverage[x, y] = 1.0
            reward += self.EXPLORE_REWARD
        else:
            # Phạt khi revisit
            reward -= min(
                self.visit_count[x, y] ** 2 * self.REVISIT_MULT, self.REVISIT_CAP
            )
        
        # ========== Bonus các ô hẹp (passages) ==========
        if (x, y) in self.passages and (x, y) not in self._visited_passages:
            reward += self.PASSAGE_BONUS
            self._visited_passages.add((x, y))
        
        # ========== Thưởng frontier (ô ranh giới) ==========
        frontier = sum(
            1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (
                0 <= x + dx < self.grid_size
                and 0 <= y + dy < self.grid_size
                and self.coverage[x + dx, y + dy] == 0
                and (x + dx, y + dy) not in self.obstacles
            )
        )
        reward += frontier * self.FRONTIER_SCALE
        
        # ========== Thưởng khoảng cách tới frontier (BFS) ==========
        d = self._bfs_nearest_frontier()
        reward += (self.BFS_SCALE / (d + 1)) if d > 0 else 3.0
        
        # ========== Thưởng coverage tỷ lệ + phạt bước ==========
        coverage_ratio = float(self.coverage.sum()) / self.free_cells
        reward += coverage_ratio * self.COVERAGE_SCALE - self.STEP_PENALTY
        
        # ========== Cập nhật bước ==========
        self.steps += 1
        
        # ========== Kiểm tra termination ==========
        terminated = truncated = False
        
        if coverage_ratio >= 0.97:
            # Hoàn thành task
            reward += self.COMPLETE_BONUS
            terminated = True
        elif self.steps >= self.max_steps:
            # Hết bước
            reward += coverage_ratio * self.PARTIAL_SCALE
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """
        Render môi trường (in ra terminal)
        
        Ký hiệu:
        - '.': Ô trống
        - '#': Chướng ngại vật
        - 'U': Vị trí UAV
        """
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=object)
        
        for o in self.obstacles:
            grid[o] = "#"
        
        grid[self.uav_pos[0], self.uav_pos[1]] = "U"
        
        for row in grid:
            print(" ".join(row))
        
        coverage_pct = self.coverage.sum() / self.free_cells * 100
        print(
            f"Coverage: {coverage_pct:.1f}%  |  Steps: {self.steps} / {self.max_steps}"
        )


if __name__ == "__main__":
    # Test môi trường
    print("=" * 50)
    print("TEST: UAVPatrolEnv")
    print("=" * 50)
    
    env = UAVPatrolEnv(grid_size=10, max_steps=100)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Grid size: {env.grid_size}")
    print(f"Free cells: {env.free_cells}")
    print(f"Passages: {len(env.passages)}")
    
    # Thử một vài bước
    print("\nChạy 10 bước ngẫu nhiên...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.1f}, "
              f"coverage={env.coverage.sum()/env.free_cells*100:.1f}%")
        if terminated or truncated:
            break
    
    print("\nEnvironment test passed! ✓")
