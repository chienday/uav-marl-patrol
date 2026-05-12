"""
UAVPatrolEnvIPPO — Research-quality Multi-Agent Environment.

Architecture: MAPPO-ready (CTDE) with a centralized critic hook.

Design principles
-----------------
* Clean separation: reward logic | observation logic | collision logic
* Simultaneous update — zero sequential bias between agents
* Local 9x9 partial observation (generalizes better than full map)
* Shared world state: coverage_map, visit_count, obstacle_map
* Gymnasium-compatible API with get_global_state() for MAPPO/VDPPO
* Used by IPPO, MAPPO, and VDPPO training notebooks.

Observation per agent (9x9 window, flattened):
    coverage_window  81
    obstacle_window  81
    visit_norm       81
    own_pos_norm      2
    other_pos_norm    2   (relative displacement, clipped to [-1,1])
    bfs_dist_norm     1
    -----------------
    Total:          248 features

Action space (per agent): Discrete(4) - Up / Down / Left / Right

Reward dict: {0: r0, 1: r1}
Obs   dict:  {0: obs0, 1: obs1}
"""

import gymnasium as gym
import numpy as np
import json
from collections import deque
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from .reward import (
    NUM_AGENTS, DIRECTIONS, OBS_RADIUS, OBS_WIN, LOCAL_SIZE,
    RewardConfig,
    compute_explore_reward,
    compute_frontier_reward,
    compute_bfs_reward,
    compute_team_coverage_bonus,
    check_collision,
    check_cross_collision,
    check_overlap,
    apply_collision_penalties,
    build_obs,
)


class UAVPatrolEnvIPPO(gym.Env):
    """
    Multi-agent UAV patrol environment for 2 UAVs.

    Key properties
    --------------
    * Simultaneous moves - no sequential bias
    * 9x9 local partial observation per agent
    * Shared coverage/visit/obstacle maps
    * Clean reward/observation/collision modules
    * get_global_state() for a MAPPO/VDPPO central critic
    * Gymnasium API: step / reset / render / close
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 600,
        map_file: Optional[str] = None,
        num_agents: int = NUM_AGENTS,
        reward_cfg: Optional[RewardConfig] = None,
        enable_diversity_reward: bool = False,
        obs_radius: int = OBS_RADIUS,
    ):
        super().__init__()

        self.reward_cfg = reward_cfg or RewardConfig()
        self.enable_diversity_reward = enable_diversity_reward

        # -- Load map config --------------------------------------------------
        if map_file:
            with open(map_file) as f:
                config = json.load(f)
            grid_size           = config.get("grid_size", grid_size)
            max_steps           = config.get("max_steps", max_steps)
            self.start_position = config.get("start_position", [0, 0])
            raw_obstacles       = config.get("obstacles", [])
        else:
            self.start_position = [0, 0]
            raw_obstacles       = []

        self.grid_size  = grid_size
        self.max_steps  = max_steps
        self.num_agents = num_agents
        self.obstacles  = set(tuple(o) for o in raw_obstacles)
        self.free_cells = grid_size * grid_size - len(self.obstacles)
        self.passages   = self._detect_passages()

        # -- Pre-build static obstacle map (never changes) --------------------
        self._obstacle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for o in self.obstacles:
            self._obstacle_map[o] = 1.0

        # -- Spaces -----------------------------------------------------------
        # obs: local_size*3 (cov + obs + visit) + 2 (own) + 2 (other_rel) + 1 (bfs)
        self.obs_radius = obs_radius
        local_size = (2 * obs_radius + 1) ** 2
        obs_size = local_size * 3 + 5
        single_obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {i: single_obs_space for i in range(self.num_agents)}
        )
        self.action_space = spaces.Dict(
            {i: spaces.Discrete(4) for i in range(self.num_agents)}
        )
        # Expose for convenience
        self.single_observation_space = single_obs_space
        self.single_action_space      = spaces.Discrete(4)

        # -- Mutable state (initialized in reset) -----------------------------
        self.agent_positions:     List[List[int]] = [[0, 0], [0, 0]]
        self.coverage:            np.ndarray      = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.visit_count:         np.ndarray      = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._visited_passages:   set             = set()
        self.steps:               int             = 0
        self.trajectory:          List[List[List[int]]] = [[], []]  # per-agent history

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _detect_passages(self) -> set:
        """Detect bottleneck cells (>=70 percent of row/col blocked)."""
        passages, g = set(), self.grid_size
        threshold = max(1, int(g * 0.7))
        for r in range(g):
            if sum(1 for c in range(g) if (r, c) in self.obstacles) >= threshold:
                for c in range(g):
                    if (r, c) not in self.obstacles:
                        passages.add((r, c))
        for c in range(g):
            if sum(1 for r in range(g) if (r, c) in self.obstacles) >= threshold:
                for r in range(g):
                    if (r, c) not in self.obstacles:
                        passages.add((r, c))
        return passages

    def _random_free_cell(self) -> List[int]:
        while True:
            x = int(self.np_random.integers(0, self.grid_size))
            y = int(self.np_random.integers(0, self.grid_size))
            if (x, y) not in self.obstacles:
                return [x, y]

    def _bfs_nearest_frontier(self, agent_id: int) -> int:
        """BFS distance from agent_id position to nearest uncovered cell."""
        sx, sy = self.agent_positions[agent_id]
        if self.coverage[sx, sy] == 0:
            return 0
        visited = {(sx, sy)}
        queue   = deque([(sx, sy, 0)])
        while queue:
            cx, cy, d = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.grid_size
                        and 0 <= ny < self.grid_size
                        and (nx, ny) not in visited
                        and (nx, ny) not in self.obstacles):
                    if self.coverage[nx, ny] == 0:
                        return d + 1
                    visited.add((nx, ny))
                    queue.append((nx, ny, d + 1))
        return self.grid_size * 2

    def _try_move(self, agent_id: int, action: int) -> Tuple[List[int], bool]:
        """Compute next position without committing. Returns (new_pos, hit_obstacle)."""
        x, y   = self.agent_positions[agent_id]
        dx, dy = DIRECTIONS[action]
        nx     = int(np.clip(x + dx, 0, self.grid_size - 1))
        ny     = int(np.clip(y + dy, 0, self.grid_size - 1))
        if (nx, ny) in self.obstacles:
            return [x, y], True
        return [nx, ny], False

    # ---------------------------------------------------------------------
    # Observation API
    # ---------------------------------------------------------------------

    def _get_obs_for_agent(self, agent_id: int) -> np.ndarray:
        bfs_dist = self._bfs_nearest_frontier(agent_id)
        return build_obs(
            coverage        = self.coverage,
            obstacle_map    = self._obstacle_map,
            visit_count     = self.visit_count,
            agent_positions = self.agent_positions,
            agent_id        = agent_id,
            bfs_dist        = bfs_dist,
            grid_size       = self.grid_size,
            obs_radius      = self.obs_radius,
        )

    def _get_obs(self) -> Dict[int, np.ndarray]:
        return {i: self._get_obs_for_agent(i) for i in range(self.num_agents)}

    # ---------------------------------------------------------------------
    # Global state for MAPPO/VDPPO central critic
    # ---------------------------------------------------------------------

    def get_global_state(self) -> np.ndarray:
        """
        Concatenated global state for a central critic.

        Layout:
            coverage_flat     G*G
            obstacle_flat     G*G
            visit_norm_flat   G*G
            agent0_pos_norm   2
            agent1_pos_norm   2
            ----------------------
            Total             G*G*3 + 4
        """
        mv = self.visit_count.max()
        vn = (self.visit_count / mv if mv > 0 else self.visit_count).flatten()
        pos0 = np.array(self.agent_positions[0], dtype=np.float32) / self.grid_size
        pos1 = np.array(self.agent_positions[1], dtype=np.float32) / self.grid_size
        return np.concatenate([
            self.coverage.flatten(),
            self._obstacle_map.flatten(),
            vn,
            pos0,
            pos1,
        ]).astype(np.float32)

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[int, np.ndarray], dict]:
        super().reset(seed=seed)

        self.coverage          = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visit_count       = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._visited_passages = set()
        self.steps             = 0
        self.trajectory        = [[], []]

        # Spawn agents at distinct free cells with minimum separation
        min_dist = max(1, self.grid_size // 2)
        while True:
            pos0 = self._random_free_cell()
            pos1 = self._random_free_cell()
            if pos0 != pos1:
                dist = abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])
                if dist >= min_dist:
                    break

        self.agent_positions = [pos0, pos1]

        for pos in self.agent_positions:
            r, c = pos
            self.coverage[r, c]    = 1.0
            self.visit_count[r, c] = 1.0

        for i, pos in enumerate(self.agent_positions):
            self.trajectory[i].append(pos.copy())

        return self._get_obs(), {}

    def step(
        self,
        actions: List[int],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, dict]:
        """
        Simultaneous step for all agents.

        Parameters
        ----------
        actions : [a0, a1]  - one action per agent

        Returns
        -------
        obs        : {agent_id: ndarray}
        rewards    : {agent_id: float}
        terminated : bool
        truncated  : bool
        info       : dict
        """
        assert len(actions) == self.num_agents, \
            f"Expected {self.num_agents} actions, got {len(actions)}"

        cfg           = self.reward_cfg
        old_positions = [p.copy() for p in self.agent_positions]
        rewards       = {i: 0.0 for i in range(self.num_agents)}

        # -- Phase 1: compute new positions simultaneously --------------------
        new_positions = []
        for i in range(self.num_agents):
            new_pos, hit_obs = self._try_move(i, int(actions[i]))
            new_positions.append(new_pos)
            if hit_obs:
                rewards[i] -= cfg.OBSTACLE_PENALTY

        # -- Phase 2: commit all positions -----------------------------------
        self.agent_positions = [p.copy() for p in new_positions]

        # -- Phase 3: collision / overlap penalties --------------------------
        apply_collision_penalties(cfg, rewards, old_positions, new_positions)

        # Anti-stacking: revert both agents to old positions on head-on collision
        if check_collision(new_positions[0], new_positions[1]):
            self.agent_positions = [p.copy() for p in old_positions]
            new_positions = [p.copy() for p in old_positions]

        # -- Phase 3b: VDPPO diversity reward (optional) ---------------------
        if self.enable_diversity_reward:
            p0, p1 = self.agent_positions[0], self.agent_positions[1]
            _manhattan = abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])
            if _manhattan <= 2:
                rewards[0] -= 3.0
                rewards[1] -= 3.0

        # -- Phase 4: per-agent rewards + update shared maps -----------------
        prev_coverage_sum = float(self.coverage.sum())

        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            self.trajectory[i].append([x, y])

            # Update shared maps
            self.visit_count[x, y] += 1

            # Explore / revisit
            rewards[i] += compute_explore_reward(cfg, self.coverage, self.visit_count, x, y)
            self.coverage[x, y] = 1.0  # mark after reward decision

            # Passage bonus (one-time per passage cell, shared set)
            if (x, y) in self.passages and (x, y) not in self._visited_passages:
                rewards[i] += cfg.PASSAGE_BONUS
                self._visited_passages.add((x, y))

            # Frontier reward
            rewards[i] += compute_frontier_reward(cfg, self.coverage, self.obstacles, self.grid_size, x, y)

            # BFS reward
            bfs_dist = self._bfs_nearest_frontier(i)
            rewards[i] += compute_bfs_reward(cfg, bfs_dist)

        # -- Phase 5: coverage reward + team bonus ---------------------------
        curr_coverage_sum = float(self.coverage.sum())
        coverage_ratio    = curr_coverage_sum / self.free_cells
        team_bonus        = compute_team_coverage_bonus(cfg, prev_coverage_sum, curr_coverage_sum)

        for i in range(self.num_agents):
            rewards[i] += coverage_ratio * cfg.COVERAGE_SCALE - cfg.STEP_PENALTY
            rewards[i] += team_bonus

        # -- Phase 6: terminal conditions ------------------------------------
        self.steps   += 1
        terminated    = False
        truncated     = False

        if coverage_ratio >= 0.97:
            for i in range(self.num_agents):
                rewards[i] += cfg.COMPLETE_BONUS
            terminated = True
        elif self.steps >= self.max_steps:
            for i in range(self.num_agents):
                rewards[i] += coverage_ratio * cfg.PARTIAL_SCALE
            truncated = True

        info = {
            "coverage_ratio":  coverage_ratio,
            "steps":           self.steps,
            "agent_positions": [p.copy() for p in self.agent_positions],
            "team_bonus":      team_bonus,
        }
        return self._get_obs(), rewards, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the grid.
        Symbols:
            .  uncovered free cell
            *  covered cell
            #  obstacle
            0  UAV-0
            1  UAV-1
            X  collision (both on same cell)
        """
        g     = self.grid_size
        # Coverage background
        grid  = np.where(self.coverage > 0, "*", ".").astype(object)
        for o in self.obstacles:
            grid[o] = "#"

        pos0 = tuple(self.agent_positions[0])
        pos1 = tuple(self.agent_positions[1])
        if pos0 == pos1:
            grid[pos0] = "X"
        else:
            grid[pos0] = "0"
            grid[pos1] = "1"

        lines = []
        # Column header
        header = "   " + " ".join(f"{c:1d}" for c in range(g))
        lines.append(header)
        for r in range(g):
            row_str = f"{r:2d} " + " ".join(grid[r])
            lines.append(row_str)

        coverage_ratio = self.coverage.sum() / self.free_cells
        lines.append(
            f"Coverage: {coverage_ratio:.1%}  Steps: {self.steps}  "
            f"UAV0: {self.agent_positions[0]}  UAV1: {self.agent_positions[1]}"
        )

        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output

    def render_trajectory(self) -> None:
        """
        Print trajectory of both agents overlaid on coverage map.
        Letters show visit-order (capped at 25).
        """
        g    = self.grid_size
        grid = np.where(self.coverage > 0, ".", " ").astype(object)
        for o in self.obstacles:
            grid[o] = "#"

        symbols_0 = "abcdefghijklmnopqrstuvwxyz"
        symbols_1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for step_i, pos in enumerate(self.trajectory[0]):
            grid[pos[0], pos[1]] = symbols_0[min(step_i, 25)]
        for step_i, pos in enumerate(self.trajectory[1]):
            grid[pos[0], pos[1]] = symbols_1[min(step_i, 25)]

        print("\nTrajectory  (agent-0: lowercase, agent-1: UPPERCASE)")
        for r in range(g):
            print(" ".join(grid[r]))

    def close(self) -> None:
        pass
