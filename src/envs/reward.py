"""
Reward & Collision module — RewardConfig + compute_* + collision functions.

Provides configurable reward shaping for UAV patrol missions.
Shared by IPPO / MAPPO / VDPPO environments.

Reward scale (normalised vs original single-agent PPO):
    COMPLETE_BONUS  3000 -> 300 (IPPO) / 100 (MAPPO/VDPPO)
    PARTIAL_SCALE    600 ->  60 (IPPO) /  20 (MAPPO/VDPPO)
    PASSAGE_BONUS     80 ->   8 (IPPO) /   3 (MAPPO/VDPPO)
All other values tuned proportionally.
"""

import numpy as np
from typing import Dict, List, Tuple


# -- Constants -----------------------------------------------------------------

NUM_AGENTS = 2
DIRECTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # U D L R
OBS_RADIUS = 4          # 9x9 local window  (2*OBS_RADIUS+1 = 9)
OBS_WIN    = 2 * OBS_RADIUS + 1
LOCAL_SIZE = OBS_WIN * OBS_WIN  # 81 (9x9)


# -----------------------------------------------------------------------------
# Reward configuration
# -----------------------------------------------------------------------------

class RewardConfig:
    """
    Configuration for reward computation.

    Default values match the MAPPO notebook.
    IPPO and VDPPO have their own subclasses with different scales.
    """
    # -- Per-step signals (tuned for return range ~[-100, 300]) ----------------
    EXPLORE_REWARD    = 5.0     # new cell: dominant early signal
    OBSTACLE_PENALTY  = 2.0     # hitting wall
    COVERAGE_SCALE    = 3.0     # coverage_ratio * scale added every step
    FRONTIER_SCALE    = 1.5     # uncovered neighbours bonus
    REVISIT_MULT      = 0.5     # visit_count^2 * mult penalty
    REVISIT_CAP       = 5.0     # max revisit penalty per step
    BFS_SCALE         = 1.0     # 1/(bfs_dist+1) reward toward frontier
    STEP_PENALTY      = 0.05    # small time pressure
    # -- Terminal signals -----------------------------------------------------
    COMPLETE_BONUS    = 100.0   # 97%+ coverage achieved
    PARTIAL_SCALE     = 20.0    # coverage_ratio * scale at timeout
    PASSAGE_BONUS     = 3.0     # one-time bottleneck crossing
    # -- Coordination signals -------------------------------------------------
    COLLISION_PENALTY = 3.0     # head-on or cross collision
    OVERLAP_PENALTY   = 1.0     # adjacent cell proximity penalty
    TEAM_ALPHA        = 2.0     # delta_coverage * alpha (shared CTDE credit)


class IPPORewardConfig(RewardConfig):
    """
    Reward configuration for IPPO (Independent PPO).

    Uses larger reward scale because SB3 PPO applies VecNormalize
    which auto-normalises returns. Raw magnitudes are ~10x MAPPO.
    OBS_RADIUS = 2 (5x5 window, 80 features) instead of 4 (9x9, 248).
    """
    EXPLORE_REWARD    = 60.0
    OBSTACLE_PENALTY  = 20.0
    COVERAGE_SCALE    = 25.0
    FRONTIER_SCALE    = 12.0
    REVISIT_MULT      = 1.5
    REVISIT_CAP       = 25.0
    BFS_SCALE         = 8.0
    STEP_PENALTY      = 0.1
    COMPLETE_BONUS    = 300.0     # was 3000 in PPO single-agent
    PARTIAL_SCALE     = 60.0     # was  600 in PPO single-agent
    PASSAGE_BONUS     = 8.0      # was   80 in PPO single-agent
    COLLISION_PENALTY = 15.0
    OVERLAP_PENALTY   = 5.0
    TEAM_ALPHA        = 0.3


class VDPPORewardConfig(RewardConfig):
    """
    Reward configuration optimised for VDPPO.

    Higher penalties to encourage better agent coordination.
    """
    REVISIT_MULT      = 1.5
    REVISIT_CAP       = 35.0
    STEP_PENALTY      = 1.5
    COLLISION_PENALTY  = 25.0
    OVERLAP_PENALTY    = 12.0
    TEAM_ALPHA         = 2.5


# -----------------------------------------------------------------------------
# Reward logic (pure functions)
# -----------------------------------------------------------------------------

def compute_explore_reward(
    cfg: RewardConfig,
    coverage: np.ndarray,
    visit_count: np.ndarray,
    x: int,
    y: int,
) -> float:
    """Explore reward or revisit penalty for a single cell visit."""
    if coverage[x, y] == 0:
        return cfg.EXPLORE_REWARD
    penalty = min(visit_count[x, y] ** 2 * cfg.REVISIT_MULT, cfg.REVISIT_CAP)
    return -penalty


def compute_frontier_reward(
    cfg: RewardConfig,
    coverage: np.ndarray,
    obstacles: set,
    grid_size: int,
    x: int,
    y: int,
) -> float:
    """Reward proportional to uncovered neighbors."""
    frontier = sum(
        1
        for dx, dy in DIRECTIONS
        if (0 <= x + dx < grid_size
            and 0 <= y + dy < grid_size
            and coverage[x + dx, y + dy] == 0
            and (x + dx, y + dy) not in obstacles)
    )
    return frontier * cfg.FRONTIER_SCALE


def compute_bfs_reward(cfg: RewardConfig, bfs_dist: int) -> float:
    """Reward for being close to the nearest frontier."""
    if bfs_dist == 0:
        return 3.0
    return cfg.BFS_SCALE / (bfs_dist + 1)


def compute_team_coverage_bonus(
    cfg: RewardConfig,
    prev_coverage_sum: float,
    curr_coverage_sum: float,
) -> float:
    """Shared bonus proportional to new cells covered this step."""
    delta = curr_coverage_sum - prev_coverage_sum
    return delta * cfg.TEAM_ALPHA


# -----------------------------------------------------------------------------
# Collision logic (pure functions)
# -----------------------------------------------------------------------------

def check_collision(pos0: List[int], pos1: List[int]) -> bool:
    """True if both agents land on the same cell."""
    return pos0[0] == pos1[0] and pos0[1] == pos1[1]


def check_cross_collision(
    old0: List[int], new0: List[int],
    old1: List[int], new1: List[int],
) -> bool:
    """True if agents swap cells (cross-collision)."""
    return (new0[0] == old1[0] and new0[1] == old1[1]
            and new1[0] == old0[0] and new1[1] == old0[1])


def check_overlap(pos0: List[int], pos1: List[int]) -> bool:
    """True if Manhattan distance < 2 (adjacent but not same cell)."""
    d = abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])
    return 0 < d < 2


def apply_collision_penalties(
    cfg: RewardConfig,
    rewards: Dict[int, float],
    old_positions: List[List[int]],
    new_positions: List[List[int]],
) -> None:
    """Apply collision/overlap penalties in-place."""
    p0, p1   = new_positions[0], new_positions[1]
    op0, op1 = old_positions[0], old_positions[1]

    if check_collision(p0, p1):
        rewards[0] -= cfg.COLLISION_PENALTY
        rewards[1] -= cfg.COLLISION_PENALTY

    if check_cross_collision(op0, p0, op1, p1):
        rewards[0] -= cfg.COLLISION_PENALTY
        rewards[1] -= cfg.COLLISION_PENALTY

    if check_overlap(p0, p1):
        rewards[0] -= cfg.OVERLAP_PENALTY
        rewards[1] -= cfg.OVERLAP_PENALTY


# -----------------------------------------------------------------------------
# Observation logic (pure functions)
# -----------------------------------------------------------------------------

def extract_local_window(
    grid: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Extract a (2r+1) x (2r+1) window centered at (cx, cy).
    Pads with pad_value outside grid boundaries.
    """
    g  = grid.shape[0]
    w  = 2 * radius + 1
    out = np.full((w, w), pad_value, dtype=np.float32)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = cx + di, cy + dj
            if 0 <= ni < g and 0 <= nj < g:
                out[di + radius, dj + radius] = grid[ni, nj]
    return out.flatten()


def build_obs(
    coverage: np.ndarray,
    obstacle_map: np.ndarray,
    visit_count: np.ndarray,
    agent_positions: List[List[int]],
    agent_id: int,
    bfs_dist: int,
    grid_size: int,
    obs_radius: int = OBS_RADIUS,
) -> np.ndarray:
    """
    Build the local-window observation vector for one agent.

    With obs_radius=4 (default, MAPPO/VDPPO):  9x9 window -> 248 features
    With obs_radius=2 (IPPO):                   5x5 window ->  80 features

    Layout (total = local_size*3 + 2 + 2 + 1):
        [0     : L  ]  coverage_window
        [L     : 2L ]  obstacle_window  (1=obstacle)
        [2L    : 3L ]  visit_norm
        [3L    : 3L+2] own_pos_norm     (x/G, y/G)
        [3L+2  : 3L+4] other_rel_pos    (dx/(2G), dy/(2G) clipped to [-1,1])
        [3L+4  : 3L+5] bfs_dist_norm    (bfs / (2G))
    """
    ax, ay   = agent_positions[agent_id]
    oid      = 1 - agent_id
    ox, oy   = agent_positions[oid]

    mv = visit_count.max()
    vn = visit_count / mv if mv > 0 else visit_count.copy()

    cov_win  = extract_local_window(coverage,     ax, ay, obs_radius, 0.0)
    obs_win  = extract_local_window(obstacle_map, ax, ay, obs_radius, 1.0)  # pad=1: wall outside
    vis_win  = extract_local_window(vn,           ax, ay, obs_radius, 0.0)

    own_pos  = np.array([ax / grid_size, ay / grid_size], dtype=np.float32)

    # Relative position of other agent, normalized and clipped
    rel      = np.clip(
        [(ox - ax) / (grid_size * 2), (oy - ay) / (grid_size * 2)],
        -1.0, 1.0,
    )
    other_pos = np.array(rel, dtype=np.float32)

    bfs_norm  = np.array(
        [min(bfs_dist, grid_size * 2) / (grid_size * 2)],
        dtype=np.float32,
    )

    return np.concatenate([cov_win, obs_win, vis_win, own_pos, other_pos, bfs_norm]).astype(np.float32)
