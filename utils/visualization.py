# -*- coding: utf-8 -*-
"""
Hàm visualization

Tác giả: [DATN Project]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import Optional, List
from ..environments import UAVPatrolEnv


def plot_trajectory(
    env: UAVPatrolEnv,
    trajectory: np.ndarray,
    coverage: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Vẽ quỹ đạo di chuyển UAV
    
    Args:
        env (UAVPatrolEnv): Môi trường
        trajectory (np.ndarray): Quỹ đạo (N, 2)
        coverage (float): Coverage %
        save_path (str): Nơi lưu ảnh
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    G = env.grid_size
    
    # Nền: Ô thăm / chưa thăm
    for r in range(G):
        for c in range(G):
            if (r, c) not in env.obstacles:
                visited = env.coverage[r, c] > 0
                color = "#D6EAF8" if visited else "#FFF9C4"
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=color,
                        edgecolor="gray",
                        linewidth=0.5,
                    )
                )
    
    # Obstacles
    for (ox, oy) in env.obstacles:
        ax.add_patch(
            Rectangle(
                (oy - 0.5, ox - 0.5), 1, 1,
                facecolor="#2C2C2C",
                edgecolor="#111111",
                linewidth=1.2,
            )
        )
    
    # Quỹ đạo - viền trắng
    for i in range(len(trajectory) - 1):
        ax.plot(
            trajectory[i : i + 2, 1],
            trajectory[i : i + 2, 0],
            color="white",
            linewidth=5.5,
            solid_capstyle="round",
            zorder=4,
        )
    
    # Quỹ đạo - màu
    for i in range(len(trajectory) - 1):
        ax.plot(
            trajectory[i : i + 2, 1],
            trajectory[i : i + 2, 0],
            color="#378ADD",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=5,
        )
    
    # Start / End
    ax.scatter(
        trajectory[0, 1],
        trajectory[0, 0],
        s=300,
        c="red",
        edgecolors="white",
        linewidths=2.5,
        zorder=11,
    )
    ax.text(
        trajectory[0, 1],
        trajectory[0, 0],
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        zorder=12,
    )
    
    ax.scatter(
        trajectory[-1, 1],
        trajectory[-1, 0],
        s=350,
        c="yellow",
        marker="*",
        edgecolors="darkorange",
        linewidths=1.5,
        zorder=11,
    )
    
    # Labels
    ax.set_title(
        f"UAV Trajectory\nCoverage: {coverage:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Trajectory saved: {save_path}")
    
    plt.show()


def plot_visit_heatmap(
    visit_count: np.ndarray,
    obstacles: set,
    save_path: Optional[str] = None,
) -> None:
    """
    Vẽ heatmap số lần thăm
    
    Args:
        visit_count (np.ndarray): Visit count map
        obstacles (set): Tập hợp vị trí obstacles
        save_path (str): Nơi lưu ảnh
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    masked = np.ma.masked_where(
        np.array(
            [
                [1 if (r, c) in obstacles else 0 for c in range(visit_count.shape[1])]
                for r in range(visit_count.shape[0])
            ]
        ),
        visit_count,
    )
    
    im = ax.imshow(masked, cmap="YlOrRd", origin="upper")
    
    # Obstacles
    for (ox, oy) in obstacles:
        ax.add_patch(
            Rectangle((oy - 0.5, ox - 0.5), 1, 1, color="#1a1a1a", alpha=0.9)
        )
    
    # Text
    for r in range(visit_count.shape[0]):
        for c in range(visit_count.shape[1]):
            if (r, c) not in obstacles and visit_count[r, c] > 0:
                ax.text(
                    c,
                    r,
                    int(visit_count[r, c]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                    if visit_count[r, c] < visit_count.max() * 0.7
                    else "white",
                )
    
    ax.set_title("Visit Count Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Số lần thăm")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Heatmap saved: {save_path}")
    
    plt.show()


def plot_learning_curve(
    timesteps: List[int],
    rewards: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Vẽ learning curve
    
    Args:
        timesteps (List[int]): Danh sách timesteps
        rewards (List[float]): Danh sách rewards trung bình
        save_path (str): Nơi lưu ảnh
    """
    timesteps_m = np.array(timesteps) / 1_000_000
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(timesteps_m, rewards, color="#378ADD", lw=2.5, label="Mean Reward")
    ax.fill_between(
        timesteps_m,
        np.array(rewards) - np.std(rewards),
        np.array(rewards) + np.std(rewards),
        alpha=0.2,
        color="#378ADD",
    )
    
    ax.set_xlabel("Timesteps (triệu)", fontsize=12)
    ax.set_ylabel("Trung bình Reward", fontsize=12)
    ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Learning curve saved: {save_path}")
    
    plt.show()
