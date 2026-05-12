"""
Visualization utilities — training curves, coverage heatmaps, and comparison plots.

Matches the plotting functions used in all 4 notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def plot_training_progress(
    rewards: List[float],
    coverages: List[float] = None,
    overlaps: List[float] = None,
    title: str = "Training Progress",
    save_path: str = None,
    window: int = 100,
):
    """
    Plot training reward/coverage curve with optional smoothing.

    Parameters
    ----------
    rewards   : raw episode rewards
    coverages : coverage percentages per episode
    overlaps  : overlap counts per episode
    title     : plot title
    save_path : save figure to file
    window    : smoothing window size
    """
    n_plots = 1 + (coverages is not None) + (overlaps is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Reward curve
    ax = axes[0]
    episodes = range(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color="#2196F3", label="Raw")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color="#2196F3", linewidth=2, label=f"Smoothed ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{title} — Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    idx = 1
    if coverages is not None:
        ax = axes[idx]; idx += 1
        ax.plot(range(len(coverages)), coverages, alpha=0.3, color="#4CAF50", label="Raw")
        if len(coverages) >= window:
            sm = np.convolve(coverages, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(coverages)), sm,
                    color="#4CAF50", linewidth=2, label=f"Smoothed ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Coverage %")
        ax.set_title(f"{title} — Coverage")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if overlaps is not None:
        ax = axes[idx]
        ax.plot(range(len(overlaps)), overlaps, alpha=0.3, color="#FF9800", label="Raw")
        if len(overlaps) >= window:
            sm = np.convolve(overlaps, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(overlaps)), sm,
                    color="#FF9800", linewidth=2, label=f"Smoothed ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Overlap count")
        ax.set_title(f"{title} — Overlap")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_coverage_heatmap(
    coverage: np.ndarray,
    obstacles: set = None,
    agent_positions: list = None,
    title: str = "Coverage Heatmap",
    save_path: str = None,
):
    """
    Plot a coverage heatmap for a single episode.

    Parameters
    ----------
    coverage        : (G, G) array with visit counts or 0/1 coverage
    obstacles       : set of (r, c) tuples
    agent_positions : list of [x, y] for current agent positions
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    display = coverage.copy().astype(float)
    if obstacles:
        for r, c in obstacles:
            display[r, c] = -1  # mark obstacles

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_under("gray")  # obstacles in gray
    im = ax.imshow(display, cmap=cmap, vmin=0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visit Count" if coverage.max() > 1 else "Covered")

    if agent_positions:
        colors = ["red", "blue", "green", "orange"]
        for i, pos in enumerate(agent_positions):
            ax.plot(pos[1], pos[0], "o", color=colors[i % len(colors)],
                    markersize=12, markeredgecolor="white", markeredgewidth=2,
                    label=f"UAV-{i}")
        ax.legend(loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    ylabel: str = "Reward",
    save_path: str = None,
    window: int = 100,
):
    """Plot comparison of multiple algorithms."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (name, values) in enumerate(results.items()):
        color = colors[i % len(colors)]
        if len(values) >= window:
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(values)), smoothed,
                    color=color, linewidth=2, label=name)
        else:
            ax.plot(values, color=color, linewidth=2, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_eval_history(
    eval_history: list,
    metric: str = "coverage",
    title: str = "Evaluation History",
    save_path: str = None,
):
    """
    Plot evaluation metrics over training updates.

    Parameters
    ----------
    eval_history : list of dicts from trainer.eval_history
    metric       : 'coverage', 'joint_reward', 'overlap', 'entropy'
    """
    updates   = [e["update"] for e in eval_history]
    map_names = [k for k in eval_history[0]["eval"].keys() if k != "mean"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, name in enumerate(map_names):
        values = [e["eval"][name][metric] for e in eval_history]
        ax.plot(updates, values, "o-", color=colors[i % len(colors)],
                linewidth=2, markersize=4, label=name)

    # Mean line
    mean_vals = [e["eval"]["mean"][metric] for e in eval_history]
    ax.plot(updates, mean_vals, "s--", color="black", linewidth=2,
            markersize=6, label="Mean")

    ax.set_xlabel("Update")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
