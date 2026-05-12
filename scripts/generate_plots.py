"""Generate separate plots for each algorithm from log data."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'log')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {'simple': '#2196F3', 'mixed': '#4CAF50', 'bottleneck': '#FF9800', 'mean': '#E91E63'}

def smooth(data, w=10):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode='valid').tolist()

# ============================================================
# PPO
# ============================================================
def plot_ppo():
    d = os.path.join(LOG_DIR, 'ppo_single_uav')
    if not os.path.exists(d): return
    out = os.path.join(OUT_DIR, 'ppo')
    os.makedirs(out, exist_ok=True)
    # Copy existing plots
    for f in os.listdir(d):
        if f.endswith('.png'):
            src = os.path.join(d, f)
            dst = os.path.join(out, f)
            if not os.path.exists(dst):
                import shutil; shutil.copy2(src, dst)
    print(f'  PPO: copied {len([f for f in os.listdir(d) if f.endswith(".png")])} existing plots')

# ============================================================
# IPPO
# ============================================================
def plot_ippo():
    d = os.path.join(LOG_DIR, 'ippo_experiment')
    out = os.path.join(OUT_DIR, 'ippo')
    os.makedirs(out, exist_ok=True)

    # 1. Episode logs (coverage, overlap, reward per map)
    with open(os.path.join(d, 'episode_logs.json')) as f:
        ep = json.load(f)

    # Coverage bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    maps = list(ep.keys())
    means = [np.mean(ep[m]['coverage']) for m in maps]
    stds = [np.std(ep[m]['coverage']) for m in maps]
    bars = ax.bar(maps, means, yerr=stds, color=[COLORS[m] for m in maps], capsize=5, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Coverage %'); ax.set_title('IPPO - Final Eval Coverage (10 episodes)')
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for b, m in zip(bars, means): ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{m:.1f}%', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(out, 'ippo_coverage_bar.png'), dpi=150); plt.close()

    # Overlap bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    means_o = [np.mean(ep[m]['overlap']) for m in maps]
    ax.bar(maps, means_o, color=[COLORS[m] for m in maps], edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Avg Overlaps'); ax.set_title('IPPO - Overlap per Map')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'ippo_overlap_bar.png'), dpi=150); plt.close()

    # Reward box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    data_r = [ep[m]['joint_reward'] for m in maps]
    bp = ax.boxplot(data_r, labels=maps, patch_artist=True)
    for patch, m in zip(bp['boxes'], maps): patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
    ax.set_ylabel('Joint Reward'); ax.set_title('IPPO - Reward Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'ippo_reward_box.png'), dpi=150); plt.close()

    # 2. Eval history (per-round)
    with open(os.path.join(d, 'eval_history.json')) as f:
        hist = json.load(f)
    rounds = [h['round'] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, agent in zip(axes, ['agent0', 'agent1', 'joint']):
        for m in ['simple', 'mixed', 'bottleneck']:
            vals = [h[agent][m] for h in hist]
            ax.plot(rounds, vals, 'o-', color=COLORS[m], label=m, linewidth=2, markersize=8)
        ax.set_xlabel('Round'); ax.set_ylabel('Coverage %'); ax.set_title(f'IPPO - {agent}')
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
    plt.suptitle('IPPO Training Progress (per round)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(out, 'ippo_training_rounds.png'), dpi=150); plt.close()

    # 3. Final eval results matrix
    with open(os.path.join(d, 'final_eval_results.json')) as f:
        final = json.load(f)
    train_maps = list(final.keys())
    eval_maps = [k for k in final[train_maps[0]].keys() if k != 'mean']
    matrix = np.array([[final[t][e] for e in eval_maps] for t in train_maps])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=50, vmax=100)
    ax.set_xticks(range(len(eval_maps))); ax.set_xticklabels(eval_maps)
    ax.set_yticks(range(len(train_maps))); ax.set_yticklabels(train_maps)
    ax.set_xlabel('Eval Map'); ax.set_ylabel('Train Map')
    for i in range(len(train_maps)):
        for j in range(len(eval_maps)):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontweight='bold', fontsize=12)
    plt.colorbar(im, label='Coverage %')
    ax.set_title('IPPO - Cross-Map Generalization')
    plt.tight_layout(); plt.savefig(os.path.join(out, 'ippo_cross_map.png'), dpi=150); plt.close()

    # Copy original plots too
    plots_dir = os.path.join(d, 'plots')
    if os.path.exists(plots_dir):
        import shutil
        for f in os.listdir(plots_dir):
            if f.endswith('.png'):
                shutil.copy2(os.path.join(plots_dir, f), os.path.join(out, f))

    print(f'  IPPO: generated 4 new plots + copied originals')

# ============================================================
# MAPPO
# ============================================================
def plot_mappo():
    d = os.path.join(LOG_DIR, 'mappo_experiment')
    out = os.path.join(OUT_DIR, 'mappo')
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(d, 'eval_history.json')) as f:
        hist = json.load(f)

    updates = [h['update'] for h in hist]

    # 1. Coverage over training
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in ['simple', 'mixed', 'bottleneck']:
        vals = [h['eval'][m]['coverage'] for h in hist]
        ax.plot(updates, vals, alpha=0.3, color=COLORS[m])
        ax.plot(updates[9:], smooth(vals, 10), color=COLORS[m], linewidth=2, label=m)
    mean_vals = [h['eval']['mean']['coverage'] for h in hist]
    ax.plot(updates, mean_vals, '--', color='black', linewidth=2, label='mean')
    ax.set_xlabel('Update'); ax.set_ylabel('Coverage %'); ax.set_title('MAPPO - Coverage vs Training Steps')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'mappo_coverage_curve.png'), dpi=150); plt.close()

    # 2. Reward curve
    fig, ax = plt.subplots(figsize=(12, 5))
    rewards = [h['train_joint_reward'] for h in hist]
    ax.plot(updates, rewards, alpha=0.3, color='#2196F3')
    if len(rewards) > 10:
        ax.plot(updates[9:], smooth(rewards, 10), color='#2196F3', linewidth=2, label='Smoothed')
    ax.set_xlabel('Update'); ax.set_ylabel('Joint Reward'); ax.set_title('MAPPO - Training Reward')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'mappo_reward_curve.png'), dpi=150); plt.close()

    # 3. Overlap + Entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    overlaps = [h['eval']['mean']['overlap'] for h in hist]
    ax1.plot(updates, overlaps, alpha=0.3, color='#FF9800')
    if len(overlaps) > 10: ax1.plot(updates[9:], smooth(overlaps, 10), color='#FF9800', linewidth=2)
    ax1.set_xlabel('Update'); ax1.set_ylabel('Overlap Count'); ax1.set_title('MAPPO - Overlap'); ax1.grid(alpha=0.3)

    entropies = [h['entropy'] for h in hist]
    ent_coefs = [h['ent_coef'] for h in hist]
    ax2.plot(updates, entropies, color='#9C27B0', linewidth=2, label='Policy Entropy')
    ax2r = ax2.twinx()
    ax2r.plot(updates, ent_coefs, '--', color='#FF5722', linewidth=1.5, label='Ent Coef')
    ax2.set_xlabel('Update'); ax2.set_ylabel('Entropy'); ax2r.set_ylabel('Ent Coef')
    ax2.set_title('MAPPO - Entropy Decay'); ax2.grid(alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'mappo_overlap_entropy.png'), dpi=150); plt.close()

    # 4. Actor/Critic loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(updates, [h['actor_loss'] for h in hist], color='#2196F3', linewidth=1.5)
    ax1.set_xlabel('Update'); ax1.set_ylabel('Actor Loss'); ax1.set_title('MAPPO - Actor Loss'); ax1.grid(alpha=0.3)
    ax2.plot(updates, [h['critic_loss'] for h in hist], color='#4CAF50', linewidth=1.5)
    ax2.set_xlabel('Update'); ax2.set_ylabel('Critic Loss'); ax2.set_title('MAPPO - Critic Loss'); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'mappo_losses.png'), dpi=150); plt.close()

    # 5. Final coverage bar
    last = hist[-1]['eval']
    fig, ax = plt.subplots(figsize=(8, 5))
    maps = ['simple', 'mixed', 'bottleneck']
    vals = [last[m]['coverage'] for m in maps]
    bars = ax.bar(maps, vals, color=[COLORS[m] for m in maps], edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coverage %'); ax.set_title('MAPPO - Final Coverage'); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'mappo_final_bar.png'), dpi=150); plt.close()

    # Copy originals
    import shutil
    for f in os.listdir(d):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(d, f), os.path.join(out, f))

    print(f'  MAPPO: generated 5 new plots + copied originals')

# ============================================================
# VDPPO
# ============================================================
def plot_vdppo():
    d = os.path.join(LOG_DIR, 'vdppo_experiment')
    out = os.path.join(OUT_DIR, 'vdppo')
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(d, 'eval_history.json')) as f:
        hist = json.load(f)
    updates = [h['update'] for h in hist]

    # 1. Coverage curve
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in ['simple', 'mixed', 'bottleneck']:
        vals = [h['eval'][m]['coverage'] for h in hist]
        ax.plot(updates, vals, alpha=0.3, color=COLORS[m])
        if len(vals) > 10: ax.plot(updates[9:], smooth(vals, 10), color=COLORS[m], linewidth=2, label=m)
    mean_vals = [h['eval']['mean']['coverage'] for h in hist]
    ax.plot(updates, mean_vals, '--', color='black', linewidth=2, label='mean')
    ax.set_xlabel('Update'); ax.set_ylabel('Coverage %'); ax.set_title('VDPPO - Coverage vs Training Steps')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'vdppo_coverage_curve.png'), dpi=150); plt.close()

    # 2. Reward curve
    fig, ax = plt.subplots(figsize=(12, 5))
    rewards = [h['train_joint_reward'] for h in hist]
    ax.plot(updates, rewards, alpha=0.3, color='#E91E63')
    if len(rewards) > 10: ax.plot(updates[9:], smooth(rewards, 10), color='#E91E63', linewidth=2, label='Smoothed')
    ax.set_xlabel('Update'); ax.set_ylabel('Joint Reward'); ax.set_title('VDPPO - Training Reward')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'vdppo_reward_curve.png'), dpi=150); plt.close()

    # 3. Overlap + Entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    overlaps = [h['eval']['mean']['overlap'] for h in hist]
    ax1.plot(updates, overlaps, color='#FF9800', linewidth=1.5)
    ax1.set_xlabel('Update'); ax1.set_ylabel('Overlap'); ax1.set_title('VDPPO - Overlap'); ax1.grid(alpha=0.3)

    entropies = [h['entropy'] for h in hist]
    ent_coefs = [h['ent_coef'] for h in hist]
    ax2.plot(updates, entropies, color='#9C27B0', linewidth=2, label='Entropy')
    ax2r = ax2.twinx()
    ax2r.plot(updates, ent_coefs, '--', color='#FF5722', linewidth=1.5, label='Ent Coef (adaptive)')
    ax2.set_xlabel('Update'); ax2.set_ylabel('Entropy'); ax2r.set_ylabel('Ent Coef')
    ax2.set_title('VDPPO - Adaptive Entropy'); ax2.grid(alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'vdppo_overlap_entropy.png'), dpi=150); plt.close()

    # 4. Losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(updates, [h['actor_loss'] for h in hist], color='#E91E63', linewidth=1.5)
    ax1.set_xlabel('Update'); ax1.set_ylabel('Actor Loss'); ax1.set_title('VDPPO - Actor Loss'); ax1.grid(alpha=0.3)
    ax2.plot(updates, [h['critic_loss'] for h in hist], color='#4CAF50', linewidth=1.5)
    ax2.set_xlabel('Update'); ax2.set_ylabel('Critic Loss (V_team + V_agent)'); ax2.set_title('VDPPO - Critic Loss'); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'vdppo_losses.png'), dpi=150); plt.close()

    # 5. Final bar
    last = hist[-1]['eval']
    fig, ax = plt.subplots(figsize=(8, 5))
    maps = ['simple', 'mixed', 'bottleneck']
    vals = [last[m]['coverage'] for m in maps]
    bars = ax.bar(maps, vals, color=[COLORS[m] for m in maps], edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coverage %'); ax.set_title('VDPPO - Final Coverage'); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'vdppo_final_bar.png'), dpi=150); plt.close()

    import shutil
    for f in os.listdir(d):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(d, f), os.path.join(out, f))

    print(f'  VDPPO: generated 5 new plots + copied originals')

# ============================================================
# Comparison
# ============================================================
def plot_comparison():
    out = os.path.join(OUT_DIR, 'comparison')
    os.makedirs(out, exist_ok=True)

    # Final coverage comparison
    results = {}
    # IPPO
    with open(os.path.join(LOG_DIR, 'ippo_experiment', 'final_eval_results.json')) as f:
        ippo = json.load(f)
    results['IPPO'] = {m: ippo[m][m] for m in ['simple', 'mixed', 'bottleneck']}

    # MAPPO
    with open(os.path.join(LOG_DIR, 'mappo_experiment', 'eval_history.json')) as f:
        mappo_h = json.load(f)
    last_m = mappo_h[-1]['eval']
    results['MAPPO'] = {m: last_m[m]['coverage'] for m in ['simple', 'mixed', 'bottleneck']}

    # VDPPO
    with open(os.path.join(LOG_DIR, 'vdppo_experiment', 'eval_history.json')) as f:
        vdppo_h = json.load(f)
    last_v = vdppo_h[-1]['eval']
    results['VDPPO'] = {m: last_v[m]['coverage'] for m in ['simple', 'mixed', 'bottleneck']}

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    maps = ['simple', 'mixed', 'bottleneck']
    algos = ['IPPO', 'MAPPO', 'VDPPO']
    algo_colors = ['#2196F3', '#4CAF50', '#E91E63']
    x = np.arange(len(maps))
    w = 0.25
    for i, (algo, color) in enumerate(zip(algos, algo_colors)):
        vals = [results[algo][m] for m in maps]
        bars = ax.bar(x + i*w, vals, w, label=algo, color=color, edgecolor='white', linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x + w); ax.set_xticklabels(maps)
    ax.set_ylabel('Coverage %'); ax.set_title('Algorithm Comparison - Coverage by Map')
    ax.legend(); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'comparison_coverage.png'), dpi=150); plt.close()

    # MAPPO vs VDPPO coverage over training
    fig, ax = plt.subplots(figsize=(12, 5))
    m_updates = [h['update'] for h in mappo_h]
    v_updates = [h['update'] for h in vdppo_h]
    m_mean = [h['eval']['mean']['coverage'] for h in mappo_h]
    v_mean = [h['eval']['mean']['coverage'] for h in vdppo_h]
    ax.plot(m_updates, m_mean, color='#4CAF50', linewidth=2, label='MAPPO (mean)')
    ax.plot(v_updates, v_mean, color='#E91E63', linewidth=2, label='VDPPO (mean)')
    ax.set_xlabel('Update'); ax.set_ylabel('Mean Coverage %')
    ax.set_title('MAPPO vs VDPPO - Coverage Over Training')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout(); plt.savefig(os.path.join(out, 'comparison_mappo_vs_vdppo.png'), dpi=150); plt.close()

    print(f'  Comparison: generated 2 plots')

# ============================================================
if __name__ == '__main__':
    print('Generating plots...')
    plot_ppo()
    plot_ippo()
    plot_mappo()
    plot_vdppo()
    plot_comparison()
    print(f'\nAll plots saved to: {os.path.abspath(OUT_DIR)}')
