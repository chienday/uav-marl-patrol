"""
VDPPOTrainer — Value-Decomposed PPO (extends MAPPOTrainer).

V_total = V_team (CentralCritic of global state)
        + mean(V_agent_i)   (per-agent local value heads)

Root-cause fixes vs vanilla MAPPO
----------------------------------
1. Decomposed value: V_total = V_team + mean(V_agent_i).
   Per-agent heads learn *marginal* improvement over team value.
2. Agent heads are trained on per-agent returns (not averaged).
3. Adaptive entropy balancing: increase ent_coef when coverage
   stalls to prevent premature convergence.
4. Optional diversity reward (Manhattan-distance based stacking penalty).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .mappo_trainer import MAPPOTrainer, DEVICE
from .networks import AgentValueHead
from ..envs.reward import NUM_AGENTS, VDPPORewardConfig, RewardConfig


class VDPPOTrainer(MAPPOTrainer):
    """
    VDPPO Trainer: MAPPO + per-agent value decomposition.

    V_total = V_team(s_global) + mean( V_agent_i(o_i) )
    """

    def __init__(
        self,
        map_file: str,
        map_paths: dict = None,
        backup_dir: str = "./checkpoints_vdppo",
        total_steps: int     = 4_000_000,
        rollout_len: int     = 256,
        n_envs: int          = 8,
        n_epochs: int        = 4,
        minibatch_size: int  = 512,
        gamma: float         = 0.99,
        gae_lambda: float    = 0.95,
        lr_actor: float      = 3e-4,
        lr_critic: float     = 1e-3,
        lr_agent_head: float = 5e-4,
        clip_range: float    = 0.2,
        hidden_dim: int      = 256,
        ent_coef: float      = 0.01,
        ent_coef_min: float  = 0.001,
        ent_decay: float     = 0.998,
        value_coef: float    = 0.5,
        max_grad_norm: float = 0.5,
        reward_norm: bool    = True,
        reward_cfg: Optional[RewardConfig] = None,
        enable_diversity_reward: bool = True,
        seed: int            = 42,
    ):
        # Default to VDPPORewardConfig (higher penalties)
        if reward_cfg is None:
            reward_cfg = VDPPORewardConfig()

        super().__init__(
            map_file=map_file,
            map_paths=map_paths,
            backup_dir=backup_dir,
            total_steps=total_steps,
            rollout_len=rollout_len,
            n_envs=n_envs,
            n_epochs=n_epochs,
            minibatch_size=minibatch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            clip_range=clip_range,
            hidden_dim=hidden_dim,
            ent_coef=ent_coef,
            ent_coef_min=ent_coef_min,
            ent_decay=ent_decay,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            reward_norm=reward_norm,
            reward_cfg=reward_cfg,
            enable_diversity_reward=enable_diversity_reward,
            seed=seed,
        )

        self.lr_agent_head = lr_agent_head

        # -- Per-agent value heads (one per UAV) ------------------------------
        self.agent_heads = nn.ModuleList([
            AgentValueHead(self.obs_dim, hidden_dim=128).to(DEVICE)
            for _ in range(NUM_AGENTS)
        ])
        self.opt_agent_heads = torch.optim.Adam(
            self.agent_heads.parameters(), lr=lr_agent_head, eps=1e-5
        )

        # -- Adaptive entropy state -------------------------------------------
        self._prev_mean_coverage = 0.0

    # -- V_total = V_team + mean(V_agent_i) --------------------------------
    def _compute_value_decomposed(
        self,
        global_state: torch.Tensor,
        obs_agents: torch.Tensor,
    ) -> torch.Tensor:
        """
        global_state : (B, gdim)
        obs_agents   : (B, A, obs_dim)
        returns      : (B,) V_total
        """
        v_team = self.critic(global_state).squeeze(-1)             # (B,)
        v_agents = torch.stack(
            [self.agent_heads[i](obs_agents[:, i, :]) for i in range(NUM_AGENTS)],
            dim=1,
        )                                                           # (B, A)
        return v_team + v_agents.mean(dim=1)                        # (B,)

    # -- Override PPO update ------------------------------------------------
    def _ppo_update(self, rollouts, n_updates_total):
        """VDPPO update: decomposed value loss with separate agent head gradient."""
        obs           = rollouts["obs"]         # (T,N,A,obs_dim)
        actions       = rollouts["actions"]     # (T,N,A)
        logp_old      = rollouts["logp"]        # (T,N,A)
        rewards       = rollouts["rewards"]     # (T,N,A) per-agent
        dones         = rollouts["dones"]       # (T,N)
        values        = rollouts["values"]      # (T,N)
        global_states = rollouts["global"]      # (T,N,gdim)
        last_values   = rollouts["last_values"] # (N,)

        advantages, returns = self._compute_gae(rewards, dones, values, last_values)

        if self.reward_norm:
            self._update_ret_rms(returns)
            returns_n = self._norm_returns(returns)
        else:
            returns_n = returns

        T, N, A, obs_dim = obs.shape
        gdim = self.global_state_dim
        B    = T * N

        obs_flat     = obs.reshape(B, A, obs_dim)
        actions_flat = actions.reshape(B, A)
        logp_flat    = logp_old.reshape(B, A)
        adv_flat     = advantages.reshape(B, A)
        ret_mean     = returns_n.mean(axis=2).reshape(B)       # (B,) shared critic target
        ret_agent    = returns_n.reshape(B, A)                  # (B,A) per-agent targets
        val_old_flat = values.reshape(B)
        global_flat  = global_states.reshape(B, gdim)

        # Per-agent advantage normalisation
        adv_norm = np.zeros_like(adv_flat)
        for a in range(A):
            mu  = adv_flat[:, a].mean()
            std = adv_flat[:, a].std()
            adv_norm[:, a] = np.clip((adv_flat[:, a] - mu) / (std + 1e-8), -5.0, 5.0)

        indices = np.arange(B)
        actor_losses = []; critic_losses = []; entropies = []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, self.minibatch_size):
                mb = indices[start: start + self.minibatch_size]

                obs_mb     = torch.as_tensor(obs_flat[mb],     dtype=torch.float32, device=DEVICE)
                act_mb     = torch.as_tensor(actions_flat[mb], dtype=torch.int64,   device=DEVICE)
                logp_mb    = torch.as_tensor(logp_flat[mb],    dtype=torch.float32, device=DEVICE)
                adv_mb     = torch.as_tensor(adv_norm[mb],     dtype=torch.float32, device=DEVICE)
                ret_mb     = torch.as_tensor(ret_mean[mb],     dtype=torch.float32, device=DEVICE)
                ret_ag_mb  = torch.as_tensor(ret_agent[mb],    dtype=torch.float32, device=DEVICE)
                val_old_mb = torch.as_tensor(val_old_flat[mb], dtype=torch.float32, device=DEVICE)
                glob_mb    = torch.as_tensor(global_flat[mb],  dtype=torch.float32, device=DEVICE)

                # -- Actor loss ------------------------------------------------
                total_actor_loss = torch.tensor(0.0, device=DEVICE)
                total_entropy    = torch.tensor(0.0, device=DEVICE)
                for agent_id in range(A):
                    logits = self.actor(obs_mb[:, agent_id, :])
                    dist   = torch.distributions.Categorical(logits=logits)
                    logp   = dist.log_prob(act_mb[:, agent_id])
                    ratio  = torch.exp(logp - logp_mb[:, agent_id])
                    surr1  = ratio * adv_mb[:, agent_id]
                    surr2  = torch.clamp(ratio,
                                         1.0 - self.clip_range,
                                         1.0 + self.clip_range) * adv_mb[:, agent_id]
                    total_actor_loss = total_actor_loss + (-torch.min(surr1, surr2).mean())
                    total_entropy    = total_entropy    + dist.entropy().mean()
                total_actor_loss /= A
                total_entropy    /= A
                actor_loss = total_actor_loss - self.ent_coef * total_entropy

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # -- VDPPO Critic loss (central V + agent heads) ----------------
                v_team = self.critic(glob_mb).squeeze(-1)                      # (mb,)
                v_agents = torch.stack(
                    [self.agent_heads[i](obs_mb[:, i, :]) for i in range(NUM_AGENTS)],
                    dim=1,
                )                                                               # (mb, A)
                v_total = v_team + v_agents.mean(dim=1)                        # (mb,)

                # Clipped central critic loss
                v_clipped = val_old_mb + torch.clamp(
                    v_total - val_old_mb, -self.clip_range, self.clip_range
                )
                central_loss = self.value_coef * torch.max(
                    (v_total   - ret_mb).pow(2),
                    (v_clipped - ret_mb).pow(2),
                ).mean()

                # Per-agent head loss
                agent_loss = sum(
                    (v_agents[:, a] - ret_ag_mb[:, a]).pow(2).mean()
                    for a in range(A)
                ) / A * 0.5

                critic_loss = central_loss + agent_loss

                self.opt_critic.zero_grad()
                self.opt_agent_heads.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),     self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent_heads.parameters(), self.max_grad_norm)
                self.opt_critic.step()
                self.opt_agent_heads.step()

                actor_losses.append(float(total_actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(total_entropy.detach()))

        # Adaptive entropy balancing
        self._adaptive_entropy()

        # Linear LR decay
        frac = max(1e-4, 1.0 - self._update_count / max(n_updates_total, 1))
        for pg in self.opt_actor.param_groups:       pg["lr"] = self.lr_actor      * frac
        for pg in self.opt_critic.param_groups:      pg["lr"] = self.lr_critic     * frac
        for pg in self.opt_agent_heads.param_groups: pg["lr"] = self.lr_agent_head * frac
        self._update_count += 1

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # -- Adaptive entropy control -------------------------------------------
    def _adaptive_entropy(self) -> None:
        """
        Increase entropy coefficient if coverage improvement stalls.
        Prevents premature convergence in bottleneck maps.
        """
        if not self.eval_history:
            self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.ent_decay)
            return

        last = self.eval_history[-1]
        curr_cov = last.get("train_coverage", 0.0)
        delta = curr_cov - self._prev_mean_coverage
        self._prev_mean_coverage = curr_cov

        if delta < 0.2:
            # Stalled — bump entropy
            self.ent_coef = min(0.05, self.ent_coef * 1.15)
        else:
            # Progressing — decay normally
            self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.ent_decay)

    # -- Save / Load -------------------------------------------------------
    def save(self, prefix: str = "vdppo_uav") -> None:
        super().save(prefix=prefix)
        torch.save(
            {i: h.state_dict() for i, h in enumerate(self.agent_heads)},
            f"{self.backup_dir}/{prefix}_agent_heads.pt"
        )
        print(f"  Saved -> {self.backup_dir}/{prefix}_agent_heads.pt")

    def load(self, prefix: str = "vdppo_uav") -> None:
        super().load(prefix=prefix)
        state = torch.load(
            f"{self.backup_dir}/{prefix}_agent_heads.pt", map_location=DEVICE
        )
        for i, h in enumerate(self.agent_heads):
            h.load_state_dict(state[i])
        print(f"  Loaded <- {self.backup_dir}/{prefix}_agent_heads.pt")

    # -- Override train banner ---------------------------------------------
    def train(self, eval_interval: int = 10) -> None:
        steps_per_update = self.rollout_len * self.n_envs
        n_updates = max(1, self.total_steps // steps_per_update)

        print("=" * 70)
        print(f"VDPPO Training -- {n_updates} updates x {steps_per_update} steps/update")
        print(f"  obs_dim={self.obs_dim}  state_dim={self.global_state_dim}")
        print(f"  V_total = V_team + mean(V_agent_i)")
        print(f"  Diversity reward: {self.envs[0].enable_diversity_reward}")
        print(f"  Map: {self.map_file}")
        print("=" * 70)

        for update in range(1, n_updates + 1):
            rollouts, stats = self._collect_rollouts()
            metrics = self._ppo_update(rollouts, n_updates)

            if update % eval_interval == 0 or update == 1 or update == n_updates:
                eval_res = self.evaluate_all_maps(n_episodes=5)
                record = {
                    "update":             update,
                    "train_joint_reward": stats["ep_joint_reward"],
                    "train_coverage":     stats["ep_coverage"],
                    "train_overlap":      stats["ep_overlap"],
                    "train_entropy":      stats["ep_entropy"],
                    "eval":               eval_res,
                    "actor_loss":         metrics["actor_loss"],
                    "critic_loss":        metrics["critic_loss"],
                    "entropy":            metrics["entropy"],
                    "ent_coef":           self.ent_coef,
                }
                self.eval_history.append(record)
                me = eval_res["mean"]
                print(
                    f"  upd {update:4d}/{n_updates} | "
                    f"cov={me['coverage']:.1f}% | "
                    f"R={me['joint_reward']:.0f} | "
                    f"ovlp={me['overlap']:.1f} | "
                    f"ent={me['entropy']:.3f} | "
                    f"ent_coef={self.ent_coef:.4f} | "
                    f"a_loss={metrics['actor_loss']:.4f}"
                )

        print("\n" + "=" * 70)
        print("VDPPO Training complete!")
        print("=" * 70)
