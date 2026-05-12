"""
MAPPOTrainer — Shared Actor + Centralised Critic (CTDE).

Root-cause fixes vs previous version
--------------------------------------
1. Per-agent GAE  : rewards_buf stores (T,N,A) not joint sum.
   advantages are computed per-agent with shared V(global_state).
2. Separate optimisers: lr_critic > lr_actor so V tracks returns fast.
3. Clipped value loss: PPO-style critic update for stability.
4. Return normalisation: running std normalises returns before critic.
5. Orthogonal init: non-random starting policy; entropy decays from 1.38.
6. Per-agent advantage normalisation (not global over agents+envs).
7. Linear LR decay + exponential ent_coef decay.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from ..envs.uav_env_multi import UAVPatrolEnvIPPO
from ..envs.reward import NUM_AGENTS, RewardConfig, check_collision, check_overlap
from .networks import SharedActor, CentralCritic

MASTER_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAPPOTrainer:
    """
    MAPPO trainer: shared actor + centralised critic (CTDE).
    """

    def __init__(
        self,
        map_file: str,
        map_paths: dict = None,
        backup_dir: str = "./checkpoints_mappo",
        total_steps: int     = 4_000_000,
        rollout_len: int     = 256,
        n_envs: int          = 8,
        n_epochs: int        = 4,
        minibatch_size: int  = 512,
        gamma: float         = 0.99,
        gae_lambda: float    = 0.95,
        lr_actor: float      = 3e-4,
        lr_critic: float     = 1e-3,
        clip_range: float    = 0.2,
        hidden_dim: int      = 256,
        ent_coef: float      = 0.01,
        ent_coef_min: float  = 0.001,
        ent_decay: float     = 0.998,
        value_coef: float    = 0.5,
        max_grad_norm: float = 0.5,
        reward_norm: bool    = True,
        reward_cfg: Optional[RewardConfig] = None,
        enable_diversity_reward: bool = False,
        seed: int            = MASTER_SEED,
    ):
        self.map_file       = map_file
        self.map_paths      = map_paths or {}
        self.backup_dir     = backup_dir
        self.total_steps    = total_steps
        self.rollout_len    = rollout_len
        self.n_envs         = n_envs
        self.n_epochs       = n_epochs
        self.minibatch_size = minibatch_size
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.lr_actor       = lr_actor
        self.lr_critic      = lr_critic
        self.clip_range     = clip_range
        self.hidden_dim     = hidden_dim
        self.ent_coef       = ent_coef
        self.ent_coef_min   = ent_coef_min
        self.ent_decay      = ent_decay
        self.value_coef     = value_coef
        self.max_grad_norm  = max_grad_norm
        self.reward_norm    = reward_norm
        self.seed           = seed

        os.makedirs(backup_dir, exist_ok=True)

        # -- Build envs -------------------------------------------------------
        self.envs = [
            UAVPatrolEnvIPPO(
                map_file=map_file,
                reward_cfg=reward_cfg,
                enable_diversity_reward=enable_diversity_reward,
            )
            for _ in range(n_envs)
        ]
        self.curr_obs = []
        for n in range(n_envs):
            obs_dict, _ = self.envs[n].reset(seed=self.seed + n)
            self.curr_obs.append(obs_dict)

        # -- Dimensions -------------------------------------------------------
        self.obs_dim          = self.envs[0].single_observation_space.shape[0]
        self.action_dim       = self.envs[0].single_action_space.n
        self.global_state_dim = self.envs[0].get_global_state().shape[0]

        # -- Networks ---------------------------------------------------------
        self.actor  = SharedActor(self.obs_dim, self.hidden_dim, self.action_dim).to(DEVICE)
        self.critic = CentralCritic(self.global_state_dim, self.hidden_dim).to(DEVICE)

        # -- Separate optimisers: critic needs 3-5x higher LR than actor ------
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=self.lr_actor,  eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        # -- Running return stats for normalisation ---------------------------
        self._ret_mean  = 0.0
        self._ret_var   = 1.0
        self._ret_count = 0

        # -- Logging ----------------------------------------------------------
        self.eval_history  = []
        self._update_count = 0

    # -- Running return normalisation ----------------------------------------
    def _update_ret_rms(self, returns: np.ndarray) -> None:
        """Welford online mean/variance over flattened returns."""
        batch = returns.flatten().astype(np.float64)
        n = len(batch)
        self._ret_count += n
        delta = batch - self._ret_mean
        self._ret_mean += delta.sum() / self._ret_count
        delta2 = batch - self._ret_mean
        self._ret_var = max(1.0, self._ret_var + (delta * delta2).sum())

    def _norm_returns(self, returns: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._ret_var / max(self._ret_count, 1)) + 1e-8
        return np.clip(returns / std, -10.0, 10.0)

    # -- Policy --------------------------------------------------------------
    def _policy(self, obs_batch: np.ndarray, deterministic: bool = False):
        obs_t  = torch.as_tensor(obs_batch, dtype=torch.float32, device=DEVICE)
        logits = self.actor(obs_t)
        dist   = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, dist.entropy()

    # -- Rollout collection --------------------------------------------------
    def _collect_rollouts(self):
        T = self.rollout_len; N = self.n_envs; A = NUM_AGENTS
        obs_dim = self.obs_dim; gdim = self.global_state_dim

        obs_buf     = np.zeros((T, N, A, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((T, N, A),          dtype=np.int64)
        logp_buf    = np.zeros((T, N, A),          dtype=np.float32)
        # KEY: per-agent rewards, NOT joint sum
        rewards_buf = np.zeros((T, N, A),          dtype=np.float32)
        dones_buf   = np.zeros((T, N),             dtype=np.float32)
        values_buf  = np.zeros((T, N),             dtype=np.float32)
        global_buf  = np.zeros((T, N, gdim),       dtype=np.float32)

        ep_rewards = []; ep_coverages = []; ep_overlaps = []; ep_entropies = []
        running_r   = np.zeros(N, dtype=np.float32)
        running_ent = np.zeros(N, dtype=np.float32)
        running_ovl = np.zeros(N, dtype=np.float32)
        running_stp = np.zeros(N, dtype=np.float32)

        for t in range(T):
            for n, env in enumerate(self.envs):
                obs_dict     = self.curr_obs[n]
                obs_batch    = np.stack([obs_dict[0], obs_dict[1]], axis=0)
                global_state = env.get_global_state()

                with torch.no_grad():
                    actions, logp, entropy = self._policy(obs_batch, deterministic=False)
                    value = self.critic(
                        torch.as_tensor(global_state, dtype=torch.float32, device=DEVICE)
                    ).squeeze(-1)

                actions_list = [int(actions[0]), int(actions[1])]
                next_obs, rewards, terminated, truncated, info = env.step(actions_list)
                done = terminated or truncated

                per_agent_r = np.array([float(rewards[0]), float(rewards[1])], dtype=np.float32)

                obs_buf[t, n]     = obs_batch
                actions_buf[t, n] = np.asarray(actions.cpu().numpy(), dtype=np.int64)
                logp_buf[t, n]    = logp.detach().cpu().numpy()
                rewards_buf[t, n] = per_agent_r
                dones_buf[t, n]   = 1.0 if done else 0.0
                values_buf[t, n]  = value.detach().cpu().item()
                global_buf[t, n]  = global_state

                running_r[n]   += float(per_agent_r.sum())
                running_ent[n] += float(entropy.mean().cpu().item())
                running_stp[n] += 1.0
                if check_collision(env.agent_positions[0], env.agent_positions[1]):
                    running_ovl[n] += 1
                elif check_overlap(env.agent_positions[0], env.agent_positions[1]):
                    running_ovl[n] += 1

                if done:
                    ep_rewards.append(float(running_r[n]))
                    ep_coverages.append(float(info["coverage_ratio"] * 100.0))
                    ep_overlaps.append(float(running_ovl[n]))
                    ep_entropies.append(float(running_ent[n] / max(running_stp[n], 1)))
                    running_r[n] = running_ent[n] = running_ovl[n] = running_stp[n] = 0.0
                    new_obs, _ = env.reset(seed=self.seed + n + t * N)
                    self.curr_obs[n] = new_obs
                else:
                    self.curr_obs[n] = next_obs

        # Bootstrap last values
        last_values = np.zeros(N, dtype=np.float32)
        for n, env in enumerate(self.envs):
            gs = env.get_global_state()
            with torch.no_grad():
                last_values[n] = self.critic(
                    torch.as_tensor(gs, dtype=torch.float32, device=DEVICE)
                ).squeeze(-1).cpu().item()

        rollouts = {
            "obs": obs_buf, "actions": actions_buf, "logp": logp_buf,
            "rewards": rewards_buf, "dones": dones_buf, "values": values_buf,
            "global": global_buf, "last_values": last_values,
        }
        stats = {
            "ep_joint_reward": float(np.mean(ep_rewards))   if ep_rewards   else 0.0,
            "ep_coverage":     float(np.mean(ep_coverages)) if ep_coverages else 0.0,
            "ep_overlap":      float(np.mean(ep_overlaps))  if ep_overlaps  else 0.0,
            "ep_entropy":      float(np.mean(ep_entropies)) if ep_entropies else 0.0,
        }
        return rollouts, stats

    # -- Per-agent GAE with shared central V ----------------------------------
    def _compute_gae(self, rewards, dones, values, last_values):
        """
        rewards : (T, N, A) per-agent
        values  : (T, N)    V(global_state) -- same V for both agents (MAPPO)
        returns : (T, N, A) per-agent discounted returns
        adv     : (T, N, A) per-agent advantages from shared critic
        """
        T, N, A = rewards.shape
        advantages = np.zeros((T, N, A), dtype=np.float32)
        lastgaelam = np.zeros((N, A),    dtype=np.float32)

        for t in reversed(range(T)):
            next_val = last_values if t == T - 1 else values[t + 1]  # (N,)
            mask  = 1.0 - dones[t]                                    # (N,)
            # Each agent uses shared V but its own reward (MAPPO)
            delta = (rewards[t]                                        # (N,A)
                     + self.gamma * next_val[:, None] * mask[:, None]
                     - values[t][:, None])
            lastgaelam = delta + self.gamma * self.gae_lambda * mask[:, None] * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values[:, :, None]
        return advantages, returns

    # -- PPO update ----------------------------------------------------------
    def _ppo_update(self, rollouts, n_updates_total):
        obs           = rollouts["obs"]         # (T,N,A,obs_dim)
        actions       = rollouts["actions"]     # (T,N,A)
        logp_old      = rollouts["logp"]        # (T,N,A)
        rewards       = rollouts["rewards"]     # (T,N,A) per-agent
        dones         = rollouts["dones"]       # (T,N)
        values        = rollouts["values"]      # (T,N)
        global_states = rollouts["global"]      # (T,N,gdim)
        last_values   = rollouts["last_values"] # (N,)

        advantages, returns = self._compute_gae(rewards, dones, values, last_values)

        # Return normalisation
        if self.reward_norm:
            self._update_ret_rms(returns)
            returns_n = self._norm_returns(returns)
        else:
            returns_n = returns

        T, N, A, obs_dim = obs.shape
        gdim      = self.global_state_dim
        B         = T * N  # samples

        obs_flat     = obs.reshape(B, A, obs_dim)
        actions_flat = actions.reshape(B, A)
        logp_flat    = logp_old.reshape(B, A)
        adv_flat     = advantages.reshape(B, A)   # (B, A) per-agent
        # Critic target: mean over agents (shared V)
        ret_crit     = returns_n.mean(axis=2).reshape(B)   # (B,)
        val_old_flat = values.reshape(B)                    # (B,) for clip
        global_flat  = global_states.reshape(B, gdim)      # (B, gdim)

        # Per-agent advantage normalisation (not mixed across agents)
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
                ret_mb     = torch.as_tensor(ret_crit[mb],     dtype=torch.float32, device=DEVICE)
                val_old_mb = torch.as_tensor(val_old_flat[mb], dtype=torch.float32, device=DEVICE)
                glob_mb    = torch.as_tensor(global_flat[mb],  dtype=torch.float32, device=DEVICE)

                # -- Actor loss (per-agent, then averaged) ----------------------
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

                total_actor_loss = total_actor_loss / A
                total_entropy    = total_entropy    / A
                actor_loss = total_actor_loss - self.ent_coef * total_entropy

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # -- Critic loss (PPO-style clipped value loss) -----------------
                val_pred = self.critic(glob_mb).squeeze(-1)
                val_pred_clipped = val_old_mb + torch.clamp(
                    val_pred - val_old_mb, -self.clip_range, self.clip_range
                )
                critic_loss = self.value_coef * torch.max(
                    (val_pred         - ret_mb).pow(2),
                    (val_pred_clipped - ret_mb).pow(2),
                ).mean()

                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                actor_losses.append(float(total_actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(total_entropy.detach()))

        # Linear LR decay
        frac = max(1e-4, 1.0 - self._update_count / max(n_updates_total, 1))
        for pg in self.opt_actor.param_groups:  pg["lr"] = self.lr_actor  * frac
        for pg in self.opt_critic.param_groups: pg["lr"] = self.lr_critic * frac
        self._update_count += 1

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # -- Evaluation ----------------------------------------------------------
    def evaluate_map(self, map_file: str, n_episodes: int = 8) -> dict:
        joint_rewards = []; coverages = []; overlaps = []; entropies = []
        for ep in range(n_episodes):
            env = UAVPatrolEnvIPPO(map_file=map_file)
            obs, _ = env.reset(seed=1000 + ep)
            done = False
            total_r = 0.0; overlap = 0; ent_sum = 0.0; steps = 0
            while not done:
                obs_batch = np.stack([obs[0], obs[1]], axis=0)
                with torch.no_grad():
                    actions, _, entropy = self._policy(obs_batch, deterministic=True)
                obs, rewards, t, tr, _ = env.step([int(actions[0]), int(actions[1])])
                total_r += float(rewards[0] + rewards[1])
                if check_collision(env.agent_positions[0], env.agent_positions[1]):
                    overlap += 1
                elif check_overlap(env.agent_positions[0], env.agent_positions[1]):
                    overlap += 1
                ent_sum += float(entropy.mean().cpu().item())
                steps += 1; done = t or tr
            joint_rewards.append(total_r)
            coverages.append(env.coverage.sum() / env.free_cells * 100.0)
            overlaps.append(overlap)
            entropies.append(ent_sum / max(steps, 1))
        return {
            "joint_reward": float(np.mean(joint_rewards)),
            "coverage":     float(np.mean(coverages)),
            "overlap":      float(np.mean(overlaps)),
            "entropy":      float(np.mean(entropies)),
        }

    def evaluate_all_maps(self, n_episodes: int = 8) -> dict:
        results = {}
        for name, mp in self.map_paths.items():
            results[name] = self.evaluate_map(mp, n_episodes=n_episodes)
        results["mean"] = {
            "joint_reward": float(np.mean([results[m]["joint_reward"] for m in self.map_paths])),
            "coverage":     float(np.mean([results[m]["coverage"]     for m in self.map_paths])),
            "overlap":      float(np.mean([results[m]["overlap"]      for m in self.map_paths])),
            "entropy":      float(np.mean([results[m]["entropy"]      for m in self.map_paths])),
        }
        return results

    # -- Main training loop --------------------------------------------------
    def train(self, eval_interval: int = 10) -> None:
        steps_per_update = self.rollout_len * self.n_envs
        n_updates = max(1, self.total_steps // steps_per_update)

        print("=" * 70)
        print(f"MAPPO Training -- {n_updates} updates x {steps_per_update} steps/update")
        print(f"  obs_dim={self.obs_dim}  state_dim={self.global_state_dim}")
        print(f"  lr_actor={self.lr_actor}  lr_critic={self.lr_critic}")
        print(f"  ent_coef={self.ent_coef:.4f}->{self.ent_coef_min}  decay={self.ent_decay}")
        print(f"  Map: {self.map_file}")
        print("=" * 70)

        for update in range(1, n_updates + 1):
            rollouts, stats = self._collect_rollouts()
            # Exponential entropy decay
            self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.ent_decay)
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
        print("MAPPO Training complete!")
        print("=" * 70)

    def save(self, prefix: str = "mappo_uav") -> None:
        torch.save(self.actor.state_dict(),  f"{self.backup_dir}/{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{self.backup_dir}/{prefix}_critic.pt")
        print(f"  Saved -> {self.backup_dir}/{prefix}_actor/critic.pt")

    def load(self, prefix: str = "mappo_uav") -> None:
        self.actor.load_state_dict(torch.load(f"{self.backup_dir}/{prefix}_actor.pt",  map_location=DEVICE))
        self.critic.load_state_dict(torch.load(f"{self.backup_dir}/{prefix}_critic.pt", map_location=DEVICE))
        print(f"  Loaded <- {self.backup_dir}/{prefix}")
