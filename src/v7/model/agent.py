import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.config.config import (
    MODELS_DIR,
    BATCH_SIZE_MULTIDAY_MINUTE,
    HIDDEN_DIM,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    LEARNING_RATE_ALPHA,
    GAMMA_MULTIDAY_MINUTE,
    TAU,
    ALPHA_INIT,
    TARGET_UPDATE_INTERVAL,
    UPDATE_RATIO,
    REPLAY_BUFFER_SIZE,
    WINDOW_SIZE,
)
from src.v7.model.networks import Actor, Critic
from src.v4.model.replay_buffer import IndexReplayBuffer
from src.utils.logger import Logger
from src.utils.utils import create_directory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(
        self,
        temporal_state_len: int,
        regime_state_len: int,
        total_episodes: int,
        action_dim: int = 5,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
        actor_lr: float = LEARNING_RATE_ACTOR,
        critic_lr: float = LEARNING_RATE_CRITIC,
        alpha_lr: float = LEARNING_RATE_ALPHA,
        gamma: float = GAMMA_MULTIDAY_MINUTE,
        tau: float = TAU,
        alpha_init: float = ALPHA_INIT,
        target_update_interval: int = TARGET_UPDATE_INTERVAL,
        update_ratio: int = UPDATE_RATIO,
        use_automatic_entropy_tuning: bool = True,
        device: torch.device = DEVICE,
        window_size: int = WINDOW_SIZE,
        buffer_capacity: int = REPLAY_BUFFER_SIZE,
        input_shape: Tuple[int, int] = None,  # v7: (n_tickers, feature_dim_per_ticker)
        portfolio_state_len: int = None,
        max_grad_norm: float = 1.0,
        target_entropy: float = None,
        logger: Optional[Logger] = None,
        # === New: accept external replay buffer for multi-ticker training ===
        replay_buffer = None,                       # if provided, use this instead of IndexReplayBuffer
        market_data: np.ndarray = None,             # only needed if replay_buffer is None (single-ticker mode)
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_init = alpha_init
        self.target_update_interval = target_update_interval
        self.update_ratio = update_ratio
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.logger = logger

        # ── Networks ─────────────────────────────────────────────────────
        net_kwargs = dict(
            input_shape=input_shape,
            portfolio_state_len=portfolio_state_len,
            temporal_state_len=temporal_state_len,
            regime_state_len=regime_state_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
            device=device,
        )

        self.actor = Actor(**net_kwargs)
        self.critic = Critic(**net_kwargs)
        self.critic_target = Critic(**net_kwargs)

        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(sp.data)

        # ── Optimizers ───────────────────────────────────────────────────
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Alpha and Entropy ────────────────────────────────────
        if self.use_automatic_entropy_tuning:
            self.current_episode = 0
            self.total_episodes = total_episodes
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor(alpha_init, device=device)

        # ── Replay buffer ────────────────────────────────────────────────
        if replay_buffer is not None:
            # Multi-ticker mode: caller provides a DailyReplayBuffer
            self.replay_buffer = replay_buffer
        else:
            # Single-ticker mode (backward compatible): create IndexReplayBuffer
            assert market_data is not None, "Must provide either replay_buffer or market_data"
            self.replay_buffer = IndexReplayBuffer(
                market_data=market_data,
                window_size=window_size,
                portfolio_state_len=portfolio_state_len,
                temporal_state_len=temporal_state_len,
                regime_state_len=regime_state_len,
                action_dim=action_dim,
                capacity=buffer_capacity
            )

        self.train_step_counter = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.entropy_values = []
        self.q_values = []

        # NaN-skip counters. Incremented when a gradient update is skipped
        # because the loss / gradients contained NaN or inf. Surfaced in the
        # loss dict returned from update_parameters so the trainer can log
        # cumulative skip counts during training. A non-trivial skip rate
        # indicates a deeper numerical-stability issue we should investigate.
        self.critic_nan_skips = 0
        self.actor_nan_skips = 0
        self.alpha_nan_skips = 0

        # ── Batch prefetching ────────────────────────────────────────────
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future = None

    def reset_prefetch(self):
        """Call between episodes to discard any stale prefetched batch."""
        if self._prefetch_future is not None:
            self._prefetch_future.result()  # drain it
            self._prefetch_future = None

    # ── Action selection ─────────────────────────────────────────────────
    def select_action(self, state: Dict[str, np.ndarray], validate: bool = False) -> np.ndarray:
        # Diagnostic: clearly report which input or param contains NaN/Inf,
        # rather than letting torch.distributions.Normal raise a generic
        # ValueError downstream that says only "found invalid values".
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                if not np.all(np.isfinite(v)):
                    bad_mask = ~np.isfinite(v)
                    n_bad = int(bad_mask.sum())
                    # Report up to the first 10 non-finite positions to
                    # pinpoint the source (which feature dim / which ticker).
                    bad_indices = np.argwhere(bad_mask)
                    sample = bad_indices[:10].tolist()
                    raise RuntimeError(
                        f"select_action: state['{k}'] has {n_bad}/{v.size} "
                        f"non-finite values; shape={v.shape}; "
                        f"first bad indices (up to 10): {sample}"
                    )
        for name, p in self.actor.named_parameters():
            if not torch.isfinite(p.data).all():
                n_bad = int((~torch.isfinite(p.data)).sum().item())
                raise RuntimeError(
                    f"select_action: actor parameter '{name}' has {n_bad}/{p.numel()} "
                    f"non-finite values; shape={tuple(p.shape)}. "
                    f"Counters: critic_skips={self.critic_nan_skips}, "
                    f"actor_skips={self.actor_nan_skips}, alpha_skips={self.alpha_nan_skips}, "
                    f"alpha={float(self.alpha.item()) if torch.is_tensor(self.alpha) else float(self.alpha):.4g}"
                )

        state_tensor = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state_tensor[key] = torch.as_tensor(
                    value, dtype=torch.float, device=self.device
                ).unsqueeze(0)
            else:
                state_tensor[key] = value.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if validate:
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_tensor)

        return action.cpu().numpy()[0]

    # ── Batch conversion helper ──────────────────────────────────────────
    def _batch_dict_to_tensor(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.as_tensor(v, dtype=torch.float, device=self.device)
            for k, v in state_dict.items()
        }
        
    def _get_alpha_floor(self) -> float:
        return max(0.01 * (1 - self.current_episode / self.total_episodes), 1e-8)

    # ── SAC parameter update ─────────────────────────────────────────────
    def update_parameters(self, batch_size: int = BATCH_SIZE_MULTIDAY_MINUTE) -> Dict[str, float]:
        """
        Run ``self.update_ratio`` gradient updates per call (UTD ratio).

        Each gradient pass samples a fresh batch from the replay buffer.
        Same-batch repetition is avoided because it amplifies noise from
        one specific batch rather than extracting signal across the buffer.

        Target soft-update happens once per call (not per gradient pass) —
        TAU is calibrated for the lower frequency. Target lag at 4× UTD
        with per-step soft-updates would destabilize the critic.

        Returned metrics are the MEAN across the N gradient passes — that
        gives the trainer a representative single-number summary of the
        call rather than just the last update's values.
        """
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha_loss": 0.0,
                "entropy": 0.0,
                "alpha": self.alpha.item(),
                'q_value': 0.0,
                "critic_nan_skips": self.critic_nan_skips,
                "actor_nan_skips":  self.actor_nan_skips,
                "alpha_nan_skips":  self.alpha_nan_skips,
            }

        # Accumulators for mean-across-N reporting.
        sum_actor_loss = 0.0
        sum_critic_loss = 0.0
        sum_alpha_loss = 0.0
        sum_entropy = 0.0
        sum_q_value = 0.0

        for _ in range(self.update_ratio):
            metrics = self._gradient_step(batch_size)
            sum_actor_loss  += metrics["actor_loss"]
            sum_critic_loss += metrics["critic_loss"]
            sum_alpha_loss  += metrics["alpha_loss"]
            sum_entropy     += metrics["entropy"]
            sum_q_value     += metrics["q_value"]

        # ── Soft target update (once per call, not per gradient step) ──
        # TAU was calibrated assuming 1 target update per env step. Higher
        # UTD with the same TAU would let the target network drift faster
        # than the online critic can stabilize. See REDQ for the proper
        # high-UTD-with-target-update treatment if we revisit.
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

        n = float(self.update_ratio)
        mean_actor_loss  = sum_actor_loss  / n
        mean_critic_loss = sum_critic_loss / n
        mean_alpha_loss  = sum_alpha_loss  / n
        mean_entropy     = sum_entropy     / n
        mean_q_value     = sum_q_value     / n

        # Bookkeeping uses the means — same shape as before, just averaged.
        self.actor_losses.append(mean_actor_loss)
        self.critic_losses.append(mean_critic_loss)
        self.alpha_losses.append(mean_alpha_loss)
        self.entropy_values.append(mean_entropy)
        self.q_values.append(mean_q_value)

        return {
            "actor_loss":  mean_actor_loss,
            "critic_loss": mean_critic_loss,
            "alpha_loss":  mean_alpha_loss,
            "entropy":     mean_entropy,
            "alpha":       self.alpha.item(),
            "q_value":     mean_q_value,
            # Cumulative skip counts (Agent-lifetime totals). Trainer can
            # log diffs / rates to surface trend over training. See
            # _gradient_step for the NaN-guard mechanism.
            "critic_nan_skips": self.critic_nan_skips,
            "actor_nan_skips":  self.actor_nan_skips,
            "alpha_nan_skips":  self.alpha_nan_skips,
        }

    def _gradient_step(self, batch_size: int) -> Dict[str, float]:
        """
        One full SAC gradient update: critic step + actor step + alpha step.

        Does NOT touch the target network — that's handled once per
        update_parameters call regardless of UTD ratio.
        """
        # Use prefetched batch if available, otherwise sample synchronously.
        # Prefetcher overlaps the next sample with this call's gradient
        # pass — at UTD=4 the second/third/fourth samples land synchronously
        # because numpy buffer-sampling is microseconds vs millisecond-scale
        # gradient passes; not worth a 4-deep prefetch pipeline.
        if self._prefetch_future is not None:
            states, actions, rewards, next_states, dones = self._prefetch_future.result()
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Kick off next batch prefetch immediately
        self._prefetch_future = self._prefetch_executor.submit(
            self.replay_buffer.sample, batch_size
        )

        batched_states = self._batch_dict_to_tensor(states)
        batched_next_states = self._batch_dict_to_tensor(next_states)

        batched_actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        batched_rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        batched_dones = torch.as_tensor(dones, dtype=torch.float, device=self.device)

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(batched_next_states)
            next_q1, next_q2 = self.critic_target(batched_next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            expected_q = batched_rewards + (1.0 - batched_dones) * self.gamma * next_q

        cur_q1, cur_q2 = self.critic(batched_states, batched_actions)
        mean_q = torch.min(cur_q1, cur_q2).mean().item()
        critic_loss = F.mse_loss(cur_q1, expected_q) + F.mse_loss(cur_q2, expected_q)

        # Critic step with belt-and-suspenders NaN guards:
        #   1. Pre-step: check loss + gradients are finite
        #   2. Snapshot params before step (cheap — just .clone())
        #   3. Post-step: check params didn't go non-finite via Adam internals
        #      (overflow in m/v accumulators, etc.). If they did, restore.
        # The two-stage check catches the v7 NaN-cascade pattern where the
        # cascade flows through paths the gradient check alone misses. See
        # dev_log_v7.md (run-1 NaN incident).
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if torch.isfinite(critic_loss) and self._grads_are_finite(self.critic.parameters()):
            snap = self._snapshot_params(self.critic.parameters())
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            if not self._params_are_finite(self.critic.parameters()):
                self._restore_params(self.critic.parameters(), snap)
                self.critic_nan_skips += 1
                self._log_nan_event("critic", critic_loss, post_step=True)
        else:
            self.critic_nan_skips += 1
            self._log_nan_event("critic", critic_loss, post_step=False)
            self.critic_optimizer.zero_grad()

        # ── Actor update ─────────────────────────────────────────────────
        new_actions, log_probs, _ = self.actor.sample(batched_states)
        q1, q2 = self.critic(batched_states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        # Actor step with the same two-stage guard.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if torch.isfinite(actor_loss) and self._grads_are_finite(self.actor.parameters()):
            snap = self._snapshot_params(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            if not self._params_are_finite(self.actor.parameters()):
                self._restore_params(self.actor.parameters(), snap)
                self.actor_nan_skips += 1
                self._log_nan_event("actor", actor_loss, post_step=True)
        else:
            self.actor_nan_skips += 1
            self._log_nan_event("actor", actor_loss, post_step=False)
            self.actor_optimizer.zero_grad()

        # ── Alpha update ─────────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            if torch.isfinite(alpha_loss) and self._grads_are_finite([self.log_alpha]):
                snap = self._snapshot_params([self.log_alpha])
                torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
                self.alpha_optimizer.step()
                if not self._params_are_finite([self.log_alpha]):
                    self._restore_params([self.log_alpha], snap)
                    self.alpha_nan_skips += 1
                    self._log_nan_event("alpha", alpha_loss, post_step=True)
                # self.alpha = torch.clamp(self.log_alpha.exp(), min=self._get_alpha_floor())
                self.alpha = self.log_alpha.exp()
                alpha_loss_val = alpha_loss.item() if torch.isfinite(alpha_loss) else 0.0
            else:
                self.alpha_nan_skips += 1
                self._log_nan_event("alpha", alpha_loss, post_step=False)
                self.alpha_optimizer.zero_grad()

        # actor_loss / critic_loss values may be inf/NaN when the guard
        # caught them above. Coerce to safe float so the trainer's logging
        # / averaging stays clean — the nan_skips counters are the
        # authoritative signal that something was skipped.
        def _safe_item(t):
            v = t.item()
            return v if (v == v and v != float('inf') and v != float('-inf')) else 0.0

        return {
            "actor_loss":  _safe_item(actor_loss),
            "critic_loss": _safe_item(critic_loss),
            "alpha_loss":  alpha_loss_val,
            "entropy":     -log_probs.mean().item(),
            "q_value":     mean_q,
        }

    @staticmethod
    def _grads_are_finite(params) -> bool:
        """
        Return True iff every parameter's .grad is finite (no NaN / inf).
        Used to guard optimizer.step() calls so a single bad gradient
        batch doesn't NaN-out the network permanently.

        Cheap — short-circuits on the first non-finite gradient found.
        """
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        return True

    @staticmethod
    def _params_are_finite(params) -> bool:
        """
        Return True iff every parameter's data is finite. Used as the
        post-step check to catch corruption that flows through optimizer
        internals (Adam m/v overflow, edge-case numerical issues) even
        when the pre-step grads-are-finite check passed.
        """
        for p in params:
            if not torch.isfinite(p.data).all():
                return False
        return True

    @staticmethod
    def _snapshot_params(params):
        """
        Take a cheap clone of every parameter tensor's data so the step
        can be reverted if the post-step finiteness check fails. The list
        is materialized eagerly because the caller passes .parameters()
        as a generator that gets consumed by clip_grad_norm_/step.
        """
        return [p.data.detach().clone() for p in params]

    @staticmethod
    def _restore_params(params, snapshot) -> None:
        """Roll back parameter values to a snapshot taken before the step."""
        for p, s in zip(params, snapshot):
            p.data.copy_(s)

    def _log_nan_event(self, which: str, loss, post_step: bool) -> None:
        """
        Log a NaN-guard event with enough detail to diagnose the cause.
        Logs only the FIRST few events per optimizer to avoid log spam
        when the issue persists across many gradient updates.
        """
        if self.logger is None:
            return
        # Only log the first 3 events per optimizer (across the run).
        attr = f"_nan_log_count_{which}"
        n = getattr(self, attr, 0)
        if n >= 3:
            return
        setattr(self, attr, n + 1)

        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        stage = "POST-step (param NaN, rolled back)" if post_step else "PRE-step (loss/grad NaN)"
        self.logger.warning(
            f"[NaN guard fired #{n+1}] optimizer={which}, stage={stage}, "
            f"loss={loss_val}, alpha={float(self.alpha.item()) if torch.is_tensor(self.alpha) else float(self.alpha):.4g}"
        )

    # ── Save / Load ──────────────────────────────────────────────────────
    def save_model(self, save_dir: Union[str, Path] = MODELS_DIR, prefix: str = "", timestamp: Optional[str] = None) -> Path:
        create_directory(save_dir)
        timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        model_path = Path(save_dir) / f"{prefix}sac_model_{timestamp}"
        create_directory(model_path)

        torch.save(self.actor.state_dict(), model_path / "actor.pth")
        torch.save(self.critic.state_dict(), model_path / "critic.pth")
        torch.save(self.critic_target.state_dict(), model_path / "critic_target.pth")

        torch.save(self.actor_optimizer.state_dict(), model_path / "actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), model_path / "critic_optimizer.pth")

        if self.use_automatic_entropy_tuning:
            torch.save(self.log_alpha, model_path / "log_alpha.pth")
            torch.save(self.current_episode, model_path / "current_episode.pth")
            torch.save(self.total_episodes, model_path / "total_episodes.pth")
            torch.save(self.alpha_optimizer.state_dict(), model_path / "alpha_optimizer.pth")

        torch.save(
            {
                "actor_losses": self.actor_losses,
                "critic_losses": self.critic_losses,
                "alpha_losses": self.alpha_losses,
                "entropy_values": self.entropy_values,
                "train_step_counter": self.train_step_counter,
            },
            model_path / "training_stats.pth",
        )

        torch.save(
            {
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha_init": self.alpha_init,
                "target_update_interval": self.target_update_interval,
                "use_automatic_entropy_tuning": self.use_automatic_entropy_tuning,
            },
            model_path / "config.pth",
        )

        if self.logger:
            self.logger.info(f"Model saved: {model_path}")
        return model_path

    def load_model(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            if self.logger:
                self.logger.error(f"Model path does not exist: {model_path}")
            return

        self.actor.load_state_dict(torch.load(model_path / "actor.pth", map_location=self.device, weights_only=False))
        self.critic.load_state_dict(torch.load(model_path / "critic.pth", map_location=self.device, weights_only=False))
        self.critic_target.load_state_dict(torch.load(model_path / "critic_target.pth", map_location=self.device, weights_only=False))

        self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device, weights_only=False))
        self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device, weights_only=False))

        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device, weights_only=False)
            self.current_episode = torch.load(model_path / "current_episode.pth", map_location=self.device, weights_only=False)
            self.total_episodes = torch.load(model_path / "total_episodes.pth", map_location=self.device, weights_only=False)
            # self.alpha = torch.clamp(self.log_alpha.exp(), min=self._get_alpha_floor())
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device, weights_only=False))

        stats = torch.load(model_path / "training_stats.pth", map_location=self.device, weights_only=False)
        self.actor_losses = stats["actor_losses"]
        self.critic_losses = stats["critic_losses"]
        self.alpha_losses = stats["alpha_losses"]
        self.entropy_values = stats["entropy_values"]
        self.train_step_counter = stats["train_step_counter"]

        if self.logger:
            self.logger.info(f"Model loaded: {model_path}")

    def get_latest_model_path(self, save_dir: Union[str, Path] = None, prefix: str = "") -> Optional[Path]:
        if save_dir is None:
            save_dir = MODELS_DIR
        save_dir = Path(save_dir)
        if not save_dir.exists():
            return None
        model_dirs = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}sac_model_")]
        if not model_dirs:
            return None
        model_dirs.sort(key=lambda d: d.name, reverse=True)
        return model_dirs[0]