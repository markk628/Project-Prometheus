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
    DEVICE,
    REPLAY_BUFFER_SIZE,
    WINDOW_SIZE,
)
from src.v4.model.networks import Actor, Critic
from src.v4.model.replay_buffer import IndexReplayBuffer
from src.utils.logger import Logger
from src.utils.utils import create_directory


class Agent:
    def __init__(
        self,
        market_data: np.ndarray,
        temporal_state_len: int ,
        regime_state_len: int,
        total_episodes: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        actor_lr: float = LEARNING_RATE_ACTOR,
        critic_lr: float = LEARNING_RATE_CRITIC,
        alpha_lr: float = LEARNING_RATE_ALPHA,
        gamma: float = GAMMA_MULTIDAY_MINUTE,
        tau: float = TAU,
        alpha_init: float = ALPHA_INIT,
        target_update_interval: int = TARGET_UPDATE_INTERVAL,
        use_automatic_entropy_tuning: bool = True,
        device: torch.device = DEVICE,
        window_size: int = WINDOW_SIZE,
        buffer_capacity: int = REPLAY_BUFFER_SIZE,
        input_shape: Tuple[int, int] = None,       # (window_size, market_feature_dim) — AE latents only
        portfolio_state_len: int = None,
        max_grad_norm: float = 1.0,
        target_entropy: float = None,
        logger: Optional[Logger] = None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_init = alpha_init
        self.target_update_interval = target_update_interval
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) # TODO add weight decay if overfitting suspected
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Alpha and Entropy ────────────────────────────────────
        if self.use_automatic_entropy_tuning:
            self.current_episode = 0
            self.total_episodes = total_episodes
            # standard Gaussian (target entropy ≈ 1.42 + ln(sigma)) where sigma is noise applied to action
            # meaning environment will randomly force anywhere from -sigma to +sigma to be applied to action
            # calculate using sigma ≈ e^(target entropy - 1.42)
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor(alpha_init, device=device)

        # ── Index-based replay buffer ────────────────────────────────────
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
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha_loss": 0.0,
                "entropy": 0.0,
                "alpha": self.alpha.item(),
                'q_value': 0.0
            }

        # Use prefetched batch if available, otherwise sample synchronously
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

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ── Actor update ─────────────────────────────────────────────────
        new_actions, log_probs, _ = self.actor.sample(batched_states)
        q1, q2 = self.critic(batched_states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # ── Alpha update ─────────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.clamp(self.log_alpha.exp(), min=self._get_alpha_floor())
            alpha_loss_val = alpha_loss.item()

        # ── Soft target update ───────────────────────────────────────────
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

        # ── Book-keeping ─────────────────────────────────────────────────
        actor_loss = actor_loss.item()
        critic_loss = critic_loss.item()
        mean_log_prob = log_probs.mean().item()
        
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.alpha_losses.append(alpha_loss_val)
        self.entropy_values.append(-mean_log_prob)
        self.q_values.append(mean_q)

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss_val,
            "entropy": -mean_log_prob,
            "alpha": self.alpha.item(),
            "q_value": mean_q
        }

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
            self.alpha = torch.clamp(self.log_alpha.exp(), min=self._get_alpha_floor())
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