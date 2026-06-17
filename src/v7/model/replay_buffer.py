import numpy as np
from typing import Any, Dict, Tuple

from src.config.config import REPLAY_BUFFER_SIZE, BATCH_SIZE


class DailyReplayBuffer:
    """
    v7 replay buffer for multi-ticker allocation training.

    Stores full flattened states because transitions across different
    fold/episode start_idx pairs see different market_data slices.

    v7 architecture is MLP-on-current-bar (no window dim), so market_data
    per state is just (N, F_per_ticker). At the v7 defaults (N=5, F=43,
    temporal=6, regime=45, portfolio=22, action=5, cap=200k), the
    per-state width is 288 floats × 4 bytes ≈ 1.1 KB × 2 sides × 200k
    entries ≈ 460 MB. Substantial reduction vs the windowed v6 layout
    (which would have been ~21 GB at the same cap with N=5, W=60).

    Sampling: recent-emphasis inverse-transform exponential decay,
    decay=3.0 default carried over from v6 pending the planned v7 decay
    sweep (see dev_log_v7.md "Recency-emphasis sweep"). decay=0 is
    special-cased to uniform.

    State dict keys: market_data (B, N, F), temporal (B, T),
                     regime (B, R), portfolio_state (B, P).
    """

    def __init__(
        self,
        n_tickers: int,
        n_market_per_ticker: int,
        n_temporal: int,
        n_regime: int,
        portfolio_state_len: int,
        action_dim: int,
        capacity: int = REPLAY_BUFFER_SIZE,
        decay: float = 3.0,
    ):
        self.capacity = capacity
        self.decay = decay
        self.ptr = 0
        self.size = 0

        self.n_tickers = n_tickers
        self.n_market_per_ticker = n_market_per_ticker
        self.n_temporal = n_temporal
        self.n_regime = n_regime
        self.portfolio_state_len = portfolio_state_len

        # Flat state dim: market (N*F) + temporal (T) + regime (R) + portfolio (P).
        self._market_flat = n_tickers * n_market_per_ticker
        self._state_dim = (
            self._market_flat + n_temporal + n_regime + portfolio_state_len
        )

        self.states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def _flatten(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Pack a state dict into a 1-D float32 row.

        Order: market_data (flattened N*F), temporal (T), regime (R),
        portfolio_state (P). Matches _unflatten_batch ordering.
        """
        return np.concatenate(
            [
                state["market_data"].reshape(-1),
                state["temporal"].ravel(),
                state["regime"].ravel(),
                state["portfolio_state"].ravel(),
            ],
            dtype=np.float32,
        )

    def _unflatten_batch(self, flat: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Inverse of _flatten for a (B, _state_dim) batch.

        Restores market_data to (B, N, F) — the shape FeatureExtractor expects.
        """
        B = flat.shape[0]
        idx = 0

        market = flat[:, idx : idx + self._market_flat].reshape(
            B, self.n_tickers, self.n_market_per_ticker
        )
        idx += self._market_flat

        temporal = flat[:, idx : idx + self.n_temporal]
        idx += self.n_temporal

        regime = flat[:, idx : idx + self.n_regime]
        idx += self.n_regime

        portfolio = flat[:, idx : idx + self.portfolio_state_len]

        return {
            "market_data": market.astype(np.float32),
            "temporal": temporal.astype(np.float32),
            "regime": regime.astype(np.float32),
            "portfolio_state": portfolio.astype(np.float32),
        }

    def push(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: float,
    ):
        self.states[self.ptr] = self._flatten(state)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = self._flatten(next_state)
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = BATCH_SIZE
    ) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        Dict[str, np.ndarray],
        np.ndarray,
    ]:
        """
        Sample a batch with exponential recency bias. Identical math to v6.

        Inverse-CDF of truncated exponential maps uniform u in [0,1] to
        a normalized age in [0,1] with heavy concentration near 0 (recent).
        At decay=3.0: ~60% of samples from newest 30% of buffer, ~10%
        from oldest third.

        Note: the v7 buffer dynamics differ from v6 (no cross-ticker
        mixing within episodes), which is why a decay sweep on the v7 setup
        is planned before locking the run-1 baseline. See dev_log_v7.md.
        """
        u = np.random.rand(batch_size)

        # decay=0 → uniform sampling. The inverse-CDF formula below has a
        # 0/0 singularity at decay=0 (limit_val=0 and division by decay).
        # Treat it as the obvious limiting case: pick any valid buffer
        # index uniformly. Matches the planned decay sweep candidate set
        # {0.0 uniform, 1.0 mild, 3.0 v6 default}.
        if self.decay <= 1e-9:
            normalized_ages = u
        else:
            limit_val = 1 - np.exp(-self.decay)
            normalized_ages = -np.log(1 - u * limit_val) / self.decay

        indices = (
            self.ptr - 1 - (normalized_ages * self.size).astype(int)
        ) % self.capacity

        return (
            self._unflatten_batch(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self._unflatten_batch(self.next_states[indices]),
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size