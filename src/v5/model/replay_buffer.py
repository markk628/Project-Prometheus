import numpy as np
from typing import Any, Dict, Tuple

from src.config.config import REPLAY_BUFFER_SIZE, BATCH_SIZE

# TODO compare results with Uniformly sampling ReplayBuffer

class DailyReplayBuffer:
    """
    Replay buffer for daily multi-ticker training with recent-emphasis sampling.

    Unlike IndexReplayBuffer (which stores indices into a single shared
    market_data array), this buffer stores full flattened states. This is
    necessary because transitions come from different tickers with
    different data arrays.

    Sampling uses inverse-transform exponential decay so recent transitions
    are replayed more frequently. Older transitions are still kept for
    diversity (crash data, different regimes) but sampled less often.
    This is the same approach as RecentEmphasisReplayBuffer but adapted
    for the 4-key state dict (market, temporal, regime, portfolio).

    Memory is manageable for daily data: a 60-day window of 128 features
    = 7,680 floats per state (~30KB). At 200k capacity that's ~6GB.

    State dict keys: market_data (B, W, F), temporal (B, T),
                     regime (B, R), portfolio_state (B, P)
    """

    def __init__(
        self,
        window_size: int,
        n_market: int,
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

        self.window_size = window_size
        self.n_market = n_market
        self.n_temporal = n_temporal
        self.n_regime = n_regime
        self.portfolio_state_len = portfolio_state_len

        # Flat state dim: market window + temporal + regime + portfolio
        self._market_flat = window_size * n_market
        self._state_dim = self._market_flat + n_temporal + n_regime + portfolio_state_len

        self.states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def _flatten(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            state["market_data"].reshape(-1),
            state["temporal"].ravel(),
            state["regime"].ravel(),
            state["portfolio_state"].ravel(),
        ], dtype=np.float32)

    def _unflatten_batch(self, flat: np.ndarray) -> Dict[str, np.ndarray]:
        B = flat.shape[0]
        idx = 0

        market = flat[:, idx:idx + self._market_flat].reshape(B, self.window_size, self.n_market)
        idx += self._market_flat

        temporal = flat[:, idx:idx + self.n_temporal]
        idx += self.n_temporal

        regime = flat[:, idx:idx + self.n_regime]
        idx += self.n_regime

        portfolio = flat[:, idx:idx + self.portfolio_state_len]

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
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Sample a batch with exponential recency bias.

        Uses inverse transform sampling of a truncated exponential
        distribution. P(transition) ~ exp(-decay * age/size), where
        age 0 = most recently pushed, age size-1 = oldest in buffer.

        With decay=3.0: ~60% of samples come from the newest 30% of
        the buffer, ~30% from the middle, ~10% from the oldest third.
        """
        u = np.random.rand(batch_size)

        # Inverse CDF of truncated exponential: maps uniform [0,1]
        # to normalized age [0,1] with heavy bias toward 0 (recent)
        limit_val = 1 - np.exp(-self.decay)
        normalized_ages = -np.log(1 - u * limit_val) / self.decay

        # Convert normalized age to circular buffer indices
        # age 0 → ptr-1 (newest), age 1 → ptr-size (oldest)
        indices = (self.ptr - 1 - (normalized_ages * self.size).astype(int)) % self.capacity

        return (
            self._unflatten_batch(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self._unflatten_batch(self.next_states[indices]),
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size