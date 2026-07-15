import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from src.config.config import REPLAY_BUFFER_SIZE, BATCH_SIZE_MULTIDAY_MINUTE


class UniformReplayBuffer:
    def __init__(
        self,
        observation_shape: Tuple[int, int],
        portfolio_state_len: int,
        action_dim: int,
        capacity: int = REPLAY_BUFFER_SIZE,
    ):
        """
        Args:
            observation_shape: (window_size, feature_dim) for market_data.
            action_dim: Size of action vector.
            capacity: Max number of transitions.
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.observation_shape = observation_shape
        window_size, feature_dim = observation_shape
        self._market_flat_len = window_size * feature_dim
        self._portfolio_state_len = portfolio_state_len
        state_dim = self._market_flat_len + self._portfolio_state_len

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        flattened_state = self._flatten_state(state)
        flattened_next_state = self._flatten_state(next_state)

        self.states[self.ptr] = flattened_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = flattened_next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = BATCH_SIZE_MULTIDAY_MINUTE) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self._unflatten_batch(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self._unflatten_batch(self.next_states[indices]),
            self.dones[indices],
        )

    def _unflatten_batch(self, flat_states: np.ndarray) -> Dict[str, np.ndarray]:
        batch_size = flat_states.shape[0]
        market_data = flat_states[:, : self._market_flat_len].reshape(
            batch_size, *self.observation_shape
        )
        portfolio_state = flat_states[:, self._market_flat_len :]
        return {
            "market_data": market_data.astype(np.float32),
            "portfolio_state": portfolio_state.astype(np.float32),
        }

    def _flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            state["market_data"].reshape(-1),
            state["portfolio_state"].ravel(),
        ], dtype=np.float32)

    def __len__(self):
        return self.size
    
    def save(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
            ptr=np.array([self.ptr]),
            size=np.array([self.size])
        )

    def load(self, path: Union[str, Path]) -> None:
        data = np.load(path)
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_states = data['next_states']
        self.dones = data['dones']
        self.ptr = int(data['ptr'][0])
        self.size = int(data['size'][0])
        self.capacity = self.states.shape[0]
        

# ---------------------------------------------------------------------------
# Recent-Emphasis Replay Buffer
# Older transitions are still kept (giving the agent a diverse history) but
# recent transitions are sampled more frequently via an exponential decay
# weight.  No TD-error bookkeeping is required, making this a drop-in
# replacement for the plain ReplayBuffer.
# ---------------------------------------------------------------------------

class RecentEmphasisReplayBuffer:
    def __init__(
        self, 
        observation_shape: Tuple[int, int],
        portfolio_state_len: int,
        action_dim, 
        capacity=REPLAY_BUFFER_SIZE, 
        decay=3.0
    ):
        self.capacity = capacity
        self.decay = decay
        self.ptr = 0
        self.size = 0
        self.observation_shape = observation_shape
        window_size, feature_dim = observation_shape
        self._market_flat_len = window_size * feature_dim
        self._portfolio_state_len = portfolio_state_len
        state_dim = self._market_flat_len + self._portfolio_state_len

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        flattened_state = self._flatten_state(state)
        flattened_next_state = self._flatten_state(next_state)
        
        self.states[self.ptr] = flattened_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = flattened_next_state
        self.dones[self.ptr] = done

        # Move pointer and update size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int=BATCH_SIZE_MULTIDAY_MINUTE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1. Create a temporary array of indices ordered by time
        # The buffer is circular, so "newest" is at ptr-1, "oldest" is at ptr
        
        # Fast geometric-like sampling without full array operations:
        # We want to sample index 'i' relative to current pointer
        # P(i) ~ exp(-decay * i / size)
        
        # Inverse Transform Sampling for speed:
        # u ~ Uniform(0,1)
        # index_offset = -ln(1 - u * (1 - exp(-decay))) / (decay / size)
        
        # Mathematical Approximation for speed (Linear decay mapping)
        # This avoids creating the full probability array every step
        
        u = np.random.rand(batch_size)
        # Maps u [0,1] to "normalized age" [0,1] with exponential bias
        # Using simple inverse transform of exponential distribution truncated at 1.0
        # CDF(x) = (1 - exp(-decay * x)) / (1 - exp(-decay))
        # This ensures we pick heavily from 0 (new) and sparsely from 1 (old)
        
        limit_val = 1 - np.exp(-self.decay)
        normalized_ages = -np.log(1 - u * limit_val) / self.decay
        
        # Convert normalized age to actual buffer indices
        # age 0 = self.ptr - 1
        # age 1 = self.ptr - size
        raw_indices = (self.ptr - 1 - (normalized_ages * self.size).astype(int)) % self.capacity
        
        return (
            self._unflatten_batch(self.states[raw_indices]),
            self.actions[raw_indices],
            self.rewards[raw_indices],
            self._unflatten_batch(self.next_states[raw_indices]),
            self.dones[raw_indices]
        )
        
    def _unflatten_batch(self, flat_states: np.ndarray) -> Dict[str, np.ndarray]:
        batch_size = flat_states.shape[0]
        market_data = flat_states[:, : self._market_flat_len].reshape(
            batch_size, *self.observation_shape
        )
        portfolio_state = flat_states[:, self._market_flat_len :]
        return {
            "market_data": market_data.astype(np.float32),
            "portfolio_state": portfolio_state.astype(np.float32),
        }

    def _flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            state["market_data"].reshape(-1),
            state["portfolio_state"].ravel(),
        ], dtype=np.float32)
        
    def __len__(self) -> int:
        return self.size

    def save(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
            ptr=np.array([self.ptr]),
            size=np.array([self.size]),
            decay=np.array([self.decay])
        )

    def load(self, path: Union[str, Path]) -> None:
        data = np.load(path)
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_states = data['next_states']
        self.dones = data['dones']
        self.ptr = int(data['ptr'][0])
        self.size = int(data['size'][0])
        self.decay = float(data['decay'][0])
        self.capacity = self.states.shape[0]