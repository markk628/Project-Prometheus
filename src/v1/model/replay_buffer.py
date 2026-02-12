import numpy as np
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, Union

from src.config.config import REPLAY_BUFFER_SIZE

# TODO make a prioritized replay buffer class 

class ReplayBuffer:
    def __init__(self, capacity: int=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: Union[str, Path]) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, path: Union[str, Path]) -> None:
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)