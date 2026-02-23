import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.config.config import MODELS_DIR, BATCH_SIZE, HIDDEN_DIM, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, LEARNING_RATE_ALPHA, GAMMA, TAU, ALPHA_INIT, TARGET_UPDATE_INTERVAL, DEVICE, REPLAY_BUFFER_SIZE
from src.v1.environment.environment import Environment
from src.v1.model.networks import Actor, Critic
from src.v1.model.replay_buffer import UniformReplayBuffer
from src.utils.logger import Logger
from src.utils.utils import create_directory, load_stock_data

class Agent:
    def __init__(
        self,
        action_dim: int=1,
        hidden_dim: int=HIDDEN_DIM,
        actor_lr: float=LEARNING_RATE_ACTOR,
        critic_lr: float=LEARNING_RATE_CRITIC,
        alpha_lr: float=LEARNING_RATE_ALPHA,
        gamma: float=GAMMA,
        tau: float=TAU,
        alpha_init: float=ALPHA_INIT,
        target_update_interval: int=TARGET_UPDATE_INTERVAL,
        use_automatic_entropy_tuning: bool=True,
        device: torch.device=DEVICE,
        buffer_capacity: int=REPLAY_BUFFER_SIZE,
        input_shape: Tuple[int, int]=None,
        portfolio_state_len: int=None,
        logger: Optional[Logger]=None
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_init = alpha_init
        self.target_update_interval = target_update_interval
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.device = device
        self.logger = logger
        
        self.actor = Actor(
            input_shape=input_shape,
            portfolio_state_length=portfolio_state_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.critic = Critic(
            input_shape=input_shape,
            portfolio_state_length=portfolio_state_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.critic_target = Critic(
            input_shape=input_shape,
            portfolio_state_length=portfolio_state_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -action_dim 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor(alpha_init, device=device)
        
        self.replay_buffer = UniformReplayBuffer(
            observation_shape=input_shape,
            portfolio_state_len=portfolio_state_len,
            action_dim=action_dim,
            capacity=buffer_capacity,
        )
        
        self.train_step_counter = 0
        
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.entropy_values = []
        
    def select_action(self, state: Any, validate: bool=False) -> np.ndarray:
        state_tensor = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state_tensor[key] = torch.tensor(value, dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                state_tensor[key] = value.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if validate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy()[0]
    
    def _batch_dict_to_tensor(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert dict of batched numpy arrays to dict of tensors on device."""
        return {
            k: torch.as_tensor(v, dtype=torch.float, device=self.device)
            for k, v in state_dict.items()
        }

    def process_state_for_network(self, state: Any) -> Any:
        if isinstance(state, dict):
            state_dict = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    state_dict[key] = torch.tensor(value, dtype=torch.float, device=self.device)
                else:
                    state_dict[key] = value.to(self.device)
            return state_dict
        else:
            if self.logger:
                self.logger.error('State type must be a dictionary')
            return None
        
    def update_parameters(self, batch_size: int=BATCH_SIZE) -> Dict[str, float]:
        # Skip if not enough samples in buffer
        if len(self.replay_buffer) < batch_size:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'alpha_loss': 0.0,
                'entropy': 0.0,
                'alpha': self.alpha.item()
            }
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        batched_states = self._batch_dict_to_tensor(states)
        batched_next_states = self._batch_dict_to_tensor(next_states)

        batched_actions = torch.as_tensor(np.vstack(actions), dtype=torch.float, device=self.device)
        batched_rewards = torch.as_tensor(np.vstack(rewards), dtype=torch.float, device=self.device)
        batched_dones = torch.as_tensor(np.vstack(dones), dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(batched_next_states)
            next_q1_target, next_q2_target = self.critic_target(batched_next_states, next_actions)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_q_target = next_q_target - self.alpha * next_log_probs
            expected_q = batched_rewards + (1.0 - batched_dones) * self.gamma * next_q_target
        
        # Update critic
        current_q1, current_q2 = self.critic(batched_states, batched_actions)
        q1_loss = F.mse_loss(current_q1, expected_q.detach())
        q2_loss = F.mse_loss(current_q2, expected_q.detach())
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.actor.sample(batched_states)
        q1, q2 = self.critic(batched_states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Automatic entropy tuning
        alpha_loss = 0.0
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
        
        # Record training results
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.alpha_losses.append(alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0)
        self.entropy_values.append(-log_probs.mean().item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item() if self.use_automatic_entropy_tuning else 0.0,
            'entropy': -log_probs.mean().item(),
            'alpha': self.alpha.item()
        }
        
    def save_model(self, save_dir: Union[str, Path]=MODELS_DIR, prefix: str='', timestamp: Optional[str]=None) -> None:        
        create_directory(save_dir)
        
        timestamp = timestamp if timestamp else time.strftime("%Y%m%d_%H%M%S") 
        model_path = Path(save_dir) / f"{prefix}sac_model_{timestamp}"
        create_directory(model_path)
        
        # NN
        torch.save(self.actor.state_dict(), model_path / "actor.pth")
        torch.save(self.critic.state_dict(), model_path / "critic.pth")
        torch.save(self.critic_target.state_dict(), model_path / "critic_target.pth")
        
        # Optimizer
        torch.save(self.actor_optimizer.state_dict(), model_path / "actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), model_path / "critic_optimizer.pth")
        
        # Alpha
        if self.use_automatic_entropy_tuning:
            torch.save(self.log_alpha, model_path / "log_alpha.pth")
            torch.save(self.alpha_optimizer.state_dict(), model_path / "alpha_optimizer.pth")
        
        # Training stats
        training_stats = {
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_losses': self.alpha_losses,
            'entropy_values': self.entropy_values,
            'train_step_counter': self.train_step_counter
        }
        torch.save(training_stats, model_path / "training_stats.pth")
        
        # Agent config
        config = {
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha_init': self.alpha_init,
            'target_update_interval': self.target_update_interval,
            'use_automatic_entropy_tuning': self.use_automatic_entropy_tuning,
        }
        torch.save(config, model_path / "config.pth")
        
        if self.logger:
            self.logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        
        if not model_path.exists():
            if self.logger:
                self.logger.error(f"Model path does not exist: {model_path}")
            return
        
        # NN
        self.actor.load_state_dict(torch.load(model_path / "actor.pth", map_location=self.device, weights_only=False))
        self.critic.load_state_dict(torch.load(model_path / "critic.pth", map_location=self.device, weights_only=False))
        self.critic_target.load_state_dict(torch.load(model_path / "critic_target.pth", map_location=self.device, weights_only=False))
        
        # Optimizer
        self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device, weights_only=False))
        self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device, weights_only=False))
        
        # Alpha
        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device, weights_only=False)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device, weights_only=False))
        
        # Training stats
        training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device, weights_only=False)
        self.actor_losses = training_stats['actor_losses']
        self.critic_losses = training_stats['critic_losses']
        self.alpha_losses = training_stats['alpha_losses']
        self.entropy_values = training_stats['entropy_values']
        self.train_step_counter = training_stats['train_step_counter']
        
        if self.logger:
            self.logger.info(f"Model loaded: {model_path}")
            
    def get_latest_model_path(self, save_dir: Union[str, Path]=None, prefix: str='') -> Optional[Path]:
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir = Path(save_dir)
        if not save_dir.exists():
            return None
        
        model_dirs = [dir for dir in save_dir.iterdir() if dir.is_dir() and dir.name.startswith(f"{prefix}sac_model_")]
        if not model_dirs:
            return None
        
        model_dirs.sort(key=lambda dir: dir.name, reverse=True)
        return model_dirs[0]
   
def main():
    import math
    
    from src.config.config import DATA_DIR
    
    ticker = 'TSLA'
    data_dir = f'{DATA_DIR}/preprocessed/v1/{ticker}/{ticker}_train.csv'
    data = load_stock_data(data_dir)
    
    env = Environment(data=data, logger=Logger())
    action_dim = env.action_space.shape[0]
    batch_size = BATCH_SIZE
    window_size = env.window_size
    feature_dim = env.feature_dim
    portfolio_dim = env.observation_space['portfolio_state'].shape[0]
    
    agent = Agent(
        action_dim=action_dim,
        input_shape=(window_size, feature_dim),
        portfolio_state_len=portfolio_dim,
        logger=Logger()
    )
    
    for _ in range(batch_size * 2):
        state = {
            'market_data': np.random.randn(window_size, feature_dim),
            'portfolio_state': np.random.randn(2)
        }
        action = np.random.randn(action_dim)
        reward = np.random.randn(1)
        next_state = {
            'market_data': np.random.randn(window_size, feature_dim),
            'portfolio_state': np.random.randn(2)
        }
        done = np.random.randint(0, 2, (1,))
        
        # agent.replay_buffer.push(state, torch.tensor(action, dtype=torch.float, device=agent.device), reward, next_state, done)
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    for _ in range(5):
        stats = agent.update_parameters()
        print(f"CNN Learning stats: {stats}")
    
    model_path = agent.save_model(prefix='cnn_')
    agent.load_model(model_path) 
    
if __name__ == "__main__":
    main()