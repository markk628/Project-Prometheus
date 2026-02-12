import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from src.config.config import HIDDEN_DIM, DEVICE

class Actor(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int], # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        super(Actor, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        self.conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        conv_output_size = 128 * (self.window_size // 4)
        
        self.portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        self.fc1 = nn.Linear(conv_output_size + hidden_dim // 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.to(device)
        
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State dictionary
                   {'market_data': market data, 'portfolio_state': portfolio state}
            
        Returns:
            Tuple of (mean tensor, log standard deviation tensor)
        """
        # Process market data
        # Convert shape from (B, W, F) to (B, F, W)
        market_data = state['market_data']
        
        if len(market_data.shape) == 3: # (B, W, F)
            market_data = market_data.permute(0, 2, 1)
        elif len(market_data.shape) == 2: # (W, F)
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)
        
        # Process portfolio state
        portfolio_state = state['portfolio_state']
        
        if len(portfolio_state.shape) == 1: # (2,)
            portfolio_state = portfolio_state.unsqueeze(0) # (1, 2)
        
        x = F.relu(self.conv1(market_data))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        p = F.relu(self.portfolio_fc(portfolio_state))
        
        combined = torch.cat([x, p], dim=1)
        
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the given state
        
        Args:
            state: State dictionary
            
        Returns:
            Tuple of (action, log probability, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from a normal distribution using the reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply Tanh transform to constrain action range to (-1, 1)
        y_t = torch.tanh(x_t)
        
        # Compute log probability of the policy
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def to(self, device: torch.device) -> 'Actor':
        self.device = device
        return super(Actor, self).to(device)


class Critic(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int], # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        device: torch.device = DEVICE
    ):
        super(Critic, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.q1_conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.q1_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.q1_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.q2_conv1 = nn.Conv1d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)
        self.q2_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.q2_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        conv_output_size = 128 * (self.window_size // 4)
        
        self.q1_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        self.q2_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        self.q1_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        self.q2_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        
        self.q1_fc1 = nn.Linear(conv_output_size + hidden_dim // 2, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(conv_output_size + hidden_dim // 2, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        self.to(device)
        
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State dictionary
                   {'market_data': market data, 'portfolio_state': portfolio state}
            action: Action tensor
            
        Returns:
            Tuple of two Q-values
        """
        # Process market data
        # Convert shape from (B, W, F) to (B, F, W)
        market_data = state['market_data']
        
        if len(market_data.shape) == 3:  # (B, W, F)
            market_data = market_data.permute(0, 2, 1)
        elif len(market_data.shape) == 2:  # (W, F)
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)
        
        # Process portfolio state
        portfolio_state = state['portfolio_state']
        
        if len(portfolio_state.shape) == 1: # (2,)
            portfolio_state = portfolio_state.unsqueeze(0) # (1, 2)
        
        if len(action.shape) == 1: # (action_dim,)
            action = action.unsqueeze(0) # (1, action_dim)
        
        q1_x = F.relu(self.q1_conv1(market_data))
        q1_x = self.pool(q1_x)
        q1_x = F.relu(self.q1_conv2(q1_x))
        q1_x = self.pool(q1_x)
        q1_x = F.relu(self.q1_conv3(q1_x))
        q1_x = q1_x.view(q1_x.size(0), -1)
        
        q1_p = F.relu(self.q1_portfolio_fc(portfolio_state))
        
        q1_a = F.relu(self.q1_action_fc(action))
        
        q1_combined = torch.cat([q1_x, q1_p, q1_a], dim=1)
        
        q1_x = F.relu(self.q1_fc1(q1_combined))
        q1_x = F.relu(self.q1_fc2(q1_x))
        q1 = self.q1(q1_x)
        
        q2_x = F.relu(self.q2_conv1(market_data))
        q2_x = self.pool(q2_x)
        q2_x = F.relu(self.q2_conv2(q2_x))
        q2_x = self.pool(q2_x)
        q2_x = F.relu(self.q2_conv3(q2_x))
        q2_x = q2_x.view(q2_x.size(0), -1)
        
        q2_p = F.relu(self.q2_portfolio_fc(portfolio_state))
        
        q2_a = F.relu(self.q2_action_fc(action))
        
        q2_combined = torch.cat([q2_x, q2_p, q2_a], dim=1)
        
        q2_x = F.relu(self.q2_fc1(q2_combined))
        q2_x = F.relu(self.q2_fc2(q2_x))
        q2 = self.q2(q2_x)
        
        return q1, q2
    
    def to(self, device: torch.device) -> 'Critic':
        self.device = device
        return super(Critic, self).to(device)