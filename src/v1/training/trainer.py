import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Union

from src.config.config import DATA_DIR, BATCH_SIZE, NUM_EPISODES, VALIDATION_INTERVAL, SAVE_MODEL_INTERVAL, MODELS_DIR, RESULTS_DIR
from src.v1.environment.environment import Environment
from src.utils.logger import Logger
from src.v1.model.agent import Agent
from src.utils.utils import create_directory, load_stock_data, format_duration

class Trainer:
    def __init__(
        self,
        agent: Agent,
        train_env: Environment,
        valid_env: Environment,
        randomize_trading_days: bool=False,
        batch_size: int=BATCH_SIZE,
        num_episodes: int = NUM_EPISODES,
        validation_interval: int = VALIDATION_INTERVAL,
        save_interval: int = SAVE_MODEL_INTERVAL,
        models_dir: Union[str, Path] = MODELS_DIR,
        results_dir: Union[str, Path] = RESULTS_DIR,
        logger: Optional[Logger]=None
    ):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.randomize_trading_days = randomize_trading_days
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.logger = logger
        
        create_directory(self.models_dir)
        create_directory(self.results_dir)
        
        self.episode_returns = []
        self.episode_rewards = []
        self.valid_rewards = []
        self.train_losses = []
        self.train_actions = []
        self.valid_actions = []
        
        if self.logger:
            self.logger.info(f'Trainer initialized: {num_episodes} episodes, {batch_size} samples per batch')
            
    def train(self) -> Dict[str, List[float]]:
        start_time = time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_randomized_start_idx_list = list(np.random.permutation(self.train_env.market_open_idx))
        valid_randomized_start_idx_list = list(np.random.permutation(self.valid_env.market_open_idx))

        if self.logger:
            self.logger.info('Training...')
        
        for episode in range(1, self.num_episodes + 1):
            if self.randomize_trading_days:
                self.train_env.current_step = train_randomized_start_idx_list.pop()
                
            state = self.train_env.reset()
            episode_reward = 0
            episode_loss = {"actor_loss": 0, "critic_loss": 0, "alpha_loss": 0, "entropy": 0, 'alpha': 0}
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.train_env.step(action)
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                self.train_actions.append(action)
                
                if len(self.agent.replay_buffer) > self.batch_size:
                    loss = self.agent.update_parameters(self.batch_size)
                    
                    for k, v in loss.items():
                        episode_loss[k] += v
                
                state = next_state
                episode_reward += reward
            
            self.episode_returns.append(info['total_return_pct'] * 100)
            self.episode_rewards.append(episode_reward)
            
            if self.train_env.current_step_in_episode > 0:
                for k in episode_loss:
                    episode_loss[k] /= self.train_env.current_step_in_episode
            self.train_losses.append(episode_loss)
            
            episode_start_date = self.train_env.timestamps[self.train_env.current_step - self.train_env.current_step_in_episode]
            episode_end_date = self.train_env.timestamps[self.train_env.current_step]
            
            if self.logger:
                # self.logger.info(f"\nEP: {episode}/{self.num_episodes}\
                #                    \nEpisode Start Date: {episode_start_date}\
                #                    \nEpisode End Date: {episode_end_date}\
                #                    \nReward: {episode_reward:.2f}\
                #                    \nBalance: ${info['balance']:.2f}\
                #                    \nTotal Return: ${info['total_return']:.2f}\
                #                    \nTotal Return PCT: {info['total_return_pct']:.2%}\
                #                    \nTotal Shares Purchased: {info['total_shares_purchased']}\
                #                    \nTotal Shares Sold: {info['total_shares_sold']}\
                #                    \nTotal Transaction Fee Penalty: {info['total_transaction_fee_penalty']}\
                #                    \nTotal Shares Held Penalty: {info['total_shares_held_penalty']}\
                #                    \nPenalty To Return PCT Ratio: {info['total_transaction_fee_penalty'] + info['total_shares_held_penalty']}/{abs(info['total_return_pct'])} = {(info['total_transaction_fee_penalty'] + info['total_shares_held_penalty'])/abs(info['total_return_pct'])}\
                #                    \n{'='*50}")
                
                self.logger.info(f"\nEP: {episode}/{self.num_episodes}\
                                   \nEpisode Start Date: {episode_start_date}\
                                   \nEpisode End Date: {episode_end_date}\
                                   \nReward: {episode_reward:.2f}\
                                   \nBalance: ${info['balance']:.2f}\
                                   \nTotal Return: {info['total_return_pct']:.2%}\
                                   \nTotal Shares Traded: {info['total_shares_purchased']}\
                                   \nTotal Transaction Fee Penalty: {info['total_transaction_fee_penalty']}\
                                   \nTotal Shares Held Penalty: {info['total_shares_held_penalty']}\
                                   \nPenalty To Return Ratio: {info['total_transaction_fee_penalty'] + info['total_shares_held_penalty']}/{abs(info['total_return_pct'])} = {(info['total_transaction_fee_penalty'] + info['total_shares_held_penalty'])/abs(info['total_return_pct'])}\
                                   \n{'='*50}")
            
            if episode % self.validation_interval == 0:
                if self.randomize_trading_days:
                    self.valid_env.current_step = valid_randomized_start_idx_list.pop()
                valid_reward = self.validate()
                self.valid_rewards.append(valid_reward)
            
            if episode % self.validation_interval == 0:
                self._plot_training_curves(timestamp)
        
        final_model_path = self.agent.save_model(self.models_dir, "final_", timestamp)
        if self.logger:
            self.logger.info(f"Final model saved: {final_model_path}")
        
        self._plot_training_curves(timestamp)
        
        if self.logger:
            total_time = time() - start_time
            self.logger.info(f"Training complete: {format_duration(total_time)})")
        
        return {
            "episode_rewards": self.episode_rewards,
            "valid_rewards": self.valid_rewards,
            "actor_losses": [loss["actor_loss"] for loss in self.train_losses],
            "critic_losses": [loss["critic_loss"] for loss in self.train_losses],
            "alpha_losses": [loss["alpha_loss"] for loss in self.train_losses],
            "entropy_values": [loss["entropy"] for loss in self.train_losses],
            "alphas": [loss["alpha"] for loss in self.train_losses]
        }
        
    def validate(self, num_episodes: int=1) -> float:
        total_reward = 0
        
        for episode in range(1, num_episodes + 1):
            state = self.valid_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, validate=True)
                next_state, reward, done, info = self.valid_env.step(action)
                state = next_state
                episode_reward += reward
                self.valid_actions.append(action)
            
            total_reward += episode_reward
            
            episode_start_date = self.valid_env.timestamps[self.valid_env.current_step - self.valid_env.current_step_in_episode]
            episode_end_date = self.valid_env.timestamps[self.valid_env.current_step]
        
            if self.logger:
                # self.logger.info(f"\nValid EP: {episode}/{num_episodes}\
                #                    \nEpisode Start Date: {episode_start_date}\
                #                    \nEpisode End Date: {episode_end_date}\
                #                    \nReward: {episode_reward:.2f}\
                #                    \nBalance: ${info['balance']:.2f}\
                #                    \nTotal Return: ${info['total_return']:.2f}\
                #                    \nTotal Return PCT: {info['total_return_pct']:.2%}%\
                #                    \nTotal Shares Purchased: {info['total_shares_purchased']}\
                #                    \nTotal Shares Sold: {info['total_shares_sold']}\
                #                    \n{'='*50}")
                
                self.logger.info(f"\nValid EP: {episode}/{num_episodes}\
                                   \nEpisode Start Date: {episode_start_date}\
                                   \nEpisode End Date: {episode_end_date}\
                                   \nReward: {episode_reward:.2f}\
                                   \nBalance: ${info['balance']:.2f}\
                                   \nTotal Return: {info['total_return_pct']:.2%}%\
                                   \nTotal Shares Traded: {info['total_shares_purchased']}\
                                   \n{'='*50}")
        
        validation_reward = total_reward / num_episodes
        if self.logger:
            self.logger.info(f"Validation Reward: {validation_reward:.2f}")
        return validation_reward
    
    def _plot_training_curves(self, timestamp: str) -> None:
        result_dir = self.results_dir / f"training_{timestamp}"
        create_directory(result_dir)
        
        def plot_learning_curve(data: List[float], title: str, xlabel: str, ylabel: str, ma_window: int = 100, save_path: Union[str, Path] = None) -> None:        
            plt.figure(figsize=(10, 6))
            plt.plot(data, alpha=0.3, color='blue', label=ylabel)
            
            if len(data) >= ma_window:
                ma_rewards = pd.Series(data).rolling(window=ma_window).mean().values
                plt.plot(ma_rewards, color='red', label=f'{ma_window} Moving Average')
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            if save_path:
                create_directory(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
            plt.close()
        
        def plot_actions() -> None:
            # Plot training prices with buy/sell actions overlaid
            if self.train_actions:
                plt.figure(figsize=(12, 6))
                # Use the first len(train_actions) prices from the training environment
                prices = self.train_env.prices[:len(self.train_actions)]
                x_idx = np.arange(len(prices))
                plt.plot(x_idx, prices.values, color='black', linewidth=1.0, label='Price')
                
                train_buys_x, train_buys_y, train_buys_c = [], [], []
                train_sells_x, train_sells_y, train_sells_c = [], [], []
                
                for i, a in enumerate(self.train_actions):
                    # a is typically a numpy array of shape (1,)
                    if isinstance(a, np.ndarray):
                        val = float(a[0])
                    elif isinstance(a, (list, tuple)):
                        val = float(a[0])
                    else:
                        val = float(a)
                    
                    if val > 0.0:
                        # Buy: green upward triangle at the price level, alpha ~ action strength
                        alpha = max(0.0, min(1.0, val))
                        train_buys_x.append(i)
                        train_buys_y.append(prices[i])
                        train_buys_c.append((0.0, 0.8, 0.0, alpha))
                    elif val < 0.0:
                        # Sell: red downward triangle at the price level, alpha ~ |action|
                        alpha = max(0.0, min(1.0, -val))
                        train_sells_x.append(i)
                        train_sells_y.append(prices[i])
                        train_sells_c.append((0.8, 0.0, 0.0, alpha))
                    # val == 0.0 is a hold -> ignore
                
                ax = plt.gca()
                if train_buys_x:
                    ax.scatter(train_buys_x, train_buys_y, marker='^', c=train_buys_c, edgecolors='none', label='Train Buy')
                if train_sells_x:
                    ax.scatter(train_sells_x, train_sells_y, marker='v', c=train_sells_c, edgecolors='none', label='Train Sell')
                
                ax.set_title('Training Price and Actions')
                ax.set_xlabel('Step')
                ax.set_ylabel('Price')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.savefig(result_dir / "train_price_actions.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot validation prices with buy/sell actions overlaid
            if self.valid_actions:
                plt.figure(figsize=(12, 6))
                prices = self.valid_env.prices[:len(self.valid_actions)]
                x_idx = np.arange(len(prices))
                plt.plot(x_idx, prices.values, color='black', linewidth=1.0, label='Price')
                
                valid_buys_x, valid_buys_y, valid_buys_c = [], [], []
                valid_sells_x, valid_sells_y, valid_sells_c = [], [], []
                
                for i, a in enumerate(self.valid_actions):
                    if isinstance(a, np.ndarray):
                        val = float(a[0])
                    elif isinstance(a, (list, tuple)):
                        val = float(a[0])
                    else:
                        val = float(a)
                    
                    if val > 0.0:
                        alpha = max(0.0, min(1.0, val))
                        valid_buys_x.append(i)
                        valid_buys_y.append(prices[i])
                        valid_buys_c.append((0.0, 0.8, 0.0, alpha))
                    elif val < 0.0:
                        alpha = max(0.0, min(1.0, -val))
                        valid_sells_x.append(i)
                        valid_sells_y.append(prices[i])
                        valid_sells_c.append((0.8, 0.0, 0.0, alpha))
                
                ax = plt.gca()
                if valid_buys_x:
                    ax.scatter(valid_buys_x, valid_buys_y, marker='^', c=valid_buys_c, edgecolors='none', label='Valid Buy')
                if valid_sells_x:
                    ax.scatter(valid_sells_x, valid_sells_y, marker='v', c=valid_sells_c, edgecolors='none', label='Valid Sell')
                
                ax.set_title('Validation Price and Actions')
                ax.set_xlabel('Step')
                ax.set_ylabel('Price')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.savefig(result_dir / "valid_price_actions.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        plot_learning_curve(
            data=self.episode_returns,
            title="Training Returns",
            xlabel="Episode",
            ylabel="Return",
            ma_window=10,
            save_path=result_dir / "episode_returns.png"
        )
        
        plot_learning_curve(
            data=self.episode_rewards,
            title="Training Rewards",
            xlabel="Episode",
            ylabel="Reward",
            ma_window=10,
            save_path=result_dir / "episode_rewards.png"
        )
        
        if self.train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot([loss["actor_loss"] for loss in self.train_losses])
            plt.title('Actor Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "actor_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot([loss["critic_loss"] for loss in self.train_losses])
            plt.title('Critic Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "critic_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot([loss["alpha_loss"] for loss in self.train_losses])
            plt.title('Alpha Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "alpha_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot([loss["entropy"] for loss in self.train_losses])
            plt.title('Policy Entropy')
            plt.xlabel('Episode')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "entropy.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot([loss["alpha"] for loss in self.train_losses])
            plt.title('Alpha')
            plt.xlabel('Episode')
            plt.ylabel('Alpha')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "alpha.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        if self.valid_rewards:
            plt.figure(figsize=(10, 6))
            x_vals = list(range(self.validation_interval, self.validation_interval * len(self.valid_rewards) + 1, self.validation_interval))
            plt.plot(x_vals, self.valid_rewards)
            plt.title('Validation Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "valid_rewards.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        if not self.randomize_trading_days:
            plot_actions()
        
        stats = {
            "episode_rewards": self.episode_rewards,
            "valid_rewards": self.valid_rewards,
            "train_losses": self.train_losses,
            "train_actions": self.train_actions,
            "valid_actions": self.valid_actions
        }
        torch.save(stats, result_dir / "training_stats.pth")
        
def main():
    np.random.seed(42)
    
    ticker = 'TSLA'
    train_data_dir = f'{DATA_DIR}/preprocessed/{ticker}/{ticker}_train.csv'
    train_data, _, _ = load_stock_data(train_data_dir)
    valid_data_dir = f'{DATA_DIR}/preprocessed/{ticker}/{ticker}_valid.csv'
    valid_data, _, _ = load_stock_data(valid_data_dir)
    
    train_env = Environment(data=train_data)
    valid_env = Environment(data=valid_data)
    
    action_dim = 1
    
    agent = Agent(
        action_dim=action_dim,
        input_shape=(train_env.window_size, train_env.feature_dim)
    )
    
    trainer = Trainer(
        agent=agent,
        train_env=train_env,
        valid_env=valid_env,
        randomize_trading_days=True,
        num_episodes=200,
        batch_size=256,
        validation_interval=10,
        save_interval=50,
        logger=Logger()
    )
    
    _ = trainer.train()
    
    print(f"Training complete: Final validation reward {trainer.valid_rewards[-1] if trainer.valid_rewards else 'N/A'}") 
            
if __name__ == "__main__":
    main()