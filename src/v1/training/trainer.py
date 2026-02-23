import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Union

from src.config.config import DATA_DIR, BATCH_SIZE, NUM_EPISODES, VALID_INTERVAL, SAVE_MODEL_INTERVAL, MODELS_DIR, RESULTS_DIR, SEED
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
        num_episodes: int=NUM_EPISODES,
        valid_interval: int=VALID_INTERVAL,
        save_interval: int=SAVE_MODEL_INTERVAL,
        models_dir: Union[str, Path]=MODELS_DIR,
        results_dir: Union[str, Path]=RESULTS_DIR,
        logger: Optional[Logger]=None
    ):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.randomize_trading_days = randomize_trading_days
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.valid_interval = valid_interval
        self.save_interval = save_interval
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.logger = logger
        
        create_directory(self.models_dir)
        create_directory(self.results_dir)
        
        self.train_returns = []
        self.valid_returns = []
        self.train_rewards = []
        self.valid_rewards = []
        self.train_fee_impacts = []
        self.valid_fee_impacts = []
        self.train_total_trade_counts = []
        self.valid_total_trade_counts = []
        self.train_turnover_ratios = []
        self.valid_turnover_ratios = []
        self.train_avg_hold_times = []
        self.valid_avg_hold_times = []
        self.train_invalid_actions_counts = []
        self.valid_invalid_actions_counts = []
        
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
            episode_train_rewards = []
            train_reward = 0
            train_loss = {"actor_loss": 0, "critic_loss": 0, "alpha_loss": 0, "entropy": 0, 'alpha': 0}
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                action = self.train_env.mask_action(action)
                next_state, reward, done, info = self.train_env.step(action)
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                self.train_actions.append(action)
                
                if len(self.agent.replay_buffer) > self.batch_size:
                    loss = self.agent.update_parameters(self.batch_size)
                    
                    for k, v in loss.items():
                        train_loss[k] += v
                
                state = next_state
                train_reward += reward
                episode_train_rewards.append(reward)
            
            net_return_pct = info['net_return_pct']
            self.train_returns.append(net_return_pct * 100)
            self.train_rewards.append(train_reward)
            
            gross_return_pct = info['gross_return_pct']
            fee_impact = (gross_return_pct - net_return_pct)
            self.train_fee_impacts.append(fee_impact * 100)
            
            trade_execution_count = info['trade_execution_count']
            self.train_total_trade_counts.append(trade_execution_count)
            
            turnover_ratio = info['turnover_ratio']
            self.train_turnover_ratios.append(turnover_ratio * 100)
            
            avg_hold_time = info['avg_hold_time']
            self.train_avg_hold_times.append(avg_hold_time)
            
            invalid_actions_count = info['invalid_acitons_count']
            self.train_invalid_actions_counts.append(invalid_actions_count)
            
            if self.train_env.current_step_in_episode > 0:
                for k in train_loss:
                    train_loss[k] /= self.train_env.current_step_in_episode
            self.train_losses.append(train_loss)
            
            episode_start_date = self.train_env.timestamps[self.train_env.current_step - self.train_env.current_step_in_episode]
            episode_end_date = self.train_env.timestamps[self.train_env.current_step]
            
            if self.logger:
                episode_train_rewards = np.array(episode_train_rewards)
                self.logger.info(f"\nEP: {episode}/{self.num_episodes}\
                                   \nEpisode Start Date: {episode_start_date}\
                                   \nEpisode End Date: {episode_end_date}\
                                   \nReward Total: {train_reward:.2f}\
                                   \nReward Mean: {episode_train_rewards.mean()}\
                                   \nReward STD: {episode_train_rewards.std()}\
                                   \nReward Min: {episode_train_rewards.min()}\
                                   \nReward Max: {episode_train_rewards.max()}\
                                   \nPositive Reward PCT: {np.mean(episode_train_rewards > 0):.2%}\
                                   \nZero-ish Reward PCT: {np.mean(np.abs(episode_train_rewards) < 1e-6):.2%}\
                                   \nBalance: ${info['balance']:.2f}\
                                   \nGross Return (Pre-Fee): {gross_return_pct:.2%}\
                                   \nNet Return (Post-Fee): {net_return_pct:.2%} {'POSITIVE' if net_return_pct > 0 else ''}\
                                   \nFee Impact: {fee_impact:.2%}\
                                   \nTotal Trade Count: {trade_execution_count}\
                                   \nTurnover Ratio: {turnover_ratio:.2%}\
                                   \nAvg Hold Time: {avg_hold_time}\
                                   \nInvalid Actions: {invalid_actions_count}\
                                   \nTotal Shares Traded: {info['total_shares_sold']}\
                                   \n{'='*50}")
            
            if episode % self.valid_interval == 0:
                if self.randomize_trading_days:
                    self.valid_env.current_step = valid_randomized_start_idx_list.pop()
                self.validate()
            
            if episode % self.valid_interval == 0:
                self._plot_training_curves(timestamp)
        
        final_model_path = self.agent.save_model(self.models_dir, "final_", timestamp)
        if self.logger:
            self.logger.info(f"Final model saved: {final_model_path}")
        
        self._plot_training_curves(timestamp)
        
        if self.logger:
            total_time = time() - start_time
            positive_return_train = 0
            positive_return_valid = 0
            
            for train_return in self.train_returns:
                if train_return > 0:
                    positive_return_train += 1
            
            for valid_return in self.valid_returns:
                if valid_return > 0:
                    positive_return_valid += 1
                    
            self.logger.info(f"Training complete: {format_duration(total_time)})\
                              \nPositive Return Training: {positive_return_train}/{self.num_episodes}\
                              \nPositive Returns Validation: {positive_return_valid}/{int(self.num_episodes / self.valid_interval)}")
        
        return {
            "train_rewards": self.train_rewards,
            "valid_rewards": self.valid_rewards,
            "actor_losses": [loss["actor_loss"] for loss in self.train_losses],
            "critic_losses": [loss["critic_loss"] for loss in self.train_losses],
            "alpha_losses": [loss["alpha_loss"] for loss in self.train_losses],
            "entropy_values": [loss["entropy"] for loss in self.train_losses],
            "alphas": [loss["alpha"] for loss in self.train_losses]
        }
        
    def validate(self, num_episodes: int=1):
        for episode in range(1, num_episodes + 1):
            state = self.valid_env.reset()
            episode_valid_rewards = []
            valid_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, validate=True)
                action = self.valid_env.mask_action(action)
                next_state, reward, done, info = self.valid_env.step(action)
                state = next_state
                valid_reward += reward
                episode_valid_rewards.append(reward)
                self.valid_actions.append(action)
            
            net_return_pct = info['net_return_pct']
            self.valid_returns.append(net_return_pct * 100)
            self.valid_rewards.append(valid_reward)
            
            gross_return_pct = info['gross_return_pct']
            fee_impact = (gross_return_pct - net_return_pct)
            self.valid_fee_impacts.append(fee_impact * 100)
            
            trade_execution_count = info['trade_execution_count']
            self.valid_total_trade_counts.append(trade_execution_count)
            
            turnover_ratio = info['turnover_ratio']
            self.valid_turnover_ratios.append(turnover_ratio * 100)
            
            avg_hold_time = info['avg_hold_time']
            self.valid_avg_hold_times.append(avg_hold_time)
            
            invalid_actions_count = info['invalid_acitons_count']
            self.valid_invalid_actions_counts.append(invalid_actions_count)
            
            episode_start_date = self.valid_env.timestamps[self.valid_env.current_step - self.valid_env.current_step_in_episode]
            episode_end_date = self.valid_env.timestamps[self.valid_env.current_step]
        
            if self.logger:
                episode_valid_rewards = np.array(episode_valid_rewards)
                self.logger.info(f"\nValid EP: {episode}/{num_episodes}\
                                   \nEpisode Start Date: {episode_start_date}\
                                   \nEpisode End Date: {episode_end_date}\
                                   \nReward Total: {valid_reward:.2f}\
                                   \nReward Mean: {episode_valid_rewards.mean()}\
                                   \nReward STD: {episode_valid_rewards.std()}\
                                   \nReward Min: {episode_valid_rewards.min()}\
                                   \nReward Max: {episode_valid_rewards.max()}\
                                   \nPositive Reward PCT: {np.mean(episode_valid_rewards > 0):.2%}\
                                   \nZero-ish Reward PCT: {np.mean(np.abs(episode_valid_rewards) < 1e-6):.2%}\
                                   \nBalance: ${info['balance']:.2f}\
                                   \nGross Return (Pre-Fee): {gross_return_pct:.2%}\
                                   \nNet Return (Post-Fee): {net_return_pct:.2%} {'POSITIVE' if net_return_pct > 0 else ''}\
                                   \nFee Impact: {fee_impact:.2%}\
                                   \nTotal Trade Count: {trade_execution_count}\
                                   \nTurnover Ratio: {turnover_ratio:.2%}\
                                   \nAvg Hold Time: {avg_hold_time}\
                                   \nInvalid Actions: {invalid_actions_count}\
                                   \nTotal Shares Traded: {info['total_shares_sold']}\
                                   \n{'='*50}")
        
    def _plot_training_curves(self, timestamp: str) -> None:
        result_dir = self.results_dir / f"training_{timestamp}"
        loss_dir = result_dir / "loss"
        metrics_dir = result_dir / "metrics"
        create_directory(result_dir)
        create_directory(loss_dir)
        create_directory(metrics_dir)
            
        metrics = [
            {
                "train_data": self.train_returns,
                "valid_data": self.valid_returns,
                "title": "Returns",
                "ylabel": "Return",
                "filename": "returns.png"
            },
            {
                "train_data": self.train_rewards,
                "valid_data": self.valid_rewards,
                "title": "Rewards",
                "ylabel": "Reward",
                "filename": "rewards.png"
            },
            {
                "train_data": self.train_fee_impacts,
                "valid_data": self.valid_fee_impacts,
                "title": "Fee Impacts",
                "ylabel": "Fee Impact",
                "filename": "fee_impacts.png"
            },
            {
                "train_data": self.train_total_trade_counts,
                "valid_data": self.valid_total_trade_counts,
                "title": "Trade Counts",
                "ylabel": "Trade Count",
                "filename": "total_trade_counts.png"
            },
            {
                "train_data": self.train_turnover_ratios,
                "valid_data": self.valid_turnover_ratios,
                "title": "Turnover Ratios",
                "ylabel": "Turnover Ratio",
                "filename": "turnover_ratios.png"
            },
            {
                "train_data": self.train_avg_hold_times,
                "valid_data": self.valid_avg_hold_times,
                "title": "Average Hold Times",
                "ylabel": "Average Hold Time",
                "filename": "avg_hold_times.png"
            },
            # {
            #     "train_data": self.train_invalid_actions_counts,
            #     "valid_data": self.valid_invalid_actions_counts,
            #     "title": "Invalid Actions",
            #     "ylabel": "Invalid Action Count",
            #     "filename": "invalid_action_counts.png"
            # }
        ]

        loss_plots = [
            {
                "key": "actor_loss",
                "title": "Actor Losses",
                "ylabel": "Loss",
                "filename": "actor_loss.png"
            },
            {
                "key": "critic_loss",
                "title": "Critic Losses",
                "ylabel": "Loss",
                "filename": "critic_loss.png"
            },
            {
                "key": "alpha_loss",
                "title": "Alpha Losses",
                "ylabel": "Loss",
                "filename": "alpha_loss.png"
            },
            {
                "key": "entropy",
                "title": "Policy Entropy",
                "ylabel": "Entropy",
                "filename": "entropy.png"
            },
            {
                "key": "alpha",
                "title": "Alpha",
                "ylabel": "Alpha",
                "filename": "alpha.png"
            },
        ]

        for metric in metrics:
            train_data = metric["train_data"]
            valid_data = metric["valid_data"]
            title = metric["title"]
            ylabel = metric["ylabel"]
            filename = metric["filename"]

            plt.figure(figsize=(10, 6))
            plt.plot(train_data, alpha=0.3, color='blue', label=f'Train {ylabel}')
            if len(train_data) >= self.valid_interval:
                ma = pd.Series(train_data).rolling(window=self.valid_interval).mean().values
                plt.plot(ma, color='blue', linewidth=1.5, label=f'Train {self.valid_interval}-ep MA')
            if valid_data:
                x_vals = list(range(self.valid_interval, self.valid_interval * len(valid_data) + 1, self.valid_interval))
                plt.plot(x_vals, valid_data, color='orange', marker='o', markersize=4, label='Validation')
            
            if title in ["Returns", "Rewards"]:
                train_above_zero = sum(1 for x in train_data if x > 0)
                train_total = len(train_data)

                train_pct = train_above_zero / train_total if train_total > 0 else 0

                if valid_data:
                    valid_above_zero = sum(1 for x in valid_data if x > 0)
                    valid_total = len(valid_data)
                    valid_pct = valid_above_zero / valid_total if valid_total > 0 else None
                else:
                    valid_above_zero = 0
                    valid_total = 0
                    valid_pct = None

                if valid_pct is not None:
                    valid_pct_str = f"{valid_pct:.1%}"
                else:
                    valid_pct_str = "N/A"

                stats_text = (
                    f"Train > 0: {train_above_zero}/{train_total} ({train_pct:.1%})\n"
                    f"Valid > 0: {valid_above_zero}/{valid_total} ({valid_pct_str})"
                )

                plt.text(
                    0.02, 0.98,
                    stats_text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )    
            
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(metrics_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        if self.train_losses:
            for loss in loss_plots:
                plt.figure(figsize=(10, 6))
                plt.plot([train_loss[loss["key"]] for train_loss in self.train_losses])
                plt.title(loss["title"])
                plt.xlabel("Episode")
                plt.ylabel(loss["ylabel"])
                plt.grid(True, alpha=0.3)
                plt.savefig(loss_dir / loss["filename"], dpi=300, bbox_inches='tight')
                plt.close()
        
        stats = {
            "train_returns": self.train_returns,
            "valid_returns": self.valid_returns,
            "train_rewards": self.train_rewards,
            "valid_rewards": self.valid_rewards,
            "train_fee_impacts": self.train_fee_impacts,
            "valid_fee_impacts": self.valid_fee_impacts,
            "train_total_trade_counts": self.train_total_trade_counts,
            "valid_total_trade_counts": self.valid_total_trade_counts,
            "train_turnover_ratios": self.train_turnover_ratios,
            "valid_turnover_ratios": self.valid_turnover_ratios,
            "train_avg_hold_times": self.train_avg_hold_times,
            "valid_avg_hold_times": self.valid_avg_hold_times,
            "train_losses": self.train_losses,
            "train_actions": self.train_actions,
            "valid_actions": self.valid_actions
        }
        torch.save(stats, result_dir / "training_stats.pth")
        
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    ticker = 'TSLA'
    train_data_dir = f'{DATA_DIR}/preprocessed/v1/{ticker}/{ticker}_train.csv'
    train_data = load_stock_data(train_data_dir)
    valid_data_dir = f'{DATA_DIR}/preprocessed/v1/{ticker}/{ticker}_valid.csv'
    valid_data = load_stock_data(valid_data_dir)
    
    train_env = Environment(data=train_data)
    valid_env = Environment(data=valid_data)
    
    action_dim = train_env.action_space.shape[0]
    portfolio_dim = train_env.observation_space['portfolio_state'].shape[0]
    
    agent = Agent(
        action_dim=action_dim,
        input_shape=(train_env.window_size, train_env.feature_dim),
        portfolio_state_len=portfolio_dim
    )
    
    models_dir = f"{MODELS_DIR}/v1"
    results_dir = f"{RESULTS_DIR}/v1"
    
    trainer = Trainer(
        agent=agent,
        train_env=train_env,
        valid_env=valid_env,
        randomize_trading_days=True,
        num_episodes=200,
        batch_size=256,
        valid_interval=10,
        save_interval=50,
        models_dir=models_dir,
        results_dir=results_dir,
        logger=Logger()
    )
    
    _ = trainer.train()
    
    print(f"Training complete: Final validation reward {trainer.valid_rewards[-1] if trainer.valid_rewards else 'N/A'}") 
            
if __name__ == "__main__":
    main()