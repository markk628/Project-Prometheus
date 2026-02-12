import pandas as pd
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import WINDOW_SIZE, INITIAL_BALANCE, MAX_TRADING_UNITS, TRANSACTION_FEE_PERCENT
from src.utils.logger import Logger
from src.utils.utils import load_stock_data

class Environment:
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int=WINDOW_SIZE,
        initial_balance: float=INITIAL_BALANCE,
        max_trading_units: int=MAX_TRADING_UNITS,
        transaction_fee_percent: float=TRANSACTION_FEE_PERCENT,
        logger: Optional[Logger]=None
    ):
        self.data = data.drop(['timestamp', 'close'], axis=1)
        self.prices = data['close'].to_numpy()
        self.timestamps = data['timestamp'].to_numpy()
        self.market_open_idx = data[data['timestamp'].str.contains('14:30:00')].index.to_numpy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        # TODO make max_tradin_units dynamic using market data
        # something like this
        '''
        vol_factor = rolling_vol / target_vol
        volume_factor = current_volume / avg_volume

        effective_max_units = base_units * clamp(
            vol_factor * volume_factor,
            0.5, 3.0
        )
        '''
        self.max_trading_units = max_trading_units
        self.transaction_fee_percent = transaction_fee_percent
        self.logger = logger
        
        # Data
        self.feature_dim = self.data.shape[1]
        self.data_length = self.data.shape[0]
        
        # Env state
        self.current_step = 0
        self.current_step_in_episode = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_transaction_fee = 0
        self.current_transaction_fee = 0
        self.shares_traded = 0
        
        # Debugging use
        self.total_transaction_fee_penalty = 0
        self.total_shares_held_penalty = 0
        
        # Episode history
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        # Action space: -1.0 ~ 1.0
        # -1.0 max sell, 0.0 hold, 1.0 max buy
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: market_data x portfolio state
        # market_data: window_size x feature_dim
        # portfolio_state: [cash held, stocks held]
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=0, high=1, shape=(self.window_size, self.feature_dim), dtype=np.float32
            ),
            'portfolio_state': spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            )
        })
        
        if self.logger:
            self.logger.info('Trading environment initialized')
            
    def reset(self) -> Dict[str, np.ndarray]:
        # State
        self.current_step_in_episode = 0
        self.balance = self.initial_balance # TODO eventually we want the previous episode's balance to carry over so remove this when model is consistently winning
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_transaction_fee = 0
        self.current_transaction_fee = 0
        self.shares_traded = 0
        
        self.total_transaction_fee_penalty = 0
        self.total_shares_held_penalty = 0
        
        # History
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        return self._get_observation()
    
    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        # Record action
        self.actions_history.append(action)
        
        # Get portfolio value before action
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_trade_action(action)
        
        # Step
        self.current_step += 1
        self.current_step_in_episode += 1
        
        # Get portfolio value after action
        current_portfolio_value = self._get_portfolio_value()
        
        # Record portfolio value
        self.portfolio_values_history.append(current_portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        
        # Record reward
        self.rewards_history.append(reward)
        
        done = '20:59:00' in self.timestamps[self.current_step - 1]
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Pad data if length is insufficient
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            actual_data = self.data.iloc[start_idx:end_idx].values
            market_data[-len(actual_data):] = actual_data
        else:
            market_data = self.data.iloc[start_idx:end_idx].values
            
            if len(market_data) < self.window_size:
                padding = np.zeros((self.window_size - len(market_data), self.feature_dim), dtype=np.float32)
                market_data = np.vstack([padding, market_data])
        
        # Calculate portfolio state
        portfolio_value = self._get_portfolio_value()
        portfolio_state = np.array([
            self.balance / portfolio_value,  # cash ratio
            (self.shares_held * self._get_current_price()) / portfolio_value  # stock ratio
        ], dtype=np.float32)
        
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio_state': portfolio_state
        }
        
        # Record state
        self.states_history.append(observation)
        
        return observation
    
    def _execute_trade_action(self, action: float) -> None:
        current_price = self._get_current_price()
        
        if current_price <= 0:
            if self.logger:
                self.logger.warning(f"Price is less than 0: {current_price}")
            return
        
        action_value = action[0] if isinstance(action, np.ndarray) else action
        
        if not '20:59:00' in self.timestamps[self.current_step]:
            if action_value > 0:  # Buy
                max_affordable = self.balance / (current_price * (1 + self.transaction_fee_percent))
                shares_to_buy = min(
                    max_affordable,
                    self.max_trading_units * action_value
                )
                
                shares_to_buy = int(shares_to_buy)
                
                if shares_to_buy > 0:
                    buy_cost = shares_to_buy * current_price
                    transaction_fee = buy_cost * self.transaction_fee_percent
                    total_cost = buy_cost + transaction_fee
                    
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.shares_held += shares_to_buy
                        self.total_shares_purchased += shares_to_buy
                        self.total_transaction_fee += transaction_fee
                        self.current_transaction_fee = transaction_fee
                        self.shares_traded = shares_to_buy
                        
                        if self.shares_held > 0:
                            self.cost_basis = ((self.cost_basis * (self.shares_held - shares_to_buy)) + buy_cost) / self.shares_held
                        
                        if self.logger:
                            self.logger.debug(f"Buy: {shares_to_buy} shares @ {current_price:.2f}, Cost: {total_cost:.2f}, Fee: {transaction_fee:.2f}")    
            elif action_value < 0:  # Sell
                shares_to_sell = min(
                    self.shares_held,
                    self.max_trading_units * abs(action_value)
                )
                
                shares_to_sell = int(shares_to_sell)
                
                if shares_to_sell > 0:
                    sell_value = shares_to_sell * current_price
                    transaction_fee = sell_value * self.transaction_fee_percent
                    net_value = sell_value - transaction_fee
                    
                    self.balance += net_value
                    self.shares_held -= shares_to_sell
                    self.total_shares_sold += shares_to_sell
                    self.total_sales_value += sell_value
                    self.total_transaction_fee += transaction_fee
                    self.current_transaction_fee = transaction_fee
                    self.shares_traded = shares_to_sell
                    
                    if self.logger:
                        self.logger.debug(f"Sell: {shares_to_sell} shares @ {current_price:.2f}, Profit: {net_value:.2f}, Fee: {transaction_fee:.2f}")
            else: # hold
                self.shares_traded = 0
        else:
            if self.shares_held > 0:
                sell_value = self.shares_held * current_price
                transaction_fee = sell_value * self.transaction_fee_percent
                net_value = sell_value - transaction_fee
                
                self.balance += net_value
                self.total_shares_sold += self.shares_held
                self.shares_held = 0
                self.total_sales_value += sell_value
                self.total_transaction_fee += transaction_fee
                self.current_transaction_fee = transaction_fee
                self.shares_traded = 0
                
    def _get_current_price(self) -> float:
        return self.prices[self.current_step]
    
    def _get_portfolio_value(self) -> float:
        return self.balance + self.shares_held * self._get_current_price()
    
    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        # if prev_portfolio_value > 0:
        #     return_rate = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        #     reward = return_rate * 100
        # else:
        #     reward = 0
        
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        # transaction_penalty = (self.current_transaction_fee * abs(self.shares_traded)) / current_portfolio_value
        # position_penalty = 0.0001 * abs(self.shares_held)
        transaction_penalty = ((self.current_transaction_fee * abs(self.shares_traded)) / current_portfolio_value) * 0.1
        position_fraction = (self.shares_held * self._get_current_price()) / current_portfolio_value
        position_penalty = 0.001 * abs(position_fraction)

        
        reward = portfolio_return - transaction_penalty - position_penalty
        
        self.total_transaction_fee_penalty += transaction_penalty
        self.total_shares_held_penalty += position_penalty
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        if self.initial_balance > 0:
            total_return = portfolio_value - self.initial_balance
            total_return_pct = (portfolio_value - self.initial_balance) / self.initial_balance
        else:
            total_return = 0
            total_return_pct = 0
        
        return {
            'step': self.current_step,
            'step_episode': self.current_step_in_episode,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'cost_basis': self.cost_basis,
            'total_shares_purchased': self.total_shares_purchased,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'total_transaction_fee': self.total_transaction_fee,
            'total_transaction_fee_penalty': self.total_transaction_fee_penalty,
            'total_shares_held_penalty': self.total_shares_held_penalty
        }
        
    def render(self) -> None:
        info = self._get_info()
        
        print(f"Step: {info['step']}")
        print(f"Step in episode: {info['step_episode']}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Shares held: {info['shares_held']}")
        print(f"Current price: ${info['current_price']:.2f}")
        print(f"Portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Total return: {info['total_return']}%")
        print(f"Total return pct: {info['total_return_pct']:.2%}")
        print(f"Total Fee paid: ${info['total_transaction_fee']:.2f}")
        print("-" * 50)
        
    def get_episode_data(self) -> Dict[str, List]:
        return {
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'portfolio_values': self.portfolio_values_history
        }
    
    def get_final_portfolio_value(self) -> float:
        return self._get_portfolio_value()
    
    # def get_total_reward(self) -> float:
    #     return sum(self.rewards_history)
    
def main():
    import matplotlib.pyplot as plt
    
    from src.config.config import DATA_DIR
    
    ticker = 'TSLA'
    data_dir = f'{DATA_DIR}/preprocessed/{ticker}/{ticker}_train.csv'
    data, _, _ = load_stock_data(data_dir)
    
    env = Environment(data=data, logger=Logger())
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = np.random.uniform(-1.0, 1.0)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if env.current_step_in_episode % 10 == 0:
            env.render()
    
    env.render()
    print("\nFinal Results:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Portfolio Value: ${env.get_final_portfolio_value():.2f}")
    print(f"Total Profit: {(env.get_final_portfolio_value() - env.initial_balance) / env.initial_balance * 100:.2f}%")
    
    episode_data = env.get_episode_data()
    plt.figure(figsize=(12, 6))
    plt.plot(episode_data['portfolio_values'])
    plt.title('Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close() 
    
if __name__ == '__main__':
    main()