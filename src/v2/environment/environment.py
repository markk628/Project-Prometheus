import pandas as pd
import numpy as np
from collections import deque
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import WINDOW_SIZE, INITIAL_BALANCE, MAX_TRADING_UNITS, SEC_FEE, SEC_FEE_PRINCIPAL, TAF_FEE, TAF_FEE_CAP, CAT_FEE, SPREAD, SLIPPAGE
from src.utils.logger import Logger
from src.utils.utils import load_stock_data

class Environment:
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int=WINDOW_SIZE,
        initial_balance: float=INITIAL_BALANCE,
        max_trading_units: int=MAX_TRADING_UNITS,
        sec_fee: float=SEC_FEE,
        sec_fee_principal: float=SEC_FEE_PRINCIPAL,
        taf_fee: float=TAF_FEE,
        taf_fee_cap: float=TAF_FEE_CAP,
        cat_fee: float=CAT_FEE,
        spread: float=SPREAD,
        slippage: float=SLIPPAGE,
        logger: Optional[Logger]=None
    ):
        self.data = data.drop(['timestamp', 'close'], axis=1).to_numpy(dtype=np.float32)
        self.feature_dim = self.data.shape[1]
        self.data_length = self.data.shape[0]
        self.prices = data['close'].to_numpy()
        self.timestamps = data['timestamp'].to_numpy()
        self.market_open_idx = (
            data.index[data['timestamp'].str.contains('09:30:00')]
            .to_numpy()
        )
        self.market_close_idx = np.roll(self.market_open_idx, -1) - 1
        self.market_close_idx[-1] = len(data) - 1
        self.window_size = window_size
        self.initial_balance = initial_balance
        # TODO make max_trading_units dynamic using market data
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
        self.sec_fee = sec_fee
        self.sec_fee_principal = sec_fee_principal
        self.taf_fee = taf_fee
        self.taf_fee_cap = taf_fee_cap
        self.cat_fee = cat_fee
        self.spread = spread
        self.slippage = slippage
        self.logger = logger
        
        # Env state (Reset these every episode)
        self.current_step = 0
        self.current_step_in_episode = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_transaction_fee = 0
        self.current_transaction_fee = 0
        self.shares_traded = 0
        self.trade_execution_count = 0
        self.total_dollar_traded = 0
        self.avg_entry_price = 0
        self.hold_time = 0
        self.hold_times = []
        
        # self.recent_returns = deque(maxlen=self.window_size)
        # self.return_volatility = 1e-9
        
        self.invalid_actions_count = 0
        self.episode_start = 0
        self.episode_end = 0
        self.episode_length = 0
        
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
                low=0, high=np.inf, shape=(4,), dtype=np.float32
            )
        })
        
        if self.logger:
            self.logger.info('Trading environment initialized')
     
    def _get_episode_bounds(self, current_step: int) -> Tuple[int, int, int]:
        """
        Returns (episode_start, episode_end, episode_length)
        """
        # Find which trading day idx
        day_idx = np.searchsorted(self.market_open_idx, current_step, side="right") - 1
        
        episode_start = self.market_open_idx[day_idx]
        episode_end = self.market_close_idx[day_idx]
        episode_length = episode_end - episode_start + 1
        
        return episode_start, episode_end, episode_length
       
    def reset(self) -> Dict[str, np.ndarray]:
        # State
        self.current_step_in_episode = 0
        self.balance = self.initial_balance # TODO eventually we want the previous episode's balance to carry over so remove this when model is consistently winning
        self.shares_held = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_transaction_fee = 0
        self.current_transaction_fee = 0
        self.shares_traded = 0
        self.trade_execution_count = 0
        self.total_dollar_traded = 0
        self.avg_entry_price = 0
        self.hold_time = 0
        self.hold_times = []
        
        # self.recent_returns = deque(maxlen=self.window_size)
        # self.return_volatility = 1e-9
        
        self.invalid_actions_count = 0
        self.episode_start, self.episode_end, self.episode_length = self._get_episode_bounds(self.current_step)
        
        # History
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        return self._get_observation()
    
    def mask_action(self, action: float) -> float:
        market_price = self._get_current_price()

        if market_price <= 0:
            return 0

        # BUY
        if action > 0:
            requested_shares = int(self.max_trading_units * action)

            if requested_shares <= 0:
                return 0

            max_affordable = int(self.balance / market_price)
            shares_to_buy = min(requested_shares, max_affordable)

            if shares_to_buy <= 0:
                return 0

            exec_price, fees, notional = self._calculate_transaction_cost(
                side="buy",
                shares=shares_to_buy,
                market_price=market_price
            )

            total_cost = notional + fees

            if self.balance < total_cost:
                return 0

        # SELL
        elif action < 0:
            requested_shares = int(self.max_trading_units * abs(action))

            if requested_shares <= 0:
                return 0

            if self.shares_held <= 0:
                return 0

        return action
    
    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        # Record action
        self.actions_history.append(action)
        
        # Get portfolio value before action
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_trade_action(action)
        
        # Check if episode is over
        done = self.current_step >= self.episode_end
        
        # Step
        self.current_step += 1
        self.current_step_in_episode += 1
        
        # Get portfolio value after action
        current_portfolio_value = self._get_portfolio_value()
        
        # Record portfolio value
        self.portfolio_values_history.append(current_portfolio_value)
        # self.recent_returns.append(current_portfolio_value)
        # if len(self.recent_returns) > 1:
        #     self.return_volatility = max(np.std(self.recent_returns) + 1e-9, 1e-4)
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        
        # Record reward
        self.rewards_history.append(reward)
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Pad data if length is insufficient
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            actual_data = self.data[start_idx:end_idx]
            market_data[-len(actual_data):] = actual_data
        else:
            market_data = self.data[start_idx:end_idx]
            
            if len(market_data) < self.window_size:
                padding = np.zeros((self.window_size - len(market_data), self.feature_dim), dtype=np.float32)
                market_data = np.vstack([padding, market_data])
        
        # Calculate portfolio state
        portfolio_value = self._get_portfolio_value()
        portfolio_state = np.array([ #TODO update self.observation_space shape each time a new portfolio state is added
            self.balance / portfolio_value,  # cash ratio
            (self.shares_held * self._get_current_price()) / portfolio_value,  # stock ratio
            self._get_unrealized_pnl_pct(),
            min(self.hold_time / self.episode_length, 1.0)
        ], dtype=np.float32)
        
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio_state': portfolio_state
        }
        
        # Record state
        self.states_history.append(observation)
        
        return observation
    
    def _calculate_execution_price(self, side: str, market_price: float) -> float:
        """
        side: 'buy' or 'sell'
        market_price: midpoint or candle close price
        
        Minute-data assumption:
        - Spread is symmetric
        - Slippage proportional to price
        """
        
        half_spread = self.spread / 2
        slippage = market_price * self.slippage
        
        if side == "buy":
            return market_price + half_spread + slippage
        
        elif side == "sell":
            return market_price - half_spread - slippage
        
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        
    def _calculate_regulatory_fees(self, side: str, shares: int, notional: float) -> float:
        """
        Calculates SEC, TAF, CAT fees.
        """
        
        sec_fee = 0.0
        taf_fee = 0.0
        cat_fee = shares * self.cat_fee
        
        # SEC fee (currently zero, but structure left in place)
        if side == "sell":
            sec_fee = notional * self.sec_fee
            
            # FINRA TAF (sell only)
            taf_fee = min(shares * self.taf_fee, self.taf_fee_cap)
        
        return sec_fee + taf_fee + cat_fee

    def _calculate_transaction_cost(self, side: str, shares: int, market_price: float):
        """
        Returns:
            execution_price
            total_fees
            total_notional
        """
        
        execution_price = self._calculate_execution_price(side, market_price)
        notional = shares * execution_price
        fees = self._calculate_regulatory_fees(side, shares, notional)
        
        return execution_price, fees, notional
    
    def _execute_trade_action(self, action: float) -> None:
        def update_avg_entry_price(old_avg, old_shares, buy_price, buy_shares):
            total_cost = old_avg * old_shares + buy_price * buy_shares
            total_shares = old_shares + buy_shares
            return total_cost / total_shares
        
        current_price = self._get_current_price()
        
        if current_price <= 0:
            if self.logger:
                self.logger.warning(f"Price is less than 0: {current_price}")
            return
        
        previous_shares = self.shares_held
        
        if not self.current_step >= self.episode_end:
            if action > 0: # Buy
                max_shares = int(self.balance / current_price)
                shares_to_buy = min(max_shares, int(self.max_trading_units * action))
                
                if shares_to_buy > 0:
                    exec_price, fees, notional = self._calculate_transaction_cost(
                        side="buy",
                        shares=shares_to_buy,
                        market_price=current_price
                    )
                    total_cost = notional + fees
                    
                    if self.balance >= total_cost:
                        self.avg_entry_price = update_avg_entry_price(self.avg_entry_price, self.shares_held, exec_price, shares_to_buy)
                        self.balance -= total_cost
                        self.shares_held += shares_to_buy
                        self.total_shares_purchased += shares_to_buy
                        self.total_transaction_fee += fees
                        self.current_transaction_fee = fees
                        self.shares_traded = shares_to_buy
                        self.trade_execution_count += 1
                        self.total_dollar_traded += notional
                    else:
                        self.invalid_actions_count += 1
                else:
                    self.invalid_actions_count += 1

            elif action < 0: # Sell
                shares_to_sell = min(
                    self.shares_held,
                    int(self.max_trading_units * abs(action))
                )
                
                if shares_to_sell > 0:
                    exec_price, fees, notional = self._calculate_transaction_cost(
                        side="sell",
                        shares=shares_to_sell,
                        market_price=current_price
                    )
                    
                    net_proceeds = notional - fees
                    
                    self.balance += net_proceeds
                    self.shares_held -= shares_to_sell
                    self.total_shares_sold += shares_to_sell
                    self.total_sales_value = notional
                    self.total_transaction_fee += fees
                    self.current_transaction_fee = fees
                    self.shares_traded = shares_to_sell
                    self.trade_execution_count += 1
                    self.total_dollar_traded += notional
                else:
                    self.invalid_actions_count += 1
                
                if self.shares_held == 0:
                    self.avg_entry_price = 0

            else: # Hold
                self.shares_traded = 0
        else:
            if action >= 0:
                self.invalid_actions_count += 1
                
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                exec_price, fees, notional = self._calculate_transaction_cost(
                    side="sell",
                    shares=shares_to_sell,
                    market_price=current_price
                )
                net_proceeds = notional - fees
                
                self.balance += net_proceeds
                self.total_shares_sold += shares_to_sell
                self.shares_held = 0
                self.total_sales_value += notional
                self.total_transaction_fee += fees
                self.current_transaction_fee = fees
                self.shares_traded = shares_to_sell
                self.trade_execution_count += 1
                self.total_dollar_traded += notional
                self.avg_entry_price = 0
                
        if self.shares_held > 0:
            self.hold_time += 1
        elif previous_shares > 0 and self.shares_held == 0:
            self.hold_times.append(self.hold_time)
            self.hold_time = 0
                
    def _get_current_price(self) -> float:
        return self.prices[self.current_step]
    
    def _get_portfolio_value(self) -> float:
        return self.balance + self.shares_held * self._get_current_price()
    
    def _get_unrealized_pnl_pct(self) -> float:
        if self.shares_held > 0:
            return (self._get_current_price() - self.avg_entry_price) / self.avg_entry_price
        return 0.0
    
    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        # if prev_portfolio_value > 0:
        #     return_rate = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        #     reward = return_rate * 100
        # else:
        #     reward = 0
        
        
        # portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        # transaction_penalty = (self.current_transaction_fee * abs(self.shares_traded)) / current_portfolio_value
        # reward = portfolio_return - transaction_penalty
        
        
        # # 1. Use Log Returns (more stable for compounding)
        # portfolio_return = np.log(current_portfolio_value / prev_portfolio_value)
        
        # # 2. Penalty as a ratio of the value
        # # Ensure shares_traded is relative (e.g., 0.1 means 10% of portfolio moved)
        # transaction_penalty = (self.current_transaction_fee * abs(self.shares_traded)) / current_portfolio_value
        
        # # 3. Scale the whole signal
        # # A multiplier of 100 transforms 0.01 (1%) into 1.0
        # reward = (portfolio_return - transaction_penalty) * 100 
        
        
        # portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        # transaction_penalty = (self.current_transaction_fee * abs(self.shares_traded)) / current_portfolio_value
        # reward = np.clip((portfolio_return - transaction_penalty) / self.return_volatility, -10, 10)
        
        
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        if self.initial_balance > 0:
            gross_return = portfolio_value + self.total_transaction_fee - self.initial_balance
            gross_return_pct = gross_return / self.initial_balance
            net_return = portfolio_value - self.initial_balance
            net_return_pct = net_return / self.initial_balance
        else:
            gross_return = 0
            gross_return_pct = 0
            net_return = 0
            net_return_pct = 0
        
        return {
            'step': self.current_step,
            'step_episode': self.current_step_in_episode,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'gross_return': gross_return,
            'gross_return_pct': gross_return_pct,
            'net_return': net_return,
            'net_return_pct': net_return_pct,
            'total_shares_purchased': self.total_shares_purchased,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'total_transaction_fee': self.total_transaction_fee,
            'trade_execution_count': self.trade_execution_count,
            'turnover_ratio': self.total_dollar_traded / self.initial_balance,
            'avg_hold_time': np.array(self.hold_times).mean() if self.hold_times else 0,
            'invalid_actions_count': self.invalid_actions_count
        }
        
    def render(self) -> None:
        info = self._get_info()
        
        print(f"Step: {info['step']}")
        print(f"Step in episode: {info['step_episode']}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Shares held: {info['shares_held']}")
        print(f"Current price: ${info['current_price']:.2f}")
        print(f"Portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Net return pct: {info['net_return_pct']:.2%}")
        print(f"Total Fee paid: ${info['total_transaction_fee']:.2f}")
        print("=" * 50)
        
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
    data_dir = f'{DATA_DIR}/preprocessed/v2/{ticker}/{ticker}_train.csv'
    data = load_stock_data(data_dir)
    
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