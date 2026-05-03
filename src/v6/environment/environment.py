import numpy as np
import math
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import (
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    MAX_POSITION_FRACTION,
    SEC_FEE,
    TAF_FEE,
    TAF_FEE_CAP,
    CAT_FEE,
    SPREAD,
    SLIPPAGE,
)
from src.utils.logger import Logger
from src.utils.utils import load_stock_data


# Default daily config (override via config.py as needed)
DAILY_WINDOW_SIZE = 60          # ~3 months of trading days
DAILY_EPISODE_DAYS = 252        # ~1 year per episode


class DailyEnvironment:
    """
    Trading environment for daily bars with multi-ticker support.

    Each episode operates on a single ticker over a contiguous window of
    trading days. The caller (Trainer) is responsible for selecting which
    ticker and start date to use, then calling reset() with those params.

    Key differences from the minute-level Environment:
    - No intraday market open/close indices — every row is one trading day
    - Episode bounds are simply (start_idx, start_idx + episode_days)
    - Temporal features: day_sin/cos, month_sin/cos, quarter_sin/cos
    - Sharpe annualized with sqrt(252) not sqrt(390 * 252)
    - Simpler fee model (daily trades are infrequent)
    """

    def __init__(
        self,
        data: np.ndarray,         # (T, F) — all features in order: market, temporal, regime
        prices: np.ndarray,       # (T,)   — close prices for execution
        n_market: int,            # number of market feature columns
        n_temporal: int,          # number of temporal feature columns
        n_regime: int,            # number of regime feature columns
        window_size: int = DAILY_WINDOW_SIZE,
        episode_days: int = DAILY_EPISODE_DAYS,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        max_position_fraction: float = MAX_POSITION_FRACTION,
        sec_fee: float = SEC_FEE,
        taf_fee: float = TAF_FEE,
        taf_fee_cap: float = TAF_FEE_CAP,
        cat_fee: float = CAT_FEE,
        spread: float = SPREAD,
        slippage: float = SLIPPAGE,
        logger: Optional[Logger] = None,
    ):
        self.PORTFOLIO_STATE_DIM = 7

        self.data = data
        self.prices = prices
        self.n_market = n_market
        self.n_temporal = n_temporal
        self.n_regime = n_regime
        self.feature_dim = n_market  # only market features for network input_shape
        self.total_bars = len(data)

        self.window_size = window_size
        self.episode_days = episode_days
        self.initial_balance = initial_balance
        self.max_trading_units = max_trading_units
        self.max_position_fraction = max_position_fraction
        self.sec_fee = sec_fee
        self.taf_fee = taf_fee
        self.taf_fee_cap = taf_fee_cap
        self.cat_fee = cat_fee
        self.spread = spread
        self.slippage = slippage
        self.logger = logger

        # Dollar-based position sizing (ticker-agnostic, fractional shares).
        self.max_position_dollars = initial_balance * max_position_fraction
        self.deadband_dollars = self.max_position_dollars * 0.07

        # Env state (Reset these every episode)
        self.current_step = 0
        self.current_step_in_episode = 0
        self.balance = initial_balance
        self.shares_held = 0.0
        self.total_shares_purchased = 0.0
        self.total_shares_sold = 0.0
        self.total_sales_value = 0.0
        self.total_transaction_fee = 0.0
        self.current_transaction_fee = 0.0
        self.shares_traded_at_step = 0.0
        self.trade_execution_count = 0
        self.total_dollar_traded = 0.0
        self.avg_entry_price = 0.0
        self.hold_time = 0
        self.hold_times = []

        # Metrics tracking
        self.step_returns = []
        self.peak_portfolio_value = initial_balance
        self.max_drawdown = 0.0
        self.completed_trades = 0
        self.winning_trades = 0

        self.invalid_actions_count = 0
        self.episode_start = 0
        self.episode_end = 0
        self.episode_length = 0

        # Episode history
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []

        # Action space: -1.0 ~ 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, start_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Reset the environment for a new episode starting at start_idx.

        The caller (Trainer) determines start_idx by picking a random
        ticker and random start date within that ticker's available range.
        """
        self.current_step = start_idx
        self.current_step_in_episode = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.total_shares_purchased = 0.0
        self.total_shares_sold = 0.0
        self.total_sales_value = 0.0
        self.total_transaction_fee = 0.0
        self.current_transaction_fee = 0.0
        self.shares_traded_at_step = 0.0
        self.trade_execution_count = 0
        self.total_dollar_traded = 0.0
        self.avg_entry_price = 0.0
        self.hold_time = 0
        self.hold_times = []

        # Metrics tracking
        self.step_returns = []
        self.peak_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0
        self.completed_trades = 0
        self.winning_trades = 0

        self.invalid_actions_count = 0

        # Episode bounds: start_idx → start_idx + episode_days
        # Clamp to data length if the ticker doesn't have enough bars
        self.episode_start = start_idx
        self.episode_end = min(start_idx + self.episode_days, self.total_bars)
        self.episode_length = self.episode_end - self.episode_start

        # History
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

        # Check if episode is over
        done = self.current_step >= self.episode_end

        # TODO(v6): early-termination on portfolio blowup.
        #
        # With the old MAX_TRADING_UNITS=30 cap on minute TSLA, the max
        # position value was `30 * price` — for most tickers a small
        # fraction of the account, so blowups were structurally limited.
        # Under MAX_POSITION_FRACTION=1.0 with daily bars, a -10% day on
        # a fully-invested position takes PV from $10k → $9k in one step.
        # Several bad days in sequence can dig the agent into a hole it
        # can't meaningfully recover from — remaining episode steps
        # produce noisy reward on a shrunken base, not useful training
        # signal.
        #
        # Proposed rule:
        #   if current_portfolio_value <= initial_balance * (1 - MAX_DRAWDOWN):
        #       done = True
        #
        # where MAX_DRAWDOWN is probably 0.30 (standard blowup threshold)
        # or 0.50 (catches only catastrophic failures). Use drawdown from
        # initial_balance, not from peak — "has the agent lost X% of
        # starting capital" is cleaner than penalizing a +50% → +20%
        # giveback that's still net positive.
        #
        # On early-termination semantics:
        #   - Set done=True here; the existing last-step liquidation
        #     branch in _execute_trade_action already handles closing
        #     any open position, so no new reward logic needed.
        #   - The cumulative losses are already in the reward stream via
        #     the per-step portfolio_return calculation.
        #
        # Distribution-skew caveat: early terminations over-represent
        # early-episode transitions in the replay buffer. If a large
        # fraction of episodes end early, the agent may become biased
        # toward early-episode strategies. Watch `avg_episode_length` in
        # logs once this is enabled.

        # Get portfolio value after action
        current_portfolio_value = self._get_portfolio_value()

        # Record portfolio value
        self.portfolio_values_history.append(current_portfolio_value)

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)

        # Track metrics
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.step_returns.append(portfolio_return)

        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        current_dd = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

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
            market_data = np.zeros((self.window_size, self.data.shape[1]), dtype=np.float32)
            actual_data = self.data[start_idx:end_idx]
            market_data[-len(actual_data):] = actual_data
        else:
            market_data = self.data[start_idx:end_idx]

            if len(market_data) < self.window_size:
                padding = np.zeros((self.window_size - len(market_data), self.data.shape[1]), dtype=np.float32)
                market_data = np.vstack([padding, market_data])

        # Calculate portfolio state
        portfolio_value = self._get_portfolio_value()
        portfolio_state = np.array([
            self.balance / portfolio_value,                                      # cash ratio [0, 1]
            (self.shares_held * self._get_current_price()) / portfolio_value,    # stock ratio [0, 1]
            self._get_unrealized_pnl_pct(),                                     # unrealized pnl
            min(self.hold_time / max(self.episode_length, 1), 1.0),             # hold time ratio [0, 1]
            self._get_episode_sharpe(),                                         # sharpe ratio [-3, 3]
            self._get_current_drawdown(),                                       # current drawdown [0, 1]
            self._get_win_rate(),                                               # win rate [0, 1]
        ], dtype=np.float32)

        observation = {
            'market_data': market_data[:, :self.n_market].astype(np.float32),
            'temporal': market_data[-1, self.n_market:self.n_market + self.n_temporal].astype(np.float32),
            'regime': market_data[-1, self.n_market + self.n_temporal:].astype(np.float32),
            'portfolio_state': portfolio_state
        }

        return observation

    def _calculate_execution_price(self, side: str, market_price: float) -> float:
        half_spread = self.spread / 2
        slippage = market_price * self.slippage

        if side == "buy":
            return market_price + half_spread + slippage
        elif side == "sell":
            return market_price - half_spread - slippage
        else:
            raise ValueError("side must be 'buy' or 'sell'")

    def _calculate_regulatory_fees(self, side: str, shares: int, notional: float) -> float:
        sec_fee = 0.0
        taf_fee = 0.0
        cat_fee = shares * self.cat_fee

        if side == "sell":
            sec_fee = math.ceil(notional * self.sec_fee * 100) / 100
            taf_fee = min(math.ceil(shares * self.taf_fee * 100) / 100, self.taf_fee_cap)

        return sec_fee + taf_fee + cat_fee

    def _calculate_transaction_cost(self, side: str, shares: int, market_price: float):
        execution_price = self._calculate_execution_price(side, market_price)
        notional = shares * execution_price
        fees = self._calculate_regulatory_fees(side, shares, notional)
        return execution_price, fees, notional

    def _execute_trade_action(self, action: float) -> None:
        """
        Target Position Action Space (dollar-exposure, ticker-agnostic):
            action in [-1, 1] maps to target dollar exposure in
            [0, initial_balance * max_position_fraction].
            -1 = hold $0 (all cash)
             1 = hold max_position_fraction of initial_balance
                 (e.g. 1.0 = fully invested)

        Shares are fractional. This makes the action's meaning invariant
        across tickers regardless of price — action = 1.0 always means
        "target 100% of starting capital in this ticker".

        Deadband is dollar-based: changes smaller than 7% of max dollar
        exposure (roughly $700 with a $10k balance) are ignored so small
        exploration-noise deltas don't trigger trades.
        """
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
        self.shares_traded_at_step = 0.0
        self.current_transaction_fee = 0.0

        # Convert [-1, 1] action to target dollar exposure, then to fractional shares
        normalized_action = (action + 1) / 2.0
        target_dollars = self.max_position_dollars * normalized_action
        target_shares = target_dollars / current_price

        current_dollars = self.shares_held * current_price
        dollars_delta = target_dollars - current_dollars
        shares_delta = target_shares - self.shares_held

        # Episode end: liquidate everything (must come before deadband)
        if self.current_step + 1 >= self.episode_end:
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                exec_price, fees, notional = self._calculate_transaction_cost(
                    side="sell", shares=shares_to_sell, market_price=current_price
                )
                net_proceeds = notional - fees

                self.balance += net_proceeds
                self.total_shares_sold += shares_to_sell
                self.shares_held = 0.0
                self.total_sales_value += notional
                self.total_transaction_fee += fees
                self.current_transaction_fee = fees
                self.shares_traded_at_step = -shares_to_sell
                self.trade_execution_count += 1
                self.total_dollar_traded += notional
                self.completed_trades += 1
                if exec_price > self.avg_entry_price:
                    self.winning_trades += 1
                self.avg_entry_price = 0.0
                self.hold_times.append(self.hold_time)
                self.hold_time = 0
            return

        # Dollar-based deadband: ignore changes smaller than ~7% of max dollar exposure.
        # Ticker-agnostic — doesn't depend on share count or price.
        if abs(dollars_delta) < self.deadband_dollars:
            if self.shares_held > 0:
                self.hold_time += 1
            return

        if shares_delta > 0:  # Buy
            # Compute max affordable fractional shares accounting for fees.
            # Buy exec price + fees per share gives true per-share cost.
            exec_price_probe = self._calculate_execution_price("buy", current_price)
            per_share_cost = exec_price_probe + self.cat_fee  # CAT is per-share; SEC/TAF are sell-only
            max_affordable = self.balance / per_share_cost if per_share_cost > 0 else 0.0
            shares_to_buy = min(max_affordable, shares_delta)

            if shares_to_buy > 0:
                exec_price, fees, notional = self._calculate_transaction_cost(
                    side="buy", shares=shares_to_buy, market_price=current_price
                )
                total_cost = notional + fees

                if self.balance >= total_cost:
                    self.avg_entry_price = update_avg_entry_price(
                        self.avg_entry_price, self.shares_held, exec_price, shares_to_buy
                    )
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    self.total_shares_purchased += shares_to_buy
                    self.total_transaction_fee += fees
                    self.current_transaction_fee = fees
                    self.shares_traded_at_step = shares_to_buy
                    self.trade_execution_count += 1
                    self.total_dollar_traded += notional
                else:
                    self.invalid_actions_count += 1

        elif shares_delta < 0:  # Sell
            shares_to_sell = min(self.shares_held, abs(shares_delta))

            if shares_to_sell > 0:
                exec_price, fees, notional = self._calculate_transaction_cost(
                    side="sell", shares=shares_to_sell, market_price=current_price
                )
                net_proceeds = notional - fees

                self.balance += net_proceeds
                self.shares_held -= shares_to_sell
                self.total_shares_sold += shares_to_sell
                self.total_sales_value += notional
                self.total_transaction_fee += fees
                self.current_transaction_fee = fees
                self.shares_traded_at_step = -shares_to_sell
                self.trade_execution_count += 1
                self.total_dollar_traded += notional

                # Use a small tolerance — fractional share float arithmetic may
                # leave a tiny residual instead of exact zero.
                if self.shares_held < 1e-9:
                    self.shares_held = 0.0
                    self.completed_trades += 1
                    if exec_price > self.avg_entry_price:
                        self.winning_trades += 1
                    self.avg_entry_price = 0.0

        # Track hold time
        if self.shares_held > 0:
            self.hold_time += 1
        elif previous_shares > 0 and self.shares_held == 0:
            self.hold_times.append(self.hold_time)
            self.hold_time = 0

    def _get_current_price(self) -> float:
        return self.prices[self.current_step]

    def _get_portfolio_value(self) -> float:
        # Mark-to-market valuation: shares valued at current price.
        # Execution costs (spread + slippage + fees) are realized at buy/sell
        # time via _execute_trade_action, so we don't double-charge them here.
        return self.balance + (self.shares_held * self._get_current_price())

    def _get_unrealized_pnl_pct(self) -> float:
        if self.shares_held > 0:
            return (self._get_current_price() - self.avg_entry_price) / self.avg_entry_price
        return 0.0

    def _get_episode_sharpe(self) -> float:
        if len(self.step_returns) < 2:
            return 0.0
        returns = np.array(self.step_returns)
        sharpe = returns.mean() / (returns.std() + 1e-9)
        return np.clip(sharpe, -3.0, 3.0)

    def _get_episode_sortino(self) -> float:
        """
        Sortino: like Sharpe but only penalizes downside volatility.
        Undefined when there are no negative returns — return a large
        positive clip value in that case (strategy had no downside so far).
        """
        if len(self.step_returns) < 2:
            return 0.0
        returns = np.array(self.step_returns)
        negative = returns[returns < 0]
        if negative.size == 0:
            return 3.0  # no downside observed; clip ceiling
        downside_std = negative.std()
        sortino = returns.mean() / (downside_std + 1e-9)
        return float(np.clip(sortino, -3.0, 3.0))

    def _get_episode_calmar(self) -> float:
        """
        Calmar: annualized return divided by max drawdown.
        Measures return per unit of worst observed loss. Undefined when
        max_drawdown is zero — return a capped positive value in that
        case (no drawdown yet means "infinitely good" relative risk).
        """
        if len(self.step_returns) < 2 or self.max_drawdown <= 0:
            # If we have meaningful returns but no drawdown, cap high.
            # Otherwise return 0 (not enough data to judge).
            if len(self.step_returns) >= 2 and np.mean(self.step_returns) > 0:
                return 10.0
            return 0.0
        mean_return = np.mean(self.step_returns)
        annualized = mean_return * 252
        calmar = annualized / self.max_drawdown
        return float(np.clip(calmar, -10.0, 10.0))

    def _get_profit_factor(self) -> float:
        """
        Profit factor: sum of positive returns / sum of |negative returns|.
        "For every $1 lost, how much was gained." >1 means net profitable,
        <1 means net loss. Capped to prevent extreme values when the
        denominator is tiny.
        """
        if len(self.step_returns) < 2:
            return 1.0
        returns = np.array(self.step_returns)
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()  # absolute value
        if losses < 1e-9:
            # No losses yet — cap high if there are any gains
            return 10.0 if gains > 0 else 1.0
        return float(np.clip(gains / losses, 0.0, 10.0))

    def _get_avg_win_loss_ratio(self) -> float:
        """
        Average win / average loss magnitude. Independent of how often
        you win (that's win_rate) — this measures "when I win vs lose,
        how do the sizes compare." Capped similarly.
        """
        if len(self.step_returns) < 2:
            return 1.0
        returns = np.array(self.step_returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if wins.size == 0 or losses.size == 0:
            return 1.0  # not enough data to form the ratio
        avg_win = wins.mean()
        avg_loss = -losses.mean()  # positive magnitude
        if avg_loss < 1e-9:
            return 10.0
        return float(np.clip(avg_win / avg_loss, 0.0, 10.0))

    def _get_current_drawdown(self) -> float:
        if self.peak_portfolio_value <= 0:
            return 0.0
        current_value = self._get_portfolio_value()
        return (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

    def _get_win_rate(self) -> float:
        if self.completed_trades == 0:
            return 0.5
        return self.winning_trades / self.completed_trades

    def _get_episode_progress(self) -> float:
        if self.episode_length <= 0:
            return 0.0
        return self.current_step_in_episode / self.episode_length

    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        return np.clip(portfolio_return * 100, -10.0, 10.0)

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
            'invalid_actions_count': self.invalid_actions_count,
            'sharpe_ratio': self._get_episode_sharpe() * np.sqrt(252),  # annualized daily sharpe
            'sortino_ratio': self._get_episode_sortino() * np.sqrt(252),  # annualized daily sortino
            'calmar_ratio': self._get_episode_calmar(),
            'profit_factor': self._get_profit_factor(),
            'avg_win_loss_ratio': self._get_avg_win_loss_ratio(),
            'current_drawdown': self._get_current_drawdown(),
            'max_drawdown': self.max_drawdown,
            'win_rate': self._get_win_rate(),
            'completed_trades': self.completed_trades,
            'episode_progress': self._get_episode_progress()
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