import math
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import (
    INITIAL_BALANCE,
    SEC_FEE,
    TAF_FEE,
    TAF_FEE_CAP,
    CAT_FEE,
    SPREAD,
    SLIPPAGE,
)
from src.utils.logger import Logger


# Default daily config (override via config.py as needed)
DAILY_EPISODE_DAYS = 252        # ~1 year per episode


def _safe_log_ratio(numer: float, denom: float, eps: float = 1e-9) -> float:
    """log(numer / denom) with floor on denom to avoid log(0) / div0."""
    return float(np.log(max(numer, eps) / max(denom, eps)))


class DailyEnvironment:
    """
    v7 multi-ticker allocation environment.

    Each episode operates on a fixed basket of N tickers (typically 5 from
    V7_BASKET) over a contiguous window of trading days. The basket is set
    at construction; the caller (Trainer) picks start_idx via reset().

    Differences from v6 single-ticker env:
    - Action space: continuous (N,)-dim in [-1, 1]^N. Softmax-with-fixed-
      cash-logit converts to (N+1) weights (N tickers + cash) summing to 1.
      action[i] strongly negative → low weight on ticker i.
      action[i] strongly positive → high weight on ticker i.
      All-negative → mostly cash. All-zero → uniform ~1/(N+1) each.
    - Portfolio state: PORTFOLIO_STATE_DIM=22 for the default N=5:
        4 per-ticker dims × 5 tickers (weight, position_return, hold_time, dist_from_target)
        + 2 portfolio-level dims (cash fraction, total value log-ratio)
    - Reward: log(pv_after / pv_before) — same shape as v6, generalized.
    - Transaction costs: v6's SPREAD/SLIPPAGE/SEC/TAF/CAT model, applied
      per-ticker per-rebalance. No additional shaped turnover penalty —
      the cost model IS the turnover discouragement.
    - Initial state: all cash. Policy must actively allocate to defend
      against passive-cash baseline.
    - Last-step liquidation: all positions closed on the final step so
      episode-end portfolio value is comparable across actions.
    """

    def __init__(
        self,
        basket_tickers: List[str],
        market_data: Dict[str, np.ndarray],   # ticker -> (T, F_per_ticker), aligned timelines
        prices: Dict[str, np.ndarray],        # ticker -> (T,) close prices for execution
        temporal_data: np.ndarray,            # (T, n_temporal) shared across tickers
        regime_data: np.ndarray,              # (T, n_regime) shared across tickers
        episode_days: int = DAILY_EPISODE_DAYS,
        initial_balance: float = INITIAL_BALANCE,
        deadband_fraction: float = 0.02,      # min rebalance |Δ$| as fraction of initial balance
        sec_fee: float = SEC_FEE,
        taf_fee: float = TAF_FEE,
        taf_fee_cap: float = TAF_FEE_CAP,
        cat_fee: float = CAT_FEE,
        spread: float = SPREAD,
        slippage: float = SLIPPAGE,
        logger: Optional[Logger] = None,
    ):
        if len(basket_tickers) < 2:
            raise ValueError(f"Basket needs ≥2 tickers, got {len(basket_tickers)}")
        for t in basket_tickers:
            if t not in market_data:
                raise KeyError(f"market_data missing ticker {t}")
            if t not in prices:
                raise KeyError(f"prices missing ticker {t}")

        # All ticker arrays + shared arrays must have the same T.
        T_set = {len(prices[t]) for t in basket_tickers}
        T_set.add(len(temporal_data))
        T_set.add(len(regime_data))
        if len(T_set) != 1:
            raise ValueError(f"Timeline mismatch across inputs: lengths={T_set}")
        self.total_bars = T_set.pop()

        # Per-ticker market feature dim must agree (all tickers in v7 have
        # the same schema by construction — 43 cols for asset-class basket).
        F_set = {market_data[t].shape[1] for t in basket_tickers}
        if len(F_set) != 1:
            raise ValueError(f"Per-ticker market feature dim mismatch: {F_set}")

        self.basket_tickers = list(basket_tickers)
        self.n_tickers = len(self.basket_tickers)
        self.market_data = {t: market_data[t] for t in self.basket_tickers}
        self.prices = {t: prices[t] for t in self.basket_tickers}
        self.temporal_data = temporal_data
        self.regime_data = regime_data

        self.n_market_per_ticker = F_set.pop()
        self.n_temporal = temporal_data.shape[1]
        self.n_regime = regime_data.shape[1]
        # Network input contract: flattened per-step market dim = n_tickers * n_market_per_ticker.
        # Observation produces (W, n_tickers, n_market_per_ticker); replay buffer / network
        # can reshape as needed.
        self.feature_dim = self.n_market_per_ticker  # per-ticker feature dim

        # Portfolio state width:
        #   per-ticker (×N): weight, position_return, hold_time, dist_from_target = 4
        #   portfolio-level: cash_fraction, total_value_log_ratio = 2
        # For N=5 this is 22.
        self.PORTFOLIO_STATE_DIM = 4 * self.n_tickers + 2

        self.episode_days = episode_days
        self.initial_balance = initial_balance
        self.deadband_dollars = initial_balance * deadband_fraction
        self.sec_fee = sec_fee
        self.taf_fee = taf_fee
        self.taf_fee_cap = taf_fee_cap
        self.cat_fee = cat_fee
        self.spread = spread
        self.slippage = slippage
        self.logger = logger

        # Action space: N-dim continuous in [-1, 1]. Softmax-with-cash-logit
        # is applied inside step(), producing (N+1) weights.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_tickers,), dtype=np.float32
        )

        self._init_episode_state()

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def _init_episode_state(self):
        """All resettable state lives here so reset() doesn't drift from __init__."""
        self.current_step = 0
        self.current_step_in_episode = 0
        self.balance = self.initial_balance

        # Per-ticker state — fresh on each episode.
        self.per_ticker: Dict[str, Dict[str, Any]] = {
            t: {
                "shares_held": 0.0,
                "avg_entry_price": 0.0,
                "hold_time": 0,
                "hold_times": [],
                "total_shares_purchased": 0.0,
                "total_shares_sold": 0.0,
                "total_sales_value": 0.0,
                "completed_trades": 0,
                "winning_trades": 0,
                "last_target_weight": 0.0,
            }
            for t in self.basket_tickers
        }
        # Aggregate trade accounting (across all tickers).
        self.total_transaction_fee = 0.0
        self.current_transaction_fee = 0.0
        self.trade_execution_count = 0
        self.total_dollar_traded = 0.0

        # Episode metrics
        self.step_returns: List[float] = []
        self.peak_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0

        self.invalid_actions_count = 0
        self.episode_start = 0
        self.episode_end = 0
        self.episode_length = 0

        # History
        self.actions_history: List[np.ndarray] = []
        self.rewards_history: List[float] = []
        self.portfolio_values_history: List[float] = []
        self.weights_history: List[np.ndarray] = []

    def reset(self, start_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Reset env for a new episode starting at start_idx.

        All 5 ticker series are aligned on the same timeline, so start_idx
        applies uniformly to every ticker plus the shared temporal/regime.
        """
        self._init_episode_state()

        self.current_step = start_idx
        self.episode_start = start_idx
        self.episode_end = min(start_idx + self.episode_days, self.total_bars)
        self.episode_length = self.episode_end - self.episode_start

        return self._get_observation()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one rebalance step.

        action: shape (n_tickers,), values in [-1, 1] (tanh-squashed Gaussian
        from SAC actor). Internally appends a fixed 0.0 cash logit and
        softmaxes to get (n_tickers + 1) target weights summing to 1.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_tickers:
            raise ValueError(
                f"action shape {action.shape} != expected ({self.n_tickers},)"
            )

        self.actions_history.append(action.copy())
        self.current_transaction_fee = 0.0

        prev_portfolio_value = self._get_portfolio_value()

        target_weights = self._action_to_weights(action)  # length n_tickers + 1 (last = cash)
        self.weights_history.append(target_weights.copy())

        # Last-step liquidation: force target = all-cash to close every position
        # cleanly. Final portfolio value then reflects realized P&L without
        # mark-to-market noise on whatever position happened to be open.
        is_last_step = (self.current_step + 1) >= self.episode_end
        if is_last_step:
            target_weights = np.zeros_like(target_weights)
            target_weights[-1] = 1.0

        self._rebalance(target_weights, prev_portfolio_value)

        # Save target weights for next-step distance_from_target computation.
        for i, t in enumerate(self.basket_tickers):
            self.per_ticker[t]["last_target_weight"] = float(target_weights[i])

        # Advance time.
        self.current_step += 1
        self.current_step_in_episode += 1
        done = self.current_step >= self.episode_end

        # Update hold_time per ticker.
        # Done AFTER step advance so hold_time reflects "days since open at
        # the new current_step" — matches v6's semantics.
        for t in self.basket_tickers:
            if self.per_ticker[t]["shares_held"] > 0:
                self.per_ticker[t]["hold_time"] += 1

        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_values_history.append(current_portfolio_value)

        # Reward: log portfolio return. Costs already deducted via balance
        # updates in _rebalance, so this is naturally after-cost.
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        self.rewards_history.append(reward)

        # Track returns + drawdown.
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1e-9)
        self.step_returns.append(portfolio_return)

        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        current_dd = (self.peak_portfolio_value - current_portfolio_value) / max(self.peak_portfolio_value, 1e-9)
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, done, info

    # ------------------------------------------------------------------ #
    # Action handling                                                    #
    # ------------------------------------------------------------------ #

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Softmax-with-fixed-cash-logit.

        Append 0.0 as the cash logit so cash is always a real allocation
        choice the policy competes against on equal footing. Strongly
        negative ticker logits → high cash. Strongly positive ticker logits
        → low cash, that ticker dominant.

        Returns shape (n_tickers + 1,) summing to 1.
        Position [:-1] are ticker weights in basket_tickers order.
        Position [-1] is the cash weight.
        """
        logits = np.concatenate([action.astype(np.float64), [0.0]])
        # Standard numerically-stable softmax.
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / exp_logits.sum()).astype(np.float32)

    def _rebalance(self, target_weights: np.ndarray, portfolio_value: float) -> None:
        """
        Execute trades to move current positions toward target weights.

        Order: sells first (free cash), buys second (use available cash).
        Per-ticker deadband filters out moves smaller than self.deadband_dollars
        to avoid noise-driven rebalancing.

        target_weights: (n_tickers + 1,), last entry is cash weight.
        portfolio_value: total PV before any trades this step (mark-to-market
        valuation at current_step's prices).
        """
        if portfolio_value <= 0:
            self.invalid_actions_count += 1
            return

        current_prices = self._get_current_prices()

        # Compute target dollar exposure per ticker, current exposure, delta.
        # Cash target is implicit — whatever's left after ticker targets.
        sells_to_execute: List[Tuple[str, float, float]] = []  # (ticker, shares_to_sell, price)
        buys_desired: List[Tuple[str, float, float]] = []      # (ticker, dollars_to_buy, price)

        for i, t in enumerate(self.basket_tickers):
            price = current_prices[t]
            if price <= 0:
                if self.logger:
                    self.logger.warning(f"Non-positive price for {t}: {price}; skipping")
                continue

            current_dollars = self.per_ticker[t]["shares_held"] * price
            target_dollars = portfolio_value * float(target_weights[i])
            dollars_delta = target_dollars - current_dollars

            if abs(dollars_delta) < self.deadband_dollars:
                # Position close enough to target — no trade.
                continue

            if dollars_delta < 0:
                # Sell. Compute shares to sell from dollar delta at current price.
                # We sell at execution price (mid - half spread - slippage),
                # which is slightly below current price — using current price
                # to size the sell is a small approximation that costs us
                # marginally accurate weight targeting. Acceptable for run 1.
                shares_to_sell = min(
                    self.per_ticker[t]["shares_held"],
                    abs(dollars_delta) / price,
                )
                if shares_to_sell > 0:
                    sells_to_execute.append((t, shares_to_sell, price))
            else:
                buys_desired.append((t, dollars_delta, price))

        # Execute sells first.
        for ticker, shares, price in sells_to_execute:
            self._execute_sell(ticker, shares, price)

        # Execute buys with available cash. If desired total > available, scale
        # proportionally so the relative target ratios are preserved.
        if buys_desired:
            total_desired = sum(d for _, d, _ in buys_desired)
            # Apply a safety margin for buy fees (CAT is per-share; SEC/TAF are
            # sell-only so don't apply here). Reserve 1% of available cash to
            # absorb spread + slippage + CAT — empirically generous.
            available_cash = self.balance * 0.99
            scale = min(1.0, available_cash / max(total_desired, 1e-9))

            for ticker, dollars_to_buy, price in buys_desired:
                scaled_dollars = dollars_to_buy * scale
                if scaled_dollars < self.deadband_dollars:
                    continue
                # Convert desired dollars to shares via execution-price probe.
                exec_price_probe = self._calculate_execution_price("buy", price)
                per_share_cost = exec_price_probe + self.cat_fee
                if per_share_cost <= 0:
                    continue
                shares_to_buy = scaled_dollars / per_share_cost
                if shares_to_buy > 0:
                    self._execute_buy(ticker, shares_to_buy, price)

    # ------------------------------------------------------------------ #
    # Per-ticker trade execution                                         #
    # ------------------------------------------------------------------ #

    def _execute_buy(self, ticker: str, shares: float, market_price: float) -> None:
        """Buy `shares` of `ticker` at current market_price. Updates state in place."""
        exec_price, fees, notional = self._calculate_transaction_cost(
            "buy", shares, market_price
        )
        total_cost = notional + fees

        if self.balance < total_cost:
            # Last-ditch cash check (rebalance already applied a 0.99 buffer).
            # Cap to actual affordable.
            if total_cost <= 0:
                return
            affordable_ratio = self.balance / total_cost
            shares = shares * affordable_ratio
            if shares <= 0:
                self.invalid_actions_count += 1
                return
            exec_price, fees, notional = self._calculate_transaction_cost(
                "buy", shares, market_price
            )
            total_cost = notional + fees
            if self.balance < total_cost:
                self.invalid_actions_count += 1
                return

        ts = self.per_ticker[ticker]
        # Update weighted-avg entry price.
        old_shares = ts["shares_held"]
        if old_shares + shares > 0:
            ts["avg_entry_price"] = (
                ts["avg_entry_price"] * old_shares + exec_price * shares
            ) / (old_shares + shares)
        ts["shares_held"] += shares
        ts["total_shares_purchased"] += shares

        self.balance -= total_cost
        self.total_transaction_fee += fees
        self.current_transaction_fee += fees
        self.trade_execution_count += 1
        self.total_dollar_traded += notional

    def _execute_sell(self, ticker: str, shares: float, market_price: float) -> None:
        """Sell `shares` of `ticker` at current market_price. Updates state in place."""
        ts = self.per_ticker[ticker]
        if shares > ts["shares_held"]:
            shares = ts["shares_held"]
        if shares <= 0:
            return

        exec_price, fees, notional = self._calculate_transaction_cost(
            "sell", shares, market_price
        )
        net_proceeds = notional - fees
        prev_avg_entry = ts["avg_entry_price"]

        ts["shares_held"] -= shares
        ts["total_shares_sold"] += shares
        ts["total_sales_value"] += notional

        self.balance += net_proceeds
        self.total_transaction_fee += fees
        self.current_transaction_fee += fees
        self.trade_execution_count += 1
        self.total_dollar_traded += notional

        # Round-trip accounting on full close.
        if ts["shares_held"] < 1e-9:
            ts["shares_held"] = 0.0
            ts["completed_trades"] += 1
            if exec_price > prev_avg_entry:
                ts["winning_trades"] += 1
            ts["avg_entry_price"] = 0.0
            ts["hold_times"].append(ts["hold_time"])
            ts["hold_time"] = 0

    def _calculate_execution_price(self, side: str, market_price: float) -> float:
        half_spread = self.spread / 2.0
        slippage = market_price * self.slippage
        if side == "buy":
            return market_price + half_spread + slippage
        elif side == "sell":
            return market_price - half_spread - slippage
        else:
            raise ValueError("side must be 'buy' or 'sell'")

    def _calculate_regulatory_fees(self, side: str, shares: float, notional: float) -> float:
        sec_fee = 0.0
        taf_fee = 0.0
        cat_fee = shares * self.cat_fee
        if side == "sell":
            sec_fee = math.ceil(notional * self.sec_fee * 100) / 100
            taf_fee = min(math.ceil(shares * self.taf_fee * 100) / 100, self.taf_fee_cap)
        return sec_fee + taf_fee + cat_fee

    def _calculate_transaction_cost(
        self, side: str, shares: float, market_price: float
    ) -> Tuple[float, float, float]:
        exec_price = self._calculate_execution_price(side, market_price)
        notional = shares * exec_price
        fees = self._calculate_regulatory_fees(side, shares, notional)
        return exec_price, fees, notional

    # ------------------------------------------------------------------ #
    # Observation & portfolio state                                       #
    # ------------------------------------------------------------------ #

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the state dict for the agent.

        v7 uses MLP-on-last-bar (committed in v7 dev log; v6 4b precedent),
        so market_data is just the current step's per-ticker features —
        no window. Per-ticker delta features added in v6 4a/4c carry the
        trajectory info that a window would otherwise provide.

        Returns:
            market_data:      (n_tickers, F_per_ticker)
            temporal:         (n_temporal,)
            regime:           (n_regime,)
            portfolio_state:  (PORTFOLIO_STATE_DIM,)
        """
        # Per-ticker last-bar features stacked: (N, F_per_ticker).
        market_data = np.zeros(
            (self.n_tickers, self.n_market_per_ticker), dtype=np.float32,
        )
        for i, t in enumerate(self.basket_tickers):
            market_data[i, :] = self.market_data[t][self.current_step]

        # Shared temporal + regime snapshots at current step.
        temporal_snapshot = self.temporal_data[self.current_step].astype(np.float32)
        regime_snapshot = self.regime_data[self.current_step].astype(np.float32)

        portfolio_state = self._compute_portfolio_state()

        return {
            "market_data": market_data,
            "temporal": temporal_snapshot,
            "regime": regime_snapshot,
            "portfolio_state": portfolio_state,
        }

    def _compute_portfolio_state(self) -> np.ndarray:
        """
        22-dim portfolio state (for n_tickers=5):
            Per-ticker (×5):
                weight:                    [0, 1]        — position fraction of PV
                position_return:           ~[-1, 1]      — log(price / entry) when held
                hold_time_log:             ~[0, 1]       — log-scaled holding period
                distance_from_target:      ~[-1, 1]      — current_weight - last_target_weight
            Portfolio-level (×2):
                cash_fraction:             [0, 1]
                total_value_log_ratio:     unbounded     — log(PV / initial_balance)
        """
        pv = self._get_portfolio_value()
        pv_safe = max(pv, 1e-9)
        current_prices = self._get_current_prices()

        dims: List[float] = []

        # Pre-compute log-scaling denominator for hold_time. Use episode_days
        # as the reference so hold_time approaches 1.0 only on a long-held
        # position over a full episode.
        hold_norm = math.log(1 + max(self.episode_days, 1))

        for t in self.basket_tickers:
            ts = self.per_ticker[t]
            price = current_prices[t]
            shares = ts["shares_held"]
            position_value = shares * price
            weight = position_value / pv_safe if pv_safe > 0 else 0.0

            if shares > 0 and ts["avg_entry_price"] > 0:
                position_return = _safe_log_ratio(price, ts["avg_entry_price"])
                # Clip to a reasonable range to keep the network input bounded.
                position_return = float(np.clip(position_return, -1.0, 1.0))
            else:
                position_return = 0.0

            hold_time_log = math.log(1 + ts["hold_time"]) / hold_norm

            dist_from_target = weight - ts["last_target_weight"]

            dims.extend([weight, position_return, hold_time_log, dist_from_target])

        cash_fraction = self.balance / pv_safe if pv_safe > 0 else 1.0
        total_value_log_ratio = _safe_log_ratio(pv, self.initial_balance)

        dims.extend([cash_fraction, total_value_log_ratio])

        return np.array(dims, dtype=np.float32)

    def _get_current_prices(self) -> Dict[str, float]:
        return {t: float(self.prices[t][self.current_step]) for t in self.basket_tickers}

    def _get_portfolio_value(self) -> float:
        """Mark-to-market: cash + sum of per-ticker position values."""
        prices = self._get_current_prices()
        position_value = sum(
            self.per_ticker[t]["shares_held"] * prices[t]
            for t in self.basket_tickers
        )
        return self.balance + position_value

    # ------------------------------------------------------------------ #
    # Reward                                                             #
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, prev_pv: float, current_pv: float) -> float:
        portfolio_return = (current_pv - prev_pv) / max(prev_pv, 1e-9)
        return float(np.clip(portfolio_return * 100, -10.0, 10.0))

    # ------------------------------------------------------------------ #
    # Episode metrics (mostly inherited from v6, applied at portfolio level) #
    # ------------------------------------------------------------------ #

    def _get_episode_sharpe(self) -> float:
        if len(self.step_returns) < 2:
            return 0.0
        returns = np.array(self.step_returns)
        sharpe = returns.mean() / (returns.std() + 1e-9)
        return float(np.clip(sharpe, -3.0, 3.0))

    def _get_episode_sortino(self) -> float:
        if len(self.step_returns) < 2:
            return 0.0
        returns = np.array(self.step_returns)
        negative = returns[returns < 0]
        if negative.size == 0:
            return 3.0
        sortino = returns.mean() / (negative.std() + 1e-9)
        return float(np.clip(sortino, -3.0, 3.0))

    def _get_episode_calmar(self) -> float:
        if len(self.step_returns) < 2 or self.max_drawdown <= 0:
            if len(self.step_returns) >= 2 and np.mean(self.step_returns) > 0:
                return 10.0
            return 0.0
        annualized = np.mean(self.step_returns) * 252
        return float(np.clip(annualized / self.max_drawdown, -10.0, 10.0))

    def _get_profit_factor(self) -> float:
        if len(self.step_returns) < 2:
            return 1.0
        returns = np.array(self.step_returns)
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        if losses < 1e-9:
            return 10.0 if gains > 0 else 1.0
        return float(np.clip(gains / losses, 0.0, 10.0))

    def _get_avg_win_loss_ratio(self) -> float:
        if len(self.step_returns) < 2:
            return 1.0
        returns = np.array(self.step_returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if wins.size == 0 or losses.size == 0:
            return 1.0
        avg_loss = -losses.mean()
        if avg_loss < 1e-9:
            return 10.0
        return float(np.clip(wins.mean() / avg_loss, 0.0, 10.0))

    def _get_current_drawdown(self) -> float:
        if self.peak_portfolio_value <= 0:
            return 0.0
        return (self.peak_portfolio_value - self._get_portfolio_value()) / self.peak_portfolio_value

    def _get_win_rate(self) -> float:
        """Aggregated win rate across all tickers' completed trades."""
        total = sum(self.per_ticker[t]["completed_trades"] for t in self.basket_tickers)
        if total == 0:
            return 0.5
        wins = sum(self.per_ticker[t]["winning_trades"] for t in self.basket_tickers)
        return wins / total

    def _get_episode_progress(self) -> float:
        if self.episode_length <= 0:
            return 0.0
        return self.current_step_in_episode / self.episode_length

    def _get_turnover_ratio(self) -> float:
        return self.total_dollar_traded / max(self.initial_balance, 1e-9)

    # ------------------------------------------------------------------ #
    # Info / introspection                                                #
    # ------------------------------------------------------------------ #

    def _get_info(self) -> Dict[str, Any]:
        pv = self._get_portfolio_value()
        prices = self._get_current_prices()

        # Per-ticker weights + position values (snapshot at current step).
        weights = {}
        position_values = {}
        for t in self.basket_tickers:
            position_values[t] = self.per_ticker[t]["shares_held"] * prices[t]
            weights[t] = position_values[t] / max(pv, 1e-9)

        # Statistics over policy TARGET weights across the episode (pre-
        # liquidation). weights_history stores the agent's target_weights
        # BEFORE the last-step liquidation override, so these capture the
        # policy's actual allocation character. The post-liquidation
        # `weights` snapshot above is always (0,0,...,cash=1) at episode
        # end, which is uninformative.
        if self.weights_history:
            weights_arr = np.array(self.weights_history)  # (n_steps, n_tickers + 1)
            mean_weights = {
                t: float(weights_arr[:, i].mean())
                for i, t in enumerate(self.basket_tickers)
            }
            std_weights = {
                t: float(weights_arr[:, i].std())
                for i, t in enumerate(self.basket_tickers)
            }
            min_weights = {
                t: float(weights_arr[:, i].min())
                for i, t in enumerate(self.basket_tickers)
            }
            max_weights = {
                t: float(weights_arr[:, i].max())
                for i, t in enumerate(self.basket_tickers)
            }
            mean_cash = float(weights_arr[:, -1].mean())
            std_cash = float(weights_arr[:, -1].std())
            min_cash = float(weights_arr[:, -1].min())
            max_cash = float(weights_arr[:, -1].max())
        else:
            mean_weights = {t: 0.0 for t in self.basket_tickers}
            std_weights = {t: 0.0 for t in self.basket_tickers}
            min_weights = {t: 0.0 for t in self.basket_tickers}
            max_weights = {t: 0.0 for t in self.basket_tickers}
            mean_cash = 1.0
            std_cash = 0.0
            min_cash = 1.0
            max_cash = 1.0

        gross_return_pct = (pv + self.total_transaction_fee - self.initial_balance) / max(self.initial_balance, 1e-9)
        net_return_pct = (pv - self.initial_balance) / max(self.initial_balance, 1e-9)

        # Aggregate completed trades across tickers.
        total_completed = sum(self.per_ticker[t]["completed_trades"] for t in self.basket_tickers)

        # Average hold time across all closed trades from all tickers.
        all_hold_times: List[int] = []
        for t in self.basket_tickers:
            all_hold_times.extend(self.per_ticker[t]["hold_times"])
        avg_hold_time = float(np.mean(all_hold_times)) if all_hold_times else 0.0

        return {
            "step": self.current_step,
            "step_episode": self.current_step_in_episode,
            "balance": self.balance,
            "portfolio_value": pv,
            "weights": weights,
            "mean_weights": mean_weights,
            "mean_cash": mean_cash,
            "std_weights": std_weights,
            "std_cash": std_cash,
            "min_weights": min_weights,
            "min_cash": min_cash,
            "max_weights": max_weights,
            "max_cash": max_cash,
            "position_values": position_values,
            "current_prices": prices,
            "gross_return_pct": float(gross_return_pct),
            "net_return_pct": float(net_return_pct),
            "total_transaction_fee": self.total_transaction_fee,
            "current_transaction_fee": self.current_transaction_fee,
            "trade_execution_count": self.trade_execution_count,
            "turnover_ratio": self._get_turnover_ratio(),
            "avg_hold_time": avg_hold_time,
            "completed_trades": total_completed,
            "win_rate": self._get_win_rate(),
            "invalid_actions_count": self.invalid_actions_count,
            "sharpe_ratio": self._get_episode_sharpe() * np.sqrt(252),
            "sortino_ratio": self._get_episode_sortino() * np.sqrt(252),
            "calmar_ratio": self._get_episode_calmar(),
            "profit_factor": self._get_profit_factor(),
            "avg_win_loss_ratio": self._get_avg_win_loss_ratio(),
            "current_drawdown": self._get_current_drawdown(),
            "max_drawdown": self.max_drawdown,
            "episode_progress": self._get_episode_progress(),
        }

    def render(self) -> None:
        info = self._get_info()
        print(f"Step {info['step']} (ep {info['step_episode']})  "
              f"PV ${info['portfolio_value']:.2f}  "
              f"NetRet {info['net_return_pct']:+.2%}  "
              f"Fees ${info['total_transaction_fee']:.2f}")
        for t in self.basket_tickers:
            w = info['weights'].get(t, 0.0)
            print(f"  {t:5s}  weight={w:+.4f}  shares={self.per_ticker[t]['shares_held']:.4f}")
        print("=" * 60)

    def get_episode_data(self) -> Dict[str, Any]:
        return {
            "actions": self.actions_history,
            "rewards": self.rewards_history,
            "portfolio_values": self.portfolio_values_history,
            "weights": self.weights_history,
        }

    def get_final_portfolio_value(self) -> float:
        return self._get_portfolio_value()