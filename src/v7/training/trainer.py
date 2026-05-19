import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
from datetime import datetime, date
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config.config import (
    DATA_DIR,
    TRAINING_LOGS_DIR,
    NUM_EPISODES,
    VALID_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    BATCH_SIZE,
)
from src.v7.environment.environment import DailyEnvironment
from src.v7.model.agent import Agent
from src.v7.preprocessing.constants import SHARED_REGIME_PREFIXES
from src.utils.logger import Logger
from src.utils.utils import create_directory, load_stock_data, format_duration, resolve_run_number


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TickerData:
    """Pre-loaded data for a single ticker, ready for environment use.

    `columns` is the ordered list of feature-column names corresponding
    to the columns of `data` (i.e. the same ordering used when
    constructing `data` from the unified parquet: market → temporal →
    per-ticker regime → shared regime). Useful for debugging,
    sanity-checking observations, and any code that needs to look up
    a feature by name rather than by positional index.
    """
    __slots__ = ('ticker', 'data', 'prices', 'timestamps',
                 'n_market', 'n_temporal', 'n_regime', 'n_bars',
                 'first_valid_idx', 'last_valid_idx', 'columns')

    def __init__(self, ticker: str, data: np.ndarray, prices: np.ndarray,
                 timestamps: np.ndarray, n_market: int, n_temporal: int, n_regime: int,
                 first_valid_idx: int, last_valid_idx: int,
                 columns: List[str]):
        self.ticker = ticker
        self.data = data
        self.prices = prices
        self.timestamps = timestamps
        self.n_market = n_market
        self.n_temporal = n_temporal
        self.n_regime = n_regime
        self.n_bars = len(data)
        # Inclusive range of rows where this ticker had real data. Rows
        # outside [first_valid_idx, last_valid_idx] are zero-padded leftovers
        # from the left-join in _build_unified (pre-IPO and post-delisting).
        self.first_valid_idx = first_valid_idx
        self.last_valid_idx = last_valid_idx
        # Feature column names in the same order as `data`'s columns.
        self.columns = columns


def _build_basket_inputs(
    basket_data: Dict[str, TickerData],
    basket_tickers: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, int, int, int]:
    """
    Extract v7 env inputs from a Dict[str, TickerData].

    v7's env wants:
      - per-ticker market_data (T, F_per_ticker) where F_per_ticker is
        the per-ticker base features PLUS per-ticker regime (rs_spy_*),
        concatenated. For the asset-class basket this is 37 + 6 = 43 cols.
      - prices dict
      - shared temporal (T, F_temporal)  — same across all basket members
      - shared regime (T, F_shared_regime) — same across all basket members,
        excludes per-ticker regime which now feeds the per-ticker encoder
        instead of the regime path.

    TickerData stores `data` in the order
        [market | temporal | per_ticker_regime | shared_regime]
    so per-ticker market for v7 is a non-contiguous concat of columns
    [0:n_market] + [n_market+n_temporal : n_market+n_temporal+n_per_ticker_regime].

    n_per_ticker_regime isn't stored on TickerData explicitly; we infer it
    by scanning columns: per-ticker regime cols don't start with any of
    SHARED_REGIME_PREFIXES, shared ones do. All basket members must have
    the same schema (verified at runtime).

    Also computes (first_valid_idx, last_valid_idx): the index range over
    which ALL basket members have finite price data. Basket members can
    have different IPO dates (e.g. UUP 2007-02 vs SPY 1993), and the
    unified.parquet's left-join leaves leading rows null for late-IPO
    members. Episodes whose `start_idx` puts current_step in a member's
    pre-IPO range produce NaN portfolio_state — the trainer uses
    first_valid_idx to clamp `_build_sample_pool`'s `lo` to a safe range.
    """
    if not basket_tickers:
        raise ValueError("basket_tickers is empty")
    sample = basket_data[basket_tickers[0]]
    n_market = sample.n_market
    n_temporal = sample.n_temporal
    n_total_regime = sample.n_regime
    T = sample.data.shape[0]

    # Detect n_per_ticker_regime from the columns list. Per-ticker regime
    # cols precede shared regime in TickerData's column order, so the
    # first SHARED_REGIME_PREFIXES match marks the boundary.
    n_per_ticker_regime = 0
    for col in sample.columns[n_market + n_temporal:]:
        if col.startswith(SHARED_REGIME_PREFIXES):
            break
        n_per_ticker_regime += 1
    n_shared_regime = n_total_regime - n_per_ticker_regime

    if n_market + n_temporal + n_total_regime != sample.data.shape[1]:
        raise ValueError(
            f"Column-count mismatch in TickerData for {basket_tickers[0]}: "
            f"n_market({n_market}) + n_temporal({n_temporal}) + n_regime({n_total_regime}) "
            f"!= data.shape[1]({sample.data.shape[1]})"
        )

    # Per-ticker arrays. Verify schema match across basket members.
    market_data: Dict[str, np.ndarray] = {}
    prices: Dict[str, np.ndarray] = {}
    for t in basket_tickers:
        td = basket_data[t]
        if td.n_market != n_market or td.n_temporal != n_temporal or td.n_regime != n_total_regime:
            raise ValueError(
                f"Schema mismatch for {t}: "
                f"(n_market={td.n_market}, n_temporal={td.n_temporal}, n_regime={td.n_regime}) "
                f"vs reference ({n_market}, {n_temporal}, {n_total_regime})"
            )
        base = td.data[:, :n_market]
        per_t_regime = td.data[:, n_market + n_temporal : n_market + n_temporal + n_per_ticker_regime]
        market_data[t] = np.concatenate([base, per_t_regime], axis=1).astype(np.float32)
        prices[t] = td.prices.astype(np.float32)

    # Shared temporal + shared regime (same across all members; take from sample).
    temporal_data = sample.data[:, n_market : n_market + n_temporal].astype(np.float32)
    shared_regime_start = n_market + n_temporal + n_per_ticker_regime
    regime_data = sample.data[:, shared_regime_start:].astype(np.float32)

    # ----------------------------------------------------------------------
    # Determine the basket-wide valid index range. A row is "basket-valid"
    # only if every member's price at that row is finite. Leading NaN runs
    # correspond to pre-IPO periods of late-listing members; trailing NaN
    # runs would correspond to delistings (not expected for asset-class
    # basket, but the check is defensive). The trainer clamps episode
    # sampling to [first_valid_idx, last_valid_idx] inclusive.
    # ----------------------------------------------------------------------
    valid_mask = np.ones(T, dtype=bool)
    for t in basket_tickers:
        valid_mask &= np.isfinite(prices[t])

    if not valid_mask.any():
        raise ValueError(
            f"No row in unified.parquet has finite prices for the entire basket "
            f"({basket_tickers}). Check preprocessing / IPO dates / cutoff."
        )

    first_valid_idx = int(np.argmax(valid_mask))                # first True
    last_valid_idx = T - 1 - int(np.argmax(valid_mask[::-1]))   # last True

    # Internal NaN check: leading + trailing should be the only sources for
    # asset-class basket. Anything else is a preprocessing bug.
    interior = valid_mask[first_valid_idx : last_valid_idx + 1]
    if not interior.all():
        n_interior_nan = int((~interior).sum())
        # Report the first few problem rows per ticker so the user can
        # investigate; don't raise — the trainer's sample-pool clamp
        # already guarantees we don't sample those rows for start_idx,
        # but interior NaN within an episode window WOULD still crash,
        # so this is a serious warning.
        first_bad = first_valid_idx + int(np.argmax(~interior))
        msg = (
            f"Internal NaN in basket prices: {n_interior_nan} interior "
            f"rows have NaN for at least one basket member. First bad row: "
            f"{first_bad}. Episodes whose start_idx leads to a window "
            f"containing these rows will fail."
        )
        # Use a print so it surfaces even when logger isn't configured.
        # The trainer logs this via its own logger when constructed.
        print(f"WARNING: {msg}")

    return (
        market_data, prices, temporal_data, regime_data,
        n_per_ticker_regime, first_valid_idx, last_valid_idx,
    )


# ---------------------------------------------------------------------------
# Walk-forward fold generation
# ---------------------------------------------------------------------------

def generate_walk_forward_folds(
    data_start_year: int,
    data_end_year: int,
    initial_train_years: int = 10,
    valid_years: int = 1,
    test_years: int = 2,
    logger: Optional[Logger] = None,
) -> List[Dict[str, date]]:
    """
    Generate expanding-window walk-forward folds.

    The training window always starts from data_start_year and expands
    forward. The validation window is the next valid_years after training.
    The final test_years of data are held out entirely.

    Example with data 2004-2025, initial_train=10, valid=1, test=2:
        Fold 1: Train 2004-2013, Valid 2014
        Fold 2: Train 2004-2014, Valid 2015
        ...
        Fold 10: Train 2004-2022, Valid 2023
        (2024-2025 reserved for final test)

    Returns list of dicts with keys:
        train_start, train_end, valid_start, valid_end
    """
    test_start_year = data_end_year - test_years + 1
    first_valid_year = data_start_year + initial_train_years

    folds = []
    valid_year = first_valid_year

    while valid_year + valid_years - 1 < test_start_year:
        folds.append({
            "train_start": date(data_start_year, 1, 1),
            "train_end": date(valid_year - 1, 12, 31),
            "valid_start": date(valid_year, 1, 1),
            "valid_end": date(valid_year + valid_years - 1, 12, 31),
        })
        valid_year += 1

    if logger:
        logger.info(
            f"Generated {len(folds)} walk-forward folds "
            f"(initial_train={initial_train_years}y, valid={valid_years}y, "
            f"test={test_years}y holdout at {test_start_year}-{data_end_year})"
        )
        for i, f in enumerate(folds, 1):
            logger.info(
                f"  Fold {i:2d}: Train {f['train_start']} → {f['train_end']} | "
                f"Valid {f['valid_start']} → {f['valid_end']}"
            )

    return folds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DailyTrainer:
    """
    Multi-ticker daily trainer with expanding-window walk-forward validation.

    Training loop:
        Each episode samples a random ticker and a random start date within
        the training time window. The agent trades for episode_days on that
        ticker's daily bars. Sampling is biased toward more recent dates
        via exponential decay.

    Validation:
        Uses the validation time window (immediately after training cutoff).
        Samples random ticker + start date. Runs deterministic policy.

    Walk-forward:
        train_walk_forward() iterates over multiple folds, each with an
        expanding training window and a sliding validation window. The model
        trains continuously across folds (no reset between folds).
    """

    def __init__(
        self,
        agent: Agent,
        basket_tickers: List[str],
        basket_market_data: Dict[str, np.ndarray],
        basket_prices: Dict[str, np.ndarray],
        shared_temporal: np.ndarray,
        shared_regime: np.ndarray,
        timestamps: np.ndarray,
        run_number: int,
        # Basket-wide valid index range (computed by _build_basket_inputs).
        # Episodes whose `current_step` would dip outside this range
        # encounter pre-IPO NaN prices for some basket member and crash.
        # Default = full range when not supplied (backwards-compatible),
        # but main() always passes the computed values.
        basket_first_valid_idx: int = 0,
        basket_last_valid_idx: Optional[int] = None,
        episode_days: int = 252,
        batch_size: int = BATCH_SIZE,
        num_episodes_per_fold: int = NUM_EPISODES,
        valid_interval: int = VALID_INTERVAL,
        valid_episodes_per_eval: int = 10,
        save_interval: int = SAVE_MODEL_INTERVAL,
        incremental_save_every_valids: int = 5,
        models_dir: Union[str, Path] = MODELS_DIR,
        results_dir: Union[str, Path] = RESULTS_DIR,
        recency_decay: float = 1.5,
        logger: Optional[Logger] = None,
    ):
        self.agent = agent
        self.basket_tickers = list(basket_tickers)
        self.basket_market_data = basket_market_data
        self.basket_prices = basket_prices
        self.shared_temporal = shared_temporal
        self.shared_regime = shared_regime
        self.timestamps = timestamps
        self.n_bars = len(timestamps)
        self.basket_first_valid_idx = max(0, int(basket_first_valid_idx))
        self.basket_last_valid_idx = (
            int(basket_last_valid_idx) if basket_last_valid_idx is not None
            else self.n_bars - 1
        )
        self.run_number = run_number
        self.episode_days = episode_days
        self.batch_size = batch_size
        self.num_episodes_per_fold = num_episodes_per_fold
        self.valid_interval = valid_interval
        self.valid_episodes_per_eval = valid_episodes_per_eval
        self.save_interval = save_interval
        self.incremental_save_every_valids = incremental_save_every_valids

        # Run-scoped paths: every artifact for this run (checkpoints, stats,
        # plots) lives under a single `run_N/` dir so crash recovery and
        # post-hoc analysis never cross streams between runs.
        self.models_dir  = Path(models_dir)  / f"run_{run_number}"
        self.results_dir = Path(results_dir) / f"run_{run_number}"
        self.plots_dir   = self.results_dir / "plots"

        create_directory(self.models_dir)
        create_directory(self.results_dir)
        create_directory(self.plots_dir)

        self.recency_decay = recency_decay
        self.logger = logger

        # Validation-call counter drives incremental _save_results.
        self._validation_call_count = 0

        # Metrics (accumulated across all folds). Keep train_* / valid_*
        # lists parallel so _plot_metric can render both curves cleanly.
        self.train_returns = []
        self.valid_returns = []
        self.train_rewards = []
        self.valid_rewards = []
        self.train_fee_impacts = []
        self.valid_fee_impacts = []
        self.train_total_trade_counts = []
        self.valid_total_trade_counts = []
        self.train_sharpe_ratios = []
        self.valid_sharpe_ratios = []
        self.train_sortino_ratios = []
        self.valid_sortino_ratios = []
        self.train_calmar_ratios = []
        self.valid_calmar_ratios = []
        self.train_profit_factors = []
        self.valid_profit_factors = []
        self.train_avg_win_loss_ratios = []
        self.valid_avg_win_loss_ratios = []
        self.train_max_drawdowns = []
        self.valid_max_drawdowns = []
        self.train_win_rates = []
        self.valid_win_rates = []
        self.train_losses = []
        self.fold_boundaries = []

        # Fixed validation set for the current fold (re-sampled at each fold
        # start). For v7 this is just a list of start_idx values — basket is
        # fixed so there's no per-ticker dim.
        self.current_valid_set: List[int] = []

        if self.logger:
            self.logger.info(
                f"DailyTrainer initialized (run {run_number}): "
                f"basket={self.basket_tickers}, "
                f"episode={episode_days} days"
            )
            first_ts = np.datetime_as_string(self.timestamps[self.basket_first_valid_idx], unit="D")
            last_ts = np.datetime_as_string(self.timestamps[self.basket_last_valid_idx], unit="D")
            self.logger.info(
                f"  basket valid range: idx [{self.basket_first_valid_idx}, "
                f"{self.basket_last_valid_idx}] = dates [{first_ts}, {last_ts}]"
            )
            self.logger.info(f"  results_dir: {self.results_dir}")
            self.logger.info(f"  models_dir:  {self.models_dir}")

    # ------------------------------------------------------------------
    # Time-filtered sample pool
    # ------------------------------------------------------------------

    def _build_sample_pool(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Tuple[int, float]]:
        """
        Build a list of (start_idx, weight) for episode sampling, restricted
        to episodes that START within [start_date, end_date].

        v7: basket is fixed (no per-ticker dim). All basket members are
        aligned on the same NYSE trading-day timeline so a single start_idx
        applies uniformly. Bounds:

          - lo: max(basket_first_valid_idx, 0). Late-IPO basket members
            (e.g. UUP listed 2007-02-20 with CUTOFF_TIMESTAMP at 2004-12-13)
            have leading NaN price rows. Starting an episode in that range
            produces NaN portfolio_state on the first observation.
          - hi: min(basket_last_valid_idx - episode_days + 1, n_bars - episode_days).
            The episode [start_idx, start_idx + episode_days) must fit
            within the valid range.

        No market-data window lookback bound is needed in v7 (MLP-on-current-
        bar means the encoder doesn't read history beyond the current step).
        Rolling-z-score warmup is already past via the unified parquet's
        CUTOFF_TIMESTAMP cutoff.

        Weight decays exponentially with age for recency bias (same recency
        mechanism as v6, but now applied over a single start_idx axis).
        """
        lo = self.basket_first_valid_idx
        hi = min(
            self.basket_last_valid_idx - self.episode_days + 1,
            self.n_bars - self.episode_days,
        )
        if lo >= hi:
            return []

        pool: List[Tuple[int, float]] = []
        for start_idx in range(lo, hi):
            ts = self.timestamps[start_idx]
            ts_date = ts.astype('datetime64[D]').astype(date) if hasattr(ts, 'astype') else (
                ts.date() if hasattr(ts, 'date') else ts
            )
            if ts_date < start_date or ts_date > end_date:
                continue
            pool.append((start_idx, 0.0))

        if not pool:
            return pool

        # Recency weighting: most recent start_idx = age 0, oldest = age 1.
        pool.sort(key=lambda x: x[0])
        n = len(pool)
        weighted = []
        for i, (start_idx, _) in enumerate(pool):
            age = 1.0 - (i / max(n - 1, 1))
            weight = np.exp(-self.recency_decay * age)
            weighted.append((start_idx, weight))

        total = sum(w for _, w in weighted)
        return [(s, w / total) for s, w in weighted]

    def _sample_episode(self, pool: List[Tuple[int, float]]) -> int:
        """Sample a start_idx from the weighted pool."""
        weights = np.array([w for _, w in pool])
        idx = np.random.choice(len(pool), p=weights)
        return pool[idx][0]

    def _sample_fixed_valid_set(
        self,
        valid_pool: List[Tuple[int, float]],
        n: int,
    ) -> List[int]:
        """
        Draw a fixed set of start_idx values from valid_pool for use across
        all validation calls within a fold.

        Sampling is uniform (no recency weighting) so the validation set
        represents the whole validation window. For v7 the basket is fixed
        — no per-ticker cap mechanism needed; we just sample n unique
        start_idx values uniformly (or all of them if the pool is small).
        """
        if not valid_pool:
            return []
        all_starts = [s for s, _ in valid_pool]
        k = min(n, len(all_starts))
        chosen = np.random.choice(all_starts, size=k, replace=False)
        return [int(s) for s in chosen]

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        start_idx: int,
        train: bool = True,
    ) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
        """
        Run a single basket-episode starting at start_idx.

        v7: env operates on all basket members simultaneously. start_idx is
        a single integer into the shared NYSE timeline; the env slices each
        basket member's market_data at the same start_idx since they're
        aligned.
        """
        env = DailyEnvironment(
            basket_tickers=self.basket_tickers,
            market_data=self.basket_market_data,
            prices=self.basket_prices,
            temporal_data=self.shared_temporal,
            regime_data=self.shared_regime,
            episode_days=self.episode_days,
        )

        state = env.reset(start_idx=start_idx)
        total_reward = 0.0
        update_count = 0
        train_loss = {
            "actor_loss": 0, "critic_loss": 0, "alpha_loss": 0,
            "entropy": 0, "alpha": 0, "q_value": 0,
            # Cumulative counters on Agent — track LATEST value across
            # the episode's UTD calls, not running sum (which would
            # double-count since each call returns the running total).
            "critic_nan_skips": 0, "actor_nan_skips": 0, "alpha_nan_skips": 0,
        }
        _AVG_KEYS = ("actor_loss", "critic_loss", "alpha_loss", "entropy", "alpha", "q_value")
        _LATEST_KEYS = ("critic_nan_skips", "actor_nan_skips", "alpha_nan_skips")
        done = False

        while not done:
            action = self.agent.select_action(state, validate=not train)
            # v7 action is a (n_tickers,) numpy array, not a scalar. Env
            # handles softmax-with-cash-logit internally.
            next_state, reward, done, info = env.step(action)

            if train:
                self.agent.replay_buffer.push(
                    state, action, reward, next_state, float(done)
                )

                if len(self.agent.replay_buffer) > self.batch_size:
                    update_count += 1
                    loss = self.agent.update_parameters(self.batch_size)
                    for k in _AVG_KEYS:
                        if k in loss:
                            train_loss[k] += loss[k]
                    for k in _LATEST_KEYS:
                        if k in loss:
                            train_loss[k] = loss[k]   # cumulative on self.agent → keep latest

            state = next_state
            total_reward += reward

        if update_count > 0:
            for k in _AVG_KEYS:
                train_loss[k] /= update_count

        return total_reward, info, train_loss

    # ------------------------------------------------------------------
    # Per-episode logging
    # ------------------------------------------------------------------

    def _log_episode_block(
        self,
        label: str,
        start_idx: int,
        total_reward: float,
        info: Dict[str, Any],
        train_loss: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Emit a vertical per-episode log block for v7 basket episodes.

        Format mirrors v6 single-ticker logs as closely as possible so train/
        valid output reads the same. Differences from v6:
          - "Episode Price Diff" reports SPY's move over the episode window
            as a regime benchmark, not a single tradable's move.
          - "Final Weights" line lists the policy's last-rebalance target
            weights per basket member (5 entries + implicit cash).
        """
        if self.logger is None:
            return

        last_idx = min(start_idx + self.episode_days, self.n_bars) - 1

        start_date_str = np.datetime_as_string(self.timestamps[start_idx], unit="D")
        end_date_str   = np.datetime_as_string(self.timestamps[last_idx],  unit="D")

        # SPY-as-benchmark: report SPY's start/end price + pct change over the
        # window. Useful as a passive-baseline reference for the episode.
        # Falls back gracefully if SPY isn't in the basket.
        benchmark = "SPY" if "SPY" in self.basket_prices else self.basket_tickers[0]
        bench_prices = self.basket_prices[benchmark]
        start_price = float(bench_prices[start_idx])
        end_price = float(bench_prices[last_idx])
        price_diff = end_price - start_price
        price_diff_pct = price_diff / start_price if start_price > 0 else 0.0

        net_return_pct   = info["net_return_pct"]
        gross_return_pct = info["gross_return_pct"]
        fee_impact       = gross_return_pct - net_return_pct

        # Mean policy-target weights across the episode (pre-liquidation).
        # The "weights" in info is the post-liquidation snapshot — always
        # (0,...,0, cash=1) on episode end since last-step liquidation
        # forces all-cash. Mean target weights from weights_history show
        # what the policy was actually doing across the 252 steps; std
        # captures HOW MUCH the policy moved (a static 0.20 allocation
        # vs constant flipping between 0.0 and 0.40 both have mean=0.20
        # but very different std), and min/max captures the range explored.
        mean_weights = info.get("mean_weights", {})
        std_weights = info.get("std_weights", {})
        min_weights = info.get("min_weights", {})
        max_weights = info.get("max_weights", {})
        mean_cash = info.get("mean_cash", 0.0)
        std_cash = info.get("std_cash", 0.0)
        min_cash = info.get("min_cash", 0.0)
        max_cash = info.get("max_cash", 0.0)

        weights_mean_std_str = ", ".join(
            f"{t}={mean_weights.get(t, 0.0):.3f}±{std_weights.get(t, 0.0):.3f}"
            for t in self.basket_tickers
        )
        weights_mean_std_str += f", cash={mean_cash:.3f}±{std_cash:.3f}"

        weights_range_str = ", ".join(
            f"{t} [{min_weights.get(t, 0.0):.3f}, {max_weights.get(t, 0.0):.3f}]"
            for t in self.basket_tickers
        )
        weights_range_str += f", cash [{min_cash:.3f}, {max_cash:.3f}]"

        lines = [
            f"\n{label} | basket={self.basket_tickers}",
            f"Episode Start Date:  {start_date_str}",
            f"Episode End Date:    {end_date_str}",
            f"Benchmark ({benchmark}): ${start_price:.2f} → ${end_price:.2f} ({price_diff_pct:+.2%})",
            f"Reward Total: {total_reward:.2f}",
            f"Balance:      ${info['balance']:.2f}",
            f"Portfolio Value: ${info['portfolio_value']:.2f}",
            f"Gross Return (Pre-Fee):  {gross_return_pct:.2%}",
            f"Net Return (Post-Fee):   {net_return_pct:.2%}{' POSITIVE' if net_return_pct > 0 else ''}",
            f"Fee Impact:              {fee_impact:.2%}",
            f"Total Trade Count:  {info['trade_execution_count']}",
            f"Turnover Ratio:     {info['turnover_ratio']:.2%}",
            f"Avg Hold Time:      {info['avg_hold_time']:.2f}",
            f"Sharpe Ratio:       {info['sharpe_ratio']:.2f}",
            f"Sortino Ratio:      {info['sortino_ratio']:.2f}",
            f"Calmar Ratio:       {info['calmar_ratio']:.2f}",
            f"Profit Factor:      {info['profit_factor']:.2f}",
            f"Avg Win/Loss:       {info['avg_win_loss_ratio']:.2f}",
            f"Max Drawdown:       {info['max_drawdown']:.2%}",
            f"Win Rate:           {info['win_rate']:.2%}",
            f"Weights (mean±std): {weights_mean_std_str}",
            f"Weight Range:       {weights_range_str}",
        ]

        if train_loss is not None:
            lines.append(
                f"Losses:  actor={train_loss.get('actor_loss', 0.0):+.4f}, "
                f"critic={train_loss.get('critic_loss', 0.0):.4f}, "
                f"alpha_loss={train_loss.get('alpha_loss', 0.0):+.4f}"
            )
            lines.append(
                f"SAC:     alpha={train_loss.get('alpha', 0.0):.4f}, "
                f"entropy={train_loss.get('entropy', 0.0):+.4f}, "
                f"q_value={train_loss.get('q_value', 0.0):+.4f}"
            )
            nan_total = (
                train_loss.get('critic_nan_skips', 0)
                + train_loss.get('actor_nan_skips', 0)
                + train_loss.get('alpha_nan_skips', 0)
            )
            if nan_total > 0:
                lines.append(
                    f"NaN skips (cumulative): "
                    f"critic={train_loss.get('critic_nan_skips', 0)}, "
                    f"actor={train_loss.get('actor_nan_skips', 0)}, "
                    f"alpha={train_loss.get('alpha_nan_skips', 0)}"
                )

        lines.append("=" * 50)
        self.logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Single-fold training
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        fold_idx: int,
        fold: Dict[str, date],
        global_episode_offset: int,
        timestamp: str,
    ) -> int:
        """
        Train one walk-forward fold. Returns the number of episodes run.

        The model is NOT reset between folds — training is continuous so the
        agent builds on what it learned in previous folds.
        """
        train_start = fold["train_start"]
        train_end = fold["train_end"]
        valid_start = fold["valid_start"]
        valid_end = fold["valid_end"]

        if self.logger:
            self.logger.info(
                f"\n{'='*60}\n"
                f"Fold {fold_idx + 1}: "
                f"Train {train_start} → {train_end} | "
                f"Valid {valid_start} → {valid_end}\n"
                f"{'='*60}"
            )

        # Build sample pools for this fold's time boundaries (basket-wide).
        train_pool = self._build_sample_pool(train_start, train_end)
        valid_pool = self._build_sample_pool(valid_start, valid_end)

        # Sample a fixed validation set once per fold — reused across all
        # _run_validation calls in this fold for clean learning curves.
        self.current_valid_set = self._sample_fixed_valid_set(
            valid_pool, n=self.valid_episodes_per_eval,
        )

        if not train_pool:
            if self.logger:
                self.logger.warning(
                    f"  No valid training episodes for fold {fold_idx + 1}, skipping"
                )
            return 0

        if self.logger:
            self.logger.info(
                f"  Train pool: {len(train_pool)} start-idx candidates | "
                f"Valid pool: {len(valid_pool)} | "
                f"Fixed valid set: {len(self.current_valid_set)} episodes"
            )

        self.fold_boundaries.append({
            "fold": fold_idx + 1,
            "episode_start": global_episode_offset + 1,
            "episode_end": global_episode_offset + self.num_episodes_per_fold,
            **fold,
        })

        for ep in range(1, self.num_episodes_per_fold + 1):
            global_ep = global_episode_offset + ep
            start_idx = self._sample_episode(train_pool)

            self.agent.actor.train()
            self.agent.current_episode = global_ep

            total_reward, info, train_loss = self._run_episode(start_idx, train=True)

            # Record metrics.
            net_return_pct = info["net_return_pct"]
            self.train_returns.append(net_return_pct * 100)
            self.train_rewards.append(total_reward)
            self.train_fee_impacts.append(
                (info["gross_return_pct"] - net_return_pct) * 100
            )
            self.train_total_trade_counts.append(info["trade_execution_count"])
            self.train_sharpe_ratios.append(info["sharpe_ratio"])
            self.train_sortino_ratios.append(info["sortino_ratio"])
            self.train_calmar_ratios.append(info["calmar_ratio"])
            self.train_profit_factors.append(info["profit_factor"])
            self.train_avg_win_loss_ratios.append(info["avg_win_loss_ratio"])
            self.train_max_drawdowns.append(info["max_drawdown"] * 100)
            self.train_win_rates.append(info["win_rate"] * 100)
            self.train_losses.append(train_loss)

            if self.logger:
                label = (
                    f"Fold {fold_idx+1} EP: {ep}/{self.num_episodes_per_fold} "
                    f"(global {global_ep})"
                )
                self._log_episode_block(
                    label=label,
                    start_idx=start_idx,
                    total_reward=total_reward,
                    info=info,
                    train_loss=train_loss,
                )

            # Validation
            if ep % self.valid_interval == 0 and self.current_valid_set:
                self._run_validation(fold_idx, global_ep)

            # Save model
            if ep % self.save_interval == 0:
                self.agent.save_model(
                    save_dir=self.models_dir,
                    prefix=f"daily_fold{fold_idx+1}_",
                    timestamp=f"{timestamp}_ep{global_ep}",
                )

        return self.num_episodes_per_fold

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, fold_idx: int, global_ep: int):
        """
        Run validation on the fold's fixed validation set and log results.

        The set of start_idx values is sampled once at fold start and reused
        here, so metric changes across validation calls reflect policy
        improvement rather than draw-to-draw noise.
        """
        if not self.current_valid_set:
            return

        self.agent.actor.eval()

        valid_returns = []
        valid_rewards = []
        valid_trades = []
        valid_sharpes = []
        valid_sortinos = []
        valid_calmars = []
        valid_profit_factors = []
        valid_avg_wl = []
        valid_max_dd = []
        valid_fee_impacts = []
        valid_win_rates = []

        for i, start_idx in enumerate(self.current_valid_set, 1):
            total_reward, info, _ = self._run_episode(start_idx, train=False)

            net_return_pct = info["net_return_pct"]
            valid_returns.append(net_return_pct * 100)
            valid_rewards.append(total_reward)
            valid_trades.append(info["trade_execution_count"])
            valid_sharpes.append(info["sharpe_ratio"])
            valid_sortinos.append(info["sortino_ratio"])
            valid_calmars.append(info["calmar_ratio"])
            valid_profit_factors.append(info["profit_factor"])
            valid_avg_wl.append(info["avg_win_loss_ratio"])
            valid_max_dd.append(info["max_drawdown"] * 100)
            valid_fee_impacts.append((info["gross_return_pct"] - net_return_pct) * 100)
            valid_win_rates.append(info["win_rate"] * 100)

            label = (
                f"VALID Fold {fold_idx+1} ep {i}/{len(self.current_valid_set)} "
                f"(global {global_ep})"
            )
            self._log_episode_block(
                label=label,
                start_idx=start_idx,
                total_reward=total_reward,
                info=info,
                train_loss=None,  # no gradient updates on valid → no alpha/entropy
            )

        # Append means to the persistent valid_* lists so _plot_metric can
        # render both train and valid curves and _save_results can persist
        # a symmetric stats dict.
        self.valid_returns.append(np.mean(valid_returns))
        self.valid_rewards.append(np.mean(valid_rewards))
        self.valid_total_trade_counts.append(np.mean(valid_trades))
        self.valid_sharpe_ratios.append(np.mean(valid_sharpes))
        self.valid_sortino_ratios.append(np.mean(valid_sortinos))
        self.valid_calmar_ratios.append(np.mean(valid_calmars))
        self.valid_profit_factors.append(np.mean(valid_profit_factors))
        self.valid_avg_win_loss_ratios.append(np.mean(valid_avg_wl))
        self.valid_max_drawdowns.append(np.mean(valid_max_dd))
        self.valid_fee_impacts.append(np.mean(valid_fee_impacts))
        self.valid_win_rates.append(np.mean(valid_win_rates))

        if self.logger:
            self.logger.info(
                f"  Validation (fold {fold_idx+1}, ep {global_ep}): "
                f"Return: {np.mean(valid_returns):.2f}% ± {np.std(valid_returns):.2f}% | "
                f"Trades: {np.mean(valid_trades):.0f} | "
                f"Sharpe: {np.mean(valid_sharpes):.2f} | "
                f"Sortino: {np.mean(valid_sortinos):.2f} | "
                f"Calmar: {np.mean(valid_calmars):.2f} | "
                f"PF: {np.mean(valid_profit_factors):.2f} | "
                f"W/L: {np.mean(valid_avg_wl):.2f} | "
                f"MaxDD: {np.mean(valid_max_dd):.1f}% | "
                f"Positive: {sum(1 for r in valid_returns if r > 0)}/{len(valid_returns)}"
            )

        self.agent.actor.train()

        # Incremental persistence — every Nth validation call, re-save
        # stats + re-render plots so a crash mid-run still leaves prior
        # folds analyzable. Overwrites in place (last-write-wins).
        self._validation_call_count += 1
        if (
            self.incremental_save_every_valids > 0
            and self._validation_call_count % self.incremental_save_every_valids == 0
        ):
            if self.logger:
                self.logger.info(
                    f"  [incremental save] valid call "
                    f"{self._validation_call_count} → {self.results_dir}"
                )
            self._save_results()

    # ------------------------------------------------------------------
    # Walk-forward training
    # ------------------------------------------------------------------

    def train_walk_forward(self, folds: List[Dict[str, date]]) -> Dict[str, List]:
        """
        Train across multiple walk-forward folds with expanding window.

        The model trains continuously — no reset between folds. Each fold
        expands the training window and slides the validation window forward.

        Parameters
        ----------
        folds : list of dict
            Each dict has keys: train_start, train_end, valid_start, valid_end
            Generated by generate_walk_forward_folds().

        Returns
        -------
        dict of training metrics
        """
        start_time = time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.logger:
            self.logger.info(
                f"Walk-forward training: {len(folds)} folds, "
                f"{self.num_episodes_per_fold} episodes per fold, "
                f"{len(folds) * self.num_episodes_per_fold} total episodes"
            )

        global_episode_offset = 0

        for fold_idx, fold in enumerate(folds):
            episodes_run = self._train_fold(fold_idx, fold, global_episode_offset, timestamp)
            global_episode_offset += episodes_run

        total_time = format_duration(time() - start_time)
        if self.logger:
            self.logger.info(f"\nWalk-forward training complete. Total time: {total_time}")

        # Final model checkpoint — guarantees an end-of-training weights
        # file exists regardless of whether the last episode landed on a
        # save_interval boundary.
        self.agent.save_model(
            save_dir=self.models_dir,
            prefix="daily_final_",
            timestamp=timestamp,
        )
        if self.logger:
            self.logger.info(f"Final model saved to {self.models_dir}")

        self._save_results()

        return {
            "train_returns": self.train_returns,
            "valid_returns": self.valid_returns,
            "train_rewards": self.train_rewards,
            "valid_rewards": self.valid_rewards,
            "fold_boundaries": self.fold_boundaries,
        }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _save_results(self):
        """
        Persist training artifacts into the run-scoped results directory.

        Writes:
            self.results_dir / training_stats.pth
            self.plots_dir   / *.png          (per-metric learning curves)

        Idempotent — safe to call multiple times during training for
        incremental snapshots as well as once at the end.
        """
        stats = {
            # Paired train/valid metrics
            "train_returns": self.train_returns,
            "valid_returns": self.valid_returns,
            "train_rewards": self.train_rewards,
            "valid_rewards": self.valid_rewards,
            "train_fee_impacts": self.train_fee_impacts,
            "valid_fee_impacts": self.valid_fee_impacts,
            "train_total_trade_counts": self.train_total_trade_counts,
            "valid_total_trade_counts": self.valid_total_trade_counts,
            "train_sharpe_ratios": self.train_sharpe_ratios,
            "valid_sharpe_ratios": self.valid_sharpe_ratios,
            "train_sortino_ratios": self.train_sortino_ratios,
            "valid_sortino_ratios": self.valid_sortino_ratios,
            "train_calmar_ratios": self.train_calmar_ratios,
            "valid_calmar_ratios": self.valid_calmar_ratios,
            "train_profit_factors": self.train_profit_factors,
            "valid_profit_factors": self.valid_profit_factors,
            "train_avg_win_loss_ratios": self.train_avg_win_loss_ratios,
            "valid_avg_win_loss_ratios": self.valid_avg_win_loss_ratios,
            "train_max_drawdowns": self.train_max_drawdowns,
            "valid_max_drawdowns": self.valid_max_drawdowns,
            "train_win_rates": self.train_win_rates,
            "valid_win_rates": self.valid_win_rates,
            # Train-only
            "train_losses": self.train_losses,
            "fold_boundaries": self.fold_boundaries,
            # Run metadata
            "run_number": self.run_number,
        }
        torch.save(stats, self.results_dir / "training_stats.pth")

        # All paired metrics — one plot per metric, both curves + stats box.
        paired_plots = [
            (self.train_returns,             self.valid_returns,             "Returns (%)",       "returns.png"),
            (self.train_rewards,             self.valid_rewards,             "Rewards",           "rewards.png"),
            (self.train_fee_impacts,         self.valid_fee_impacts,         "Fee Impact (%)",    "fee_impacts.png"),
            (self.train_total_trade_counts,  self.valid_total_trade_counts,  "Trade Count",       "trade_counts.png"),
            (self.train_sharpe_ratios,       self.valid_sharpe_ratios,       "Sharpe Ratio",      "sharpe.png"),
            (self.train_sortino_ratios,      self.valid_sortino_ratios,      "Sortino Ratio",     "sortino.png"),
            (self.train_calmar_ratios,       self.valid_calmar_ratios,       "Calmar Ratio",      "calmar.png"),
            (self.train_profit_factors,      self.valid_profit_factors,      "Profit Factor",     "profit_factor.png"),
            (self.train_avg_win_loss_ratios, self.valid_avg_win_loss_ratios, "Avg Win/Loss",      "avg_win_loss.png"),
            (self.train_max_drawdowns,       self.valid_max_drawdowns,       "Max Drawdown (%)",  "drawdowns.png"),
            (self.train_win_rates,           self.valid_win_rates,           "Win Rate (%)",      "win_rates.png"),
        ]
        for train_data, valid_data, ylabel, filename in paired_plots:
            self._plot_metric(
                train_data, valid_data, ylabel, "Episode", filename, self.plots_dir
            )

        # Training-only loss diagnostics
        if self.train_losses:
            for key in ["actor_loss", "critic_loss", "alpha_loss",
                        "alpha", "entropy", "q_value"]:
                values = [l[key] for l in self.train_losses]
                self._plot_metric(
                    values, [], key, "Episode", f"{key}.png", self.plots_dir
                )

        if self.logger:
            self.logger.info(f"Results saved to {self.results_dir}")

    def _plot_metric(
        self, train_data, valid_data, ylabel, xlabel, filename, save_dir,
        ma_window=10,
        train_benchmark=None, valid_benchmark=None,
    ):
        """
        Plot a train/valid learning curve with moving average, fold
        boundaries, and a right-side stats text box. Ported from the
        minute-level trainer's curve-plotting style.

        train_data / valid_data: per-episode (train) and per-validation
            (valid) scalar metrics, same shape as the rest of the
            self.*_* lists.
        train_benchmark / valid_benchmark: optional buy-and-hold (or
            similar) reference series to overlay — aligned to train_data
            and valid_data respectively. Used on Returns/Rewards plots.
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        fig.tight_layout(rect=[0, 0, 0.75, 1])

        ax.plot(train_data, alpha=0.3, color="blue", label="Train")
        if len(train_data) >= ma_window:
            ma = np.convolve(
                train_data, np.ones(ma_window) / ma_window, mode="valid"
            )
            ax.plot(range(ma_window - 1, len(train_data)), ma,
                    color="blue", linewidth=1.5,
                    label=f"Train MA({ma_window})")

        if valid_data:
            valid_x = [
                (i + 1) * self.valid_interval
                for i in range(len(valid_data))
            ]
            ax.plot(valid_x, valid_data, "o-", color="orange",
                    markersize=4, label="Validation")

        # Optional benchmark overlays (buy-and-hold, etc.)
        if train_benchmark:
            ax.plot(train_benchmark, alpha=0.3, color="gray",
                    label="Train Buy & Hold")
        if valid_benchmark:
            x_vals_bm = [
                (i + 1) * self.valid_interval
                for i in range(len(valid_benchmark))
            ]
            ax.plot(x_vals_bm, valid_benchmark, color="red", marker="x",
                    markersize=4, linestyle="--", label="Valid Buy & Hold")

        # Fold boundary verticals
        for fb in self.fold_boundaries:
            ep_start = fb["episode_start"]
            if ep_start > 1:
                ax.axvline(x=ep_start - 1, color="gray", linestyle="--",
                           alpha=0.5)
                ax.text(ep_start, ax.get_ylim()[1] * 0.95,
                        f"F{fb['fold']}", fontsize=8, alpha=0.7)

        # Right-side stats text box
        train_arr = np.array(train_data) if train_data else np.array([])
        valid_arr = np.array(valid_data) if valid_data else np.array([])
        lines: List[str] = []

        # Contextual summaries for specific metric types
        if ylabel in ("Returns (%)", "Rewards"):
            if train_arr.size > 0:
                tp = (train_arr > 0).sum()
                lines.append(f"Train > 0: {tp}/{train_arr.size} "
                             f"({tp / train_arr.size:.1%})")
            if valid_arr.size > 0:
                vp = (valid_arr > 0).sum()
                lines.append(f"Valid > 0: {vp}/{valid_arr.size} "
                             f"({vp / valid_arr.size:.1%})")
            if ylabel == "Returns (%)" and train_benchmark:
                n = min(len(train_data), len(train_benchmark))
                beat = sum(1 for r, b in zip(train_data[:n], train_benchmark[:n]) if r > b)
                lines.append(f"Train Beat B&H: {beat}/{n} ({beat / max(n, 1):.1%})")
            if ylabel == "Returns (%)" and valid_benchmark:
                n = min(len(valid_data), len(valid_benchmark))
                beat = sum(1 for r, b in zip(valid_data[:n], valid_benchmark[:n]) if r > b)
                lines.append(f"Valid Beat B&H: {beat}/{n} ({beat / max(n, 1):.1%})")

        if ylabel == "Win Rate (%)":
            if train_arr.size > 0:
                t50 = (train_arr > 50).sum()
                lines.append(f"Train > 50%: {t50}/{train_arr.size} "
                             f"({t50 / train_arr.size:.1%})")
            if valid_arr.size > 0:
                v50 = (valid_arr > 50).sum()
                lines.append(f"Valid > 50%: {v50}/{valid_arr.size} "
                             f"({v50 / valid_arr.size:.1%})")

        # Standard descriptive stats
        def _fmt(v):
            return f"{v:.4f}"

        if train_arr.size > 0:
            lines.extend([
                f"Train Mean:   {_fmt(train_arr.mean())}",
                f"Train STD:    {_fmt(train_arr.std())}",
                f"Train Max:    {_fmt(train_arr.max())}",
                f"Train Min:    {_fmt(train_arr.min())}",
                f"Train Median: {_fmt(np.median(train_arr))}",
            ])
        if valid_arr.size > 0:
            lines.extend([
                f"Valid Mean:   {_fmt(valid_arr.mean())}",
                f"Valid STD:    {_fmt(valid_arr.std())}",
                f"Valid Max:    {_fmt(valid_arr.max())}",
                f"Valid Min:    {_fmt(valid_arr.min())}",
                f"Valid Median: {_fmt(np.median(valid_arr))}",
            ])

        if lines:
            fig.text(
                0.78, 0.85, "\n".join(lines),
                fontsize=9, family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_title(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.savefig(Path(save_dir) / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_price_with_actions(
        self, prices, actions, shares_traded, save_path, title: str,
    ) -> None:
        """
        Plot the price series with buy/sell markers (action visualization).
        Port of the minute-level trainer's action plot, unchanged in
        semantics: marker color by side, marker opacity scaled by trade
        size relative to the episode's max.
        """
        if prices is None or actions is None:
            return
        if len(prices) == 0 or len(actions) == 0:
            return

        prices = np.asarray(prices)
        actions = np.asarray(actions)
        shares_traded = (
            np.asarray(shares_traded) if shares_traded is not None
            else np.ones(len(actions))
        )
        min_len = min(len(prices), len(actions), len(shares_traded))
        prices = prices[:min_len]
        actions = actions[:min_len]
        shares_traded = shares_traded[:min_len]

        plt.figure(figsize=(12, 6))
        plt.plot(prices, label="Price", color="blue", linewidth=1)

        buy_mask = shares_traded > 0
        sell_mask = shares_traded < 0
        buy_idx = np.where(buy_mask)[0]
        sell_idx = np.where(sell_mask)[0]

        abs_traded = np.abs(shares_traded)
        max_traded = abs_traded.max() if abs_traded.max() > 0 else 1.0

        if len(buy_idx) > 0:
            plt.scatter(
                buy_idx, prices[buy_idx],
                marker="^", c="green",
                alpha=np.clip(abs_traded[buy_idx] / max_traded, 0.2, 1.0),
                s=60, label="Buy",
            )
        if len(sell_idx) > 0:
            plt.scatter(
                sell_idx, prices[sell_idx],
                marker="v", c="red",
                alpha=np.clip(abs_traded[sell_idx] / max_traded, 0.2, 1.0),
                s=60, label="Sell",
            )

        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Helper: Load all tickers from unified parquet
# ---------------------------------------------------------------------------

def load_tickers_from_unified(
    parquet_path: str,
    tickers: List[str],
    temporal_col_names: List[str] = None,
) -> Dict[str, TickerData]:
    """
    Load the unified parquet file and extract per-ticker TickerData objects.

    Performance note
    ----------------
    The naive approach — one per-ticker pass that reclassifies the full
    column list (tens of thousands of names) and materializes a separate
    numpy array per ticker — scales badly. This implementation does the
    work the other way around:

      1. Classify every column in the unified frame ONCE, grouping
         per-ticker columns into a dict keyed by ticker symbol and
         collecting shared-regime / temporal columns separately.
      2. Convert the entire dataframe to numpy ONCE.
      3. Per-ticker work is reduced to fancy-indexing column slices
         out of the pre-built numpy array.

    On wide unified frames (3k+ tickers, 80k+ columns) this is two
    orders of magnitude faster than the per-ticker approach and uses
    no extra memory beyond the single full-frame numpy buffer.
    """
    if temporal_col_names is None:
        temporal_col_names = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']

    df = pl.read_parquet(parquet_path)
    all_cols = df.columns
    temporal_set = set(temporal_col_names)

    per_ticker_regime_suffixes = (
        "_cs_zscore",
        "_sector_zscore",
        # RS-vs-SPY levels (short + long horizons)
        "_rs_spy_5d", "_rs_spy_20d", "_rs_spy_60d", "_rs_spy_252d",
        # RS-vs-SPY medium deltas on the long-horizon level. Short-horizon
        # RS levels don't get deltas in the v6 design (per the matching
        # in feature_engineer.py), so no _rs_spy_5d_delta_* etc. here.
        "_rs_spy_252d_delta_20", "_rs_spy_252d_delta_60",
    )

    # --- Phase 1: classify every column once -------------------------------
    #
    # Per-ticker groupings hold (market_col_indices, per_ticker_regime_col_indices)
    # so we can assemble each ticker's (market + regime) slice later.
    # Shared regime + temporal indices are collected once and reused.
    per_ticker_market: Dict[str, List[int]] = {}
    per_ticker_regime: Dict[str, List[int]] = {}
    shared_regime_idx: List[int] = []
    temporal_idx: List[int] = []
    close_col_idx: Dict[str, int] = {}

    # Build a ticker set we actually care about for O(1) lookup. We still
    # walk every column in the frame (to find shared + close cols) but can
    # short-circuit per-ticker classification to the requested tickers only.
    requested = set(tickers)

    for idx, c in enumerate(all_cols):
        if c == "timestamp":
            continue
        if c in temporal_set:
            temporal_idx.append(idx)
            continue
        if c.startswith(SHARED_REGIME_PREFIXES):
            shared_regime_idx.append(idx)
            continue
        if c.endswith("_close"):
            tic = c[:-len("_close")]
            if tic in requested:
                close_col_idx[tic] = idx
            continue

        # Figure out which ticker this column belongs to. Column names are
        # "<TICKER>_<rest>", so split on the first underscore.
        us = c.find("_")
        if us <= 0:
            continue
        tic = c[:us]
        if tic not in requested:
            continue

        stripped = c[us + 1:]
        if stripped.startswith("sector_") or c.endswith(per_ticker_regime_suffixes):
            per_ticker_regime.setdefault(tic, []).append(idx)
        else:
            per_ticker_market.setdefault(tic, []).append(idx)

    # --- Phase 2: convert the whole frame to numpy once --------------------
    #
    # This is the single heavy operation. The feature-engineering auditor
    # downcasts *most* Float64 columns to Float32 (those where the roundtrip
    # error stays within tolerance) but leaves high-precision columns as
    # Float64. That's the right call for audit-time analysis, but the
    # training pipeline ultimately casts everything to Float32 at the
    # network boundary anyway, so propagating Float64 through here just
    # wastes memory — and the naive `df.to_numpy().astype(np.float32)` path
    # upcasts the whole mixed-dtype frame to Float64 first, peaking around
    # 2× what we actually need.
    #
    # Casting the remaining Float64 columns to Float32 *inside polars*
    # before `to_numpy` avoids that intermediate spike. For this pipeline
    # that precision loss is unobservable — the network layer would do the
    # same cast moments later.
    if "timestamp" in all_cols:
        timestamps = df.select("timestamp").to_numpy().ravel()
        df_numeric = df.drop("timestamp")
    else:
        timestamps = np.arange(df.height)
        df_numeric = df

    float64_cols = [c for c, dt in zip(df_numeric.columns, df_numeric.dtypes)
                    if dt == pl.Float64]
    if float64_cols:
        df_numeric = df_numeric.with_columns([
            pl.col(c).cast(pl.Float32) for c in float64_cols
        ])

    full_arr = df_numeric.to_numpy()
    if full_arr.dtype != np.float32:
        full_arr = full_arr.astype(np.float32)

    # Column indices captured in Phase 1 were relative to the original `df`
    # (which included "timestamp"). After dropping timestamp they shift by
    # the timestamp column's position. Remap to the new column order.
    if "timestamp" in all_cols:
        ts_col_idx = all_cols.index("timestamp")
        def _remap(i):
            return i if i < ts_col_idx else i - 1
        temporal_idx = [_remap(i) for i in temporal_idx]
        shared_regime_idx = [_remap(i) for i in shared_regime_idx]
        for tic in per_ticker_market:
            per_ticker_market[tic] = [_remap(i) for i in per_ticker_market[tic]]
        for tic in per_ticker_regime:
            per_ticker_regime[tic] = [_remap(i) for i in per_ticker_regime[tic]]
        close_col_idx = {tic: _remap(i) for tic, i in close_col_idx.items()}

    # Pre-convert the shared/temporal index arrays for fancy indexing.
    shared_regime_idx_arr = np.asarray(shared_regime_idx, dtype=np.int64)
    temporal_idx_arr = np.asarray(temporal_idx, dtype=np.int64)
    n_temporal = len(temporal_idx)
    n_shared_regime = len(shared_regime_idx)

    # Column names corresponding 1:1 to columns of full_arr. We use this
    # below to record each TickerData's column names in the same order
    # as its `data` matrix, by indexing into this list with col_idx.
    full_arr_columns = df_numeric.columns

    # --- Phase 3: per-ticker assembly via numpy slicing --------------------
    result: Dict[str, TickerData] = {}

    for ticker in tickers:
        if ticker not in close_col_idx:
            continue

        market_idx = per_ticker_market.get(ticker, [])
        pt_regime_idx = per_ticker_regime.get(ticker, [])

        # Order: market → temporal → per-ticker regime → shared regime.
        # The environment relies on this ordering to slice observation
        # dicts using (n_market, n_temporal, n_regime), so it must match.
        col_idx = np.fromiter(
            (*market_idx, *temporal_idx_arr.tolist(),
             *pt_regime_idx, *shared_regime_idx_arr.tolist()),
            dtype=np.int64,
            count=len(market_idx) + n_temporal + len(pt_regime_idx) + n_shared_regime,
        )
        data = full_arr[:, col_idx]
        prices = full_arr[:, close_col_idx[ticker]].ravel()

        valid_mask = prices > 0
        if valid_mask.any():
            first_valid_idx = int(np.argmax(valid_mask))
            last_valid_idx = int(len(prices) - 1 - np.argmax(valid_mask[::-1]))
        else:
            first_valid_idx, last_valid_idx = 0, -1

        n_market = len(market_idx)
        n_regime = len(pt_regime_idx) + n_shared_regime

        ticker_columns = [full_arr_columns[i] for i in col_idx.tolist()]

        result[ticker] = TickerData(
            ticker=ticker,
            data=data,
            prices=prices,
            timestamps=timestamps,
            n_market=n_market,
            n_temporal=n_temporal,
            n_regime=n_regime,
            first_valid_idx=first_valid_idx,
            last_valid_idx=last_valid_idx,
            columns=ticker_columns,
        )

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    create_directory(TRAINING_LOGS_DIR)

    # Resolve run number by scanning the v6 results root. Propagates to
    # the logger filename, models_dir, and results_dir so every artifact
    # for this run lives under a single run_N/ bucket.
    v5_results_base = Path(RESULTS_DIR) / "v7"
    run_number = resolve_run_number(v5_results_base)

    # Single logger instance for the entire run — captures setup (ticker
    # discovery, data loading, fold generation) plus all training/
    # validation output. Passed to both generate_walk_forward_folds and
    # DailyTrainer so everything lands in one file.
    logger = Logger(f"{TRAINING_LOGS_DIR}/v7/daily_wf_log_run_{run_number}.txt")
    logger.info(f"Run number: {run_number}")

    # --- Config ---
    episode_days = 252
    episodes_per_fold = 200

    temporal_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']

    logger.info(
        f"Run config: episode_days={episode_days}, "
        f"episodes_per_fold={episodes_per_fold}, seed={SEED}"
    )

    # --- Load data ---
    unified_path = f"{DATA_DIR}/preprocessed/v7/unified/unified.parquet"

    # v7 trades the fixed basket from config — no ticker discovery / filtering.
    from src.config.config import V7_BASKET
    available_tickers = list(V7_BASKET)
    logger.info(f"v7 basket: {available_tickers}")

    logger.info("Loading data from unified parquet...")
    all_tickers = load_tickers_from_unified(unified_path, available_tickers, temporal_cols)
    logger.info(f"Loaded {len(all_tickers)} tickers")

    # --- Walk-forward folds ---
    # Start at 2005 rather than 2004 so fold 1's training window lines up
    # cleanly with the usable data range. The CUTOFF_TIMESTAMP in config
    # is 2004-12-13 (60-day + 252-day rolling z-score warmup), so only
    # ~3 weeks of 2004 would have been available as training data and it
    # sits right on the feature warmup boundary. Starting at 2005-01-01
    # gives fold 1 a full 10 real calendar years of fully-warmed-up data.
    folds = generate_walk_forward_folds(
        data_start_year=2005,
        data_end_year=2025,
        initial_train_years=10,
        valid_years=1,
        test_years=2,
        logger=logger,
    )

    # --- Create agent + buffer ---
    # Extract v7 env inputs (per-ticker market, per-ticker prices,
    # shared temporal, shared regime) from the loaded basket TickerData.
    (
        basket_market_data, basket_prices, shared_temporal, shared_regime,
        n_per_ticker_regime, basket_first_valid, basket_last_valid,
    ) = _build_basket_inputs(all_tickers, available_tickers)
    n_tickers = len(available_tickers)
    n_market_per_ticker = next(iter(basket_market_data.values())).shape[1]
    n_temporal = shared_temporal.shape[1]
    n_shared_regime = shared_regime.shape[1]
    sample_td = all_tickers[available_tickers[0]]
    timestamps = sample_td.timestamps

    # Portfolio state width is determined by n_tickers (4 per-ticker dims + 2 portfolio-level)
    portfolio_state_len = 4 * n_tickers + 2

    logger.info(
        f"v7 data dims: "
        f"n_tickers={n_tickers}, "
        f"n_market_per_ticker={n_market_per_ticker} "
        f"(base {sample_td.n_market} + per-ticker regime {n_per_ticker_regime}), "
        f"n_temporal={n_temporal}, "
        f"n_shared_regime={n_shared_regime}, "
        f"portfolio_state={portfolio_state_len}"
    )

    from src.v7.model.replay_buffer import DailyReplayBuffer

    replay_buffer = DailyReplayBuffer(
        n_tickers=n_tickers,
        n_market_per_ticker=n_market_per_ticker,
        n_temporal=n_temporal,
        n_regime=n_shared_regime,
        portfolio_state_len=portfolio_state_len,
        action_dim=n_tickers,
        capacity=200_000,
        decay=3.0,          # v6 default; planned v7 decay sweep before run-1 MC
    )

    agent = Agent(
        replay_buffer=replay_buffer,
        temporal_state_len=n_temporal,
        regime_state_len=n_shared_regime,
        total_episodes=episodes_per_fold * len(folds),
        action_dim=n_tickers,
        input_shape=(n_tickers, n_market_per_ticker),
        portfolio_state_len=portfolio_state_len,
    )

    # --- Train ---
    trainer = DailyTrainer(
        agent=agent,
        basket_tickers=available_tickers,
        basket_market_data=basket_market_data,
        basket_prices=basket_prices,
        shared_temporal=shared_temporal,
        shared_regime=shared_regime,
        timestamps=timestamps,
        run_number=run_number,
        basket_first_valid_idx=basket_first_valid,
        basket_last_valid_idx=basket_last_valid,
        episode_days=episode_days,
        num_episodes_per_fold=episodes_per_fold,
        valid_interval=10,
        valid_episodes_per_eval=10,
        save_interval=50,
        models_dir=f"{MODELS_DIR}/v7",
        results_dir=f"{RESULTS_DIR}/v7",
        recency_decay=1.5,
        logger=logger,
    )

    _ = trainer.train_walk_forward(folds)
    logger.info("Walk-forward training complete.")


if __name__ == "__main__":
    main()