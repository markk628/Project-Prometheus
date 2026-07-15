"""
Out-of-sample backtester for the v7 daily-bar SAC allocation model.

Runs a trained policy deterministically over the held-out test window (the
final TEST_YEARS calendar years, which the walk-forward never trains or
validates on) under TWO protocols, and reports realized performance against
an equal-weight buy-and-hold of the basket:

  - HEADLINE  "annual_reset": consecutive DAYS_PER_EPISODE-length episodes
    mirroring the training protocol exactly. Each episode's final step
    liquidates to cash (the env's standard terminal liquidation), the next
    episode starts from 100% cash on the following bar, and portfolio value
    COMPOUNDS across the boundary. The boundary round-trip's exit and
    re-entry costs are charged — the honest price of the system's real
    protocol. This is "the system as built, deployed as designed."
  - SECONDARY "continuous": one single episode spanning the whole window.
    The policy never trained past DAYS_PER_EPISODE, so episode-length-
    dependent portfolio-state components (hold_time, the compounding
    total-value scalar) drift out-of-distribution in the back half. The
    headline-minus-secondary gap is itself a measurement of that drift.

Both protocols run in a single invocation, per the pre-registered protocol
in dev_log_v7.md ("Backtest" section).

Everything that defines "what the policy would have done" is reused from
training so the backtest is faithful:
  - data + features : load_tickers_from_unified / _build_basket_inputs (the
                      same functions the trainer uses) -> identical observation
                      construction.
  - trading + costs : DailyEnvironment as-is (spread, slippage, SEC/TAF/CAT
                      fees, 2% rebalance deadband, final-step liquidation).
  - test window     : DATA_END_YEAR / TEST_YEARS from config, shared with the
                      trainer's generate_walk_forward_folds.

Metrics (return, Sharpe, Sortino, Calmar, profit factor, max drawdown) are
computed cleanly and UNCLIPPED from the realized daily net-PV series using
ANNUAL_RISK_FREE_RATE and TRADING_DAYS_PER_YEAR. They deliberately do NOT
reuse the env's internal Sharpe/Sortino/Calmar helpers, which clip to +/-3
for reward shaping and are not meant for reporting.

Buy-and-hold benchmark: equal dollar weight (1/N) across the basket, bought
once at the test-window open and held to the end (no rebalancing), frictionless.

NOTE on model/data matching: the unified.parquet loaded here must match the
config the checkpoint was trained on. A baseline (te=3) checkpoint expects the
regime channel WITHOUT the discrete regime label (45 dims); a regime-feature
checkpoint expects it WITH (49 dims). Mismatch -> a clear load error.

Usage:
    python -m src.v7.backtesting.backtester /path/to/models/v7/daily_final_backtest_<timestamp>
  or set MODEL_PATH below and run with no arguments.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; we only save the figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

from src.config.config import (
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    INITIAL_BALANCE,
    ANNUAL_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DAYS_PER_EPISODE,
    DATA_END_YEAR,
    TEST_YEARS,
    V7_BASKET,
    SEED
)
from src.v7.environment.environment import DailyEnvironment
from src.v7.model.agent import Agent
from src.v7.model.replay_buffer import DailyReplayBuffer
from src.utils.utils import create_directory
from src.utils.logger import Logger

# load_tickers_from_unified and _build_basket_inputs are reused from the
# trainer so the backtest's data / observation path is byte-identical to
# training's.
from src.v7.training.trainer import load_tickers_from_unified, _build_basket_inputs


# Set this to the model directory to backtest, or pass it as argv[1].
MODEL_PATH = f"{MODELS_DIR}/v7/run_21/daily_final_backtest_sac_model_20260714_063900"

TEMPORAL_COLS = ["day_sin", "day_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos"]


# --------------------------------------------------------------------------- #
# Test-window resolution
# --------------------------------------------------------------------------- #

def resolve_test_window(
    timestamps: np.ndarray,
    data_end_year: int = 2026, # DATA_END_YEAR, # 2022
    test_years: int = 3, # TEST_YEARS, # 17
    first_valid_idx: int = 0,
    last_valid_idx: int = None,
) -> Tuple[int, int]:
    """
    Inclusive index range [start, end] of the held-out test window — the final
    ``test_years`` complete calendar years through ``data_end_year`` — matching
    the holdout that generate_walk_forward_folds reserves.

    Clamped to the basket's valid (finite-price) range so the window never
    starts before every member has data or runs past the last usable bar.
    """
    years = timestamps.astype("datetime64[Y]").astype(int) + 1970
    test_start_year = data_end_year - test_years + 1

    in_window = np.where((years >= test_start_year) & (years <= data_end_year))[0]
    if in_window.size == 0:
        raise ValueError(
            f"No bars in test window {test_start_year}-{data_end_year}; "
            f"data spans {int(years.min())}-{int(years.max())}."
        )
    start, end = int(in_window[0]), int(in_window[-1])

    if last_valid_idx is None:
        last_valid_idx = len(timestamps) - 1
    start = max(start, int(first_valid_idx))
    end = min(end, int(last_valid_idx), len(timestamps) - 1)
    if end <= start:
        raise ValueError(f"Degenerate test window after valid-range clamp: [{start}, {end}].")
    return start, end


# --------------------------------------------------------------------------- #
# Agent (inference only)
# --------------------------------------------------------------------------- #

def load_agent_for_inference(
    model_path: Path,
    n_tickers: int,
    n_market_per_ticker: int,
    n_temporal: int,
    n_shared_regime: int,
    portfolio_state_len: int,
) -> Agent:
    """
    Build the network with the same dims used in training and load the saved
    weights. A tiny dummy replay buffer satisfies the Agent constructor; it is
    never used. The actor is put in eval() mode (disables dropout, etc.).

    A load failure almost always means the unified.parquet loaded here doesn't
    match the checkpoint's training config (e.g. regime-channel width differs).
    """
    dummy_buffer = DailyReplayBuffer(
        n_tickers=n_tickers,
        n_market_per_ticker=n_market_per_ticker,
        n_temporal=n_temporal,
        n_regime=n_shared_regime,
        portfolio_state_len=portfolio_state_len,
        action_dim=n_tickers,
        capacity=2,
    )
    agent = Agent(
        replay_buffer=dummy_buffer,
        temporal_state_len=n_temporal,
        regime_state_len=n_shared_regime,
        total_episodes=1,
        action_dim=n_tickers,
        input_shape=(n_tickers, n_market_per_ticker),
        portfolio_state_len=portfolio_state_len,
    )
    try:
        agent.load_model(model_path)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load model weights from {model_path}.\n"
            f"This is usually a dim mismatch between the checkpoint and the "
            f"unified.parquet loaded here (regime channel = {n_shared_regime} dims). "
            f"Make sure the parquet matches the model's training config "
            f"(baseline vs regime-feature).\nOriginal error: {e}"
        ) from e
    agent.actor.eval()
    return agent


# --------------------------------------------------------------------------- #
# Backtest run
# --------------------------------------------------------------------------- #

def plan_protocol_segments(
    test_start_idx: int,
    test_end_idx: int,
    segment_days: int,
) -> List[Tuple[int, int]]:
    """
    Split the inclusive bar range [test_start_idx, test_end_idx] into
    consecutive episodes of at most segment_days steps.

    Returns a list of (start_idx, steps). Each episode consumes `steps`
    bars after its start bar, so episode k+1 starts on the exact bar where
    episode k ended (in cash, post-liquidation). The final segment takes
    whatever remainder is left (possibly < segment_days — shorter than the
    trained episode length, which is within-distribution; longer never is).
    Total steps across segments == test_end_idx - test_start_idx, so the
    combined curve has one value per bar in the window.
    """
    segments: List[Tuple[int, int]] = []
    total_steps = test_end_idx - test_start_idx
    start = test_start_idx
    done_steps = 0
    while done_steps < total_steps:
        steps = min(segment_days, total_steps - done_steps)
        segments.append((start, steps))
        start += steps
        done_steps += steps
    return segments


def compound_segment_curves(
    segment_pvs: List[np.ndarray],
    initial_balance: float,
) -> np.ndarray:
    """
    Chain per-episode PV series into one compounded equity curve.

    Each raw segment PV comes from an env run that starts from
    initial_balance internally (one entry per step, no leading prefix).
    The real capital entering segment k is initial_balance scaled by the
    product of all previous segments' end-to-start ratios, so segment k's
    curve is scaled by that running ratio. The returned curve is prefixed
    with initial_balance (value at the window's first bar) — same alignment
    convention as the single-episode path.
    """
    curve = [float(initial_balance)]
    scale = 1.0
    for pv in segment_pvs:
        pv = np.asarray(pv, dtype=float)
        curve.extend((pv * scale).tolist())
        scale *= float(pv[-1]) / float(initial_balance)
    return np.asarray(curve, dtype=float)


def run_model_backtest(
    agent: Agent,
    basket_tickers: List[str],
    basket_market_data: Dict[str, np.ndarray],
    basket_prices: Dict[str, np.ndarray],
    shared_temporal: np.ndarray,
    shared_regime: np.ndarray,
    test_start_idx: int,
    test_end_idx: int,
    protocol: str = "annual_reset",
) -> Tuple[np.ndarray, List[int]]:
    """
    Run the deterministic policy over the inclusive range
    [test_start_idx, test_end_idx] under the given protocol and return
    (pv_curve, boundary_offsets).

    protocol="annual_reset" (HEADLINE): consecutive DAYS_PER_EPISODE
    episodes mirroring training. Each episode's final env step liquidates
    to cash (charging exit costs); the next episode starts from 100% cash
    on that same bar (paying entry costs as it rebuilds) — the boundary
    round-trip is the real cost of the real protocol. PV compounds across
    boundaries via compound_segment_curves.

    protocol="continuous" (SECONDARY): one single episode spanning the
    whole window — no mid-window resets, only the end-of-window
    liquidation. Exercises episode lengths the policy never trained on.

    pv_curve has one value per bar, prefixed with INITIAL_BALANCE at the
    first bar. boundary_offsets are bar offsets (from test_start_idx) where
    an annual reset occurred — empty for continuous — for plot markers.
    """
    if protocol not in ("annual_reset", "continuous"):
        raise ValueError(f"Unknown protocol: {protocol!r}")

    total_steps = test_end_idx - test_start_idx
    segment_days = DAYS_PER_EPISODE if protocol == "annual_reset" else total_steps
    segments = plan_protocol_segments(test_start_idx, test_end_idx, segment_days)

    segment_pvs: List[np.ndarray] = []
    for seg_start, steps in segments:
        env = DailyEnvironment(
            basket_tickers=basket_tickers,
            market_data=basket_market_data,
            prices=basket_prices,
            temporal_data=shared_temporal,
            regime_data=shared_regime,
            episode_days=steps,
        )
        state = env.reset(start_idx=seg_start)
        done = False
        while not done:
            action = agent.select_action(state, validate=True)   # deterministic: tanh(mean)
            state, _reward, done, _info = env.step(action)
        segment_pvs.append(np.asarray(env.portfolio_values_history, dtype=float))

    pv = compound_segment_curves(segment_pvs, float(INITIAL_BALANCE))

    # Reset boundaries (bar offsets from test_start_idx), excluding the
    # window end itself.
    boundary_offsets: List[int] = []
    acc = 0
    for _seg_start, steps in segments[:-1]:
        acc += steps
        boundary_offsets.append(acc)
    return pv, boundary_offsets


def buy_and_hold_curve(
    basket_tickers: List[str],
    basket_prices: Dict[str, np.ndarray],
    test_start_idx: int,
    test_end_idx: int,
) -> np.ndarray:
    """
    Frictionless equal-weight buy-and-hold over [test_start_idx, test_end_idx].
    Equal dollar weight (1/N) at the open, bought once, held to the end with no
    rebalancing (weights allowed to drift). value[0] == INITIAL_BALANCE.
    """
    n = len(basket_tickers)
    alloc = float(INITIAL_BALANCE) / n
    window = slice(test_start_idx, test_end_idx + 1)
    pv = np.zeros(test_end_idx - test_start_idx + 1, dtype=float)
    for t in basket_tickers:
        p = np.asarray(basket_prices[t][window], dtype=float)
        shares = alloc / p[0]
        pv = pv + shares * p
    return pv


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(pv: np.ndarray) -> Dict[str, float]:
    """
    Performance metrics from a daily portfolio-value series (length D+1 for D
    daily returns). Annualized with TRADING_DAYS_PER_YEAR; risk-free from
    ANNUAL_RISK_FREE_RATE.

        total_return  : pv[-1]/pv[0] - 1
        annual_return : CAGR = (pv[-1]/pv[0])**(periods_per_year/D) - 1
        sharpe        : annualized, daily excess over the risk-free
        sortino       : annualized, daily excess over rf, downside deviation
                        measured against rf
        calmar        : annual_return / max_drawdown
        profit_factor : sum(positive daily PnL) / sum(|negative daily PnL|)
        max_drawdown  : peak-to-trough on the PV curve (positive fraction)

    Degenerate cases (no variance / no losses / no drawdown) return nan or inf
    rather than raising.
    """
    pv = np.asarray(pv, dtype=float)
    ppy = float(TRADING_DAYS_PER_YEAR)
    rf_daily = ANNUAL_RISK_FREE_RATE / ppy

    rets = pv[1:] / pv[:-1] - 1.0
    D = rets.size

    total_return = pv[-1] / pv[0] - 1.0 if pv[0] > 0 else float("nan")
    annual_return = (pv[-1] / pv[0]) ** (ppy / D) - 1.0 if (D > 0 and pv[0] > 0) else float("nan")

    excess = rets - rf_daily
    sd = rets.std(ddof=1) if D > 1 else 0.0
    sharpe = (excess.mean() / sd) * np.sqrt(ppy) if sd > 0 else float("nan")

    downside = np.minimum(excess, 0.0)
    dd_dev = np.sqrt(np.mean(downside ** 2)) if D > 0 else 0.0
    sortino = (excess.mean() / dd_dev) * np.sqrt(ppy) if dd_dev > 0 else float("inf")

    running_peak = np.maximum.accumulate(pv)
    drawdowns = (running_peak - pv) / running_peak
    max_dd = float(drawdowns.max()) if drawdowns.size else 0.0
    calmar = (annual_return / max_dd) if max_dd > 0 else float("inf")

    pnl = np.diff(pv)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    profit_factor = (gains / losses) if losses > 0 else float("inf")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def format_metrics_table(columns: List[Tuple[str, Dict[str, float]]],
                         start_date: str, end_date: str) -> str:
    """Build the metrics table as a single string for the run logger.
    `columns` is an ordered list of (label, metrics_dict) — e.g.
    [("model annual-reset", ...), ("model continuous", ...),
    ("buy&hold(eqwt)", ...)]. Returned (rather than printed) so the caller
    can hand it to the run logger in one call — logging it as a single
    record keeps the table cleanly aligned instead of stamping a timestamp
    onto every row."""
    pct = lambda x: ("nan" if np.isnan(x) else f"{x * 100:+.2f}%")
    num = lambda x: ("inf" if np.isinf(x) else ("nan" if np.isnan(x) else f"{x:.3f}"))
    ddp = lambda x: f"{x * 100:.2f}%"

    col_w = max(18, max(len(lbl) for lbl, _ in columns) + 2)
    width = 17 + col_w * len(columns)

    def row(label, key, fmt):
        cells = "".join(f"{fmt(m[key]):>{col_w}}" for _, m in columns)
        return f"  {label:<15}{cells}"

    header_cells = "".join(f"{lbl:>{col_w}}" for lbl, _ in columns)
    lines = [
        "=" * width,
        f"  v7 OUT-OF-SAMPLE BACKTEST   {start_date} -> {end_date}",
        "=" * width,
        f"  {'metric':<15}{header_cells}",
        "-" * width,
        row("total return", "total_return", pct),
        row("annual return", "annual_return", pct),
        row("sharpe", "sharpe", num),
        row("sortino", "sortino", num),
        row("calmar", "calmar", num),
        row("profit factor", "profit_factor", num),
        row("max drawdown", "max_drawdown", ddp),
        "=" * width,
    ]
    return "\n".join(lines)


def plot_equity_curves(dates: np.ndarray,
                       ar_pv: np.ndarray, cont_pv: np.ndarray, bh_pv: np.ndarray,
                       ar_m: Dict[str, float], cont_m: Dict[str, float],
                       bh_m: Dict[str, float],
                       reset_offsets: List[int],
                       out_path: Path) -> None:
    d = dates.astype("datetime64[s]").tolist()   # -> python datetimes for matplotlib

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        d, ar_pv, lw=1.9, color="#1f77b4",
        label=(f"v7 model, annual reset [headline]   (total {ar_m['total_return'] * 100:+.1f}%, "
               f"Sharpe {ar_m['sharpe']:.2f}, MaxDD {ar_m['max_drawdown'] * 100:.1f}%)"),
    )
    ax.plot(
        d, cont_pv, lw=1.6, color="#7fb3d9", ls="-.",
        label=(f"v7 model, continuous [secondary]   (total {cont_m['total_return'] * 100:+.1f}%, "
               f"Sharpe {cont_m['sharpe']:.2f}, MaxDD {cont_m['max_drawdown'] * 100:.1f}%)"),
    )
    ax.plot(
        d, bh_pv, lw=1.9, color="#888888", ls="--",
        label=(f"buy & hold, eq-wt   (total {bh_m['total_return'] * 100:+.1f}%, "
               f"Sharpe {bh_m['sharpe']:.2f}, MaxDD {bh_m['max_drawdown'] * 100:.1f}%)"),
    )
    for off in reset_offsets:
        ax.axvline(d[off], color="#1f77b4", lw=0.9, ls=":", alpha=0.55)
        ax.annotate("annual reset", xy=(d[off], ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color="#1f77b4", alpha=0.8, rotation=90)
    ax.axhline(float(INITIAL_BALANCE), color="black", lw=0.8, alpha=0.35)

    ax.set_title("v7 out-of-sample backtest — annual-reset (headline) vs continuous vs buy & hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    if not model_path:
        raise SystemExit(
            "No model path given. Pass it as an argument:\n"
            "  python -m src.v7.model.backtester /path/to/models/v7/daily_final_<timestamp>\n"
            "or set MODEL_PATH at the top of this file."
        )
    model_path = Path(model_path)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    # One logger for the whole run: mirrors every message to the console (as
    # before) and writes it into a per-model directory (created on first use)
    # alongside the equity-curve PNG, so each backtest's console output and plot
    # live together and are preserved verbatim for the record. Start fresh each
    # run (like the PNG, which is overwritten) rather than appending.
    #
    # Directory name is "{run}_{model}" when the checkpoint lives under a run_N
    # folder (e.g. .../v7/run_8/fold_7_best_sac_model_ckpt -> the dir
    # "run_8_fold_7_best_sac_model_ckpt"), so the same model name from different
    # runs doesn't collide. The nearest run_N to the checkpoint wins; a
    # checkpoint with no run_N ancestor (e.g. a final daily_final_<ts> model)
    # just uses the model name.
    run_tag = next((p for p in reversed(model_path.parts) if re.fullmatch(r"run_\d+", p)), None)
    backtest_dir_name = f"{run_tag}_{model_path.name}" if run_tag else model_path.name
    out_dir = Path(RESULTS_DIR) / "v7" / "backtest" / backtest_dir_name
    log_path = out_dir / f"backtest.txt"
    log_path.unlink(missing_ok=True)
    logger = Logger(log_file=str(log_path))

    unified_path = f"{DATA_DIR}/preprocessed/v7/unified/unified.parquet"
    basket = list(V7_BASKET)

    logger.info(f"Loading unified data + basket {basket} ...")
    all_tickers = load_tickers_from_unified(unified_path, basket, TEMPORAL_COLS)

    (
        basket_market_data, basket_prices, shared_temporal, shared_regime,
        n_per_ticker_regime, basket_first_valid, basket_last_valid,
    ) = _build_basket_inputs(all_tickers, basket)

    n_tickers = len(basket)
    n_market_per_ticker = next(iter(basket_market_data.values())).shape[1]
    n_temporal = shared_temporal.shape[1]
    n_shared_regime = shared_regime.shape[1]
    portfolio_state_len = 4 * n_tickers + 2
    timestamps = all_tickers[basket[0]].timestamps

    test_start_idx, test_end_idx = resolve_test_window(
        timestamps,
        first_valid_idx=basket_first_valid,
        last_valid_idx=basket_last_valid,
    )
    start_date = np.datetime_as_string(timestamps[test_start_idx], unit="D")
    end_date = np.datetime_as_string(timestamps[test_end_idx], unit="D")
    logger.info(f"Test window: {start_date} -> {end_date} "
                f"({test_end_idx - test_start_idx + 1} bars) | model: {model_path.name}")

    agent = load_agent_for_inference(
        model_path, n_tickers, n_market_per_ticker,
        n_temporal, n_shared_regime, portfolio_state_len,
    )

    # Both pre-registered protocols in one invocation (dev_log_v7.md,
    # "Backtest" section): annual_reset is the headline, continuous the
    # named secondary; their gap is itself a measurement of episode-length
    # drift.
    ar_pv, reset_offsets = run_model_backtest(
        agent, basket, basket_market_data, basket_prices,
        shared_temporal, shared_regime, test_start_idx, test_end_idx,
        protocol="annual_reset",
    )
    cont_pv, _ = run_model_backtest(
        agent, basket, basket_market_data, basket_prices,
        shared_temporal, shared_regime, test_start_idx, test_end_idx,
        protocol="continuous",
    )
    bh_pv = buy_and_hold_curve(basket, basket_prices, test_start_idx, test_end_idx)
    dates = timestamps[test_start_idx: test_end_idx + 1]

    ar_m = compute_metrics(ar_pv)
    cont_m = compute_metrics(cont_pv)
    bh_m = compute_metrics(bh_pv)
    logger.info(
        f"Protocols: annual_reset = HEADLINE ({len(reset_offsets)} reset(s) at "
        f"{DAYS_PER_EPISODE}-bar boundaries, mirrors training); "
        f"continuous = secondary (protocol-generalization check)."
    )
    logger.info(f"\n{format_metrics_table([('model annual-reset', ar_m), ('model continuous', cont_m), ('buy&hold(eqwt)', bh_m)], start_date, end_date)}")
    logger.info(
        f"Protocol gap (annual_reset - continuous): "
        f"total return {100 * (ar_m['total_return'] - cont_m['total_return']):+.2f}pt, "
        f"Sharpe {ar_m['sharpe'] - cont_m['sharpe']:+.3f} "
        f"(large gap => episode-length drift is real and continuous deployment "
        f"would degrade this model; ~zero => continuous deployment is safe)."
    )

    create_directory(out_dir)
    plot_path = out_dir / f"equity_curve.png"
    plot_equity_curves(dates, ar_pv, cont_pv, bh_pv, ar_m, cont_m, bh_m,
                       reset_offsets, plot_path)
    logger.info(f"Equity-curve plot saved to: {plot_path}")
    logger.info(f"Backtest log saved to: {log_path}")


if __name__ == "__main__":
    main()