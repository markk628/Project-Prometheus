"""
Shared regime features computed across all tradable tickers from the
~25 regime ETFs loaded for that purpose. Added in v6 run 3a, motivated
by the fold-7 (COVID) fragility seen in v6 runs 1 and 2 — adding macro
regime context that the existing breadth + VIX-term-structure + RS-vs-SPY
features don't capture.

Five families:
    yield_curve       — log(TLT / SHY)         ← rate regime
    credit_spread     — log(HYG / LQD)         ← risk-on vs risk-off
    size_factor       — log(IWM / SPY)         ← small vs large cap
    growth_factor     — log(QQQ / SPY)         ← growth vs broad market
    sector_rotation   — dispersion + spread of XL* sector returns

Convention (matches existing breadth / vix_term_structure):
    - Smoothing via rolling_mean (simple moving average).
    - Two horizons per family: short and long.
    - Z-score normalize the smoothed level features.
    - Single medium delta on the long-horizon level for regime-transition
      signal (the "long base + medium delta" pattern from the v5 run-3
      feature design).

Each compute_* function takes the wide stacked DataFrame and returns
``(stacked, new_column_names)``. They are wired into
``_build_unified`` after the RS-vs-SPY block, before the existing
regime-momentum-deltas block.
"""

from typing import List, Optional, Tuple

import numpy as np
import polars as pl

from src.utils.logger import Logger

from .constants import NORMALIZATION_WINDOW
from .normalization import _rolling_zscore_normalize_vectorized


# 11 GICS sector ETFs used for sector_rotation. Hardcoded here because
# the feature is intrinsically defined over this exact set; changing the
# composition would change what "sector rotation" means.
SECTOR_ETF_TICKERS: List[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLRE", "XLC",
]


def _resolve_close_col(stacked: pl.DataFrame, ticker: str) -> Optional[str]:
    """
    Find the close column for a regime ticker, accounting for lifecycle
    segments (e.g. HYG → HYG.2_close when HYG.1 was too short to save).

    Returns the column name or None if no segment exists. When multiple
    segments are present, picks the one with the most non-null values
    (the segment that covers the largest chunk of training time).

    Handles three cases observed in practice:
      - Unsegmented ticker:  TLT_close          → returns "TLT_close"
      - Single surviving segment: HYG.2_close  → returns "HYG.2_close"
      - Discontinuity skip:  no QQQ* column     → returns None
    """
    direct = f"{ticker}_close"
    if direct in stacked.columns:
        return direct

    seg_prefix = f"{ticker}."
    candidates = [
        c for c in stacked.columns
        if c.startswith(seg_prefix) and c.endswith("_close")
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return max(candidates, key=lambda c: stacked[c].is_not_null().sum())


def _compute_log_ratio_regime_feature(
    stacked: pl.DataFrame,
    numerator_ticker: str,
    denominator_ticker: str,
    feature_prefix: str,
    short_window: int = 5,
    long_window: int = 60,
    delta_window: int = 20,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Compute a 3-feature log-ratio regime triple:

        {prefix}_{short}d                     - short-window smoothed log-ratio
        {prefix}_{long}d                      - long-window smoothed log-ratio (regime baseline)
        {prefix}_{long}d_delta_{delta}        - regime-transition signal

    The smoothed levels are z-score normalized via the standard
    ``_rolling_zscore_normalize_vectorized``. The delta is then taken
    on the (normalized) long-window level — matching how the existing
    breadth and vix_term_structure deltas are computed (delta-of-z-score,
    not z-score-of-delta).

    Resolves ticker names to close columns via ``_resolve_close_col`` so
    lifecycle-segmented regime tickers (HYG.2, QQQ.2, etc.) work
    transparently. Returns ``(stacked, [])`` if either ticker has no
    surviving segment (e.g. EEM/EFA dropped by the discontinuity gate).
    """
    numerator_col = _resolve_close_col(stacked, numerator_ticker)
    denominator_col = _resolve_close_col(stacked, denominator_ticker)
    if numerator_col is None or denominator_col is None:
        return stacked, []

    eps = 1e-8

    short_name = f"{feature_prefix}_{short_window}d"
    long_name = f"{feature_prefix}_{long_window}d"
    delta_name = f"{long_name}_delta_{delta_window}"

    # Step 1: compute raw log-ratio inline, smooth at both windows.
    raw = (pl.col(numerator_col) / (pl.col(denominator_col) + eps)).log()
    stacked = stacked.with_columns([
        raw.rolling_mean(window_size=short_window).alias(short_name),
        raw.rolling_mean(window_size=long_window).alias(long_name),
    ])

    # Step 2: z-score normalize the smoothed levels.
    level_cols = [short_name, long_name]
    arr = stacked.select(level_cols).to_numpy().astype(np.float64)
    scaled = _rolling_zscore_normalize_vectorized(arr, window=NORMALIZATION_WINDOW)
    stacked = stacked.with_columns([
        pl.Series(col, scaled[:, i]) for i, col in enumerate(level_cols)
    ])

    # Step 3: delta on the (now-normalized) long-window level.
    stacked = stacked.with_columns(
        pl.col(long_name).diff(n=delta_window).fill_null(0.0).alias(delta_name)
    )

    return stacked, [short_name, long_name, delta_name]


def compute_yield_curve(
    stacked: pl.DataFrame,
    logger: Optional[Logger] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Yield-curve regime: log(TLT / SHY).

    TLT = 20+ year treasuries, SHY = 1-3 year treasuries. Ratio rises in
    bull-flattening / risk-off (long end rallies harder), falls in
    bear-steepening / inflation regimes. The 60-day delta captures the
    speed of yield-curve shifts that often precede recessions.
    """
    if logger:
        logger.info("  Computing yield curve regime (TLT/SHY)...")
    stacked, new_cols = _compute_log_ratio_regime_feature(
        stacked,
        numerator_ticker="TLT",
        denominator_ticker="SHY",
        feature_prefix="yield_curve",
    )
    if not new_cols and logger:
        logger.warning("    TLT or SHY not available — skipping yield curve")
    return stacked, new_cols


def compute_credit_spread(
    stacked: pl.DataFrame,
    logger: Optional[Logger] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Credit-spread regime: log(HYG / LQD).

    HYG = high-yield corporates, LQD = investment-grade corporates.
    Ratio expands in risk-on regimes (junk outperforms IG), compresses
    in risk-off (junk drops harder). Credit spread blowout is the
    canonical recession leading-indicator — the long-window smooth
    captures the regime baseline, the delta captures spread widening
    or tightening.
    """
    if logger:
        logger.info("  Computing credit spread regime (HYG/LQD)...")
    stacked, new_cols = _compute_log_ratio_regime_feature(
        stacked,
        numerator_ticker="HYG",
        denominator_ticker="LQD",
        feature_prefix="credit_spread",
    )
    if not new_cols and logger:
        logger.warning("    HYG or LQD not available — skipping credit spread")
    return stacked, new_cols


def compute_size_factor(
    stacked: pl.DataFrame,
    logger: Optional[Logger] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Size-factor regime: log(IWM / SPY).

    IWM = Russell 2000, SPY = S&P 500. Ratio rises when small caps
    outperform (typically risk-on / early cycle), falls when large caps
    lead (typically late cycle / risk-off). The Fama-French SMB factor
    in ratio form.
    """
    if logger:
        logger.info("  Computing size factor regime (IWM/SPY)...")
    stacked, new_cols = _compute_log_ratio_regime_feature(
        stacked,
        numerator_ticker="IWM",
        denominator_ticker="SPY",
        feature_prefix="size_factor",
    )
    if not new_cols and logger:
        logger.warning("    IWM or SPY not available — skipping size factor")
    return stacked, new_cols


def compute_growth_factor(
    stacked: pl.DataFrame,
    logger: Optional[Logger] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Growth-factor regime: log(QQQ / SPY).

    QQQ = Nasdaq-100 (growth-tilted), SPY = S&P 500. Ratio rises when
    growth/tech leads (low-rate regime, risk appetite), falls when
    value rotates back in or rates rise. Captures the growth-vs-broad
    leadership rotation that drives much of the cross-sectional return
    dispersion in modern markets.
    """
    if logger:
        logger.info("  Computing growth factor regime (QQQ/SPY)...")
    stacked, new_cols = _compute_log_ratio_regime_feature(
        stacked,
        numerator_ticker="QQQ",
        denominator_ticker="SPY",
        feature_prefix="growth_factor",
    )
    if not new_cols and logger:
        logger.warning("    QQQ or SPY not available — skipping growth factor")
    return stacked, new_cols


def compute_sector_rotation(
    stacked: pl.DataFrame,
    short_window: int = 60,
    long_window: int = 200,
    delta_window: int = 20,
    logger: Optional[Logger] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Sector-rotation regime: dispersion and spread of cumulative returns
    across the 11 GICS sector ETFs.

    Two measures, each at two horizons:
        sector_dispersion_{w}d  - cross-sectional std of XL* w-day cumulative
                                   log-returns. High = sectors moving apart
                                   (rotation regime). Low = sectors moving
                                   together (broad market regime).
        sector_topbottom_{w}d   - max minus min of XL* w-day cumulative
                                   log-returns. Captures the magnitude of
                                   the leadership spread regardless of the
                                   distribution shape.

    Plus medium delta on the long-horizon versions:
        sector_dispersion_{long}d_delta_{delta}
        sector_topbottom_{long}d_delta_{delta}

    Cumulative returns are computed directly as ``log(close_t / close_{t-w})``
    rather than via rolling_sum of short returns — gives clean N-day
    cumulative semantics needed for std/range to be interpretable.
    """
    eps = 1e-8

    # Resolve each sector ETF's close column. Lifecycle-segmented sector
    # ETFs would surface here as e.g. XLC.2_close — handled transparently
    # by _resolve_close_col. ETFs with no surviving segment (discontinuity
    # skip) get None and are dropped from the rotation calc.
    available_etfs: List[Tuple[str, str]] = []
    for t in SECTOR_ETF_TICKERS:
        col = _resolve_close_col(stacked, t)
        if col is not None:
            available_etfs.append((t, col))
    if len(available_etfs) < 2:
        if logger:
            logger.warning(
                f"    Sector rotation needs ≥2 sector ETFs, found "
                f"{len(available_etfs)} — skipping"
            )
        return stacked, []

    if logger:
        logger.info(
            f"  Computing sector rotation across {len(available_etfs)} "
            f"ETFs ({short_window}d / {long_window}d horizons)..."
        )

    # Step 1: per-ETF cumulative log-returns at both horizons. Stored as
    # temp columns, dropped after the dispersion/topbottom aggregations.
    short_ret_cols: List[str] = []
    long_ret_cols: List[str] = []
    cum_ret_exprs = []
    for t, cc in available_etfs:
        short_tmp = f"_sector_{t}_ret_{short_window}"
        long_tmp = f"_sector_{t}_ret_{long_window}"
        cum_ret_exprs.append(
            (pl.col(cc) / (pl.col(cc).shift(short_window) + eps)).log().alias(short_tmp)
        )
        cum_ret_exprs.append(
            (pl.col(cc) / (pl.col(cc).shift(long_window) + eps)).log().alias(long_tmp)
        )
        short_ret_cols.append(short_tmp)
        long_ret_cols.append(long_tmp)
    stacked = stacked.with_columns(cum_ret_exprs)

    # Step 2: cross-sectional aggregations — std and (max - min) per row.
    short_disp = f"sector_dispersion_{short_window}d"
    long_disp = f"sector_dispersion_{long_window}d"
    short_tb = f"sector_topbottom_{short_window}d"
    long_tb = f"sector_topbottom_{long_window}d"

    short_list = pl.concat_list([pl.col(c) for c in short_ret_cols])
    long_list = pl.concat_list([pl.col(c) for c in long_ret_cols])

    stacked = stacked.with_columns([
        short_list.list.std().alias(short_disp),
        long_list.list.std().alias(long_disp),
        (short_list.list.max() - short_list.list.min()).alias(short_tb),
        (long_list.list.max() - long_list.list.min()).alias(long_tb),
    ]).drop(short_ret_cols + long_ret_cols)

    # Step 3: z-score normalize the four level features.
    level_cols = [short_disp, long_disp, short_tb, long_tb]
    arr = stacked.select(level_cols).to_numpy().astype(np.float64)
    scaled = _rolling_zscore_normalize_vectorized(arr, window=NORMALIZATION_WINDOW)
    stacked = stacked.with_columns([
        pl.Series(col, scaled[:, i]) for i, col in enumerate(level_cols)
    ])

    # Step 4: medium delta on the long-window levels.
    delta_exprs = []
    delta_cols: List[str] = []
    for base in [long_disp, long_tb]:
        delta_name = f"{base}_delta_{delta_window}"
        delta_exprs.append(
            pl.col(base).diff(n=delta_window).fill_null(0.0).alias(delta_name)
        )
        delta_cols.append(delta_name)
    stacked = stacked.with_columns(delta_exprs)

    return stacked, level_cols + delta_cols