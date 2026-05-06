"""
Normalization helpers used inside the per-ticker worker pipeline:

- `_rolling_zscore_normalize_vectorized`: causal rolling z-score on a
  (T, F) array using Polars rolling functions.
- `_normalize_data`: applies the rolling z-score to a fixed list of
  per-ticker features.
- `_prefix_columns`: renames feature columns with a ticker prefix,
  leaving the timestamp and the shared temporal columns alone.

Note on imports: `_prefix_columns` uses `temporal_features` from
`features.py`. That list is mutated in-place by `_add_temporal_patterns`
during the worker pipeline, so we use `from . import features` (rather
than `from .features import temporal_features`) and reference
`features.temporal_features` to always read the current value.
"""

import numpy as np
import polars as pl

from . import features
from .constants import NORMALIZATION_WINDOW


def _rolling_zscore_normalize_vectorized(
    full_arr: np.ndarray,
    window: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Causal rolling z-score using Polars' native rolling functions.

    At each bar t the lookback is [t-window, t-1] (strictly past data), so
    valid and test rows are normalized using only historical data that precedes
    them — no look-ahead bias.

    Replaces the previous stride-tricks implementation which materialized a
    (T, window, F) array in RAM, causing OOM errors on large datasets.  Polars computes each rolling
    statistic in a single O(T) pass using O(window) working memory per column,
    keeping peak RAM at O(T * F) — the size of the data itself.

    Steps per bar:
        1. Shift by 1 so each row's window covers [t-window, t-1] (past only).
        2. Rolling quantiles (1st / 99th) for winsorization bounds.
        3. Rolling mean and std for z-scoring.
        4. Clip current value to [lo, hi], z-score, apply tanh.
        5. Zero out the first `window` warmup bars (no full history yet).

    Parameters
    ----------
    full_arr : np.ndarray, shape (T, F)
        All bars in time order (train + valid + test concatenated).
    window : int
        Lookback length in bars.
    eps : float
        Std floor to prevent division by zero.

    Returns
    -------
    np.ndarray, shape (T, F), float64
    """
    T, F = full_arr.shape

    # Build a Polars DataFrame — one column per feature.
    col_names = [f"f{i}" for i in range(F)]
    df = pl.DataFrame(
        {col_names[i]: full_arr[:, i].astype(np.float64) for i in range(F)}
    )

    # shift(1) makes each row's value the *previous* bar, so the rolling
    # window [t-window, t-1] is achieved with min_samples=window.
    shifted = df.select([
        pl.col(c).shift(1).alias(c) for c in col_names
    ])

    mu  = shifted.select([pl.col(c).rolling_mean(window_size=window, min_samples=window).alias(c) for c in col_names])
    sig = shifted.select([pl.col(c).rolling_std( window_size=window, min_samples=window).alias(c) for c in col_names])
    lo  = shifted.select([pl.col(c).rolling_quantile(0.01, window_size=window, min_samples=window).alias(c) for c in col_names])
    hi  = shifted.select([pl.col(c).rolling_quantile(0.99, window_size=window, min_samples=window).alias(c) for c in col_names])

    mu_arr  = mu.to_numpy()
    sig_arr = np.where(sig.to_numpy() < eps, eps, sig.to_numpy())
    lo_arr  = lo.to_numpy()
    hi_arr  = hi.to_numpy()

    x_arr   = np.clip(full_arr.astype(np.float64), lo_arr, hi_arr)
    z_arr   = np.tanh((x_arr - mu_arr) / sig_arr)

    # Warmup rows have NaN stats — set them to 0.0.
    out = np.where(np.isnan(mu_arr), 0.0, z_arr)

    return out


def _normalize_data(
    df: pl.DataFrame,
    log: bool = False,
) -> pl.DataFrame:
    """
    Causal rolling z-score normalization.

    At each bar t, statistics (mean, std, winsorization bounds) are computed
    exclusively from bars [t-WINDOW, t-1] — strictly past data.
    No global fit on the training set; no look-ahead bias.

    Features intentionally excluded here (CS-normalized later):
        log_return_5, log_return_20, log_return_60,
        overnight_gap, intraday_return,
        price_vwap_distance, volume_price_corr_20, adx_14
    """
    # NOTE: log_return_5, log_return_20, log_return_60,
    #       overnight_gap, intraday_return,
    #       price_vwap_distance, volume_price_corr_20, and adx_14 are
    #       intentionally excluded here. They are normalised cross-sectionally
    #       inside DataFeatureEngineer._cross_sectional_normalization(), which has
    #       access to all tickers simultaneously and can therefore compute
    #       meaningful market-relative statistics.
    features_to_scale = [
        # Trend
        "ema_close_ratio_5", "ema_close_ratio_20", "ema_close_ratio_60",
        "ema_5_20_ratio", "ema_20_60_ratio", "adx_5_20_ratio", "adx_20_60_ratio",
        # Volatility
        "volatility_5_20_ratio", "volatility_20_60_ratio",
        # Volume
        "volume_ema_ratio_20", "volume_5_20_ratio", "volume_20_60_ratio",
        "avg_trade_size_ratio", "range_per_trade_ratio", "trade_intensity_ratio",
        # Candlestick (bounded [0,1] but rolling z-score captures "unusual vs recent history")
        "upper_wick_ratio", "lower_wick_ratio",
    ]

    # Rolling lookback: ~1 year of trading days (252 bars).
    # Long enough to capture regime context, short enough to adapt
    # to vol-regime changes over years.

    if log:
        print(f"Rolling z-score normalizing {len(features_to_scale)} features "
              f"(window={NORMALIZATION_WINDOW} bars, causal)...")

    full_arr = df.select(features_to_scale).to_numpy().astype(np.float64)
    scaled = _rolling_zscore_normalize_vectorized(full_arr, window=NORMALIZATION_WINDOW)

    return df.with_columns([
        pl.Series(col, scaled[:, i])
        for i, col in enumerate(features_to_scale)
    ])


def _prefix_columns(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """Rename all feature columns with a ticker prefix, leaving timestamp intact."""
    return df.rename({
        col: f"{ticker}_{col}"
        for col in df.columns
        if col != "timestamp" and col not in features.temporal_features
    })
