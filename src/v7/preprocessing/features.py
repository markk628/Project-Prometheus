"""
Per-ticker feature computations: VWAP, candlestick shape, temporal
patterns (sin/cos), and TA-Lib-driven volatility / trend / volume
features. Each `_add_*` function appends columns to the per-ticker
DataFrame and is composed by `worker.py::_process_ticker`.

Note on module-level state
--------------------------
The lists below (temporal_features, volatility_features, etc.) are
populated in-place by the corresponding `_add_*` functions and read
later by `_prefix_columns` (in `normalization.py`) to decide which
columns are *not* to be ticker-prefixed (only `temporal_features` is
read in practice; the others are populated but currently unused).

This is shared mutable module state — preserved as-is for the v6
refactor since it's pure code motion. ProcessPoolExecutor workers each
get their own copy, so concurrency isn't an issue. A future cleanup
could pass these lists explicitly through the call chain instead.
"""

import numpy as np
import polars as pl
import talib as ta


temporal_features = []
volatility_features = []
trend_features = []
momentum_features = []
volume_features = []
candlestick_features = []


def _calculate_vwap(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Calculate daily VWAP as typical price: (high + low + close) / 3.

    For daily bars we don't have intraday volume distribution, so the
    typical price is the standard proxy. This captures where the "fair
    value" of the day was — price_vwap_distance then measures whether
    the stock closed near its highs or lows.
    """
    if log:
        print("Calculating VWAP...")

    return df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap")
    )


def _add_candlestick_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Derive inter-day and intraday structure features from OHLC data.

    These recover signals that minute-level data captures naturally but
    daily bars lose — overnight sentiment, intraday conviction, and
    candle shape. All features use only current-bar OHLC and the
    previous bar's close (via shift(1)), so there is no data leakage.

    Features:
        overnight_gap     — log(open / prev_close), overnight sentiment
        intraday_return   — log(close / open), intraday direction
        upper_wick_ratio  — (high - max(open,close)) / (high-low), rejection above [0,1]
        lower_wick_ratio  — (min(open,close) - low) / (high-low), rejection below [0,1]

    Note: close_location ((close-low)/(high-low)) was removed — it
    correlates >0.96 with price_vwap_distance_cs_zscore for 2/3 of tickers
    and VWAP distance is the cleaner signal (uses volume-weighted price
    as reference instead of noisy H/L extremes).
    """
    if log:
        print("Adding candlestick features...")

    original_features = df.columns

    eps = 1e-8
    prev_close = pl.col("close").shift(1)
    range_ = (pl.col("high") - pl.col("low")).clip(lower_bound=eps)
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")

    df = df.with_columns([
        # Overnight gap: pre-market sentiment, earnings reactions
        (pl.col("open") / (prev_close + eps)).log().fill_null(0.0).alias("overnight_gap"),

        # Intraday return: what happened during market hours
        (pl.col("close") / (pl.col("open") + eps)).log().alias("intraday_return"),

        # Upper wick: rejection at highs (shooting star signal)
        ((pl.col("high") - body_top) / range_).alias("upper_wick_ratio"),

        # Lower wick: rejection at lows (hammer signal)
        ((body_bottom - pl.col("low")) / range_).alias("lower_wick_ratio"),
    ])

    candlestick_features.extend([f for f in df.columns if f not in original_features])
    return df


def _add_temporal_patterns(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Encode cyclical time-based patterns for daily bars.

    Markets exhibit weekly and monthly seasonality (e.g., Monday effects,
    month-end rebalancing, quarterly window-dressing).

    Features:
        day_sin / day_cos          — day of week (Mon=0, Fri=4)
        month_sin / month_cos      — month of year (Jan=1, Dec=12)
        quarter_sin / quarter_cos  — position within the calendar quarter
    """
    if log:
        print("Adding temporal patterns...")
        
    original_features = df.columns

    TAU = 2 * np.pi

    df = df.with_columns([
        # Day of week cyclical (Mon=0 .. Fri=4)
        (TAU * pl.col("timestamp").dt.weekday() / 5).sin().alias("day_sin"),
        (TAU * pl.col("timestamp").dt.weekday() / 5).cos().alias("day_cos"),
        # Month of year cyclical (1-12)
        (TAU * pl.col("timestamp").dt.month() / 12).sin().alias("month_sin"),
        (TAU * pl.col("timestamp").dt.month() / 12).cos().alias("month_cos"),
        # Quarter progress cyclical — position within the calendar quarter
        # Captures end-of-quarter rebalancing / window-dressing effects
        (TAU * ((pl.col("timestamp").dt.ordinal_day() - 1) % 91) / 91).sin().alias("quarter_sin"),
        (TAU * ((pl.col("timestamp").dt.ordinal_day() - 1) % 91) / 91).cos().alias("quarter_cos"),
    ])

    temporal_features.extend([feature for feature in df.columns if feature not in original_features])
    return df


def _talib_series(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract a column as a contiguous float64 array suitable for TA-Lib."""
    return df[col].to_numpy().astype(np.float64)


def _add_volatility_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    if log:
        print("Adding volatility features...")
        
    original_features = df.columns

    close = _talib_series(df, "close")

    log_return_5 = np.log(close / np.roll(close, 5))
    log_return_5[:5] = np.nan
    log_return_20 = np.log(close / np.roll(close, 20))
    log_return_20[:20] = np.nan
    log_return_60 = np.log(close / np.roll(close, 60))
    log_return_60[:60] = np.nan
    # Long-horizon cumulative returns (v6 run 4c).
    log_return_120 = np.log(close / np.roll(close, 120))
    log_return_120[:120] = np.nan
    log_return_252 = np.log(close / np.roll(close, 252))
    log_return_252[:252] = np.nan

    vol_5d = ta.STDDEV(close, timeperiod=5)
    vol_20d = ta.STDDEV(close, timeperiod=20)
    vol_60d = ta.STDDEV(close, timeperiod=60)
    vol_252d = ta.STDDEV(close, timeperiod=252)
    eps = 1e-8

    df = df.with_columns([
        pl.Series("log_return_5",            log_return_5),
        pl.Series("log_return_20",           log_return_20),
        pl.Series("log_return_60",           log_return_60),
        pl.Series("log_return_120",          log_return_120),
        pl.Series("log_return_252",          log_return_252),
        pl.Series("volatility_5_20_ratio",   np.log((vol_5d + eps) / (vol_20d + eps))),
        pl.Series("volatility_20_60_ratio",  np.log((vol_20d + eps) / (vol_60d + eps))),
        pl.Series("volatility_60_252_ratio", np.log((vol_60d + eps) / (vol_252d + eps))),
    ])
    
    volatility_features.extend([feature for feature in df.columns if feature not in original_features])
    return df


def _add_trend_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    if log:
        print("Adding trend features...")
        
    original_features = df.columns

    close = _talib_series(df, "close")
    high  = _talib_series(df, "high")
    low   = _talib_series(df, "low")
    
    ema_5d   = ta.EMA(close, timeperiod=5)
    ema_20d  = ta.EMA(close, timeperiod=20)
    ema_60d  = ta.EMA(close, timeperiod=60)
    # Long-horizon EMAs (v6 run 4c).
    ema_120d = ta.EMA(close, timeperiod=120)
    ema_252d = ta.EMA(close, timeperiod=252)
    adx_5  = ta.ADX(high, low, close, timeperiod=5) / 100.0
    adx_20 = ta.ADX(high, low, close, timeperiod=20) / 100.0
    adx_60 = ta.ADX(high, low, close, timeperiod=60) / 100.0
    eps = 1e-8

    df = df.with_columns([
        pl.Series("ema_close_ratio_5",   np.log((close   + eps) / (ema_5d   + eps))),
        pl.Series("ema_close_ratio_20",  np.log((close   + eps) / (ema_20d  + eps))),
        pl.Series("ema_close_ratio_60",  np.log((close   + eps) / (ema_60d  + eps))),
        pl.Series("ema_close_ratio_120", np.log((close   + eps) / (ema_120d + eps))),
        pl.Series("ema_close_ratio_252", np.log((close   + eps) / (ema_252d + eps))),
        pl.Series("ema_5_20_ratio",      np.log((ema_5d  + eps) / (ema_20d  + eps))),
        pl.Series("ema_20_60_ratio",     np.log((ema_20d + eps) / (ema_60d  + eps))),
        pl.Series("adx_5_20_ratio",      np.log((adx_5   + eps) / (adx_20   + eps))),
        pl.Series("adx_20_60_ratio",     np.log((adx_20  + eps) / (adx_60   + eps))),
        pl.Series("adx_14",              ta.ADX(high, low, close, timeperiod=14) / 100.0),
    ])
    
    trend_features.extend([feature for feature in df.columns if feature not in original_features])
    return df


def _add_volume_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    if log:
        print("Adding volume features...")
        
    original_features = df.columns

    close  = _talib_series(df, "close")
    volume = _talib_series(df, "volume")
    vwap   = _talib_series(df, "vwap")
    transactions = _talib_series(df, "transactions")
    
    ema_5d = ta.EMA(volume, timeperiod=5)
    ema_20d = ta.EMA(volume, timeperiod=20)
    ema_60d = ta.EMA(volume, timeperiod=60)
    eps = 1e-8

    avg_trade_size = volume / np.maximum(transactions, 1)
    range_per_trade = (df["high"].to_numpy() - df["low"].to_numpy()) / np.maximum(transactions, 1)

    df = df.with_columns([
        pl.Series("volume_ema_ratio_20", np.log((volume + eps) / (ema_20d + eps))),
        pl.Series("volume_5_20_ratio",   np.log((ema_5d + eps) / (ema_20d + eps))),
        pl.Series("volume_20_60_ratio",  np.log((ema_20d + eps) / (ema_60d + eps))),
        pl.Series("price_roc",           ta.ROC(close, timeperiod=1)),
        pl.Series("price_vwap_distance", (close - vwap) / (vwap + eps)),
        pl.Series("avg_trade_size_ratio",  np.log((avg_trade_size + eps) / (ta.EMA(avg_trade_size, timeperiod=60) + eps))),
        pl.Series("range_per_trade_ratio", np.log((range_per_trade + eps) / (ta.EMA(range_per_trade, timeperiod=60) + eps))),
        pl.Series("trade_intensity_ratio", np.log((transactions + eps) / (ta.EMA(transactions, timeperiod=60) + eps))),
    ])

    # Rolling correlation between volume and price change (20-day window)
    df = df.with_columns(
        pl.rolling_corr("volume", "price_roc", window_size=20)
        .fill_nan(0.0)
        .fill_null(0.0)
        .clip(-1.0, 1.0)
        .alias("volume_price_corr_20")
    ).drop(["price_roc"])

    volume_features.extend([feature for feature in df.columns if feature not in original_features])
    return df


def _drop_unnecessary_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    if log:
        print("Dropping unnecessary features...")
    drop_cols = [c for c in ["ticker", "volume", "open", "high", "low", "transactions", "vwap"] if c in df.columns]
    return df.drop(drop_cols)