import numpy as np
import polars as pl
import pandas_market_calendars as mcal
import talib as ta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import DATA_DIR, CUTOFF_TIMESTAMP, TICKERS, TICKER_ALIASES
from src.utils.database_manager import DatabaseManager
from src.utils.logger import Logger
from src.utils.utils import create_directory, save_to_parquet


# ---------------------------------------------------------------------------
# Module-level helpers (must be top-level so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------

temporal_features = []
volatility_features = []
trend_features = []
momentum_features = []
volume_features = []

NORMALIZATION_WINDOW = 3900


def _build_nyse_valid_minutes(start_date, end_date) -> pl.DataFrame:
    """
    Return a DataFrame of every valid NYSE trading minute in the date range.

    Timestamps are kept in UTC and cast to microsecond precision so they
    align exactly with the raw data (which is also normalised to UTC/us).
    """
    import pandas as pd

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    rows: list[dict] = []
    for _, row in schedule.iterrows():
        open_ts  = row["market_open"]   # already UTC from mcal
        close_ts = row["market_close"]  # already UTC from mcal
        idx = pd.date_range(
            start=open_ts,
            end=close_ts - pd.Timedelta(minutes=1),
            freq="1min",
            tz="UTC",
        )
        for ts in idx:
            rows.append({"timestamp": ts.to_pydatetime()})

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us")
        )
    )


def _apply_split_adjustments(
    df: pl.DataFrame,
    ticker: str,
    data_dir: Path,
    log: bool = False,
) -> pl.DataFrame:
    """
    Adjust raw OHLCV bars for historical stock splits.

    For each bar, the API's ``historical_adjustment_factor`` gives the
    cumulative multiplier that converts an unadjusted price to its
    split-adjusted equivalent.  The correct factor for a bar on date D
    is the one belonging to the *first* split whose ``execution_date``
    is strictly after D — i.e. we look forward to find which splits
    have not yet been "absorbed" into the raw price at that point in
    time.

    Because the splits parquet is sorted ascending by ``execution_date``
    (guaranteed by ``stock_splits.py``), a ``join_asof`` with
    ``strategy="forward"`` efficiently finds that first future split for
    every bar in a single vectorised pass.

    Price columns (open, high, low, close, vwap) are multiplied by the
    factor.  Volume is divided by the same factor so that the total
    dollar value of each bar (price × volume) is preserved and relative
    volume signals remain comparable across the split boundary.

    If no splits file exists for the ticker (e.g. the stock has never
    split), the DataFrame is returned unchanged.

    Parameters
    ----------
    df : pl.DataFrame
        Raw minute bars for a single ticker.  Must contain the columns
        ``timestamp``, ``open``, ``high``, ``low``, ``close``,
        ``volume``, and optionally ``vwap``.
    ticker : str
        Ticker symbol — used to locate ``<data_dir>/raw/splits/<ticker>_splits.parquet``.
    data_dir : Path
        Root data directory (same value passed to ``_process_ticker``).
    log : bool
        If True, print progress messages.

    Returns
    -------
    pl.DataFrame
        DataFrame with split-adjusted price and volume columns.
        All other columns are passed through unchanged.
    """
    splits_path = data_dir / "raw" / "splits" / f"{ticker}_splits.parquet"

    if not splits_path.exists():
        if log:
            print(f"No splits file found for {ticker}, skipping adjustment.")
        return df

    splits = pl.read_parquet(splits_path)

    if splits.is_empty():
        if log:
            print(f"No splits found for {ticker}, skipping adjustment.")
        return df

    if log:
        print(f"Applying {len(splits)} split adjustment(s) for {ticker}...")

    # join_asof requires a Date key on the left; extract one from timestamp.
    # We keep the original timestamp column intact throughout.
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("_bar_date")
    )

    # join_asof(strategy="forward") attaches, for each bar, the
    # historical_adjustment_factor of the next split *after* that bar's date.
    # Bars that fall on or after the last split get null (no future split
    # exists) — those bars are already at the current, fully-adjusted price
    # so a factor of 1.0 is the correct fill.
    df = df.join_asof(
        splits.select(["execution_date", "historical_adjustment_factor"]),
        left_on="_bar_date",
        right_on="execution_date",
        strategy="forward",
    ).with_columns(
        pl.col("historical_adjustment_factor").fill_null(1.0)
    )

    price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in df.columns]

    df = df.with_columns(
        [pl.col(c) * pl.col("historical_adjustment_factor") for c in price_cols] +
        # Volume moves inversely: a 4-for-1 split quadruples share count going
        # forward, so pre-split bars must also be multiplied — but the factor
        # already encodes the *price* direction (< 1 for forward splits on
        # historical bars).  Dividing volume by the same factor is equivalent
        # to multiplying by the inverse ratio, restoring the correct pre-split
        # share count in adjusted terms.
        [pl.col("volume") / pl.col("historical_adjustment_factor")]
    ).drop(["execution_date", "_bar_date", "historical_adjustment_factor"])
    return df


def _process_ticker(
    ticker: str,
    ticker_df: pl.DataFrame,
    data_dir: Path,
    log: bool,
) -> str:
    """
    Full feature-engineering pipeline for one ticker.
    Runs in a worker process — no shared state.

    Normalization happens here on the full timeline (no trimming, no splitting)
    so that the rolling warmup rows that come out as zeros land in the
    pre-cutoff warmup period, not in the actual training data.

    _drop_rows_before_timestamp and _split_data are deferred all the way to
    the end of _build_unified_splits, after every normalization pass
    (per-ticker rolling z-score, cross-sectional z-score, breadth, RS)
    has had the full history available.

    Returns the ticker name on success or an error string on failure.
    """
    try:
        df = _apply_split_adjustments(ticker_df, ticker, data_dir, log)
        df = _handle_gaps(df, log)
        df = _add_temporal_patterns(df, log)
        df = _add_volatility_features(df, log)
        df = _add_trend_features(df, log)
        df = _add_volume_features(df, log)
        df = _drop_unnecessary_features(df, log)
        df = _normalize_data(df, log)
        df = _prefix_columns(df, ticker)

        base_dir = data_dir / "preprocessed" / "v4" / "tickers"
        create_directory(base_dir)
        save_to_parquet(df, f"{base_dir}/{ticker}.parquet", index=False)

        return ticker

    except Exception as e:
        return f"ERROR:{ticker}:{e}"


# ---------------------------------------------------------------------------
# Stateless pipeline steps (pure functions — easy to test, no pandas)
# ---------------------------------------------------------------------------

def _handle_gaps(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Enforce a continuous 1-minute NYSE index per ticker.

    - Forward-fills price columns to maintain price continuity.
    - Zeros out volume / transaction counts for missing minutes.
    - Discards sessions where the first bar has no price data.

    Data stays in UTC throughout — no timezone conversion needed.
    The only normalisation is casting nanoseconds → microseconds so
    both sides of the join share the same datetime precision.
    """
    if log:
        print("Handling gaps...")

    # Normalise raw timestamps to UTC microseconds (Polars default precision).
    # The minute grid is built in the same unit, so the join key matches exactly.
    df = df.with_columns(
        pl.col("timestamp").dt.cast_time_unit("us")
    )

    start_date = df["timestamp"].min().date()
    end_date   = df["timestamp"].max().date()

    valid_minutes = _build_nyse_valid_minutes(start_date, end_date)

    # Left-join full minute grid onto raw data
    df = (
        valid_minutes
        .join(df, on="timestamp", how="left")
        .sort("timestamp")
    )

    price_cols  = ["open", "high", "low", "close"]
    volume_cols = ["volume", "transactions"]

    # Drop sessions whose first bar has no price (no data at all that day)
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("_date")
    )
    first_close = (
        df.group_by("_date")
        .agg(pl.col("close").drop_nulls().first().alias("_first_close"))
    )
    df = (
        df.join(first_close, on="_date", how="left")
        .filter(pl.col("_first_close").is_not_null())
        .drop(["_first_close", "_date"])
    )

    # Forward-fill prices within each calendar day, zero-fill volumes
    df = df.sort("timestamp").with_columns(
        [pl.col(c).forward_fill() for c in price_cols] +
        [pl.col(c).fill_null(0)   for c in volume_cols]
    )

    # -----------------------------------------------------------------
    # Compute session VWAP — typical price * volume, cumulated intraday.
    #   typical_price  = (high + low + close) / 3
    #   vwap[t]        = Σ(tp * vol)[0..t] / Σvol[0..t]   (resets each day)
    #
    # For zero-volume bars (filled gaps) the bar contributes 0 to both
    # numerator and denominator, so the session VWAP is naturally
    # unaffected — no special-casing needed.
    # -----------------------------------------------------------------
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("_date"),
        (((pl.col("high") + pl.col("low") + pl.col("close")) / 3) * pl.col("volume"))
        .alias("_tp_vol"),
    ).with_columns([
        pl.col("_tp_vol").cum_sum().over("_date").alias("_cum_tp_vol"),
        pl.col("volume").cum_sum().over("_date").alias("_cum_vol"),
    ]).with_columns(
        # Guard against the (rare) edge case where cumulative volume is 0
        # (e.g., entire session has no trades). Fall back to mid-price.
        pl.when(pl.col("_cum_vol") > 0)
        .then(pl.col("_cum_tp_vol") / pl.col("_cum_vol"))
        .otherwise((pl.col("high") + pl.col("low") + pl.col("close")) / 3)
        .alias("vwap")
    ).drop(["_date", "_tp_vol", "_cum_tp_vol", "_cum_vol"])

    return df


def _add_temporal_patterns(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Encode cyclical time-based market patterns using sine / cosine transforms.

    Intraday markets exhibit strong periodic behaviour driven by
    human activity, institutional schedules, and market microstructure
    (e.g., open/close volatility, lunch-hour slowdowns).

    Precalculus FTW bois 😤📐
    """
    if log:
        print("Adding temporal patterns...")
        
    original_features = df.columns

    TAU = 2 * np.pi

    df = df.with_columns([
        # --- cyclical encodings ---
        (TAU * pl.col("timestamp").dt.minute() / 60).sin().alias("minute_sin"),
        (TAU * pl.col("timestamp").dt.minute() / 60).cos().alias("minute_cos"),
        (TAU * pl.col("timestamp").dt.hour()   / 24).sin().alias("hour_sin"),
        (TAU * pl.col("timestamp").dt.hour()   / 24).cos().alias("hour_cos"),
        (TAU * pl.col("timestamp").dt.weekday() / 5).sin().alias("day_sin"),
        (TAU * pl.col("timestamp").dt.weekday() / 5).cos().alias("day_cos"),
        # helper column for intraday position
        pl.col("timestamp").dt.date().alias("_date"),
    ])

    # minutes_since_open / minutes_to_close normalised by the actual session
    # length of each trading day.
    #
    # Using _day_len (the real bar count per day) rather than a hardcoded 390
    # correctly handles NYSE half-days (e.g. day-before-Thanksgiving = 210 bars)
    # without any look-ahead: the exchange calendar is published in advance so
    # half-days are known before trading begins.
    #
    # Result: both features are in [0, 1] on full days and half-days alike.
    # No rolling warmup is needed — this is a deterministic rescale.
    df = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).over("_date").alias("_day_idx"),
        pl.len().over("_date").alias("_day_len"),
    ).with_columns([
        # Normalise to [0, 1]: first bar = 0.0, last bar = 1.0
        (pl.col("_day_idx") / (pl.col("_day_len") - 1).clip(lower_bound=1))
        .alias("minutes_since_open"),
        # Complement: first bar = 1.0, last bar = 0.0
        (1.0 - pl.col("_day_idx") / (pl.col("_day_len") - 1).clip(lower_bound=1))
        .alias("minutes_to_close"),
    ]).drop(["_date", "_day_idx", "_day_len"])

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
    log_return_15 = np.log(close / np.roll(close, 15))
    log_return_15[:15] = np.nan
    log_return_60 = np.log(close / np.roll(close, 60))
    log_return_60[:60] = np.nan

    vol_5m = ta.STDDEV(close, timeperiod=5)
    vol_15m = ta.STDDEV(close, timeperiod=15)
    vol_60m = ta.STDDEV(close, timeperiod=60)
    eps = 1e-8
    
    df = df.with_columns([
        pl.Series("log_return_5",           log_return_5),
        pl.Series("log_return_15",          log_return_15),
        pl.Series("log_return_60",          log_return_60),
        pl.Series("volatility_5_15_ratio",  np.log((vol_5m + eps) / (vol_15m + eps))),
        pl.Series("volatility_15_60_ratio", np.log((vol_15m + eps) / (vol_60m + eps)))
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
    
    ema_5m = ta.EMA(close, timeperiod=5)
    ema_15m = ta.EMA(close, timeperiod=15)
    ema_60m = ta.EMA(close, timeperiod=60)
    adx_5 = ta.ADX(high, low, close, timeperiod=5) / 100.0
    adx_15 = ta.ADX(high, low, close, timeperiod=15) / 100.0
    adx_60 = ta.ADX(high, low, close, timeperiod=60) / 100.0
    eps = 1e-8

    df =  df.with_columns([
        pl.Series("ema_close_ratio_5",  np.log((close + eps) / (ema_5m + eps))),
        pl.Series("ema_close_ratio_15", np.log((close + eps) / (ema_15m + eps))),
        pl.Series("ema_close_ratio_60", np.log((close + eps) / (ema_60m + eps))),
        pl.Series("ema_5_15_ratio",     np.log((ema_5m + eps) / (ema_15m + eps))),
        pl.Series("ema_15_60_ratio",    np.log((ema_15m + eps) / (ema_60m + eps))),
        pl.Series("adx_5_15_ratio",     np.log((adx_5 + eps) / (adx_15 + eps))),
        pl.Series("adx_15_60_ratio",    np.log((adx_15 + eps) / (adx_60 + eps))),
        pl.Series("adx_20",             ta.ADX(high, low, close, timeperiod=20) / 100.0),
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
    
    ema_5m = ta.EMA(volume, timeperiod=5)
    ema_15m = ta.EMA(volume, timeperiod=15)
    ema_60m = ta.EMA(volume, timeperiod=60)
    eps = 1e-8

    # Add the numpy arrays first
    df = df.with_columns([
        pl.Series("volume_ema_ratio_15", np.log((volume + eps) / (ema_15m + eps))),
        pl.Series("volume_5_15_ratio",   np.log((ema_5m + eps) / (ema_15m + eps))),
        pl.Series("volume_15_60_ratio",  np.log((ema_15m + eps) / (ema_60m + eps))),
        pl.Series("price_roc",           ta.ROC(close, timeperiod=1)),
        pl.Series("price_vwap_distance", (close - vwap) / vwap),
    ])

    # Let Polars handle the rolling correlation natively and cleanly
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("_date")
    ).with_columns(
        pl.rolling_corr("volume", "price_roc", window_size=15)
        .over("_date")
        .fill_nan(0.0)
        .fill_null(0.0)
        .clip(-1.0, 1.0)  # guard against ±inf on zero-variance windows
        .alias("volume_price_corr_15")
    ).drop(["price_roc", "_date"])
    
    transactions = _talib_series(df, "transactions")
    avg_trade_size = volume / np.maximum(transactions, 1)
    range_per_trade = (df["high"].to_numpy() - df["low"].to_numpy()) / np.maximum(transactions, 1)
    avg_trade_size_ratio = np.log((avg_trade_size + eps) / (ta.EMA(avg_trade_size, timeperiod=60) + eps))
    range_per_trade_ratio = np.log((range_per_trade + eps) / (ta.EMA(range_per_trade, timeperiod=60) + eps))
    trade_intensity_ratio = np.log((transactions + eps) / (ta.EMA(transactions, timeperiod=60) + eps))
    
    df = df.with_columns([
        pl.Series("avg_trade_size_ratio",  avg_trade_size_ratio),
        pl.Series("range_per_trade_ratio", range_per_trade_ratio),
        pl.Series("trade_intensity_ratio", trade_intensity_ratio)
    ])

    volume_features.extend([feature for feature in df.columns if feature not in original_features])
    return df


def _drop_unnecessary_features(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    if log:
        print("Dropping unnecessary features...")
    return df.drop(["ticker", "volume", "open", "high", "low", "transactions", "vwap"])


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
    (T, window, F) array in RAM, causing OOM errors on large tickers
    (e.g. shape (608190, 1950, 15) = ~16 GB).  Polars computes each rolling
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
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Causal rolling z-score normalization.

    All three splits are concatenated in time order so that rolling windows
    at the train→valid and valid→test boundaries have real history behind
    them (no warmup rows are lost from valid or test).

    At each bar t, statistics (mean, std, winsorization bounds) are computed
    exclusively from bars [t-WINDOW, t-1] — strictly past data.  This means:
      • Train bars  are normalized with their own rolling past.
      • Valid bars  see only training history + earlier valid bars.
      • Test  bars  see training + valid history + earlier test bars.

    No global fit on the training set; no look-ahead bias.

    Features intentionally excluded here (CS-normalized later):
        log_return_5, log_return_15, log_return_60,
        price_vwap_distance, volume_price_corr_15, adx_20
    """
    # NOTE: log_return_5, log_return_15, log_return_60,
    #       price_vwap_distance, volume_price_corr_15, and adx_20 are
    #       intentionally excluded here. They are normalised cross-sectionally
    #       inside DataFeatureEngineer._cross_sectional_normalization(), which has
    #       access to all tickers simultaneously and can therefore compute
    #       meaningful market-relative statistics.
    features_to_scale = [
        # Temporal features (minutes_since_open / minutes_to_close) are excluded:
        # they are already normalised to [0, 1] by _add_temporal_patterns using
        # the per-day session length, so no further scaling is needed here.
        # Trend
        "ema_close_ratio_5", "ema_close_ratio_15", "ema_close_ratio_60",
        "ema_5_15_ratio", "ema_15_60_ratio", "adx_5_15_ratio", "adx_15_60_ratio",
        # Volatility
        "volatility_5_15_ratio", "volatility_15_60_ratio",
        # Volume
        "volume_ema_ratio_15", "volume_5_15_ratio", "volume_15_60_ratio",
        "avg_trade_size_ratio", "range_per_trade_ratio", "trade_intensity_ratio",
    ]

    # Rolling lookback: ~1 trading week of 1-min bars (390 * 5 bars).
    # Long enough to capture intraday regime context, short enough to adapt
    # to vol-regime changes over weeks.

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
        if col != "timestamp" and col not in temporal_features
    })

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataFeatureEngineer:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        timestamp: str = CUTOFF_TIMESTAMP,
        tickers: List[str] = TICKERS,
        ticker_aliases: Dict[str, Any] = TICKER_ALIASES,
        database_manager: DatabaseManager = None,
        logger: Optional[Logger] = None,
        max_workers: Optional[int] = None,   # None → os.cpu_count()
    ):
        self.data_dir         = data_dir
        self.tickers_dir      = Path(self.data_dir) / "preprocessed" / "v4" / "tickers"
        self.files            = (
            list(Path(f"{self.data_dir}/raw/minute_data").glob("2019/12/**/*.csv.gz")) +
            list(Path(f"{self.data_dir}/raw/minute_data").glob("202[0-9]/**/*.csv.gz"))
        )
        self.timestamp        = timestamp
        self.tickers          = tickers
        self.ticker_aliases   = ticker_aliases
        self.database_manager = database_manager
        self.logger           = logger
        self.max_workers      = max_workers

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def _check_for_missing_tickers(self) -> List[str]:
        if self.logger:
            self.logger.info('Checking for missing tickers...')
            
        tickers_present = set(
            pl.scan_csv(self.files)
            .select("ticker")
            .unique()
            .collect(engine="streaming")["ticker"]
        )

        missing = []
        for ticker in self.tickers:
            if isinstance(ticker, str):
                if ticker not in tickers_present:
                    missing.append(ticker)
            else:
                if not any(t in tickers_present for t in ticker):
                    missing.append(ticker)

        return missing

    def _get_ticker_batch(self, batch: List[str]) -> Dict[str, pl.DataFrame]:
        """
        Scan the raw CSVs and collect only the tickers in `batch`.

        Keeping batches small is the primary memory-control lever: the
        entire dataset never lands in RAM all at once — only one batch
        worth of rows is live at any moment.

        Ticker-alias stitching (e.g., FB → META) is applied per-batch
        so the caller always receives fully resolved DataFrames.
        """
        # Gather every symbol we might need for this batch, including
        # historical aliases so their rows are pulled in the same scan.
        alias_map: Dict[str, str] = {}   # previous_symbol → current_ticker
        symbols_needed = set(batch)
        for alias in self.ticker_aliases:
            if alias["current_ticker"] in batch:
                for t in alias["previous_tickers"]:
                    symbols_needed.add(t["symbol"])
                    alias_map[t["symbol"]] = alias["current_ticker"]

        df = (
            pl.scan_csv(self.files)
            .filter(pl.col("ticker").is_in(list(symbols_needed)))
            .with_columns(
                pl.from_epoch("window_start", time_unit="ns")
                .dt.replace_time_zone("UTC")
                .alias("timestamp")
            )
            .drop("window_start")
            .collect(engine="streaming")
            .sort(["ticker", "timestamp"])
            .rechunk()
        )

        ticker_dfs: Dict[str, pl.DataFrame] = {
            ticker[0]: frame
            for ticker, frame in df.partition_by("ticker", as_dict=True).items()
        }
        del df  # release the combined frame immediately

        # Resolve aliases that are relevant to this batch
        for alias in self.ticker_aliases:
            current_ticker   = alias["current_ticker"]
            previous_tickers = alias["previous_tickers"]

            if current_ticker not in batch:
                continue

            dfs = []
            latest_date = previous_tickers[0]["change_date"]

            for ticker_info in previous_tickers:
                symbol      = ticker_info["symbol"]
                change_date = ticker_info["change_date"]

                if symbol not in ticker_dfs:
                    continue

                dfs.append(ticker_dfs.pop(symbol).filter(pl.col("timestamp") < change_date))
                latest_date = max(latest_date, change_date)

            if current_ticker in ticker_dfs:
                dfs.append(ticker_dfs[current_ticker].filter(pl.col("timestamp") >= latest_date))

            if not dfs:
                continue

            ticker_dfs[current_ticker] = (
                pl.concat(dfs)
                .sort("timestamp")
                .with_columns(pl.lit(current_ticker).alias("ticker"))
            )

        return ticker_dfs
    
    def _drop_rows_before_timestamp(self, df: pl.DataFrame, timestamp: str) -> pl.DataFrame:
        """
        Remove early data to ensure relevant historical context.

        Trims the dataset to exclude rows where features may be unreliable
        or incomplete (rows during pre-covid era).
        """
        if self.logger:
            self.logger.info(f"Dropping rows before {timestamp}...")

        cutoff = pl.Series([timestamp]).str.to_datetime(format="%Y-%m-%d %H:%M:%S%z").dt.cast_time_unit("us")[0]
        return df.filter(pl.col("timestamp") >= cutoff)

    def _clip_first_and_last_day(self, df: pl.DataFrame) -> pl.DataFrame:
        dates = df.select(pl.col("timestamp").dt.date().unique().sort()).to_series()
        if len(dates) > 2:
            keep = dates[1:-1]
            df = df.filter(pl.col("timestamp").dt.date().is_in(keep.to_list()))
        return df

    def _split_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if self.logger:
            self.logger.info("Splitting data...")

        TRAIN_RATIO = 0.70
        VALID_RATIO = 0.15

        n          = len(df)
        train_end  = int(n * TRAIN_RATIO)
        valid_end  = int(n * (TRAIN_RATIO + VALID_RATIO))

        df_train = self._clip_first_and_last_day(df[:train_end])
        df_valid = self._clip_first_and_last_day(df[train_end:valid_end])
        df_test  = self._clip_first_and_last_day(df[valid_end:])

        return df_train, df_valid, df_test
    
    def _cross_sectional_normalization(self) -> None:
        """
        Second-pass normalisation that operates across all tickers simultaneously.

        After `feature_engineer_and_save_tickers` writes per-ticker parquet files, this
        method loads them all, joins on timestamp, and for each minute bar computes:

        Global (all-ticker) cross-sectional features
        ---------------------------------------------
        For each feature in `CS_FEATURES`:
          • <ticker>_<feat>_cs_zscore  — z-score across all tickers at that timestamp

        Additionally, the raw features themselves are replaced with their cross-
        sectional z-scores (they were intentionally left un-normalised in
        `_normalize_data` so their scales are preserved for this step).

        Sector (intra-sector) cross-sectional features
        -----------------------------------------------
        The same z-score features are computed a second time but only within
        each ticker's sector peer group, giving the model a relative-strength
        signal that is cleaner than the broad-market one for names that live in
        very different vol / return regimes (e.g., UVXY vs TLT).
        Columns are named:
          • <ticker>_<feat>_sector_zscore

        Train / valid / test integrity
        --------------------------------
        Scaler statistics (mean, std) for the raw-feature replacement are fit
        exclusively on the training split and applied to valid/test — exactly
        the same discipline as `_normalize_data`.  Cross-sectional ranks and
        z-scores are computed purely within each timestamp row, so they are
        leak-free by construction.

        Output
        ------
        Overwrites the existing per-ticker parquet files in-place.
        """

        # ------------------------------------------------------------------
        # Sector mapping
        # ------------------------------------------------------------------
        # Each ticker belongs to exactly one sector string. Tickers that span
        # multiple categories (e.g., broad-market ETFs) are placed in the
        # sector that best describes their primary exposure for relative-
        # strength purposes.  Tickers absent from this map are assigned the
        # catch-all "other" sector so they still participate in sector ranks
        # with other ungrouped names rather than being dropped.
        SECTOR_MAP: Dict[str, str] = {
            # broad market / size
            "SPY": "broad_market", "VOO": "broad_market", "IVV": "broad_market",
            "VTI": "broad_market", "QQQ": "broad_market",
            "IWM": "broad_market", "MDY": "broad_market",
            # sector ETFs
            "XLK": "sector_etf", "XLF": "sector_etf", "XLE": "sector_etf",
            "XLV": "sector_etf", "XLI": "sector_etf", "XLY": "sector_etf",
            "XLP": "sector_etf", "XLU": "sector_etf", "XLB": "sector_etf",
            "XLRE": "sector_etf", "XLC": "sector_etf",
            # international
            "EFA": "international", "EEM": "international",
            "FXI": "international", "EWJ": "international",
            # rates / bonds
            "TLT": "fixed_income", "IEF": "fixed_income", "SHY": "fixed_income",
            "LQD": "fixed_income", "HYG": "fixed_income",
            # volatility
            "VIXY": "volatility", "UVXY": "volatility",
            # dollar
            "UUP": "fx",
            # commodities
            "GLD": "commodities", "SLV": "commodities",
            "USO": "commodities", "DBA": "commodities",
            # defense / industrial
            "ITA": "defense_industrial",
            # real assets / REIT
            "VNQ": "real_estate", "AMT": "real_estate", "PLD": "real_estate",
            # financials
            "JPM": "financials", "BAC": "financials", "GS": "financials",
            "KKR": "financials", "BRK.B": "financials",
            # energy / materials
            "XOM": "energy_materials", "CVX": "energy_materials",
            "COP": "energy_materials", "FCX": "energy_materials",
            "NEM": "energy_materials",
            # semiconductors
            "TSM": "semiconductors", "ASML": "semiconductors",
            "AMAT": "semiconductors", "LRCX": "semiconductors",
            "KLAC": "semiconductors", "AMKR": "semiconductors",
            "ASX": "semiconductors", "MU": "semiconductors",
            "SOXX": "semiconductors", "SMH": "semiconductors",
            # mega-cap tech
            "AAPL": "mega_cap_tech", "MSFT": "mega_cap_tech",
            "NVDA": "mega_cap_tech", "AMZN": "mega_cap_tech",
            "GOOGL": "mega_cap_tech", "META": "mega_cap_tech",
            "TSLA": "mega_cap_tech", "AMD": "mega_cap_tech",
            # enterprise tech
            "IBM": "enterprise-tech", "ORCL": "enterprise-tech",
            "CSCO": "enterprise-tech", "ACN": "enterprise-tech",
            # defensive / healthcare
            "PG": "defensive_healthcare", "KO": "defensive_healthcare",
            "WMT": "defensive_healthcare", "JNJ": "defensive_healthcare",
            "UNH": "defensive_healthcare",
            # high-beta / emerging tech
            "ARKK": "high_beta_tech", "MRNA": "high_beta_tech",
            "SHOP": "high_beta_tech", "NET": "high_beta_tech",
            "ROKU": "high_beta_tech", "TWLO": "high_beta_tech",
        }

        # Features that are left un-normalised by _normalize_data and whose
        # ticker-specific scale differences make cross-sectional z-scoring
        # strictly more informative than per-ticker StandardScaler.
        CS_FEATURES = [
            "log_return_5",
            "log_return_15",
            "log_return_60",
            "price_vwap_distance",
            "volume_price_corr_15",  # raw rolling corr: bounded [-1,1] but can have ±inf edge cases
            "adx_20",                # already in [0,1] (divided by 100), but cross-sectional
                                     # z-score captures relative trend strength across tickers
        ]

        # ------------------------------------------------------------------
        # 1. Load all per-ticker parquet files
        # ------------------------------------------------------------------
        if self.logger:
            self.logger.info("Cross-sectional normalisation: loading parquet files...")

        # ticker_dfs[ticker] = single full-timeline DataFrame (trim/split deferred
        # to the very end of _build_unified_splits so every normalization pass
        # has the full warmup history available).
        ticker_dfs: Dict[str, pl.DataFrame] = {}

        for path in sorted(self.tickers_dir.glob("*.parquet")):
            ticker = path.stem
            ticker_dfs[ticker] = pl.read_parquet(path)

        tickers_loaded = sorted(ticker_dfs.keys())
        if self.logger:
            self.logger.info(f"Loaded {len(tickers_loaded)} tickers.")

        # ------------------------------------------------------------------
        # 2. Build a wide table: one row per timestamp, one column per
        #    (ticker, feature).  Used for row-wise cross-sectional stats.
        # ------------------------------------------------------------------

        def _wide_table() -> pl.DataFrame:
            """
            Horizontally join all tickers on timestamp.
            Each ticker contributes columns: <ticker>_<feat> for feat in CS_FEATURES.
            """
            frames = []
            for ticker in tickers_loaded:
                df = ticker_dfs[ticker]
                keep = ["timestamp"] + [f"{ticker}_{f}" for f in CS_FEATURES if f"{ticker}_{f}" in df.columns]
                frames.append(df.select(keep))

            wide = frames[0]
            for f in frames[1:]:
                wide = wide.join(f, on="timestamp", how="inner")
            return wide

        # ------------------------------------------------------------------
        # 3. Per-ticker causal rolling z-score for CS features.
        #
        #    Each ticker's full timeline is used so the 1950-bar warmup lands
        #    in the pre-cutoff period, not in training data.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info(
                f"Rolling z-score normalizing CS features per ticker "
                f"(window={NORMALIZATION_WINDOW} bars, causal)..."
            )

        CS_ROLLING_SKIP = {"volume_price_corr_15"}

        for ticker in tickers_loaded:
            full_df = ticker_dfs[ticker]

            for feat in CS_FEATURES:
                if feat in CS_ROLLING_SKIP:
                    continue

                col = f"{ticker}_{feat}"
                if col not in full_df.columns:
                    continue

                full_arr = full_df[col].to_numpy().reshape(-1, 1).astype(np.float64)
                scaled   = _rolling_zscore_normalize_vectorized(full_arr, window=NORMALIZATION_WINDOW)

                ticker_dfs[ticker] = ticker_dfs[ticker].with_columns(
                    pl.Series(col, scaled.flatten())
                )

        # ------------------------------------------------------------------
        # 4. Row-wise cross-sectional statistics (z-score per timestamp)
        # ------------------------------------------------------------------

        def _compute_cs_stats(
            tickers_in_group: List[str],
            feat: str,
            suffix_zscore: str,
            wide: pl.DataFrame,
        ) -> Dict[str, Dict[str, pl.Series]]:
            """
            Compute row-wise z-score for `feat` across tickers_in_group.

            Returns a nested dict:
                { ticker: { col_name: pl.Series, ... } }

            The series are aligned to `wide`'s row order. Callers are responsible
            for aligning to each ticker's own timestamp index before attaching.
            """
            feat_cols = [
                f"{t}_{feat}"
                for t in tickers_in_group
                if f"{t}_{feat}" in wide.columns
            ]
            if not feat_cols:
                return {}

            arr = wide.select(feat_cols).to_numpy().astype(np.float64)  # (T, N)

            cs_mean = np.nanmean(arr, axis=1, keepdims=True)
            cs_std  = np.nanstd(arr,  axis=1, keepdims=True)
            cs_std  = np.where(cs_std < 1e-8, 1.0, cs_std)

            z_arr = np.tanh((arr - cs_mean) / cs_std)  # (T, N)

            result: Dict[str, Dict[str, pl.Series]] = {}
            for i, col in enumerate(feat_cols):
                ticker = col[: -(len(feat) + 1)]
                result[ticker] = {
                    f"{ticker}_{feat}{suffix_zscore}": pl.Series(
                        f"{ticker}_{feat}{suffix_zscore}", z_arr[:, i]
                    ),
                }
            return result

        # pending_cols[ticker] = list of pl.Series to attach
        pending_cols: Dict[str, List[pl.Series]] = {ticker: [] for ticker in tickers_loaded}

        if self.logger:
            self.logger.info("Computing global cross-sectional stats...")

        wide        = _wide_table()
        wide_ts_np  = wide["timestamp"].to_numpy()
        ticker_ts_np = {t: ticker_dfs[t]["timestamp"].to_numpy() for t in tickers_loaded}

        for feat in CS_FEATURES:
            stats = _compute_cs_stats(
                tickers_in_group=tickers_loaded,
                feat=feat,
                suffix_zscore="_cs_zscore",
                wide=wide,
            )
            for ticker, series_dict in stats.items():
                if ticker not in pending_cols:
                    continue
                mask = np.isin(wide_ts_np, ticker_ts_np[ticker])
                for col_name, series in series_dict.items():
                    pending_cols[ticker].append(series.filter(mask).alias(col_name))

        # ------------------------------------------------------------------
        # 5. Sector-level cross-sectional stats
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("Computing sector cross-sectional stats...")

        sector_to_tickers: Dict[str, List[str]] = {}
        for ticker in tickers_loaded:
            sector = SECTOR_MAP.get(ticker, "other")
            sector_to_tickers.setdefault(sector, []).append(ticker)

        for sector, sector_tickers in sector_to_tickers.items():
            present = [t for t in sector_tickers if t in tickers_loaded]
            if len(present) < 2:
                continue

            for feat in CS_FEATURES:
                stats = _compute_cs_stats(
                    tickers_in_group=present,
                    feat=feat,
                    suffix_zscore="_sector_zscore",
                    wide=wide,
                )
                for ticker, series_dict in stats.items():
                    if ticker not in pending_cols:
                        continue
                    mask = np.isin(wide_ts_np, ticker_ts_np[ticker])
                    for col_name, series in series_dict.items():
                        pending_cols[ticker].append(series.filter(mask).alias(col_name))

        # Flush all accumulated columns in one with_columns call per ticker
        for ticker in tickers_loaded:
            cols = pending_cols[ticker]
            if cols:
                ticker_dfs[ticker] = ticker_dfs[ticker].with_columns(cols)

        # ------------------------------------------------------------------
        # 6. Persist — overwrite existing parquet files
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("Writing cross-sectionally normalised parquet files...")

        for ticker in tickers_loaded:
            save_to_parquet(
                ticker_dfs[ticker],
                str(self.tickers_dir / f"{ticker}.parquet"),
                index=False,
            )

        if self.logger:
            self.logger.info("Cross-sectional normalisation complete.")

    def _build_unified_splits(self) -> None:
        """
        Build unified train / valid / test parquet files by horizontally
        stacking all per-ticker DataFrames and adding two cross-ticker features:

        Market Breadth
        --------------
        Fraction of tickers with a positive return over the last N minutes,
        computed at every bar using a rolling window aligned to timestamp.
        Windows: 5, 15, 30, 60 minutes.
        Column names (shared across all tickers in the unified frame):
            breadth_5m, breadth_15m, breadth_30m, breadth_60m

        Relative Strength vs SPY
        ------------------------
        For each ticker and each window N, the N-minute cumulative return of
        that ticker minus the N-minute cumulative return of SPY.
            <ticker>_rs_spy_5m, <ticker>_rs_spy_15m,
            <ticker>_rs_spy_30m, <ticker>_rs_spy_60m

        Trim and split
        --------------
        After all normalization passes are complete, _drop_rows_before_timestamp
        removes the pre-cutoff warmup period (where rolling z-score windows
        produce zeros) and _split_data divides the remaining data into
        train / valid / test by ratio.

        Output
        ------
        <data_dir>/preprocessed/v4/unified/unified_train.parquet
        <data_dir>/preprocessed/v4/unified/unified_valid.parquet
        <data_dir>/preprocessed/v4/unified/unified_test.parquet
        """

        BREADTH_WINDOWS = [5, 15, 30, 60]   # minutes
        RS_WINDOWS      = [5, 15, 30, 60]   # minutes
        SPLITS          = ("train", "valid", "test")

        # Discover tickers from the flat parquet files written by _cross_sectional_normalization
        tickers_present = sorted(
            p.stem for p in self.tickers_dir.glob("*.parquet")
        )

        if self.logger:
            self.logger.info(
                f"Building unified splits for {len(tickers_present)} tickers..."
            )

        # Columns that are identical across all tickers (shared temporal features
        # plus timestamp). Keep them from the first ticker only; strip from all
        # subsequent ones before joining to avoid DuplicateError.
        SHARED_COLS = {"timestamp", "minute_sin", "minute_cos", "hour_sin", "hour_cos",
                    "day_sin", "day_cos", "minutes_since_open", "minutes_to_close"}

        # ------------------------------------------------------------------
        # Phase 1: horizontal stack — one pass over the full timeline
        # (no split yet; trim and split happen at the very end after all
        # normalization passes have had the full warmup history).
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Horizontally stacking tickers on full timeline...")

        stacked: Optional[pl.DataFrame] = None

        for ticker in tickers_present:
            path = self.tickers_dir / f"{ticker}.parquet"
            if not path.exists():
                if self.logger:
                    self.logger.warning(f"  missing {path}, skipping")
                continue

            df = pl.read_parquet(path)

            if stacked is None:
                stacked = df
            else:
                ticker_only_cols = [c for c in df.columns if c not in SHARED_COLS]
                stacked = stacked.join(
                    df.select(["timestamp"] + ticker_only_cols),
                    on="timestamp",
                    how="inner",
                )
                del df

        if stacked is None:
            if self.logger:
                self.logger.warning("  no ticker data found, aborting")
            return

        stacked = stacked.sort("timestamp")

        # ------------------------------------------------------------------
        # Market breadth: fraction of tickers with positive return over N min
        #
        # For each window W, we need the W-bar lagged close for every ticker.
        # The log_return_n columns are already CS z-scored, so we reconstruct
        # the rolling N-minute return from the raw close column — but close
        # was dropped in _drop_unnecessary_features.  Instead, accumulate
        # N-bar rolling sums of log_return_n (z-scored, but the *sign* is
        # preserved, which is all breadth needs).
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Computing market breadth on full timeline...")

        return_cols = [c for c in stacked.columns if c.endswith("_log_return_5")]

        breadth_exprs = []
        for w in BREADTH_WINDOWS:
            if w == 1:
                # single-bar return: just check sign
                positive = pl.concat_list([
                    (pl.col(c) > 0).cast(pl.Float64)
                    for c in return_cols
                ]).list.mean()
            else:
                # rolling sum of log returns ≈ cumulative return over W bars
                # sign of sum = sign of net move; mean gives breadth fraction
                positive = pl.concat_list([
                    (pl.col(c).rolling_sum(window_size=w) > 0).cast(pl.Float64)
                    for c in return_cols
                ]).list.mean()
            breadth_exprs.append(positive.alias(f"breadth_{w}m"))

        stacked = stacked.with_columns(breadth_exprs)

        # ------------------------------------------------------------------
        # Relative strength vs SPY
        #
        # rs_spy_Nm[t] = cumulative_return_ticker(t, N) - cumulative_return_SPY(t, N)
        #
        # Since log_return_n columns are z-scored (sign preserved, magnitude
        # is cross-sectional), the rolling sum difference is a valid relative-
        # strength measure: positive means the ticker is trending stronger
        # than SPY over the window on a market-normalised basis.
        # ------------------------------------------------------------------

        spy_col = "SPY_log_return_5"
        if spy_col not in stacked.columns:
            if self.logger:
                self.logger.warning(
                    "SPY not found in unified frame — skipping RS vs SPY features"
                )
        else:
            if self.logger:
                self.logger.info("  Computing RS vs SPY on full timeline...")

            rs_exprs = []
            for w in RS_WINDOWS:
                spy_ret = pl.col(spy_col) if w == 1 else pl.col(spy_col).rolling_sum(window_size=w)

                for ticker in tickers_present:
                    if ticker == 'SPY':
                        continue
                    t_col = f"{ticker}_log_return_5"
                    if t_col not in stacked.columns:
                        continue
                    t_ret = pl.col(t_col) if w == 1 else pl.col(t_col).rolling_sum(window_size=w)
                    rs_exprs.append((t_ret - spy_ret).alias(f"{ticker}_rs_spy_{w}m"))

            stacked = stacked.with_columns(rs_exprs)

            # ------------------------------------------------------------------
            # Normalize rs_spy columns with causal rolling z-score.
            # The stacked frame already spans train+valid+test in time order,
            # so rolling stats at each bar t are computed from [t-RS_WINDOW, t-1]
            # only — valid/test rows are normalized using only past data.
            # ------------------------------------------------------------------
            rs_cols = [c for c in stacked.columns if "_rs_spy_" in c]
            if rs_cols:
                if self.logger:
                    self.logger.info("  Normalizing RS vs SPY features (rolling z-score)...")

                full_arr = stacked.select(rs_cols).to_numpy().astype(np.float64)
                scaled   = _rolling_zscore_normalize_vectorized(full_arr, window=NORMALIZATION_WINDOW)

                stacked = stacked.with_columns([
                    pl.Series(col, scaled[:, i])
                    for i, col in enumerate(rs_cols)
                ])

        # ------------------------------------------------------------------
        # Phase 3: trim warmup period and split
        #
        # Now that all normalization passes are done, drop the pre-cutoff
        # rows (where rolling z-score windows produced zeros) and divide
        # the remaining data into train / valid / test.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Dropping warmup rows and splitting...")

        stacked = self._drop_rows_before_timestamp(stacked, self.timestamp)
        df_train, df_valid, df_test = self._split_data(stacked)
        del stacked

        out_dir = Path(self.data_dir) / "preprocessed" / "v4" / "unified"
        create_directory(out_dir)

        for split, df in zip(SPLITS, (df_train, df_valid, df_test)):
            if self.logger:
                self.logger.info(f"  [{split}] writing unified_{split}.parquet...")
            save_to_parquet(df, f"{out_dir}/unified_{split}.parquet", index=False)
            del df

        if self.logger:
            self.logger.info("Unified splits complete.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def feature_engineer_and_save_tickers(self, batch_size: int = 8, check_for_missing_tickers: bool=False) -> None:
        """
        Run the full feature-engineering pipeline for all tickers,
        processing them in memory-bounded batches.

        Memory strategy
        ---------------
        Keep batch_size ≤ max_workers so workers stay busy without
        stacking up unprocessed DataFrames in the main process.
        A batch_size of 8 on 32 GB RAM leaves comfortable headroom.
        Tune it down to 4 if you observe pressure.

        Each batch is fully processed and written to disk before the
        next batch is scanned, so peak in-flight memory is bounded.

        :param batch_size: Number of tickers to load and process per round
        :type batch_size: int
        """
        if check_for_missing_tickers:
            if missing := self._check_for_missing_tickers():
                if self.logger:
                    self.logger.error(f"Missing tickers: {missing}")
                return
            else:
                if self.logger:
                    self.logger.info('No missing tickers')

        # Build batches, keeping alias-linked tickers together so a single
        # scan can cover both the old and new symbol.
        batches: List[List[str]] = []
        batch: List[str] = []
        for ticker in self.tickers:
            batch.append(ticker)
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)

        total   = len(self.tickers)
        done    = 0

        for batch_num, batch in enumerate(batches, 1):
            if self.logger:
                self.logger.info(
                    f"Batch {batch_num}/{len(batches)} — "
                    f"loading {batch}..."
                )

            ticker_dfs = self._get_ticker_batch(batch)

            if self.logger:
                self.logger.info(
                    f"Batch {batch_num} — dispatching "
                    f"{len(ticker_dfs)} tickers to workers..."
                )

            futures = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                for ticker, df in ticker_dfs.items():
                    fut = pool.submit(
                        _process_ticker,
                        ticker,
                        df,
                        self.data_dir,
                        False,
                    )
                    futures[fut] = ticker

                # Drain results as they complete so worker memory is freed
                for fut in as_completed(futures):
                    result = fut.result()
                    if result.startswith("ERROR:"):
                        _, failed_ticker, msg = result.split(":", 2)
                        if self.logger:
                            self.logger.error(f"[{failed_ticker}] pipeline failed: {msg}")
                    else:
                        done += 1
                        if self.logger:
                            self.logger.info(f"[{result}] done ✓  ({done}/{total})")

            # Explicitly drop the batch dict before the next scan
            del ticker_dfs

        if self.logger:
            self.logger.info("Per-ticker feature engineering complete. Starting cross-sectional normalisation...")

        # Delete ghost parquet files for alias tickers (e.g. FB → META).
        # Their data has already been stitched into the current ticker's parquet
        # by _get_ticker_batch. Leaving the ghost files on disk would cause
        # _cross_sectional_normalization to load them as independent tickers,
        # producing a shorter timeline that corrupts the wide-table inner join.
        for alias in self.ticker_aliases:
            for ticker_info in alias["previous_tickers"]:
                ghost_path = self.tickers_dir / f"{ticker_info['symbol']}.parquet"
                if ghost_path.exists():
                    ghost_path.unlink()
                    if self.logger:
                        self.logger.info(
                            f"Deleted ghost file: {ghost_path} "
                            f"(alias of {alias['current_ticker']})"
                        )

        self._cross_sectional_normalization()

        if self.logger:
            self.logger.info("Cross-sectional normalisation complete. Building unified splits...")

        self._build_unified_splits()

        if self.logger:
            self.logger.info("Feature engineering complete. Godspeed brotherman")


def main():
    feature_engineer = DataFeatureEngineer(logger=Logger())
    feature_engineer.feature_engineer_and_save_tickers()


if __name__ == "__main__":
    main()