import numpy as np
import polars as pl
import pandas_market_calendars as mcal
import talib as ta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
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


def _process_ticker(
    ticker: str,
    ticker_df: pl.DataFrame,
    timestamp: str,
    data_dir: Path,
    log: bool,
) -> str:
    """
    Full feature-engineering pipeline for one ticker.
    Runs in a worker process — no shared state.
    Returns the ticker name on success or an error string on failure.
    """
    try:
        df = _handle_gaps(ticker_df, log)
        df = _add_temporal_patterns(df, log)
        df = _add_volatility_features(df, log)
        df = _add_trend_features(df, log)
        df = _add_volume_features(df, log)
        df = _drop_unnecessary_features(df, log)
        df = _drop_rows_before_timestamp(df, timestamp, log)

        df_train, df_valid, df_test = _split_data(df, log)
        df_train, df_valid, df_test = _normalize_data(df_train, df_valid, df_test, log)
        
        df_train = _prefix_columns(df_train, ticker)
        df_valid = _prefix_columns(df_valid, ticker)
        df_test  = _prefix_columns(df_test,  ticker)

        base_dir = data_dir / "preprocessed" / "v4" / ticker
        create_directory(base_dir)
        save_to_parquet(df_train, f"{base_dir}/{ticker}_train.parquet", index=False)
        save_to_parquet(df_valid, f"{base_dir}/{ticker}_valid.parquet", index=False)
        save_to_parquet(df_test,  f"{base_dir}/{ticker}_test.parquet",  index=False)

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

    # minutes_since_open / minutes_to_close require a within-day row index
    df = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).over("_date").alias("_day_idx"),
        pl.len().over("_date").alias("_day_len"),
    ).with_columns([
        pl.col("_day_idx").alias("minutes_since_open"),
        (pl.col("_day_len") - pl.col("_day_idx") - 1).alias("minutes_to_close"),
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


def _drop_rows_before_timestamp(df: pl.DataFrame, timestamp: str, log: bool = False) -> pl.DataFrame:
    """
    Remove early data to ensure relevant historical context.

    Trims the dataset to exclude rows where features may be unreliable
    or incomplete (rows during pre-covid era).
    """
    if log:
        print(f"Dropping rows before {timestamp}...")

    cutoff = pl.Series([timestamp]).str.to_datetime(format="%Y-%m-%d %H:%M:%S%z").dt.cast_time_unit("us")[0]
    return df.filter(pl.col("timestamp") >= cutoff)


def _clip_first_and_last_day(df: pl.DataFrame) -> pl.DataFrame:
    dates = df.select(pl.col("timestamp").dt.date().unique().sort()).to_series()
    if len(dates) > 2:
        keep = dates[1:-1]
        df = df.filter(pl.col("timestamp").dt.date().is_in(keep))
    return df


def _split_data(
    df: pl.DataFrame,
    log: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if log:
        print("Splitting data...")

    TRAIN_RATIO = 0.70
    VALID_RATIO = 0.15

    n          = len(df)
    train_end  = int(n * TRAIN_RATIO)
    valid_end  = int(n * (TRAIN_RATIO + VALID_RATIO))

    df_train = _clip_first_and_last_day(df[:train_end])
    df_valid = _clip_first_and_last_day(df[train_end:valid_end])
    df_test  = _clip_first_and_last_day(df[valid_end:])

    return df_train, df_valid, df_test


def _normalize_data(
    df_train: pl.DataFrame,
    df_valid: pl.DataFrame,
    df_test:  pl.DataFrame,
    log: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    temporal_features_to_scale = ["minutes_since_open", "minutes_to_close"]
    volatility_features_to_scale = [
        # Log vol ratios: these are ticker-relative (own short/long window comparison),
        # so their mean/std differ meaningfully by ticker — per-ticker StandardScaler
        # is correct. The raw values can reach ±20–30 on gap/halt bars.
        "volatility_5_15_ratio", "volatility_15_60_ratio",
    ]
    trend_features_to_scale = [
        # Log price/EMA ratios: self-referential (close vs own EMA), ticker-neutral
        # by construction but tails can exceed ±5 during large moves.
        "ema_close_ratio_5", "ema_close_ratio_15", "ema_close_ratio_60",
        "ema_5_15_ratio", "ema_15_60_ratio", "adx_5_15_ratio", "adx_15_60_ratio"
    ]
    volume_features_to_scale = [
        # Log volume/EMA ratios: ticker-relative (own volume vs own EMA windows).
        # Absolute volume scale varies enormously across tickers, but these ratios
        # are already self-normalised — per-ticker StandardScaler removes residual
        # ticker-level mean/std differences and clips outliers.
        "volume_ema_ratio_15", "volume_5_15_ratio", "volume_15_60_ratio",
        "avg_trade_size_ratio", "range_per_trade_ratio", "trade_intensity_ratio"
    ]
    # NOTE: log_return_5, log_return_15, log_return_60,
    #       price_vwap_distance, volume_price_corr_15, and adx_20 are
    #       intentionally excluded here. They are normalised cross-sectionally
    #       inside DataPreprocessor._cross_sectional_normalization(), which has
    #       access to all tickers simultaneously and can therefore compute
    #       meaningful market-relative statistics.
    features_to_standard_scale = temporal_features_to_scale + trend_features_to_scale
    features_to_robust_scale = volatility_features_to_scale + volume_features_to_scale
    
    all_scaled_features = features_to_standard_scale + features_to_robust_scale

    if log:
        print(f"Normalizing features {all_scaled_features}...")

    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler(quantile_range=(1.0, 99.0))
    
    def fit_winsorize(arr, lower=0.01, upper=0.99):
        lo = np.quantile(arr, lower, axis=0)
        hi = np.quantile(arr, upper, axis=0)
        return lo, hi

    def apply_winsorize(arr, lo, hi):
        return np.clip(arr, lo, hi)
    
    # --- STANDARD FEATURES ---
    X_train_std = df_train.select(features_to_standard_scale).to_numpy()
    X_valid_std = df_valid.select(features_to_standard_scale).to_numpy()
    X_test_std  = df_test.select(features_to_standard_scale).to_numpy()

    lo_std, hi_std = fit_winsorize(X_train_std)
    X_train_std = apply_winsorize(X_train_std, lo_std, hi_std)
    X_valid_std = apply_winsorize(X_valid_std, lo_std, hi_std)
    X_test_std  = apply_winsorize(X_test_std, lo_std, hi_std)

    # X_train_std_s = np.clip(standard_scaler.fit_transform(X_train_std), -5, 5)
    # X_valid_std_s = np.clip(standard_scaler.transform(X_valid_std),     -5, 5)
    # X_test_std_s  = np.clip(standard_scaler.transform(X_test_std),      -5, 5)
    X_train_std_s = np.tanh(standard_scaler.fit_transform(X_train_std))
    X_valid_std_s = np.tanh(standard_scaler.transform(X_valid_std))
    X_test_std_s  = np.tanh(standard_scaler.transform(X_test_std))
    
    # --- ROBUST FEATURES ---
    X_train_rob = df_train.select(features_to_robust_scale).to_numpy()
    X_valid_rob = df_valid.select(features_to_robust_scale).to_numpy()
    X_test_rob  = df_test.select(features_to_robust_scale).to_numpy()

    lo_rob, hi_rob = fit_winsorize(X_train_rob)
    X_train_rob = apply_winsorize(X_train_rob, lo_rob, hi_rob)
    X_valid_rob = apply_winsorize(X_valid_rob, lo_rob, hi_rob)
    X_test_rob  = apply_winsorize(X_test_rob, lo_rob, hi_rob)
    
    # X_train_rob_s = np.clip(robust_scaler.fit_transform(X_train_rob), -5, 5)
    # X_valid_rob_s = np.clip(robust_scaler.transform(X_valid_rob),     -5, 5)
    # X_test_rob_s  = np.clip(robust_scaler.transform(X_test_rob),      -5, 5)
    X_train_rob_s = np.tanh(robust_scaler.fit_transform(X_train_rob))
    X_valid_rob_s = np.tanh(robust_scaler.transform(X_valid_rob))
    X_test_rob_s  = np.tanh(robust_scaler.transform(X_test_rob))
    
    X_train_s = np.hstack([X_train_std_s, X_train_rob_s])
    X_valid_s = np.hstack([X_valid_std_s, X_valid_rob_s])
    X_test_s  = np.hstack([X_test_std_s,  X_test_rob_s])

    def _replace_scaled(df: pl.DataFrame, arr: np.ndarray) -> pl.DataFrame:
        scaled_cols = {
            col: pl.Series(col, arr[:, i]) for i, col in enumerate(all_scaled_features)
        }
        return df.with_columns([v.alias(k) for k, v in scaled_cols.items()])

    return (
        _replace_scaled(df_train, X_train_s),
        _replace_scaled(df_valid, X_valid_s),
        _replace_scaled(df_test,  X_test_s),
    )

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

class DataPreprocessor:
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
        self.files            = (
            list(Path(f"{self.data_dir}/raw/minute_data").glob("2019/1[0-2]/**/*.csv.gz")) +
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
    
    def _cross_sectional_normalization(self) -> None:
        """
        Second-pass normalisation that operates across all tickers simultaneously.

        After `preprocess_and_save_tickers` writes per-ticker parquet files, this
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
            "ASX": "semiconductors", "SOXX": "semiconductors",
            "SMH": "semiconductors",
            # mega-cap tech
            "AAPL": "mega_cap_tech", "MSFT": "mega_cap_tech",
            "NVDA": "mega_cap_tech", "AMZN": "mega_cap_tech",
            "GOOGL": "mega_cap_tech", "META": "mega_cap_tech",
            "TSLA": "mega_cap_tech", "AMD": "mega_cap_tech",
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
            "adx_20",                 # already in [0,1] (divided by 100), but cross-sectional
                                      # z-score captures relative trend strength across tickers
        ]

        SPLITS = ("train", "valid", "test")

        _data_dir = Path(self.data_dir) / "preprocessed" / "v4"

        # ------------------------------------------------------------------
        # 1. Load all per-ticker parquet files
        # ------------------------------------------------------------------
        if self.logger:
            self.logger.info("Cross-sectional normalisation: loading parquet files...")

        # ticker_dfs[ticker][split] = DataFrame (with prefixed columns + timestamp)
        ticker_dfs: Dict[str, Dict[str, pl.DataFrame]] = {}

        for ticker_dir in sorted(_data_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            splits: Dict[str, pl.DataFrame] = {}
            for split in SPLITS:
                path = ticker_dir / f"{ticker}_{split}.parquet"
                if path.exists():
                    splits[split] = pl.read_parquet(path)
            if len(splits) == len(SPLITS):
                ticker_dfs[ticker] = splits

        tickers_loaded = sorted(ticker_dfs.keys())
        if self.logger:
            self.logger.info(f"Loaded {len(tickers_loaded)} tickers.")

        # ------------------------------------------------------------------
        # 2. Build a per-split wide table: one row per timestamp, one column
        #    per (ticker, feature).  This is the efficient path for computing
        #    row-wise stats (mean/std/rank across tickers at the same instant).
        # ------------------------------------------------------------------

        def _wide_table(split: str) -> pl.DataFrame:
            """
            Horizontally join all tickers for `split` on timestamp.
            Each ticker contributes columns: <ticker>_<feat> for feat in CS_FEATURES.
            """
            frames = []
            for ticker in tickers_loaded:
                df = ticker_dfs[ticker][split]
                keep = ["timestamp"] + [f"{ticker}_{f}" for f in CS_FEATURES if f"{ticker}_{f}" in df.columns]
                frames.append(df.select(keep))

            wide = frames[0]
            for f in frames[1:]:
                wide = wide.join(f, on="timestamp", how="inner")
            return wide

        # ------------------------------------------------------------------
        # 3. Fit cross-sectional scalers on train, apply to all splits.
        #
        #    "Cross-sectional scaler" means: for each feature f and each
        #    timestamp t, compute mean_f(t) and std_f(t) across all tickers.
        #    Then z-score = (x - mean) / std.
        #
        #    We fit by computing, for each feature f across the entire training
        #    set, the *distribution of cross-sectional means and stds* and then
        #    use those meta-statistics to decide whether the raw z-score is
        #    reasonable.  In practice the cleanest approach that prevents
        #    look-ahead is:
        #      - z-scores are computed purely row-wise (no fitting needed)
        #      - ranks are percentile ranks within the row (no fitting needed)
        #    The only place where train-only fitting applies is the replacement
        #    of the raw feature values with a StandardScaler z-score, which we
        #    continue to do per-ticker to preserve the existing contract of
        #    _normalize_data.  Here we stack the training split for each ticker,
        #    fit one scaler per (ticker, feature), and transform all splits.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("Fitting per-ticker scalers for CS features on train split...")

        # scaler_store[ticker][feat] = (fitted StandardScaler, lo_bound, hi_bound)
        scaler_store: Dict[str, Dict[str, Tuple[StandardScaler, float, float]]] = {}
        for ticker in tickers_loaded:
            scaler_store[ticker] = {}
            train_df = ticker_dfs[ticker]["train"]
            for feat in CS_FEATURES:
                col = f"{ticker}_{feat}"
                if col not in train_df.columns:
                    continue
                arr = train_df[col].drop_nulls().to_numpy().reshape(-1, 1).astype(np.float64)
                
                # Fit winsorize bounds on train
                lo = np.quantile(arr, 0.01)
                hi = np.quantile(arr, 0.99)
                arr_win = np.clip(arr, lo, hi)
                
                sc = StandardScaler()
                sc.fit(arr_win)
                scaler_store[ticker][feat] = (sc, lo, hi)

        # Apply scalers to all splits (replace raw values in-place)
        for ticker in tickers_loaded:
            for split in SPLITS:
                df = ticker_dfs[ticker][split]
                replacements = []
                for feat in CS_FEATURES:
                    col = f"{ticker}_{feat}"
                    if col not in df.columns or feat not in scaler_store[ticker]:
                        continue
                    sc, lo, hi = scaler_store[ticker][feat]
                    arr = df[col].to_numpy().reshape(-1, 1).astype(np.float64)
                    
                    # Apply winsorize -> scale -> tanh
                    arr_win = np.clip(arr, lo, hi)
                    scaled = np.tanh(sc.transform(arr_win).flatten())
                    replacements.append(pl.Series(col, scaled))
                if replacements:
                    ticker_dfs[ticker][split] = df.with_columns(replacements)

        # ------------------------------------------------------------------
        # 4. Row-wise cross-sectional statistics (z-score and rank per timestamp)
        #
        #    For each split, for each feature f:
        #      cs_mean[t] = mean over tickers of <ticker>_f[t]
        #      cs_std[t]  = std  over tickers of <ticker>_f[t]
        #      <ticker>_f_cs_zscore[t] = (<ticker>_f[t] - cs_mean[t]) / cs_std[t]
        #
        #    Z-scores are computed purely within each timestamp row, so they are
        #    leak-free by construction.
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
            for slicing / aligning to each split before attaching.
            """
            feat_cols = [
                f"{t}_{feat}"
                for t in tickers_in_group
                if f"{t}_{feat}" in wide.columns
            ]
            if not feat_cols:
                return {}

            arr = wide.select(feat_cols).to_numpy().astype(np.float64)  # (T, N)

            # row-wise mean / std (NaN-safe)
            cs_mean = np.nanmean(arr, axis=1, keepdims=True)
            cs_std  = np.nanstd(arr,  axis=1, keepdims=True)
            cs_std  = np.where(cs_std < 1e-8, 1.0, cs_std)

            z_arr = np.tanh((arr - cs_mean) / cs_std)  # (T, N)

            result: Dict[str, Dict[str, pl.Series]] = {}
            for i, col in enumerate(feat_cols):
                ticker = col[: -(len(feat) + 1)]   # strip _<feat>
                result[ticker] = {
                    f"{ticker}_{feat}{suffix_zscore}": pl.Series(
                        f"{ticker}_{feat}{suffix_zscore}", z_arr[:, i]
                    ),
                }
            return result

        # pending_cols[split][ticker] = list of pl.Series to attach
        pending_cols: Dict[str, Dict[str, List[pl.Series]]] = {
            split: {ticker: [] for ticker in tickers_loaded}
            for split in SPLITS
        }

        if self.logger:
            self.logger.info("Computing global cross-sectional stats...")

        for split in SPLITS:
            wide = _wide_table(split)
            wide_ts_np = wide["timestamp"].to_numpy()
            # Pre-compute split timestamp arrays once per (split, ticker)
            split_ts_np_cache = {
                t: ticker_dfs[t][split]["timestamp"].to_numpy()
                for t in tickers_loaded
            }

            for feat in CS_FEATURES:
                stats = _compute_cs_stats(
                    tickers_in_group=tickers_loaded,
                    feat=feat,
                    suffix_zscore="_cs_zscore",
                    wide=wide,
                )
                for ticker, series_dict in stats.items():
                    if ticker not in pending_cols[split]:
                        continue
                    mask = np.isin(wide_ts_np, split_ts_np_cache[ticker])
                    for col_name, series in series_dict.items():
                        pending_cols[split][ticker].append(
                            series.filter(mask).alias(col_name)
                        )

        # ------------------------------------------------------------------
        # 5. Sector-level cross-sectional stats
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("Computing sector cross-sectional stats...")

        # Build reverse map: sector → [tickers]
        sector_to_tickers: Dict[str, List[str]] = {}
        for ticker in tickers_loaded:
            sector = SECTOR_MAP.get(ticker, "other")
            sector_to_tickers.setdefault(sector, []).append(ticker)

        for split in SPLITS:
            wide = _wide_table(split)
            wide_ts = wide["timestamp"]
            wide_ts_np = wide_ts.to_numpy()

            for sector, sector_tickers in sector_to_tickers.items():
                present = [t for t in sector_tickers if t in tickers_loaded]
                if len(present) < 2:
                    # Single-member sector: z-score 0 / rank 0.5 would be noise.
                    continue

                for feat in CS_FEATURES:
                    stats = _compute_cs_stats(
                        tickers_in_group=present,
                        feat=feat,
                        suffix_zscore="_sector_zscore",
                        wide=wide,
                    )
                    for ticker, series_dict in stats.items():
                        if ticker not in pending_cols[split]:
                            continue
                        split_ts_np = ticker_dfs[ticker][split]["timestamp"].to_numpy()
                        mask = np.isin(wide_ts_np, split_ts_np)
                        for col_name, series in series_dict.items():
                            pending_cols[split][ticker].append(
                                series.filter(mask).alias(col_name)
                            )

        # Flush all accumulated columns in one with_columns call per ticker/split
        for split in SPLITS:
            for ticker in tickers_loaded:
                cols = pending_cols[split][ticker]
                if cols:
                    ticker_dfs[ticker][split] = ticker_dfs[ticker][split].with_columns(cols)

        # ------------------------------------------------------------------
        # 6. Persist — overwrite existing parquet files
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("Writing cross-sectionally normalised parquet files...")

        for ticker in tickers_loaded:
            base_dir = _data_dir / ticker
            for split in SPLITS:
                save_to_parquet(
                    ticker_dfs[ticker][split],
                    str(base_dir / f"{ticker}_{split}.parquet"),
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
            breadth_1m, breadth_5m, breadth_15m, breadth_30m

        Relative Strength vs SPY
        ------------------------
        For each ticker and each window N, the N-minute cumulative return of
        that ticker minus the N-minute cumulative return of SPY.
            <ticker>_rs_spy_1m, <ticker>_rs_spy_5m,
            <ticker>_rs_spy_15m, <ticker>_rs_spy_30m

        Memory strategy
        ---------------
        Processed split-by-split.  Within each split, tickers are loaded and
        horizontally joined one at a time, freeing each individual DataFrame
        from memory as soon as it has been merged into the growing unified
        frame.  Peak RAM is bounded to roughly:
            unified_so_far  (grows incrementally)
        + one_ticker_df   (released after each join)
        which comfortably fits within 32 GB for a ~20 GB total dataset.

        Breadth and RS are computed on the full vertically-stacked timeline
        (train + valid + test) so that rolling windows at split boundaries
        have real history behind them and no warmup rows are lost from
        valid or test.  Only the very first WARMUP_BARS rows of train are
        trimmed.

        Output
        ------
        <data_dir>/preprocessed/v4/unified_train.parquet
        <data_dir>/preprocessed/v4/unified_valid.parquet
        <data_dir>/preprocessed/v4/unified_test.parquet
        """

        BREADTH_WINDOWS = [5, 15, 30, 60]   # minutes
        RS_WINDOWS      = [5, 15, 30, 60]   # minutes
        WARMUP_BARS     = max(BREADTH_WINDOWS + RS_WINDOWS)
        SPLITS          = ("train", "valid", "test")

        _data_dir = Path(self.data_dir) / "preprocessed" / "v4"

        # Discover tickers from the directory layout (same set as CS step)
        tickers_present = sorted(
            d.name for d in _data_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}_train.parquet").exists()
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
        # Phase 1: horizontal stack per split, store in memory
        # ------------------------------------------------------------------

        unified_splits: dict[str, pl.DataFrame] = {}

        for split in SPLITS:
            if self.logger:
                self.logger.info(f"  [{split}] horizontally stacking tickers...")

            unified: Optional[pl.DataFrame] = None

            for ticker in tickers_present:
                path = _data_dir / ticker / f"{ticker}_{split}.parquet"
                if not path.exists():
                    if self.logger:
                        self.logger.warning(f"  [{split}] missing {path}, skipping")
                    continue

                df = pl.read_parquet(path)

                if unified is None:
                    unified = df
                else:
                    # Drop shared columns from every subsequent ticker so they
                    # don't collide with the copies already in `unified`.
                    ticker_only_cols = [c for c in df.columns if c not in SHARED_COLS]
                    unified = unified.join(
                        df.select(["timestamp"] + ticker_only_cols),
                        on="timestamp",
                        how="inner",
                    )
                    del df  # release immediately

            if unified is None:
                if self.logger:
                    self.logger.warning(f"  [{split}] no data found, skipping")
                continue

            unified_splits[split] = unified
            del unified

        # ------------------------------------------------------------------
        # Phase 2: vertically stack all splits so rolling windows at split
        # boundaries have real history behind them.  Tag each row with its
        # split label so we can recover the boundaries after computing.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Vertically stacking splits for breadth / RS computation...")

        stacked = pl.concat([
            unified_splits[split].with_columns(pl.lit(split).alias("_split"))
            for split in SPLITS
            if split in unified_splits
        ]).sort("timestamp")

        # Free the per-split frames — stacked owns all the data now
        del unified_splits

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
            # Normalize rs_spy columns: fit winsorization and StandardScaler on 
            # train rows only, transform all splits, then tanh — consistent with 
            # the per-ticker normalization contract.  Rolling sums of z-scored 
            # values can accumulate well outside ±5 over longer windows (e.g., 30 bars),
            # so this step is necessary to keep the feature scale bounded.
            # ------------------------------------------------------------------
            rs_cols = [c for c in stacked.columns if "_rs_spy_" in c]
            if rs_cols:
                if self.logger:
                    self.logger.info("  Normalizing RS vs SPY features...")

                # 1. Identify training data
                train_mask = stacked["_split"] == "train"
                
                # 2. Extract train array and DROP NaNs (the cause of your RuntimeWarning)
                # The first WARMUP_BARS will be NaN due to rolling_sum
                train_df_raw = stacked.filter(train_mask).select(rs_cols).drop_nulls()
                train_arr = train_df_raw.to_numpy().astype(np.float64)

                if train_arr.shape[0] == 0:
                    self.logger.error("No valid training rows for RS normalization!")
                else:
                    # 3. Fit Winsorization bounds on valid training data
                    lo = np.quantile(train_arr, 0.01, axis=0)
                    hi = np.quantile(train_arr, 0.99, axis=0)
                    train_arr_win = np.clip(train_arr, lo, hi)

                    # 4. Fit Scaler on winsorized training data
                    rs_scaler = StandardScaler()
                    rs_scaler.fit(train_arr_win)

                    # 5. Transform the FULL dataset (all splits)
                    full_arr = stacked.select(rs_cols).to_numpy().astype(np.float64)
                    
                    # Apply winsorize -> scale -> tanh
                    # We use np.nan_to_num to handle the leading NaNs safely before tanh
                    full_arr_win = np.clip(full_arr, lo, hi)
                    transformed = rs_scaler.transform(np.nan_to_num(full_arr_win))
                    scaled_arr = np.tanh(transformed)

                    # 6. Write back to Polars
                    stacked = stacked.with_columns([
                        pl.Series(col, scaled_arr[:, i])
                        for i, col in enumerate(rs_cols)
                    ])

        # ------------------------------------------------------------------
        # Trim rolling warmup rows
        # Only the very first WARMUP_BARS rows of the full stacked frame
        # (i.e. the start of train) are nulled out by the rolling windows.
        # valid and test inherit real history from the splits before them,
        # so no rows are lost there.
        # ------------------------------------------------------------------

        stacked = stacked.slice(WARMUP_BARS)

        # ------------------------------------------------------------------
        # Phase 3: re-split on the _split tag and persist
        # ------------------------------------------------------------------

        for split in SPLITS:
            if self.logger:
                self.logger.info(f"  [{split}] writing unified_{split}.parquet...")

            unified = stacked.filter(pl.col("_split") == split).drop("_split")

            # Drop the first partial trading day from train.
            # The per-ticker pipeline includes a warmup day (Dec 31 2019) that
            # starts mid-session; strip it here so train begins on Jan 2 2020.
            if split == "train":
                dates = unified.select(pl.col("timestamp").dt.date().unique().sort()).to_series()
                if len(dates) > 1:
                    unified = unified.filter(pl.col("timestamp").dt.date() != dates[0])

            out_path = str(_data_dir / f"unified_{split}.parquet")
            save_to_parquet(unified, out_path, index=False)
            del unified

        del stacked

        if self.logger:
            self.logger.info("Unified splits complete.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def preprocess_and_save_tickers(self, batch_size: int = 8, check_for_missing_tickers: bool=False) -> None:
        """
        Run the full feature-engineering pipeline for all tickers,
        processing them in memory-bounded batches.

        Memory strategy
        ---------------
        77 GB dataset / ~70 tickers ≈ 1.1 GB per ticker on average.
        With `batch_size=8` and `max_workers=8`, peak RAM usage is
        roughly:

            batch_size × 1.1 GB (raw)   ← scan + collect for this batch
          + max_workers × 1.1 GB (processed copies in worker processes)

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
        alias_current = {
            a["current_ticker"] for a in self.ticker_aliases
        }
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
                        self.timestamp,
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
            self.logger.info("Per-ticker preprocessing complete. Starting cross-sectional normalisation...")

        self._cross_sectional_normalization()

        if self.logger:
            self.logger.info("Cross-sectional normalisation complete. Building unified splits...")

        self._build_unified_splits()

        if self.logger:
            self.logger.info("Preprocessing complete. Godspeed brotherman")


def main():
    data_preprocessor = DataPreprocessor(logger=Logger())
    data_preprocessor.preprocess_and_save_tickers()


if __name__ == "__main__":
    main()