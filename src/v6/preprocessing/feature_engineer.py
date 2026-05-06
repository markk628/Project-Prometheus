import os
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import polars as pl
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import DATA_DIR, CUTOFF_TIMESTAMP, REGIME_TICKERS, MASSIVE_APIKEY
from src.utils.database_manager import DatabaseManager
from src.utils.logger import Logger
from src.utils.utils import create_directory, save_to_parquet

from .constants import (
    GAP_SPLIT_TRADING_DAYS,
    GAP_SUSPICIOUS_TRADING_DAYS,
    NORMALIZATION_WINDOW,
)
from .gaps import _handle_gaps
from .normalization import _normalize_data, _rolling_zscore_normalize_vectorized
from .nyse_calendar import _build_nyse_valid_days
from .sectors import (
    ETF_SECTOR_OVERRIDES,
    N_SECTORS,
    SECTOR_NAMES,
    _sic_to_sector,
)
from .splits import _apply_split_adjustments
from .worker import _process_ticker


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataFeatureEngineer:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        timestamp: str = CUTOFF_TIMESTAMP,
        tickers: List[str] = REGIME_TICKERS,
        ticker_aliases: List[Dict[str, Any]] = None,  # built dynamically from Polygon API; pass manually to override
        database_manager: DatabaseManager = None,
        logger: Optional[Logger] = None,
        max_workers: Optional[int] = None,   # None → os.cpu_count()
        min_dollar_volume: float = 10_000_000,  # $10M avg daily dollar volume
        min_history_days: int = 504,             # ~2 years of trading days
        min_median_price: float = 5.0,           # penny-stock floor (lifetime-median close)
        api_delay: float = 0.15,                 # seconds between Polygon API calls
    ):
        self.data_dir         = data_dir
        self.tickers_dir      = Path(self.data_dir) / "preprocessed" / "v6" / "tickers"
        self.day_data_dir     = Path(self.data_dir) / "raw" / "day_data"
        self.files            = sorted(self.day_data_dir.rglob("*.csv.gz"))
        self.timestamp        = timestamp
        self.tickers          = tickers
        self.ticker_aliases   = ticker_aliases or []
        self.database_manager = database_manager
        self.logger           = logger
        self.max_workers      = max_workers
        self.min_dollar_volume = min_dollar_volume
        self.min_history_days  = min_history_days
        self.min_median_price  = min_median_price
        self.api_delay         = api_delay

        # Populated by _split_lifecycle_segments: segment-name -> source
        # Polygon symbol. For single-segment tickers the mapping is the
        # identity (AAPL -> AAPL). For lifecycle-split symbols (CIT,
        # etc.) multiple segment names resolve to the same source symbol.
        # Used by _apply_split_adjustments and _fetch_sector_labels to
        # key into source-symbol-indexed caches.
        self.segment_to_source: Dict[str, str] = {}

        # Populated by _split_lifecycle_segments for multi-segment tickers
        # only. Maps segment name -> last calendar date belonging to that
        # segment, used by _apply_split_adjustments to exclude splits
        # belonging to a later lifetime.
        self.segment_end_dates: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_all_data(self) -> pl.DataFrame:
        if self.logger:
            self.logger.info(f"Loading all daily data from {self.day_data_dir}...")

        all_data = (
            pl.scan_csv(self.files)
            .with_columns(
                # Truncate to date (midnight UTC) — for daily bars we don't
                # need intraday precision, and this ensures all code paths
                # (raw load vs gap-filling reconstruction) produce identical
                # timestamps that can be joined across tickers reliably.
                pl.from_epoch("window_start", time_unit="ns")
                .dt.date()
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC")
                .alias("timestamp")
            )
            .drop("window_start")
            .collect()
            .sort(["ticker", "timestamp"])
            .rechunk()
        )

        if self.logger:
            mem_gb = all_data.estimated_size("gb")
            n_tickers = all_data["ticker"].n_unique()
            self.logger.info(
                f"Loaded {len(all_data):,} rows, {n_tickers:,} tickers, "
                f"{mem_gb:.2f} GB in memory"
            )
        
        return all_data

    # ------------------------------------------------------------------
    # Lifecycle segmentation
    # ------------------------------------------------------------------

    def _split_lifecycle_segments(self, all_data: pl.DataFrame) -> pl.DataFrame:
        """
        Detect multi-day trading gaps that indicate a delist / relist
        lifecycle boundary and rewrite the ``ticker`` column so each
        continuous lifetime becomes its own "ticker" for the rest of
        the pipeline.

        Motivation
        ----------
        A symbol that trades, stops for months (bankruptcy + emergence,
        exchange transfer with long interruption, symbol reuse by an
        unrelated entity), and then resumes is not a single continuous
        price series. Treating it as one produces:
          * a forward-filled flat-line across the gap in ``_handle_gaps``
          * a spurious single-day "jump" on the relist day
          * blended cross-lifetime liquidity stats that can let one
            lifetime drag the other past or fail the filter

        By splitting here — before liquidity filtering, splits lookup,
        or feature engineering — every downstream step sees each
        lifetime as first-class, and the filled-in shared state
        ``self.segment_to_source`` routes source-symbol-keyed API
        lookups (splits, sectors) back to the original Polygon symbol.

        Detection
        ---------
        A gap is the number of **trading days** between two consecutive
        bars for the same ticker, measured against the NYSE calendar
        spanning the ticker's observed range. Gaps of
        ``GAP_SPLIT_TRADING_DAYS`` (10) or more trigger a segmentation;
        gaps in [5, 10) are logged as "suspicious" for manual review
        but NOT split. Shorter gaps are left for ``_handle_gaps`` to
        forward-fill within the segment, which is the defensible use
        of that function.

        Naming
        ------
        Multi-segment tickers get ``{ticker}.{N}`` (1-indexed, e.g.
        ``CIT.1``, ``CIT.2``). Single-segment tickers keep their raw
        symbol. This leaves the vast majority of the universe (AAPL,
        MSFT, …) unchanged, and the ``.`` suffix keeps the underscore-
        delimited column-parsing in the unified parquet / trainer
        working without modification.

        Returns
        -------
        pl.DataFrame
            The input frame with its ``ticker`` column rewritten to
            segment names. Row count is unchanged — splitting is a
            pure relabel.
        """
        if self.logger:
            self.logger.info(
                f"Detecting lifecycle segments "
                f"(split threshold: {GAP_SPLIT_TRADING_DAYS} trading days, "
                f"suspicious: {GAP_SUSPICIOUS_TRADING_DAYS}-{GAP_SPLIT_TRADING_DAYS-1})..."
            )

        # Build the NYSE trading-day calendar once, covering the full
        # span of all_data. Converting a timestamp to "trading-day index"
        # via a join lets us compute gaps in trading days (not calendar
        # days) without per-ticker calendar rebuilds.
        global_start = all_data["timestamp"].min().date()
        global_end   = all_data["timestamp"].max().date()
        nyse_days = _build_nyse_valid_days(global_start, global_end).with_row_index("trading_day_idx")

        # Attach the trading-day index to every bar. Join by date to
        # match _handle_gaps' convention.
        with_idx = (
            all_data
            .with_columns(pl.col("timestamp").dt.date().alias("date"))
            .join(nyse_days, on="date", how="left")
            .sort(["ticker", "trading_day_idx"])
        )

        # Per-ticker gap = (this bar's trading-day index) - (previous
        # bar's trading-day index) - 1. Values >= 1 indicate missing
        # trading days between consecutive observed bars.
        with_gaps = with_idx.with_columns(
            (
                pl.col("trading_day_idx")
                - pl.col("trading_day_idx").shift(1).over("ticker")
                - 1
            ).alias("gap_days")
        )

        # Count suspicious-bucket gaps (for logging only)
        suspicious_count = with_gaps.filter(
            (pl.col("gap_days") >= GAP_SUSPICIOUS_TRADING_DAYS)
            & (pl.col("gap_days") < GAP_SPLIT_TRADING_DAYS)
        ).height

        if suspicious_count > 0 and self.logger:
            # Per-ticker rollup of suspicious gaps so the analyst can
            # eyeball whether the 5-10 band is empty or noisy.
            suspicious_rollup = (
                with_gaps
                .filter(
                    (pl.col("gap_days") >= GAP_SUSPICIOUS_TRADING_DAYS)
                    & (pl.col("gap_days") < GAP_SPLIT_TRADING_DAYS)
                )
                .group_by("ticker")
                .agg([
                    pl.len().alias("n_suspicious_gaps"),
                    pl.col("gap_days").max().alias("max_gap"),
                ])
                .sort("n_suspicious_gaps", descending=True)
            )
            self.logger.warning(
                f"Found {suspicious_count} 'suspicious' gap(s) in "
                f"[{GAP_SUSPICIOUS_TRADING_DAYS}, {GAP_SPLIT_TRADING_DAYS}) "
                f"trading-day band across {suspicious_rollup.height} tickers "
                f"— NOT splitting these (inspect manually if unexpected):"
            )
            for row in suspicious_rollup.head(20).iter_rows(named=True):
                self.logger.warning(
                    f"  {row['ticker']:<10} | "
                    f"{row['n_suspicious_gaps']} gap(s), max {row['max_gap']}d"
                )

        # Mark segment-start rows: a bar starts a new segment if it's
        # either the first bar for its ticker OR follows a split-worthy
        # gap. `cum_sum().over("ticker")` then assigns a monotonically
        # increasing segment index within each ticker.
        with_segs = with_gaps.with_columns(
            (
                pl.col("gap_days").is_null()                          # first bar
                | (pl.col("gap_days") >= GAP_SPLIT_TRADING_DAYS)      # post-gap
            )
            .cast(pl.Int64)
            .cum_sum()
            .over("ticker")
            .alias("segment_idx")
        )

        # Build one row per (ticker, segment_idx) with first/last date.
        # Single-segment tickers get segment_idx == 1. Multi-segment
        # tickers get 1, 2, 3...
        segment_ranges = (
            with_segs
            .group_by(["ticker", "segment_idx"])
            .agg([
                pl.col("date").min().alias("seg_start"),
                pl.col("date").max().alias("seg_end"),
                pl.len().alias("n_bars"),
            ])
            .sort(["ticker", "segment_idx"])
        )

        # Also compute how many segments each ticker has, so we only
        # rename tickers that actually got split. Most tickers stay
        # single-segment and keep their raw symbol.
        segment_counts = (
            segment_ranges
            .group_by("ticker")
            .agg(pl.len().alias("n_segments"))
        )

        segment_ranges = segment_ranges.join(segment_counts, on="ticker", how="left")

        # Derive the segment name per row.
        # Single-segment (n_segments == 1): keep raw ticker symbol.
        # Multi-segment:                     {ticker}.{segment_idx}
        #
        # The .N suffix (e.g. CIT.1, CIT.2) is chosen over a date-range
        # suffix because the unified parquet flattens everything to
        # {ticker}_{feature} columns, and downstream code in the trainer
        # parses the ticker via a single split on "_". A date-based name
        # like CIT_2003_2009 would break that parse (every segment would
        # alias back to "CIT" with a feature name of "2003_2009_<...>").
        # The "." suffix can't collide with Polygon class-share symbols
        # like BRK.B because those are source symbols loaded before
        # segmentation, and "." is always followed by an integer here.
        segment_ranges = segment_ranges.with_columns(
            pl.when(pl.col("n_segments") == 1)
            .then(pl.col("ticker"))
            .otherwise(
                pl.col("ticker") + pl.lit(".") + pl.col("segment_idx").cast(pl.Utf8)
            )
            .alias("segment_name")
        )

        # Log the split decisions for visibility.
        split_tickers = segment_ranges.filter(pl.col("n_segments") > 1)
        if split_tickers.height > 0 and self.logger:
            # Unique source symbols that got split, with their segments.
            source_symbols = split_tickers["ticker"].unique().to_list()
            self.logger.warning(
                f"Lifecycle-splitting {len(source_symbols)} ticker(s) into "
                f"{split_tickers.height} segment(s) total:"
            )
            for src in sorted(source_symbols):
                segs = split_tickers.filter(pl.col("ticker") == src).sort("segment_idx")
                self.logger.warning(f"  {src}:")
                for row in segs.iter_rows(named=True):
                    self.logger.warning(
                        f"    {row['segment_name']:<20} "
                        f"{row['seg_start']} → {row['seg_end']} "
                        f"({row['n_bars']:,} bars)"
                    )
        elif self.logger:
            self.logger.info("No tickers required lifecycle splitting.")

        # Build the segment_to_source mapping (used by splits / sector
        # lookups downstream).
        self.segment_to_source = {
            row["segment_name"]: row["ticker"]
            for row in segment_ranges.iter_rows(named=True)
        }

        # Build the segment_end_date mapping. Only populated for
        # multi-segment tickers — single-segment tickers leave the mapping
        # empty so _apply_split_adjustments receives segment_end_date=None
        # and behaves identically to the pre-segmentation code path. For
        # multi-segment tickers, the end date is used to filter out splits
        # that belong to a LATER segment (economically a different entity)
        # before the join_asof.
        self.segment_end_dates: Dict[str, Any] = {
            row["segment_name"]: row["seg_end"]
            for row in segment_ranges.iter_rows(named=True)
            if row["n_segments"] > 1
        }

        # Attach segment_name back onto the data frame. Join on
        # (ticker, segment_idx) — which is the natural key — then
        # replace the ticker column.
        relabelled = (
            with_segs
            .join(
                segment_ranges.select(["ticker", "segment_idx", "segment_name"]),
                on=["ticker", "segment_idx"],
                how="left",
            )
            .drop(["ticker", "segment_idx", "gap_days", "trading_day_idx", "date"])
            .rename({"segment_name": "ticker"})
            .sort(["ticker", "timestamp"])
        )

        return relabelled

    # ------------------------------------------------------------------
    # Ticker filtering
    # ------------------------------------------------------------------

    def _filter_tickers_by_liquidity(self, all_data: pl.DataFrame) -> List[str]:
        """
        Filter tickers by:
          1. Average daily dollar volume >= min_dollar_volume
          2. Total trading days >= min_history_days
          3. Lifetime-median close price >= min_median_price

        The price floor is a separate gate from dollar volume because
        the two measure different things: dollar volume is "is there
        enough trading activity to fill orders?" while the price floor
        is "is the per-share price meaningful enough that fixed-tick
        frictions don't dominate?". A penny stock pumping at $0.50
        on 50M shares/day passes the $10M dollar-volume floor while
        still being uninvestable — the env's spread/slippage assumptions
        (calibrated for liquid mid/large-cap names) silently break down
        below ~$5/share. Median (not mean / not min) is robust to brief
        pumps and brief dips while reflecting where the ticker "lived"
        most of its history.

        Operates on an in-memory DataFrame (already loaded by the caller)
        rather than scanning CSV files from disk. Regime tickers from
        self.tickers are always included regardless of any filter.

        Returns a sorted list of ticker symbols that pass all filters.
        """
        if self.logger:
            self.logger.info(
                f"Filtering tickers: min_dollar_volume=${self.min_dollar_volume:,.0f}, "
                f"min_history_days={self.min_history_days}, "
                f"min_median_price=${self.min_median_price:.2f}..."
            )

        ticker_stats = (
            all_data
            .with_columns(
                (pl.col("close") * pl.col("volume")).alias("dollar_volume")
            )
            .group_by("ticker")
            .agg([
                pl.col("dollar_volume").mean().alias("avg_dollar_volume"),
                pl.col("close").median().alias("median_close"),
                pl.len().alias("trading_days"),
            ])
        )

        # Apply thresholds
        qualified = ticker_stats.filter(
            (pl.col("avg_dollar_volume") >= self.min_dollar_volume) &
            (pl.col("trading_days")      >= self.min_history_days) &
            (pl.col("median_close")      >= self.min_median_price)
        )

        filtered_tickers = sorted(qualified["ticker"].to_list())

        if self.logger:
            # Per-reason breakdown so the next preprocessing run reports
            # which gate is dropping the most tickers — useful when
            # tuning thresholds.
            n_total = ticker_stats.height
            n_pass_volume = ticker_stats.filter(
                pl.col("avg_dollar_volume") >= self.min_dollar_volume
            ).height
            n_pass_volume_history = ticker_stats.filter(
                (pl.col("avg_dollar_volume") >= self.min_dollar_volume) &
                (pl.col("trading_days") >= self.min_history_days)
            ).height
            n_pass_all = qualified.height
            self.logger.info(
                f"Liquidity filter: {n_total} total → "
                f"{n_pass_volume} passed dollar-volume → "
                f"{n_pass_volume_history} also passed history → "
                f"{n_pass_all} also passed price floor"
            )

        # Ensure regime tickers are always included even if they don't pass
        # the liquidity filter (they provide market context, not trading targets)
        regime_tickers = set(self.tickers)
        unqualified_regime_tickers = [t for t in regime_tickers if t not in filtered_tickers]
        filtered_tickers.extend(unqualified_regime_tickers)
        filtered_tickers = sorted(set(filtered_tickers))

        if self.logger:
            self.logger.info(f"Added {len(unqualified_regime_tickers)} ETF tickers: {unqualified_regime_tickers}")
            self.logger.info(
                f"Processing {len(filtered_tickers)} tickers total "
                f"({len(filtered_tickers) - len(regime_tickers)} tradable + "
                f"{len(regime_tickers)} regime)"
            )

        return filtered_tickers

    # ------------------------------------------------------------------
    # Splits fetching
    # ------------------------------------------------------------------

    def _fetch_splits(self, tickers: List[str]) -> None:
        """
        Fetch stock split data from the Polygon API for all tickers in bulk.

        Uses ticker_any_of to batch tickers into single API calls (up to
        ~100 tickers per call to stay within URL length limits). Results
        are cached in a single splits.parquet file. On subsequent runs,
        only tickers not already in the cache are fetched.

        Cache file: <data_dir>/raw/splits/splits.parquet
        Columns: ticker, execution_date, adjustment_type, split_from,
                 split_to, effective_ratio, historical_adjustment_factor
        """
        from massive import RESTClient

        cache_path = Path(self.data_dir) / "raw" / "splits" / "splits.parquet"
        create_directory(cache_path.parent)

        # Load existing cache
        if cache_path.exists():
            cached = pl.read_parquet(cache_path)
            cached_tickers = set(cached["ticker"].to_list())
        else:
            cached = pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "execution_date": pl.Series([], dtype=pl.Date),
                "adjustment_type": pl.Series([], dtype=pl.Utf8),
                "split_from": pl.Series([], dtype=pl.Float32),
                "split_to": pl.Series([], dtype=pl.Float32),
                "effective_ratio": pl.Series([], dtype=pl.Float32),
                "historical_adjustment_factor": pl.Series([], dtype=pl.Float64),
            })
            cached_tickers = set()

        # Also need splits for alias previous tickers
        symbols_needed = set(tickers)
        for alias in self.ticker_aliases:
            if alias["current_ticker"] in symbols_needed:
                for t in alias["previous_tickers"]:
                    symbols_needed.add(t["symbol"])

        # Only fetch tickers not already cached
        to_fetch = sorted(symbols_needed - cached_tickers)

        if not to_fetch:
            if self.logger:
                self.logger.info(
                    f"Splits: all {len(symbols_needed)} tickers already cached "
                    f"({len(cached)} splits on disk)"
                )
            return

        if self.logger:
            self.logger.info(
                f"Fetching splits for {len(to_fetch)} tickers "
                f"({len(cached_tickers)} already cached)..."
            )

        client = RESTClient(MASSIVE_APIKEY)
        all_rows = []

        # Batch tickers — ~100 per API call to stay within URL limits
        BATCH_SIZE = 100
        for batch_start in range(0, len(to_fetch), BATCH_SIZE):
            batch = to_fetch[batch_start:batch_start + BATCH_SIZE]

            try:
                for s in client.list_stocks_splits(
                    ticker_any_of=batch,
                    limit=5000,
                    sort="execution_date.asc",
                ):
                    all_rows.append({
                        "ticker": s.ticker,
                        "execution_date": s.execution_date,
                        "adjustment_type": s.adjustment_type,
                        "split_from": s.split_from,
                        "split_to": s.split_to,
                        "effective_ratio": (s.split_to / s.split_from) if s.split_from else None,
                        "historical_adjustment_factor": s.historical_adjustment_factor,
                    })

            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Failed to fetch splits for batch "
                        f"{batch_start}-{batch_start + len(batch)}: {e}"
                    )

            if self.logger:
                self.logger.info(
                    f"  Splits batch {batch_start + len(batch)}/{len(to_fetch)} done"
                )

            time.sleep(self.api_delay)

        # Merge with cache and save
        if all_rows:
            new_df = pl.DataFrame(all_rows).with_columns([
                pl.col("execution_date").str.strptime(pl.Date, strict=False),
                pl.col("split_from").cast(pl.Float32),
                pl.col("split_to").cast(pl.Float32),
                pl.col("effective_ratio").cast(pl.Float32),
                pl.col("historical_adjustment_factor").cast(pl.Float64),
            ])
            cached = pl.concat([cached, new_df]).unique()

        # Also mark tickers that had no splits so we don't re-fetch them.
        # We do this by noting which fetched tickers got results.
        tickers_with_splits = set(r["ticker"] for r in all_rows) if all_rows else set()
        tickers_without_splits = set(to_fetch) - tickers_with_splits

        # Add placeholder rows for tickers with no splits (factor=1.0)
        # so they appear in the cache and don't get re-fetched
        if tickers_without_splits:
            placeholder = pl.DataFrame({
                "ticker": sorted(tickers_without_splits),
                "execution_date": [None] * len(tickers_without_splits),
                "adjustment_type": ["none"] * len(tickers_without_splits),
                "split_from": [1.0] * len(tickers_without_splits),
                "split_to": [1.0] * len(tickers_without_splits),
                "effective_ratio": [1.0] * len(tickers_without_splits),
                "historical_adjustment_factor": [1.0] * len(tickers_without_splits),
            }).with_columns([
                pl.col("execution_date").cast(pl.Date),
                pl.col("split_from").cast(pl.Float32),
                pl.col("split_to").cast(pl.Float32),
                pl.col("effective_ratio").cast(pl.Float32),
            ])
            cached = pl.concat([cached, placeholder])

        save_to_parquet(cached, str(cache_path), index=False)

        if self.logger:
            self.logger.info(
                f"Splits fetch complete: {len(all_rows)} splits found, "
                f"{len(tickers_without_splits)} tickers with no splits, "
                f"{len(cached)} total rows cached"
            )

        # Now that splits are on disk, rebuild ticker_aliases with suspect
        # filtering. _detect_suspect_aliases needs both splits.parquet and
        # ticker_events.parquet to exist, which is true at this point.
        # The rebuild drops aliases where the previous_ticker has splits
        # recorded AFTER the rename date (indicates ticker reuse or
        # Polygon data inconsistency).
        self.ticker_aliases = self._build_ticker_aliases()

    # ------------------------------------------------------------------
    # Sector classification
    # ------------------------------------------------------------------

    def _fetch_sector_labels(self, tickers: List[str]) -> None:
        """
        Fetch SIC codes from Polygon and map to trading-relevant sectors.

        Results are cached in a parquet file so subsequent runs only fetch
        tickers that aren't already in the cache. ETFs and known edge cases
        use manual overrides from ETF_SECTOR_OVERRIDES.

        Cache file: <data_dir>/raw/sectors/ticker_sectors.parquet
        Columns: ticker, sic_code, sector_id, sector_name, ticker_type, ticker_name
        """
        from massive import RESTClient

        cache_path = Path(self.data_dir) / "raw" / "sectors" / "ticker_sectors.parquet"
        create_directory(cache_path.parent)

        # Load existing cache
        if cache_path.exists():
            cached = pl.read_parquet(cache_path)
            # Handle schema migration — add new columns if missing
            if "ticker_type" not in cached.columns:
                cached = cached.with_columns([
                    pl.lit("").alias("ticker_type"),
                    pl.lit("").alias("ticker_name"),
                ])
            cached_tickers = set(cached["ticker"].to_list())
        else:
            cached = pl.DataFrame({
                "ticker": pl.Series([], dtype=pl.Utf8),
                "sic_code": pl.Series([], dtype=pl.Utf8),
                "sector_id": pl.Series([], dtype=pl.Int32),
                "sector_name": pl.Series([], dtype=pl.Utf8),
                "ticker_type": pl.Series([], dtype=pl.Utf8),
                "ticker_name": pl.Series([], dtype=pl.Utf8),
            })
            cached_tickers = set()

        # Figure out which tickers need fetching
        tickers_to_fetch = [t for t in tickers if t not in cached_tickers]

        # Apply manual overrides first (ETFs, known edge cases)
        override_rows = []
        remaining = []
        for t in tickers_to_fetch:
            if t in ETF_SECTOR_OVERRIDES:
                sid, sname = ETF_SECTOR_OVERRIDES[t]
                override_rows.append({
                    "ticker": t, "sic_code": "ETF",
                    "sector_id": sid, "sector_name": sname,
                    "ticker_type": "ETF", "ticker_name": t,
                })
            else:
                remaining.append(t)

        if self.logger:
            self.logger.info(
                f"Sector labels: {len(cached_tickers)} cached, "
                f"{len(override_rows)} ETF overrides, "
                f"{len(remaining)} to fetch from API..."
            )

        # Fetch from API
        api_rows = []
        failed_tickers = []
        if remaining:
            client = RESTClient(MASSIVE_APIKEY)

            for i, ticker in enumerate(remaining):
                try:
                    details = client.get_ticker_details(ticker)
                    sic_code = getattr(details, "sic_code", None) or ""
                    ticker_type = getattr(details, "type", None) or ""
                    ticker_name = getattr(details, "name", None) or ""
                    sector_id, sector_name = _sic_to_sector(sic_code)

                    api_rows.append({
                        "ticker": ticker,
                        "sic_code": str(sic_code),
                        "sector_id": sector_id,
                        "sector_name": sector_name,
                        "ticker_type": str(ticker_type),
                        "ticker_name": str(ticker_name),
                    })

                except Exception as e:
                    # get_ticker_details fails for most delisted tickers.
                    # Record as failed — we'll try list_tickers fallback next.
                    failed_tickers.append(ticker)

                if self.logger and (i + 1) % 500 == 0:
                    self.logger.info(f"  Fetched {i + 1}/{len(remaining)} ticker details...")

                time.sleep(self.api_delay)

            # ----------------------------------------------------------
            # Fallback: list_tickers for delisted tickers.
            # get_ticker_details often fails for delisted tickers but
            # list_tickers(active=false) returns them with type and name.
            # We won't get SIC codes (so sector stays "other") but at
            # least we can filter correctly by type.
            # ----------------------------------------------------------
            if failed_tickers:
                if self.logger:
                    self.logger.info(
                        f"Trying list_tickers fallback for {len(failed_tickers)} failed lookups..."
                    )

                recovered = self._fallback_list_tickers(client, failed_tickers)

                if self.logger:
                    self.logger.info(
                        f"  Recovered {len(recovered)}/{len(failed_tickers)} via list_tickers"
                    )

                for ticker in failed_tickers:
                    if ticker in recovered:
                        info = recovered[ticker]
                        api_rows.append({
                            "ticker": ticker,
                            "sic_code": "",
                            "sector_id": 11,
                            "sector_name": "other",
                            "ticker_type": info["type"],
                            "ticker_name": info["name"],
                        })
                    else:
                        # Truly not found via either endpoint — mark with
                        # UNKNOWN type so the filter excludes them. These
                        # tickers have data in the flat files but no metadata
                        # in Polygon's reference endpoints, so we can't verify
                        # they're tradable common stock.
                        api_rows.append({
                            "ticker": ticker,
                            "sic_code": "",
                            "sector_id": 11,
                            "sector_name": "other",
                            "ticker_type": "UNKNOWN",
                            "ticker_name": "",
                        })

                unrecovered = sorted([t for t in failed_tickers if t not in recovered])
                if self.logger and unrecovered:
                    self.logger.warning(
                        f"Could not recover {len(unrecovered)} tickers via either endpoint "
                        f"(marked UNKNOWN, will be excluded by type filter): {unrecovered}"
                    )

        # Merge everything and save
        new_rows = override_rows + api_rows
        if new_rows:
            new_df = pl.DataFrame(new_rows).with_columns([
                pl.col("sector_id").cast(pl.Int32),
            ])
            cached = pl.concat([cached, new_df])
            save_to_parquet(cached, str(cache_path), index=False)

            if self.logger:
                sector_counts = cached.group_by("sector_name").len().sort("len", descending=True)
                self.logger.info(f"Sector distribution ({len(cached)} tickers):")
                for row in sector_counts.iter_rows(named=True):
                    self.logger.info(f"  {row['sector_name']}: {row['len']}")

    def _fallback_list_tickers(self, client, tickers: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fallback for tickers that get_ticker_details couldn't find.

        Uses list_tickers(ticker=<symbol>, active=false) as a per-ticker
        lookup for delisted stocks. This endpoint returns type and name
        for delisted tickers that aren't available via the details endpoint.

        We don't get SIC codes here (so sector stays "other") but at
        least we can filter correctly by type.

        Returns {ticker: {"type": str, "name": str}} for tickers found.
        """
        recovered = {}

        for i, ticker in enumerate(tickers):
            try:
                # list_tickers returns an iterator; for exact symbol lookup
                # we take the first result (should be exactly one match)
                results = list(client.list_tickers(
                    ticker=ticker,
                    market="stocks",
                    active="false",
                    limit=1,
                ))

                if results:
                    t = results[0]
                    recovered[ticker] = {
                        "type": getattr(t, "type", None) or "",
                        "name": getattr(t, "name", None) or "",
                    }

            except Exception as e:
                # Truly not found — skip
                pass

            if self.logger and (i + 1) % 500 == 0:
                self.logger.info(f"  Fallback {i + 1}/{len(tickers)}...")

            time.sleep(self.api_delay)

        return recovered

    def _filter_by_ticker_details(self, tickers: List[str]) -> List[str]:
        """
        Filter tickers using type and name from cached Polygon ticker details.

        Type filter: only keep tradable security types (CS, ADRC, OS).
        ETFs are kept only if they're in the regime ticker list (a
        curated list the user maintains — responsibility for not including
        leveraged/inverse products falls on the config, not the filter).
        Everything else (warrants, rights, bonds, ETNs, structured
        products, units, single-security ETFs, etc.) is excluded.

        This replaces both the old naming-pattern heuristics (W/U/R suffixes)
        and the hardcoded leveraged ticker list with objective API data.
        """
        cache_path = Path(self.data_dir) / "raw" / "sectors" / "ticker_sectors.parquet"
        if not cache_path.exists():
            if self.logger:
                self.logger.warning("No ticker_sectors.parquet — skipping type filter")
            return tickers

        df = pl.read_parquet(cache_path)
        if "ticker_type" not in df.columns:
            return tickers

        type_map = dict(zip(df["ticker"].to_list(), df["ticker_type"].to_list()))

        # Types that are tradable
        TRADABLE_TYPES = {"CS", "ADRC", "OS"}

        # Regime tickers are allowed to be ETFs (curated list — user is
        # responsible for not including leveraged/inverse products)
        regime_set = set(self.tickers)

        filtered = []
        excluded_type = []

        for t in tickers:
            ticker_type = type_map.get(t, "")

            if t in regime_set:
                # Regime tickers pass through regardless of type
                filtered.append(t)
            elif ticker_type in TRADABLE_TYPES:
                # Common stocks, ADRs, ordinary shares — keep
                filtered.append(t)
            else:
                # Everything else: ETF, ETN, ETV, SP, WARRANT, RIGHT,
                # BOND, PFD, UNIT, LT, FUND, BASKET, ETS, UNKNOWN, ""
                # UNKNOWN = couldn't verify via either endpoint
                # "" = truly missing (shouldn't happen after fallback)
                excluded_type.append((t, ticker_type or "EMPTY"))

        if self.logger:
            if excluded_type:
                type_counts = {}
                for _, typ in excluded_type:
                    type_counts[typ] = type_counts.get(typ, 0) + 1
                type_summary = ", ".join(f"{typ}: {cnt}" for typ, cnt in sorted(type_counts.items()))
                self.logger.info(
                    f"Type filter: removed {len(excluded_type)} non-tradable tickers ({type_summary})"
                )
            self.logger.info(f"{len(filtered)} tickers remaining after type filter")

        return filtered

    def _load_sector_map(self) -> Dict[str, str]:
        """
        Load the cached ticker→sector mapping as a dict.

        Returns {ticker: sector_name}. Used by _cross_sectional_normalization
        for intra-sector z-scores.
        """
        cache_path = Path(self.data_dir) / "raw" / "sectors" / "ticker_sectors.parquet"
        if not cache_path.exists():
            if self.logger:
                self.logger.warning("No ticker_sectors.parquet found — all tickers will be 'other'")
            return {}

        df = pl.read_parquet(cache_path)
        return dict(zip(df["ticker"].to_list(), df["sector_name"].to_list()))

    # ------------------------------------------------------------------
    # Ticker symbol changes
    # ------------------------------------------------------------------

    def _fetch_ticker_events(self, tickers: List[str]) -> None:
        """
        Fetch ticker change events from Polygon and cache as parquet.

        For each ticker, queries the ticker events endpoint to discover
        historical symbol changes (e.g., FB → META). Results are cached
        in ticker_events.parquet so subsequent runs skip already-fetched
        tickers.

        Only records tickers that actually had a name change — tickers
        with a single event (IPO only) or NOT_FOUND responses are tracked
        in a separate "checked" set so they aren't re-fetched.

        Cache file: <data_dir>/raw/ticker_events/ticker_events.parquet
        Columns: current_ticker, previous_ticker, change_date
        """
        from massive import RESTClient

        cache_path = Path(self.data_dir) / "raw" / "ticker_events" / "ticker_events.parquet"
        checked_path = Path(self.data_dir) / "raw" / "ticker_events" / "ticker_events_checked.parquet"
        create_directory(cache_path.parent)

        # Load existing cache
        if cache_path.exists():
            cached = pl.read_parquet(cache_path)
            # Tickers we already know about (both current and previous names)
            known_tickers = set(cached["current_ticker"].to_list()) | set(cached["previous_ticker"].to_list())
        else:
            cached = pl.DataFrame({
                "current_ticker": pl.Series([], dtype=pl.Utf8),
                "previous_ticker": pl.Series([], dtype=pl.Utf8),
                "change_date": pl.Series([], dtype=pl.Date),
            })
            known_tickers = set()

        # Load the "already checked, no changes found" set
        if checked_path.exists():
            checked_set = set(pl.read_parquet(checked_path)["ticker"].to_list())
        else:
            checked_set = set()

        # Only fetch tickers we haven't checked yet
        to_fetch = [t for t in tickers if t not in known_tickers and t not in checked_set]

        if not to_fetch:
            if self.logger:
                self.logger.info(
                    f"Ticker events: all {len(tickers)} tickers already checked "
                    f"({len(known_tickers)} with changes, {len(checked_set)} without)"
                )
            return

        if self.logger:
            self.logger.info(
                f"Fetching ticker events for {len(to_fetch)} tickers "
                f"({len(known_tickers)} known, {len(checked_set)} checked)..."
            )

        client = RESTClient(MASSIVE_APIKEY)
        new_rows = []
        newly_checked = []

        for i, ticker in enumerate(to_fetch):
            try:
                response = client.get_ticker_events(ticker)
                events_list = getattr(response, "events", None) or []

                # Extract ticker_change events. The endpoint is experimental —
                # all events are typed "ticker_change" even when they're really
                # IPO events, so we just collect every entry and sort them.
                ticker_changes = []
                for event in events_list:
                    if event.get("type") == "ticker_change":
                        tc = event.get("ticker_change") or {}
                        tc_ticker = tc.get("ticker")
                        if tc_ticker:
                            ticker_changes.append({
                                "date": event.get("date"),
                                "ticker": tc_ticker,
                            })

                if len(ticker_changes) <= 1:
                    # No rename history (just IPO or single event)
                    newly_checked.append(ticker)
                else:
                    # Sort by date descending — most recent first
                    ticker_changes.sort(key=lambda x: x["date"], reverse=True)

                    # The most recent ticker_change is the current name
                    current_name = ticker_changes[0]["ticker"]

                    # Each earlier event is a previous name; the change_date
                    # is when it stopped being that name (= date of next event)
                    for j in range(1, len(ticker_changes)):
                        prev_name = ticker_changes[j]["ticker"]
                        changed_on = ticker_changes[j - 1]["date"]

                        if prev_name and current_name and prev_name != current_name:
                            new_rows.append({
                                "current_ticker": current_name,
                                "previous_ticker": prev_name,
                                "change_date": changed_on,
                            })

            except Exception as e:
                # NOT_FOUND or API error — mark as checked so we don't retry
                newly_checked.append(ticker)

            if self.logger and (i + 1) % 500 == 0:
                self.logger.info(f"  Fetched {i + 1}/{len(to_fetch)} ticker events...")

            time.sleep(self.api_delay)

        # Save new changes
        if new_rows:
            new_df = pl.DataFrame(new_rows).with_columns(
                pl.col("change_date").str.strptime(pl.Date, strict=False)
            )
            cached = pl.concat([cached, new_df]).unique()
            save_to_parquet(cached, str(cache_path), index=False)

            if self.logger:
                for row in new_df.iter_rows(named=True):
                    self.logger.info(
                        f"  Found rename: {row['previous_ticker']} → "
                        f"{row['current_ticker']} on {row['change_date']}"
                    )

        # Save checked set
        if newly_checked:
            all_checked = sorted(checked_set | set(newly_checked))
            checked_df = pl.DataFrame({"ticker": all_checked})
            save_to_parquet(checked_df, str(checked_path), index=False)

        if self.logger:
            self.logger.info(
                f"Ticker events complete: {len(new_rows)} renames found, "
                f"{len(newly_checked)} no changes, "
                f"{len(known_tickers)} previously known"
            )

    def _detect_suspect_aliases(self) -> set:
        """
        Identify previous_ticker symbols that have splits recorded AFTER
        their change_date. This indicates either ticker reuse (a different
        company took over the symbol) or data inconsistency on Polygon's
        side. Either way, stitching the previous ticker's data into the
        current ticker would corrupt the history.

        Returns a set of previous_ticker symbols to exclude from alias
        resolution. The current tickers still keep their post-rename data
        intact — only the pre-rename stitching is skipped.
        """
        events_path = Path(self.data_dir) / "raw" / "ticker_events" / "ticker_events.parquet"
        splits_path = Path(self.data_dir) / "raw" / "splits" / "splits.parquet"

        if not events_path.exists() or not splits_path.exists():
            return set()

        events = pl.read_parquet(events_path)
        splits = pl.read_parquet(splits_path)

        if events.is_empty() or splits.is_empty():
            return set()

        # Join splits to events on previous_ticker. Any split whose
        # execution_date is AFTER the change_date is suspect.
        joined = splits.join(
            events,
            left_on="ticker",
            right_on="previous_ticker",
            how="inner",
        )

        suspect = joined.filter(
            pl.col("execution_date").is_not_null() &
            (pl.col("execution_date") > pl.col("change_date"))
        )

        suspect_prev_tickers = set(suspect["ticker"].to_list())

        if self.logger and suspect_prev_tickers:
            pairs = (
                suspect
                .select(["ticker", "current_ticker"])
                .unique()
                .iter_rows(named=True)
            )
            pair_strs = [f"{r['ticker']}→{r['current_ticker']}" for r in pairs]
            self.logger.warning(
                f"Detected {len(suspect_prev_tickers)} suspect aliases "
                f"(previous ticker has splits post-rename, likely ticker reuse): "
                f"{pair_strs}"
            )

        return suspect_prev_tickers

    def _build_ticker_aliases(self) -> List[Dict[str, Any]]:
        """
        Build the ticker_aliases structure from the cached ticker_events parquet.

        Excludes aliases where the previous_ticker appears to have been
        reused by a different company after the rename (detected by
        _detect_suspect_aliases). The current tickers are still preserved,
        they just won't have pre-rename data stitched in.

        Returns a list in the same format as the old hardcoded TICKER_ALIASES:
        [
            {
                "current_ticker": "META",
                "previous_tickers": [
                    {"symbol": "FB", "change_date": datetime(2022, 6, 9, ...)}
                ]
            },
            ...
        ]
        """
        from datetime import timezone

        cache_path = Path(self.data_dir) / "raw" / "ticker_events" / "ticker_events.parquet"
        if not cache_path.exists():
            if self.logger:
                self.logger.info("No ticker_events.parquet found — no aliases to resolve")
            return []

        df = pl.read_parquet(cache_path)
        if df.is_empty():
            return []

        # Filter out aliases whose previous_ticker has suspect split history
        suspect = self._detect_suspect_aliases()
        if suspect:
            df = df.filter(~pl.col("previous_ticker").is_in(list(suspect)))

        # Group by current_ticker
        aliases = []
        for current_ticker, group in df.group_by("current_ticker"):
            current = current_ticker[0] if isinstance(current_ticker, tuple) else current_ticker
            previous_tickers = []

            for row in group.iter_rows(named=True):
                change_date = row["change_date"]
                # Convert date to datetime with UTC timezone
                from datetime import datetime
                dt = datetime(change_date.year, change_date.month, change_date.day, tzinfo=timezone.utc)

                previous_tickers.append({
                    "symbol": row["previous_ticker"],
                    "change_date": dt,
                })

            # Sort by change_date ascending (oldest rename first)
            previous_tickers.sort(key=lambda x: x["change_date"])

            aliases.append({
                "current_ticker": current,
                "previous_tickers": previous_tickers,
            })

        if self.logger:
            self.logger.info(f"Built {len(aliases)} ticker aliases from cached events")

        return aliases

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

    def _resolve_aliases(self, ticker_dfs: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        """
        Stitch historical alias tickers into their current symbol.

        E.g., FB rows before 2022-06-09 get merged into META's DataFrame.
        The alias ticker's DataFrame is consumed (popped) and should not
        appear in the output.
        """
        for alias in self.ticker_aliases:
            current_ticker   = alias["current_ticker"]
            previous_tickers = alias["previous_tickers"]

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

    def _cross_sectional_normalization(self) -> Dict[str, pl.DataFrame]:
        """
        Second-pass normalisation that operates across all tickers simultaneously.

        After `feature_engineer_and_save_tickers` writes per-ticker parquet files, this
        method loads them all, joins on timestamp, and for each daily bar computes:

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

        Leakage safety
        --------------
        Per-ticker rolling z-scores use strictly causal windows (bars
        [t-window, t-1] only). Row-wise cross-sectional stats are computed
        within each timestamp using only same-bar data from other tickers,
        so they are leak-free by construction.

        Output
        ------
        Overwrites the existing per-ticker parquet files in-place.
        """

        # ------------------------------------------------------------------
        # Sector mapping — loaded from cached ticker_sectors.parquet
        # (built by _fetch_sector_labels from Polygon SIC codes)
        # ------------------------------------------------------------------
        SECTOR_MAP: Dict[str, str] = self._load_sector_map()

        # Features that are left un-normalised by _normalize_data and whose
        # ticker-specific scale differences make cross-sectional z-scoring
        # strictly more informative than per-ticker normalization.
        CS_FEATURES = [
            "log_return_5",
            "log_return_20",
            "log_return_60",
            "overnight_gap",             # cross-sectional: "this stock gapped up 3% while market was flat"
            "intraday_return",           # cross-sectional: "this stock rallied intraday while peers faded"
            "price_vwap_distance",
            "volume_price_corr_20",      # raw rolling corr: bounded [-1,1] but can have ±inf edge cases
            "adx_14",                    # already in [0,1] (divided by 100), but cross-sectional
                                         # z-score captures relative trend strength across tickers
        ]

        # ------------------------------------------------------------------
        # 1. Load all per-ticker parquet files
        # ------------------------------------------------------------------
        if self.logger:
            self.logger.info("Cross-sectional normalisation: loading parquet files...")

        # ticker_dfs[ticker] = single full-timeline DataFrame (warmup trim deferred
        # to the very end of _build_unified so every normalization pass
        # has the full warmup history available).
        ticker_dfs: Dict[str, pl.DataFrame] = {}

        for path in sorted(self.tickers_dir.glob("*.parquet")):
            ticker = path.stem
            ticker_dfs[ticker] = pl.read_parquet(path)

        tickers_loaded = sorted(ticker_dfs.keys())
        if self.logger:
            self.logger.info(f"Loaded {len(tickers_loaded)} tickers.")

        # Cross-sectional and sector z-scores represent "how does this stock
        # compare to the tradable universe right now" — regime tickers (ETFs
        # like SPY, XLK, VIXY) are market/sector aggregates, not independent
        # observations, so pooling them into the CS population would
        # contaminate the signal (XLK's return gets double-counted with its
        # constituent tech stocks, VIXY skews the mean because it's
        # negatively correlated with equity, etc.).
        #
        # The z-scores are only attached to tradable tickers — regime
        # tickers never appear as training episodes and their columns get
        # dropped from the unified parquet downstream.
        regime_set = set(self.tickers)
        tradable_tickers = [t for t in tickers_loaded if t not in regime_set]
        if self.logger:
            self.logger.info(
                f"  CS/sector z-scores will be computed over "
                f"{len(tradable_tickers)} tradable tickers "
                f"({len(tickers_loaded) - len(tradable_tickers)} regime tickers excluded)"
            )

        # ------------------------------------------------------------------
        # 2. Build a wide table: one row per timestamp, one column per
        #    (ticker, feature).  Used for row-wise cross-sectional stats.
        # ------------------------------------------------------------------

        def _wide_table() -> pl.DataFrame:
            """
            Horizontally join all tradable tickers on timestamp. Each
            tradable ticker contributes columns: <ticker>_<feat> for feat
            in CS_FEATURES. Regime tickers are excluded from the wide table
            because they're not part of the CS z-score population.

            Uses left join from the longest ticker so that tickers with shorter
            histories have null columns for timestamps before their inception.
            _compute_cs_stats uses np.nanmean/nanstd which handles this correctly.
            """
            # Sort by length descending so the first frame has the most timestamps
            sorted_tickers = sorted(tradable_tickers, key=lambda t: len(ticker_dfs[t]), reverse=True)

            frames = []
            for ticker in sorted_tickers:
                df = ticker_dfs[ticker]
                keep = ["timestamp"] + [f"{ticker}_{f}" for f in CS_FEATURES if f"{ticker}_{f}" in df.columns]
                frames.append(df.select(keep))

            wide = frames[0]
            for f in frames[1:]:
                wide = wide.join(f, on="timestamp", how="left")
            return wide

        # ------------------------------------------------------------------
        # 3. Per-ticker causal rolling z-score for CS features.
        #
        #    Each ticker's full timeline is used so the 252-bar warmup lands
        #    in the pre-cutoff period, not in training data.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info(
                f"Rolling z-score normalizing CS features per ticker "
                f"(window={NORMALIZATION_WINDOW} bars, causal)..."
            )

        CS_ROLLING_SKIP = {"volume_price_corr_20"}

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

        # pending_cols[ticker] = list of pl.Series to attach.
        # Only tradable tickers get z-score columns — regime tickers are
        # dropped from the unified parquet later, so computing/attaching
        # z-scores for them would just be wasted work.
        pending_cols: Dict[str, List[pl.Series]] = {ticker: [] for ticker in tradable_tickers}

        if self.logger:
            self.logger.info("Computing global cross-sectional stats...")

        wide        = _wide_table()
        wide_ts_np  = wide["timestamp"].to_numpy()
        ticker_ts_np = {t: ticker_dfs[t]["timestamp"].to_numpy() for t in tradable_tickers}

        for feat in CS_FEATURES:
            stats = _compute_cs_stats(
                tickers_in_group=tradable_tickers,
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

        # Sector grouping is over tradables only — an ETF like XLK should
        # not pool with the tech stocks it tracks, because that would
        # double-count the same underlying signal.
        sector_to_tickers: Dict[str, List[str]] = {}
        for ticker in tradable_tickers:
            sector = SECTOR_MAP.get(ticker, "other")
            sector_to_tickers.setdefault(sector, []).append(ticker)

        for sector, sector_tickers in sector_to_tickers.items():
            if len(sector_tickers) < 2:
                continue

            for feat in CS_FEATURES:
                stats = _compute_cs_stats(
                    tickers_in_group=sector_tickers,
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
        for ticker in tradable_tickers:
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

        return ticker_dfs

    def _build_unified(self, ticker_dfs: Dict[str, pl.DataFrame]) -> None:
        """
        Build a single unified parquet file by horizontally stacking all
        per-ticker DataFrames and adding cross-ticker regime features:

        Market Breadth
        --------------
        Fraction of tickers with a positive return over the last N days,
        computed at every bar using a rolling window aligned to timestamp.
        Windows: 5, 20, 60 trading days.
        Column names (shared across all tickers in the unified frame):
            breadth_5d, breadth_20d, breadth_60d

        Relative Strength vs SPY
        ------------------------
        For each ticker and each window N, the N-day cumulative return of
        that ticker minus the N-day cumulative return of SPY.
            <ticker>_rs_spy_5d, <ticker>_rs_spy_20d, <ticker>_rs_spy_60d

        Sector One-Hot Encoding
        -----------------------
        Each ticker gets N_SECTORS binary columns from its SIC code mapping
        (cached in ticker_sectors.parquet). These are static per-ticker and
        enter the model as regime features for sector-conditional behavior.
            <ticker>_sector_0, <ticker>_sector_1, ..., <ticker>_sector_11

        Warmup trim
        -----------
        After all normalization passes are complete, _drop_rows_before_timestamp
        removes the pre-cutoff warmup period (where rolling z-score windows
        produce zeros). No train/valid/test split is performed — the trainer
        owns time-based splitting for walk-forward validation.

        Output
        ------
        <data_dir>/preprocessed/v6/unified/unified.parquet
        """

        # ------------------------------------------------------------------
        # TODO(v6): add long-horizon regime levels + medium-scale deltas
        #
        # Current 5/20/60 skews short for "regime" signal. True regime shifts
        # (secular bull/bear, structural vol regimes, breadth deterioration)
        # develop over 3–12+ months. Also: several of the short-horizon
        # regime windows overlap conceptually with per-ticker features
        # (log_return_5/20/60, volatility_5_20_ratio, etc.), wasting the
        # complementarity that makes regime features valuable.
        #
        # Key design note: to add regime-scale info, add long-horizon
        # *levels* (smoothed series), not long-horizon deltas on short
        # windows. A long delta of a short window (e.g. breadth_5d_delta_200)
        # is a 2-point estimate where both endpoints are noisy short-term
        # readings — it loses the middle history. A long *level*
        # (e.g. breadth_200d) integrates the whole period into a smoothed
        # regime-state reading. Medium deltas on that long level
        # (breadth_200d_delta_20, breadth_200d_delta_60) then give
        # regime-transition signal without the endpoint-noise problem.
        #
        # What to add, per feature family:
        #
        #   BREADTH       -> add 200d (secular regime), keep 5/20/60
        #                    add breadth_200d_delta_20, breadth_200d_delta_60
        #
        #   RS vs SPY     -> add 252d (classic 12-month momentum factor),
        #                    keep 5/20/60
        #                    add rs_spy_252d_delta_20, rs_spy_252d_delta_60
        #
        #   VIX TERM      -> leave at 5/20/60. Unlike breadth, VIX term
        #                    structure is event-driven by design — it flips
        #                    on regime transitions in days, not months.
        #                    A 200d smooth would destroy the transitions
        #                    you care about and sit near its long-run mean
        #                    most of the time. Low signal-to-effort.
        #
        #   Short deltas  -> keep existing 5/20 deltas on 5/20/60 bases.
        #                    Short delta of a short window captures quick
        #                    shifts.
        #
        # Feature count: ~3 new per shared regime channel (breadth) and
        # ~3 new per-ticker (rs_spy). Modest increase.
        #
        # Validation as diagnostic: if the baseline model performs poorly
        # specifically on fold years containing regime transitions (2008,
        # 2020, 2022), that's direct evidence the current horizons miss
        # regime signal — and a reason to prioritize this over other work.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Window choice (post-v6 redesign per the long-horizon plan).
        # Short windows (5/20/60) capture day-to-week regime shifts;
        # the long windows (200d breadth, 252d RS) integrate over
        # quarter-to-year secular regime changes. VIX term structure
        # stays at short windows only — see the design TODO above.
        # ------------------------------------------------------------------
        BREADTH_WINDOWS_SHORT = [5, 20, 60]
        BREADTH_WINDOWS_LONG  = [200]
        BREADTH_WINDOWS       = BREADTH_WINDOWS_SHORT + BREADTH_WINDOWS_LONG

        RS_WINDOWS_SHORT = [5, 20, 60]
        RS_WINDOWS_LONG  = [252]
        RS_WINDOWS       = RS_WINDOWS_SHORT + RS_WINDOWS_LONG

        tickers_present = sorted(ticker_dfs.keys())

        # Split into tradable vs regime. Regime tickers (ETFs like SPY,
        # XLK, VIXY) participate in feature engineering (their closes/
        # returns drive breadth, VIX term structure, RS vs SPY) but are
        # not trained on or traded — their per-ticker columns are dropped
        # from the final unified parquet right before write. Only SPY_close
        # is preserved as a benchmark reference.
        regime_set = set(self.tickers)
        tradable_tickers = [t for t in tickers_present if t not in regime_set]
        regime_tickers_present = [t for t in tickers_present if t in regime_set]

        if self.logger:
            self.logger.info(
                f"Building unified file: {len(tradable_tickers)} tradable + "
                f"{len(regime_tickers_present)} regime tickers "
                f"({len(tickers_present)} total)..."
            )

        # Columns that are identical across all tickers (shared temporal features
        # plus timestamp). Keep them from the first ticker only; strip from all
        # subsequent ones before joining to avoid DuplicateError.
        SHARED_COLS = {"timestamp", "day_sin", "day_cos",
                    "month_sin", "month_cos", "quarter_sin", "quarter_cos"}

        # ------------------------------------------------------------------
        # Phase 1: horizontal stack — one pass over the full timeline
        #
        # Uses left join from the ticker with the most data so that tickers
        # with shorter histories (e.g., XLC from 2018) get null columns for
        # timestamps before their inception. Nulls are filled with 0 after
        # all regime features are computed, which means "no signal" in
        # z-score space.
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Horizontally stacking tickers on full timeline...")

        # Sort tickers by data length (descending) so the first one defines
        # the full timestamp range
        tickers_by_length = sorted(tickers_present, key=lambda t: len(ticker_dfs[t]), reverse=True)

        stacked: Optional[pl.DataFrame] = None

        for ticker in tickers_by_length:
            df = ticker_dfs[ticker]

            if stacked is None:
                stacked = df
            else:
                ticker_only_cols = [c for c in df.columns if c not in SHARED_COLS]
                stacked = stacked.join(
                    df.select(["timestamp"] + ticker_only_cols),
                    on="timestamp",
                    how="left",
                )
                del df

        # Release ticker_dfs — all data is now in the stacked frame
        del ticker_dfs

        if stacked is None:
            if self.logger:
                self.logger.warning("  no ticker data found, aborting")
            return

        stacked = stacked.sort("timestamp")

        # ------------------------------------------------------------------
        # Sector one-hot encoding
        #
        # Each ticker gets N_SECTORS binary columns indicating its sector.
        # These are static per-ticker and enter the model as regime features
        # so the policy can learn sector-conditional behavior (e.g., "tech
        # stocks in rising-rate regimes should reduce exposure").
        # ------------------------------------------------------------------

        sector_map = self._load_sector_map()

        if sector_map:
            if self.logger:
                self.logger.info(f"  Adding sector one-hot encoding ({N_SECTORS} sectors)...")

            # Tradables only — regime tickers don't need sector one-hot
            # because they'll be dropped from the unified parquet.
            sector_exprs = []
            for ticker in tradable_tickers:
                sector_name = sector_map.get(ticker, "other")
                try:
                    sector_id = SECTOR_NAMES.index(sector_name)
                except ValueError:
                    sector_id = SECTOR_NAMES.index("other")

                for sid in range(N_SECTORS):
                    val = 1.0 if sid == sector_id else 0.0
                    sector_exprs.append(
                        pl.lit(val).alias(f"{ticker}_sector_{sid}")
                    )

            stacked = stacked.with_columns(sector_exprs)

        # ------------------------------------------------------------------
        # Market breadth: fraction of tickers with positive return over N days
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info("  Computing market breadth on full timeline...")

        # Breadth represents "fraction of investable stocks with positive
        # returns" — so we pool over tradable tickers only. Including
        # regime ETFs would double-count their constituents (XLK ≈ sum of
        # tech stocks) and skew the ratio with negatively-correlated
        # vehicles (VIXY).
        return_cols = [
            f"{t}_log_return_5" for t in tradable_tickers
            if f"{t}_log_return_5" in stacked.columns
        ]

        breadth_exprs = []
        for w in BREADTH_WINDOWS:
            if w == 1:
                positive = pl.concat_list([
                    (pl.col(c) > 0).cast(pl.Float64)
                    for c in return_cols
                ]).list.mean()
            else:
                positive = pl.concat_list([
                    (pl.col(c).rolling_sum(window_size=w) > 0).cast(pl.Float64)
                    for c in return_cols
                ]).list.mean()
            breadth_exprs.append(positive.alias(f"breadth_{w}d"))

        stacked = stacked.with_columns(breadth_exprs)

        # ------------------------------------------------------------------
        # VIX Term Structure: log(VIXY / VIXM)
        # ------------------------------------------------------------------

        VIX_TERM_WINDOWS = [5, 20, 60]

        vixy_col = "VIXY_close"
        vixm_col = "VIXM_close"

        if vixy_col in stacked.columns and vixm_col in stacked.columns:
            if self.logger:
                self.logger.info("  Computing VIX term structure on full timeline...")

            eps = 1e-8
            stacked = stacked.with_columns(
                (pl.col(vixy_col) / (pl.col(vixm_col) + eps)).log().alias("vix_term_structure")
            )

            vix_ts_exprs = [
                pl.col("vix_term_structure")
                .rolling_mean(window_size=w)
                .alias(f"vix_term_structure_{w}d")
                for w in VIX_TERM_WINDOWS
            ]
            stacked = (
                stacked
                .with_columns(vix_ts_exprs)
                .drop("vix_term_structure")
            )

            vix_ts_cols = [f"vix_term_structure_{w}d" for w in VIX_TERM_WINDOWS]
            vix_arr = stacked.select(vix_ts_cols).to_numpy().astype(np.float64)
            vix_scaled = _rolling_zscore_normalize_vectorized(vix_arr, window=NORMALIZATION_WINDOW)

            stacked = stacked.with_columns([
                pl.Series(col, vix_scaled[:, i])
                for i, col in enumerate(vix_ts_cols)
            ])
        else:
            if self.logger:
                self.logger.warning(
                    "VIXY/VIXM close columns not found — skipping VIX term structure"
                )

        # ------------------------------------------------------------------
        # Relative strength vs SPY
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

                # Tradables only. Regime tickers never need <ticker>_rs_spy_*
                # because their columns are dropped from the unified parquet.
                for ticker in tradable_tickers:
                    t_col = f"{ticker}_log_return_5"
                    if t_col not in stacked.columns:
                        continue
                    t_ret = pl.col(t_col) if w == 1 else pl.col(t_col).rolling_sum(window_size=w)
                    rs_exprs.append((t_ret - spy_ret).alias(f"{ticker}_rs_spy_{w}d"))

            stacked = stacked.with_columns(rs_exprs)

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
        # Regime momentum deltas
        #
        # Two delta-window families:
        #   * SHORT bases (breadth 5/20/60d, VIX term 5/20/60d) get
        #     short deltas [1, 5, 20]. These capture quick regime shifts
        #     within a short-horizon level.
        #   * LONG bases (breadth_200d, rs_spy_252d) get medium deltas
        #     [20, 60]. A 1-bar delta on a 200-day rolling-mean series
        #     barely moves and adds nothing new. Medium deltas measure
        #     "is the long-run regime in transition" — the actual signal.
        #
        # Why not both delta families on both base scales: short deltas
        # on long bases are near-zero (low signal-to-noise); long deltas
        # on short bases are noisy 2-point estimates that drop the middle
        # of the window (high variance, low value). The matched-scale
        # design avoids both pathologies.
        # ------------------------------------------------------------------
        SHORT_DELTA_WINDOWS  = [1, 5, 20]
        MEDIUM_DELTA_WINDOWS = [20, 60]

        # Build the (column, delta-window) pairs explicitly per scale.
        breadth_short_cols = [
            f"breadth_{w}d" for w in BREADTH_WINDOWS_SHORT
            if f"breadth_{w}d" in stacked.columns
        ]
        breadth_long_cols = [
            f"breadth_{w}d" for w in BREADTH_WINDOWS_LONG
            if f"breadth_{w}d" in stacked.columns
        ]
        vix_ts_cols_delta = [
            f"vix_term_structure_{w}d" for w in VIX_TERM_WINDOWS
            if f"vix_term_structure_{w}d" in stacked.columns
        ]

        delta_exprs = []
        for c in breadth_short_cols + vix_ts_cols_delta:
            for d in SHORT_DELTA_WINDOWS:
                delta_exprs.append(
                    pl.col(c).diff(n=d).fill_null(0.0).alias(f"{c}_delta_{d}")
                )
        for c in breadth_long_cols:
            for d in MEDIUM_DELTA_WINDOWS:
                delta_exprs.append(
                    pl.col(c).diff(n=d).fill_null(0.0).alias(f"{c}_delta_{d}")
                )

        # Long-horizon RS deltas: per-ticker for each tradable. Apply only
        # MEDIUM_DELTA_WINDOWS to the long RS bases — same rationale.
        # Short RS bases are not delta-ed currently and we keep that
        # convention (per-ticker rolling z-scores already capture short-
        # horizon trend, applied during the RS normalization step above).
        for w_long in RS_WINDOWS_LONG:
            for ticker in tradable_tickers:
                t_col = f"{ticker}_rs_spy_{w_long}d"
                if t_col not in stacked.columns:
                    continue
                for d in MEDIUM_DELTA_WINDOWS:
                    delta_exprs.append(
                        pl.col(t_col).diff(n=d).fill_null(0.0)
                            .alias(f"{t_col}_delta_{d}")
                    )

        if delta_exprs:
            if self.logger:
                self.logger.info(
                    f"  Computing regime momentum deltas: "
                    f"{len(delta_exprs)} new columns "
                    f"(short-base × {len(SHORT_DELTA_WINDOWS)}, "
                    f"long-base × {len(MEDIUM_DELTA_WINDOWS)})..."
                )
            stacked = stacked.with_columns(delta_exprs)

        # ------------------------------------------------------------------
        # Fill nulls from left-join (tickers with shorter histories)
        #
        # Tickers that didn't exist before a certain date have null columns
        # for all timestamps before their inception. Fill with 0.0, which
        # represents "neutral / no signal" in z-score space. This must
        # happen AFTER all regime features are computed so that breadth,
        # VIX term structure, and RS vs SPY calculations can correctly
        # ignore missing tickers via null-aware aggregations.
        #
        # IMPORTANT — _close columns are EXCLUDED from the 0-fill. A null
        # close means "the ticker was not trading on this day" (pre-IPO,
        # post-delisting, between lifecycle segments).
        #
        # Note: the trainer's ticker-loading already did `prices > 0` to
        # compute first_valid_idx / last_valid_idx, so the old 0-fill was
        # being implicitly masked at episode-sampling time. This change
        # isn't what fixes the 2088%-return pathology (that's the
        # lifecycle-segmentation fix in _split_lifecycle_segments). But
        # the old behavior quietly conflated "no data" with "traded at
        # exactly $0.00" and relied on every downstream consumer to
        # defensively filter. Making null mean "not tradable" and reserving
        # 0.0 for actual-zero readings is the honest encoding. Feature
        # columns are still filled because features are z-scored and 0.0
        # is a semantically valid "neutral reading" there.
        # ------------------------------------------------------------------

        if self.logger:
            null_count = stacked.select(pl.all().null_count()).sum_horizontal().item()
            self.logger.info(f"  Filling {null_count:,} null values from shorter-history tickers...")

        # Fill numeric feature columns with 0.0, but leave timestamp and
        # every {ticker}_close column untouched (null == "not tradable").
        close_cols = {c for c in stacked.columns if c.endswith("_close")}
        numeric_cols = [
            c for c in stacked.columns
            if c not in close_cols
            and stacked[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]
        stacked = stacked.with_columns([
            pl.col(c).fill_null(0.0).fill_nan(0.0) for c in numeric_cols
        ])

        if self.logger:
            remaining_nulls = stacked.select(pl.all().null_count()).sum_horizontal().item()
            self.logger.info(
                f"  After fill: {remaining_nulls:,} nulls remaining "
                f"(expected — these are non-tradable days in _close columns)"
            )

        # ------------------------------------------------------------------
        # Phase 3: trim regime-ticker columns, warmup rows, and write output
        #
        # Regime tickers (SPY, XLK, VIXY, etc.) are used as feature-
        # engineering inputs — their returns drive breadth/VIX/RS — but
        # they're not trained on or traded. Drop their per-ticker columns
        # here so the unified parquet contains only:
        #   - timestamp
        #   - temporal features (day_sin, ...)
        #   - shared regime features (breadth_*, vix_term_*, *_delta_*)
        #   - tradable-ticker columns (<tradable>_*)
        #   - SPY_close (kept as a lightweight benchmark reference for
        #     post-hoc analysis; never consumed by training)
        #
        # No train/valid/test split here — that's owned by the trainer,
        # which defines walk-forward fold boundaries at runtime.
        # ------------------------------------------------------------------

        if regime_tickers_present:
            if self.logger:
                self.logger.info(
                    f"  Pruning regime-ticker columns "
                    f"({len(regime_tickers_present)} tickers)..."
                )

            n_before = len(stacked.columns)
            regime_prefixes = tuple(f"{t}_" for t in regime_tickers_present)
            # Keep SPY_close explicitly; drop everything else starting with
            # any regime ticker's prefix.
            cols_to_drop = [
                c for c in stacked.columns
                if c.startswith(regime_prefixes) and c != "SPY_close"
            ]
            if cols_to_drop:
                stacked = stacked.drop(cols_to_drop)
            if self.logger:
                self.logger.info(
                    f"    {n_before} → {len(stacked.columns)} columns "
                    f"({len(cols_to_drop):,} dropped)"
                )

        if self.logger:
            self.logger.info("  Dropping warmup rows...")

        stacked = self._drop_rows_before_timestamp(stacked, self.timestamp)

        out_dir = Path(self.data_dir) / "preprocessed" / "v6" / "unified"
        create_directory(out_dir)
        out_path = f"{out_dir}/unified.parquet"

        if self.logger:
            self.logger.info(
                f"  Writing {out_path} — "
                f"{len(stacked):,} rows, {len(stacked.columns)} columns"
            )

        save_to_parquet(stacked, out_path, index=False)
        del stacked

        if self.logger:
            self.logger.info("Unified file complete.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def feature_engineer_and_save_tickers(self, batch_size: int = 8) -> None:
        """
        Run the full feature-engineering pipeline for all tickers.

        Pipeline order:
        1. Load all daily CSVs into memory (~865MB compressed → ~1-3GB in RAM)
        2. Filter tickers by liquidity (dollar volume + history length)
        3. Fetch sector labels + ticker type/name from Polygon API
        4. Filter by ticker type (keep CS/ADRC/OS, exclude warrants/bonds/ETNs/ETFs)
        5. Fetch ticker events and build aliases dynamically (symbol changes)
        6. Fetch splits from Polygon API (includes alias symbols from step 5)
        7. Partition data by ticker, resolve aliases, dispatch to workers
        8. Cross-sectional normalization across all processed tickers
        9. Build unified file with regime features + sector one-hot

        Memory strategy
        ---------------
        All daily data is loaded once and partitioned by ticker in memory.
        Worker processes receive individual ticker DataFrames — no
        redundant CSV decompression or parsing. Batching controls how many
        workers run concurrently, not how much data is loaded.

        :param batch_size: Number of tickers to process concurrently
        :type batch_size: int
        """
        # ==================================================================
        # Step 1: Load all daily CSVs once
        # ==================================================================
        all_data = self._load_all_data()

        # ==================================================================
        # Step 1.5: Detect lifecycle segments and relabel the ticker column
        # ==================================================================
        all_data = self._split_lifecycle_segments(all_data)

        # ==================================================================
        # Step 2: Filter tickers by liquidity
        # ==================================================================
        filtered_tickers = self._filter_tickers_by_liquidity(all_data)

        # Segments are what downstream feature engineering and the
        # unified-frame layout operate on, but Polygon API calls
        # (sectors, ticker type, events, splits) are keyed by source
        # symbol. Collapse segment names back to their source symbols
        # for any API-hitting step so we don't double-fetch CIT twice.
        filtered_source_tickers = sorted({
            self.segment_to_source.get(t, t) for t in filtered_tickers
        })

        # ==================================================================
        # Step 3: Fetch sector labels from Polygon API (cached to parquet)
        # ==================================================================
        self._fetch_sector_labels(filtered_source_tickers)

        # ==================================================================
        # Step 4: Filter by ticker type (keep CS/ADRC/OS only)
        # ==================================================================
        kept_source_tickers = set(self._filter_by_ticker_details(filtered_source_tickers))
        # Project the type-filter decision back onto segment names.
        filtered_tickers = [
            seg for seg in filtered_tickers
            if self.segment_to_source.get(seg, seg) in kept_source_tickers
        ]

        # ==================================================================
        # Step 5: Fetch ticker events and build initial aliases
        #
        # First pass: build aliases WITHOUT suspect filtering so _fetch_splits
        # knows which historical symbols to also fetch (e.g. FB for META).
        # Suspects can't be detected yet because splits.parquet doesn't exist.
        # ==================================================================
        # Re-derive source symbols after the ticker-type filter dropped some.
        filtered_source_tickers = sorted({
            self.segment_to_source.get(t, t) for t in filtered_tickers
        })
        self._fetch_ticker_events(filtered_source_tickers)
        self.ticker_aliases = self._build_ticker_aliases()

        # ==================================================================
        # Step 6: Fetch splits + rebuild aliases with suspect filtering
        #
        # _fetch_splits fetches splits for all tickers including aliases,
        # then internally rebuilds self.ticker_aliases with suspect
        # detection now that splits.parquet exists. Aliases where the
        # previous ticker has splits recorded AFTER the rename date
        # (indicating ticker reuse) are dropped from the final mapping.
        # ==================================================================
        self._fetch_splits(filtered_source_tickers)

        # ==================================================================
        # Step 7: Partition by ticker, resolve aliases, dispatch to workers
        # ==================================================================

        # Also include alias symbols (e.g. FB) so they're available for stitching.
        #
        # Aliases are defined in source-symbol space (FB → META). Splitting
        # runs on all_data BEFORE this block, so every row in all_data has
        # a segment name in its "ticker" column. For an alias to be
        # findable here, the previous-symbol's lifetime must produce a
        # single segment (in which case the segment name equals the raw
        # symbol). Multi-segment alias symbols would break this assumption
        # — guard it explicitly so we fail loudly if it ever happens.
        alias_symbols = set()
        for alias in self.ticker_aliases:
            if alias["current_ticker"] in filtered_tickers:
                for t in alias["previous_tickers"]:
                    prev_symbol = t["symbol"]
                    source_of_prev = self.segment_to_source.get(prev_symbol)
                    if source_of_prev is not None and source_of_prev != prev_symbol:
                        raise RuntimeError(
                            f"Alias previous-symbol {prev_symbol!r} is itself a "
                            f"lifecycle segment (source={source_of_prev!r}). "
                            f"_resolve_aliases looks up raw symbols, not segment "
                            f"names — this case isn't supported. If this ever "
                            f"fires, either the alias or the split threshold "
                            f"needs re-thinking."
                        )
                    alias_symbols.add(prev_symbol)

        symbols_needed = set(filtered_tickers) | alias_symbols

        # Filter to only needed tickers, then partition
        filtered_data = all_data.filter(pl.col("ticker").is_in(list(symbols_needed)))
        del all_data  # release the full dataset

        if self.logger:
            self.logger.info(
                f"Partitioning {len(filtered_data):,} rows into per-ticker DataFrames..."
            )

        ticker_dfs: Dict[str, pl.DataFrame] = {
            ticker[0]: frame
            for ticker, frame in filtered_data.partition_by("ticker", as_dict=True).items()
        }
        del filtered_data

        # Resolve aliases (e.g. FB → META)
        ticker_dfs = self._resolve_aliases(ticker_dfs)

        # Keep only the tickers we want to process
        ticker_dfs = {t: df for t, df in ticker_dfs.items() if t in filtered_tickers}

        if self.logger:
            self.logger.info(f"Partitioned into {len(ticker_dfs)} ticker DataFrames")

        # Dispatch to workers in batches (for CPU concurrency, not memory)
        ticker_list = sorted(ticker_dfs.keys())
        batches: List[List[str]] = []
        batch: List[str] = []
        for ticker in ticker_list:
            batch.append(ticker)
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)

        total = len(ticker_list)
        done = 0
        n_errors = 0
        skip_counts: Dict[str, int] = {}
        skip_examples: Dict[str, List[str]] = {}

        for batch_num, batch in enumerate(batches, 1):
            if self.logger:
                self.logger.info(
                    f"Batch {batch_num}/{len(batches)} — "
                    f"dispatching {len(batch)} tickers to workers..."
                )

            futures = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                for ticker in batch:
                    if ticker not in ticker_dfs:
                        continue
                    fut = pool.submit(
                        _process_ticker,
                        ticker,                                              # segment name (output identity)
                        ticker_dfs[ticker],
                        self.data_dir,
                        False,
                        self.segment_to_source.get(ticker, ticker),          # source symbol for splits lookup
                        self.segment_end_dates.get(ticker),                  # None for single-segment tickers
                    )
                    futures[fut] = ticker

                for fut in as_completed(futures):
                    result = fut.result()
                    if result.startswith("ERROR:"):
                        _, failed_ticker, msg = result.split(":", 2)
                        if self.logger:
                            self.logger.error(f"[{failed_ticker}] pipeline failed: {msg}")
                        n_errors += 1
                    elif result.startswith("SKIP:"):
                        _, skipped_ticker, reason = result.split(":", 2)
                        if self.logger:
                            self.logger.info(f"[{skipped_ticker}] skipped ({reason})")
                        # Categorize for end-of-run summary. The first
                        # word of the reason names the bucket.
                        bucket = reason.split(None, 1)[0].rstrip(":")
                        skip_counts[bucket] = skip_counts.get(bucket, 0) + 1
                        skip_examples.setdefault(bucket, []).append(skipped_ticker)
                    else:
                        done += 1
                        if self.logger:
                            self.logger.info(f"[{result}] done ✓  ({done}/{total})")

            # Free the DataFrames for this batch
            for ticker in batch:
                ticker_dfs.pop(ticker, None)

        del ticker_dfs

        # End-of-dispatch summary so the per-reason skip counts are
        # findable without grepping the per-ticker log lines.
        if self.logger:
            self.logger.info(
                f"Per-ticker pipeline complete: {done} succeeded, "
                f"{sum(skip_counts.values())} skipped, {n_errors} errors"
            )
            for bucket, count in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
                examples = skip_examples.get(bucket, [])
                sample = ", ".join(sorted(examples)[:10])
                suffix = "..." if len(examples) > 10 else ""
                self.logger.info(f"  Skipped [{bucket}]: {count}  (sample: {sample}{suffix})")

        # ==================================================================
        # Step 8: Cross-sectional normalization
        # ==================================================================
        if self.logger:
            self.logger.info("Per-ticker feature engineering complete. Starting cross-sectional normalisation...")

        ticker_dfs = self._cross_sectional_normalization()

        # ==================================================================
        # Step 9: Build unified file (single parquet, no train/valid/test split)
        # ==================================================================
        if self.logger:
            self.logger.info("Cross-sectional normalisation complete. Building unified file...")

        self._build_unified(ticker_dfs)

        if self.logger:
            self.logger.info("Feature engineering complete. Godspeed brotherman")


def main():
    feature_engineer = DataFeatureEngineer(logger=Logger())
    feature_engineer.feature_engineer_and_save_tickers()


if __name__ == "__main__":
    main()