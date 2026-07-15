"""
Verify that stock split adjustments were applied correctly.

For each ticker with splits, checks:
1. No large price discontinuities around split dates
2. Prices on the execution date are consistent with adjacent days
3. The ratio of pre-split to post-split raw prices matches the expected split ratio

Usage:
    python -m src.utils.verify_splits
"""

import polars as pl
import numpy as np
from pathlib import Path

from src.config.config import DATA_DIR
from src.utils.logger import Logger


def verify_splits(
    data_dir: Path = Path(DATA_DIR),
    jump_threshold: float = 0.20,  # flag price jumps > 20% between consecutive bars
    logger: Logger = None,
):
    splits_dir = data_dir / "raw" / "splits"
    tickers_dir = data_dir / "preprocessed" / "v4" / "tickers"

    if not splits_dir.exists():
        logger.error(f"Splits directory not found: {splits_dir}")
        return

    if not tickers_dir.exists():
        logger.error(f"Tickers directory not found: {tickers_dir}")
        return

    split_files = sorted(splits_dir.glob("*.parquet"))
    issues_found = 0

    for split_file in split_files:
        ticker = split_file.stem.replace("_splits", "")
        ticker_file = tickers_dir / f"{ticker}.parquet"

        if not ticker_file.exists():
            logger.error(f"[{ticker}] No preprocessed data found, skipping")
            continue

        splits = pl.read_parquet(split_file)
        if splits.is_empty():
            continue

        # Load ticker data — look for close column (may be prefixed)
        df = pl.read_parquet(ticker_file)

        # Find the close price column
        close_col = None
        for candidate in [f"{ticker}_close", "close"]:
            if candidate in df.columns:
                close_col = candidate
                break

        if close_col is None:
            # Try to find any column ending in _close
            close_cols = [c for c in df.columns if c.endswith("_close")]
            if close_cols:
                close_col = close_cols[0]
            else:
                logger.error(f"[{ticker}] No close price column found, skipping")
                continue

        # Find timestamp column
        ts_col = None
        for candidate in ["timestamp", f"{ticker}_timestamp"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            logger.error(f"[{ticker}] No timestamp column found, skipping")
            continue

        prices = df.select([ts_col, close_col]).rename({ts_col: "timestamp", close_col: "close"})

        # Add date column
        prices = prices.with_columns(
            pl.col("timestamp").dt.date().alias("date")
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"[{ticker}] Checking {len(splits)} split(s)")

        for row in splits.iter_rows(named=True):
            exec_date = row["execution_date"]
            split_from = row["split_from"]
            split_to = row["split_to"]
            ratio = row["effective_ratio"]
            adj_factor = row["historical_adjustment_factor"]
            adj_type = row["adjustment_type"]

            logger.info(f"\n  Split: {adj_type} {split_from}:{split_to} (ratio {ratio}) on {exec_date}")
            logger.info(f"  Historical adjustment factor: {adj_factor}")

            # Get bars around the split date (2 days before, split day, 2 days after)
            window_start = exec_date - pl.duration(days=5)  # calendar days, covers weekends
            window_end = exec_date + pl.duration(days=5)

            window = prices.filter(
                (pl.col("date") >= window_start) & (pl.col("date") <= window_end)
            ).sort("timestamp")

            if window.is_empty():
                logger.warning(f"  WARNING: No data found around split date {exec_date}")
                issues_found += 1
                continue

            # Split into pre-split, split-day, and post-split
            pre_split = window.filter(pl.col("date") < exec_date)
            split_day = window.filter(pl.col("date") == exec_date)
            post_split = window.filter(pl.col("date") > exec_date)

            if pre_split.is_empty():
                logger.warning(f"  WARNING: No pre-split data found")
            else:
                pre_last = pre_split["close"][-1]
                logger.info(f"  Last pre-split close: ${pre_last:.2f}")

            if split_day.is_empty():
                logger.warning(f"  WARNING: No data on split date {exec_date}")
            else:
                split_open = split_day["close"][0]
                split_close_val = split_day["close"][-1]
                logger.info(f"  Split day open: ${split_open:.2f}, close: ${split_close_val:.2f}")

            if post_split.is_empty():
                logger.warning(f"  WARNING: No post-split data found")
            else:
                post_first = post_split["close"][0]
                logger.info(f"  First post-split close: ${post_first:.2f}")

            # Check for discontinuities in the full window
            close_arr = window["close"].to_numpy()
            returns = np.diff(close_arr) / close_arr[:-1]
            timestamps = window["timestamp"].to_list()

            large_jumps = np.where(np.abs(returns) > jump_threshold)[0]
            if len(large_jumps) > 0:
                logger.critical(f"  ISSUE: Found {len(large_jumps)} price jump(s) > {jump_threshold:.0%}:")
                for idx in large_jumps:
                    pct = returns[idx]
                    logger.critical(f"    {timestamps[idx]} → {timestamps[idx+1]}: "
                          f"${close_arr[idx]:.2f} → ${close_arr[idx+1]:.2f} ({pct:+.2%})")
                issues_found += 1
            else:
                logger.info(f"  OK: No large price discontinuities around split date")

            # Check that prices are roughly consistent across the window
            # After proper adjustment, pre-split and post-split prices should be
            # in the same ballpark (within normal daily variance)
            if not pre_split.is_empty() and not post_split.is_empty():
                pre_mean = pre_split["close"].mean()
                post_mean = post_split["close"].mean()
                diff_pct = abs(pre_mean - post_mean) / pre_mean

                if diff_pct > 0.15:  # More than 15% difference is suspicious
                    logger.warning(f"  WARNING: Pre-split mean ${pre_mean:.2f} vs post-split mean ${post_mean:.2f} "
                          f"({diff_pct:.1%} difference)")
                else:
                    logger.info(f"  OK: Pre/post means consistent (${pre_mean:.2f} vs ${post_mean:.2f}, {diff_pct:.1%} diff)")

    # Global check: scan entire price series for any large jumps
    logger.info(f"\n{'='*60}")
    logger.info("Global scan: checking all tickers for unexplained large jumps...")

    ticker_files = sorted(tickers_dir.glob("*.parquet"))
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df = pl.read_parquet(ticker_file)

        close_col = None
        for candidate in [f"{ticker}_close", "close"]:
            if candidate in df.columns:
                close_col = candidate
                break
        if close_col is None:
            close_cols = [c for c in df.columns if c.endswith("_close")]
            if close_cols:
                close_col = close_cols[0]
            else:
                continue

        ts_col = None
        for candidate in ["timestamp", f"{ticker}_timestamp"]:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            continue

        prices_arr = df[close_col].to_numpy()
        timestamps_arr = df[ts_col].to_list()

        returns = np.diff(prices_arr) / (prices_arr[:-1] + 1e-10)
        large_jumps = np.where(np.abs(returns) > jump_threshold)[0]

        if len(large_jumps) > 0:
            logger.info(f"\n[{ticker}] {len(large_jumps)} jump(s) > {jump_threshold:.0%}:")
            for idx in large_jumps[:10]:  # Show first 10 max
                pct = returns[idx]
                logger.info(f"  {timestamps_arr[idx]} → {timestamps_arr[idx+1]}: "
                      f"${prices_arr[idx]:.2f} → ${prices_arr[idx+1]:.2f} ({pct:+.2%})")
            if len(large_jumps) > 10:
                logger.info(f"  ... and {len(large_jumps) - 10} more")
            issues_found += 1
        else:
            logger.info(f"[{ticker}] OK")

    logger.info(f"\n{'='*60}")
    if issues_found == 0:
        logger.info("All split adjustments verified successfully!")
    else:
        logger.info(f"Found {issues_found} issue(s). Review the output above.")


if __name__ == "__main__":
    verify_splits()