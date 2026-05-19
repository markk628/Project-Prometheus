"""
Trading-calendar gap handling for one ticker series.

`_handle_gaps` left-joins per-ticker bars onto the NYSE trading-day
grid, forward-fills price columns through halts, zero-fills volume,
and discards leading rows before the first valid bar.

Used inside the per-ticker worker pipeline (`worker.py::_process_ticker`)
immediately after `_apply_split_adjustments`.
"""

import polars as pl

from .nyse_calendar import _build_nyse_valid_days


def _handle_gaps(df: pl.DataFrame, log: bool = False) -> pl.DataFrame:
    """
    Enforce a continuous daily NYSE trading calendar for one ticker.

    - Left-joins raw bars onto the NYSE trading day grid.
    - Forward-fills price columns (close, open, high, low) so gaps
      from halts or missing data maintain price continuity.
    - Zeros out volume / transaction counts for missing days.
    - Discards leading rows where no price data exists yet (before IPO).
    """
    if log:
        print("Handling gaps...")

    # Extract date from timestamp for joining
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Date).alias("date")
    )

    start_date = df["date"].min()
    end_date   = df["date"].max()

    valid_days = _build_nyse_valid_days(start_date, end_date)

    # Left-join full trading day grid onto raw data
    df = (
        valid_days
        .join(df, on="date", how="left")
        .sort("date")
    )

    price_cols  = [c for c in ["open", "high", "low", "close", "vwap"] if c in df.columns]
    volume_cols = [c for c in ["volume", "transactions"] if c in df.columns]

    # Drop leading rows where close is null (before ticker had any data)
    first_valid_idx = df.select(pl.col("close").is_not_null().arg_max()).item()
    df = df.slice(first_valid_idx)

    # Forward-fill prices, zero-fill volumes
    df = df.with_columns(
        [pl.col(c).forward_fill() for c in price_cols] +
        [pl.col(c).fill_null(0)   for c in volume_cols]
    )

    # Reconstruct timestamp from date (set to market close time 20:00 UTC / 4pm ET)
    # so downstream code that expects a timestamp column still works
    if "timestamp" not in df.columns or df["timestamp"].null_count() > 0:
        df = df.with_columns(
            pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("timestamp")
        )

    return df.drop("date")
