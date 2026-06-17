"""
Split-adjustment logic for OHLCV bars.

`_apply_split_adjustments` is the one function here, called from
`worker.py::_process_ticker` for every ticker (or lifecycle segment)
before any feature engineering. Reads from the Polygon splits cache
at `<data_dir>/raw/splits/splits.parquet`.
"""

from pathlib import Path
from typing import Any, Optional

import polars as pl


def _apply_split_adjustments(
    df: pl.DataFrame,
    ticker: str,
    data_dir: Path,
    log: bool = False,
    source_ticker: Optional[str] = None,
    segment_end_date: Optional[Any] = None,
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

    Reads from the unified splits cache at <data_dir>/raw/splits/splits.parquet
    and filters to the requested ticker. Tickers with no splits (or only
    placeholder rows with adjustment_type="none") are returned unchanged.

    Price columns (open, high, low, close) are multiplied by the
    factor.  Volume is divided by the same factor so that the total
    dollar value of each bar (price × volume) is preserved and relative
    volume signals remain comparable across the split boundary.

    Lifecycle-segment handling
    --------------------------
    For symbols that were split into multiple lifecycle segments (e.g.
    CIT → CIT.1, CIT.2), the splits cache groups all of ``CIT``'s splits
    under one symbol — including splits that economically belong to the
    post-BK entity. Without care, ``join_asof`` with ``strategy="forward"``
    would let bars near the end of the pre-BK segment pick up forward-
    looking factors from splits that happened after BK to a different
    entity. That's incorrect: pre-BK shareholders were wiped; post-BK
    splits didn't affect the pre-BK price series at all.

    The fix is to filter the per-segment splits table to those whose
    ``execution_date <= segment_end_date`` BEFORE the ``join_asof``.
    Then ``strategy="forward"`` naturally uses the nearest in-segment
    split (or gives a null factor → 1.0 for bars after the last in-
    segment split, which is correct since no further adjustment applies
    inside this lifetime).

    Parameters
    ----------
    df : pl.DataFrame
        Raw daily bars for a single ticker (or ticker segment).
    ticker : str
        Identifier used for logging only. Typically the segment name
        (e.g. ``CIT.1``) for clarity in worker logs.
    data_dir : Path
        Root data directory.
    log : bool
        If True, print progress messages.
    source_ticker : Optional[str]
        The original Polygon symbol to filter the splits cache by.
        Defaults to ``ticker`` when not provided — that keeps the
        single-segment case (AAPL, MSFT, …) working unchanged. For
        lifecycle-split segments the caller passes the source symbol
        (e.g. ``CIT``).
    segment_end_date : Optional[date]
        If provided, splits with ``execution_date > segment_end_date``
        are excluded from the adjustment. Only meaningful for lifecycle-
        split segments where the raw symbol has splits that belong to
        a later lifetime. For single-segment tickers this can safely
        be left ``None``.

    Returns
    -------
    pl.DataFrame
        DataFrame with split-adjusted price and volume columns.
    """
    lookup_ticker = source_ticker if source_ticker is not None else ticker
    splits_path = data_dir / "raw" / "splits" / "splits.parquet"

    if not splits_path.exists():
        if log:
            print(f"No splits cache found, skipping adjustment for {ticker}.")
        return df

    all_splits = pl.read_parquet(splits_path)

    # Filter to this ticker's real splits (exclude placeholder rows).
    # Lookup is by source symbol, but log messages use the segment name
    # to make worker output self-describing.
    splits_filter = (
        (pl.col("ticker") == lookup_ticker)
        & (pl.col("adjustment_type") != "none")
        & (pl.col("execution_date").is_not_null())
    )
    if segment_end_date is not None:
        splits_filter = splits_filter & (pl.col("execution_date") <= segment_end_date)

    splits = all_splits.filter(splits_filter).sort("execution_date")

    if splits.is_empty():
        if log:
            print(f"No splits found for {ticker}, skipping adjustment.")
        return df

    if log:
        print(f"Applying {len(splits)} split adjustment(s) for {ticker}...")

    df = df.with_columns(
        (pl.col("timestamp").dt.date() + pl.duration(days=1)).alias("_bar_date")
    )

    df = df.join_asof(
        splits.select(["execution_date", "historical_adjustment_factor"]),
        left_on="_bar_date",
        right_on="execution_date",
        strategy="forward",
    ).with_columns(
        pl.col("historical_adjustment_factor").fill_null(1.0)
    )

    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    df = df.with_columns(
        [pl.col(c) * pl.col("historical_adjustment_factor") for c in price_cols] +
        [pl.col("volume") / pl.col("historical_adjustment_factor")]
    ).drop(["execution_date", "_bar_date", "historical_adjustment_factor"])
    return df
