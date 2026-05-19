"""
Per-ticker worker pipeline entry point.

`_process_ticker` runs in a ProcessPoolExecutor worker — it must be a
top-level (importable, picklable) function with no closure state.
The worker reads/writes the per-segment parquet under
`<data_dir>/preprocessed/v7/tickers/`.

Pipeline order:
    _apply_split_adjustments  (splits.py)
    _handle_gaps              (gaps.py)
    DISCONTINUITY GATE        (constants.py)
    _calculate_vwap           (features.py)
    _add_candlestick_features (features.py)
    _add_temporal_patterns    (features.py)
    _add_volatility_features  (features.py)
    _add_trend_features       (features.py)
    _add_volume_features      (features.py)
    _drop_unnecessary_features(features.py)
    _normalize_data           (normalization.py)
    _prefix_columns           (normalization.py)
    MIN_TICKER_LENGTH GATE    (constants.py)
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from src.utils.utils import create_directory, save_to_parquet

from .constants import DISCONTINUITY_LOG_THRESHOLD, MIN_TICKER_LENGTH
from .features import (
    _add_candlestick_features,
    _add_temporal_patterns,
    _add_trend_features,
    _add_volatility_features,
    _add_volume_features,
    _calculate_vwap,
    _drop_unnecessary_features,
)
from .gaps import _handle_gaps
from .normalization import _normalize_data, _prefix_columns
from .splits import _apply_split_adjustments


def _process_ticker(
    ticker: str,
    ticker_df: pl.DataFrame,
    data_dir: Path,
    log: bool,
    source_ticker: Optional[str] = None,
    segment_end_date: Optional[Any] = None,
) -> str:
    """
    Full feature-engineering pipeline for one ticker (or lifecycle segment).
    Runs in a worker process — no shared state.

    Normalization happens here on the full timeline (no trimming, no splitting)
    so that the rolling warmup rows that come out as zeros land in the
    pre-cutoff warmup period, not in the actual training data.

    _drop_rows_before_timestamp is deferred to the end of _build_unified,
    after every normalization pass (per-ticker rolling z-score, cross-sectional
    z-score, breadth, RS) has had the full history available.

    :param ticker: Identity of this series in the processed output (segment
        name for split tickers, plain symbol otherwise). Used for log output
        and for the column prefix on the saved parquet.
    :param source_ticker: The underlying Polygon symbol for API-keyed lookups
        (splits, and later sectors). Defaults to ``ticker`` so single-segment
        tickers work unchanged.
    :param segment_end_date: For lifecycle-split segments, the last date
        belonging to this segment. Passed to _apply_split_adjustments so
        splits belonging to a later segment (e.g. post-bankruptcy reverse
        splits of the new-entity's shares) don't leak into this segment's
        adjustment factors. ``None`` for single-segment tickers.

    Returns the ticker name on success or an error string on failure.
    """
    try:
        df = _apply_split_adjustments(
            ticker_df,
            ticker,
            data_dir,
            log,
            source_ticker=source_ticker,
            segment_end_date=segment_end_date,
        )
        df = _handle_gaps(df, log)

        # Critical-discontinuity gate. Computed on post-split, post-gap
        # closes — i.e. the actual continuous price series this segment
        # would feed to training. Any single-day |log_return| >= the
        # threshold (default ≈ 100%) almost certainly indicates an
        # unadjusted corporate action Polygon doesn't have records for.
        # Bail out before paying for VWAP / candlestick / volatility /
        # trend / volume / normalization on a ticker that won't survive
        # to training anyway.
        closes = df["close"].to_numpy()
        if len(closes) > 1:
            log_rets = np.diff(np.log(closes))
            max_abs_log_ret = float(np.max(np.abs(log_rets)))
            if max_abs_log_ret >= DISCONTINUITY_LOG_THRESHOLD:
                return (
                    f"SKIP:{ticker}:discontinuity (max |log_ret|="
                    f"{max_abs_log_ret:.4f} ≥ {DISCONTINUITY_LOG_THRESHOLD:.4f})"
                )

        df = _calculate_vwap(df, log)
        df = _add_candlestick_features(df, log)
        df = _add_temporal_patterns(df, log)
        df = _add_volatility_features(df, log)
        df = _add_trend_features(df, log)
        df = _add_volume_features(df, log)
        df = _drop_unnecessary_features(df, log)
        df = _normalize_data(df, log)
        df = _prefix_columns(df, ticker)

        # Drop tickers that are too short to produce even one training
        # episode. Filtering at save time means the tickers/ directory
        # only contains viable files — nothing downstream needs to re-check.
        if len(df) < MIN_TICKER_LENGTH:
            return f"SKIP:{ticker}:{len(df)} rows < {MIN_TICKER_LENGTH}"

        base_dir = data_dir / "preprocessed" / "v7" / "tickers"
        create_directory(base_dir)
        save_to_parquet(df, f"{base_dir}/{ticker}.parquet", index=False)

        return ticker

    except Exception as e:
        return f"ERROR:{ticker}:{e}"
