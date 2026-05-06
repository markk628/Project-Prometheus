"""
Module-level constants for the v6 preprocessing pipeline.

Centralized here so they can be imported by both the orchestrator
(`feature_engineer.py`) and the per-ticker worker pipeline (`worker.py`,
`gaps.py`, `splits.py`, `normalization.py`) without circular imports.
"""

from src.config.config import WINDOW_SIZE


NORMALIZATION_WINDOW = 252  # ~1 year of trading days

# Gap threshold for lifecycle-splitting a ticker. A run of consecutive
# missing trading days of this length or more is treated as a delist/
# relist boundary (bankruptcy & emergence, exchange transfer with long
# interruption, symbol reuse after dormancy). Shorter gaps are assumed
# to be trading halts and are forward-filled by _handle_gaps within the
# segment. 10 is conservative: legitimate halts on listed US equities
# essentially never exceed 5 days, while real corporate-action gaps are
# typically weeks to months. Everything in the 5-10 bucket is logged
# but NOT split — flag for manual inspection, don't auto-segment.
GAP_SPLIT_TRADING_DAYS = 10
GAP_SUSPICIOUS_TRADING_DAYS = 5

# Minimum viable ticker length after feature engineering. A ticker needs
# at least NORMALIZATION_WINDOW bars of warmup (for rolling z-scores) plus
# one full training episode (60-day observation window + 252-day episode).
# Tickers shorter than this produce zero usable training episodes and are
# skipped at save time to keep the tickers/ directory clean.
MIN_TICKER_LENGTH = NORMALIZATION_WINDOW + WINDOW_SIZE + 252  # 564 bars

# Critical single-day log-return threshold for excluding tickers with
# unadjusted-split / data-quality discontinuities. log(2.0) ≈ 0.6931 is
# a 2x up or 50% down single-day move on split-adjusted prices. After
# _apply_split_adjustments has done its job, any remaining breach almost
# always means an unadjusted corporate action that Polygon's splits
# endpoint doesn't know about (WB.1, WELL, SDRL.1, ICON.1, etc). These
# produce pathological per-episode returns when sampled in training
# (the 2088%-style outliers seen in run_1). Same threshold the auditor
# reports against in _check_price_discontinuities — after this filter
# runs, the auditor's critical-breach count should be ~0.
DISCONTINUITY_LOG_THRESHOLD = 0.6931
