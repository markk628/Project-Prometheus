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

# Shared regime feature column prefixes.
#
# These name the families of regime features that are computed once per
# timestamp (NOT once per ticker) in _build_unified. Centralized here so
# both the auditor (categorization + multicollinearity check) and the
# trainer (column classification when loading unified.parquet into
# TickerData) use the same source of truth. Adding a new shared-regime
# family means adding its prefix here once.
#
# Two groupings because they were introduced at different times and the
# auditor reports them separately in its feature-count summary:
#   - BREADTH_VIX_PREFIXES: original v5 regime features (breadth, vix term)
#   - MACRO_REGIME_PREFIXES: v6 run 3a additions (yield curve, credit
#     spread, size/growth factor, sector rotation)
# SHARED_REGIME_PREFIXES is the union, suitable for any code that just
# needs to know "is this a shared regime column?" without caring which
# family.
BREADTH_VIX_PREFIXES = (
    "breadth_",
    "vix_term_",
)

MACRO_REGIME_PREFIXES = (
    "yield_curve_",
    "credit_spread_",
    "size_factor_",
    "growth_factor_",
    "sector_dispersion_",
    "sector_topbottom_",
)

SHARED_REGIME_PREFIXES = BREADTH_VIX_PREFIXES + MACRO_REGIME_PREFIXES