# Preprocessing pipeline — design doc

This document explains how raw daily-bar CSVs become the unified parquet that
the trainer consumes. It's organized as: what the thing produces, how the
stages compose and why they're ordered the way they are, what compensates for
what (data-vendor caveats), the full filtering decision tree a ticker passes
through, the on-disk artifacts each stage writes, and what the auditor
verifies after the fact.

Function and class references use names rather than line numbers
so the doc survives refactors.

---

## TL;DR

The pipeline takes ~35,000 raw daily-bar CSVs (one per source ticker, multiple
years), filters down to ~3,000 tradable + ~25 regime tickers, computes ~37
per-ticker features + a shared regime feature set, normalizes, and emits a
single horizontally-stacked parquet plus a per-ticker parquet directory.

The per-ticker feature set grew during v6:
- 25 base features carried from v5
- 5 medium-horizon deltas (v6 run 4a): `ema_close_ratio_20_delta_20`,
  `ema_20_60_ratio_delta_20`, `adx_5_20_ratio_delta_20`,
  `volatility_5_20_ratio_delta_20`, `volume_5_20_ratio_delta_20`
- 5 long-horizon base features (v6 run 4c): `log_return_120`,
  `log_return_252`, `ema_close_ratio_120`, `ema_close_ratio_252`,
  `volatility_60_252_ratio`
- 2 long-horizon deltas (v6 run 4c): `ema_close_ratio_252_delta_60`,
  `volatility_60_252_ratio_delta_60`

```
                ┌────────────────────┐
raw CSVs        │  feature_engineer  │       unified.parquet
~35k tickers    │                    │       ~3k columns × ~37 features
─────────────►  │  (pipeline stages) │   ─►  + regime features + temporal
2003–present    │                    │       2004-12-13 → present
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │      auditor       │       (read-only verification)
                └────────────────────┘
```

The trainer is a pure consumer of `unified.parquet`. It does no filtering of
its own — every "this ticker shouldn't be trained on" decision is made in the
preprocessing pipeline. If something looks wrong in training, it's either a
genuine model issue or it's coming from upstream of `unified.parquet`; the
trainer is never the place to add data-quality logic.

---

## High-level flow

```
                ┌──────────────────────┐
                │   _load_all_data     │  raw CSVs → one wide frame
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────────────┐
                │ _split_lifecycle_segments    │  detect ≥10-day gaps,
                │                              │  relabel ticker column
                └──────────┬───────────────────┘  e.g. CIT → CIT.1, CIT.2
                           │
                           ▼
                ┌──────────────────────────────┐
                │ _filter_tickers_by_liquidity │  3 gates: $-volume,
                │                              │  history, median price
                └──────────┬───────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ _fetch_sector_labels │  Polygon API → cached parquet
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ _filter_by_ticker_details│  keep CS / ADRC / OS only
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ _fetch_ticker_events │  Polygon → alias mapping
                │ _build_ticker_aliases│  (FB → META etc.)
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   _fetch_splits      │  Polygon → cached parquet,
                │                      │  rebuild aliases w/ suspects
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────────────────────┐
                │         _process_ticker              │  one worker per
                │  (ProcessPoolExecutor, batched)      │  segment, in parallel
                │                                      │
                │  for each segment:                   │
                │   ├─ _apply_split_adjustments        │
                │   ├─ _handle_gaps (forward-fill)     │
                │   ├─ DISCONTINUITY GATE  ◄─ skip if  │  unadjusted-split
                │   │                       breached   │  detector
                │   ├─ _calculate_vwap                 │
                │   ├─ _add_candlestick_features       │
                │   ├─ _add_temporal_patterns          │
                │   ├─ _add_volatility_features        │
                │   ├─ _add_trend_features             │
                │   ├─ _add_volume_features            │
                │   ├─ _drop_unnecessary_features      │
                │   ├─ _normalize_data                 │  rolling z-score
                │   ├─ _prefix_columns ({TICKER}_*)    │
                │   └─ MIN_TICKER_LENGTH GATE          │  ≥ 564 bars
                │                                      │
                │   write per-segment parquet          │
                └──────────────┬───────────────────────┘
                               │
                               ▼
                ┌────────────────────────────────┐
                │ _cross_sectional_normalization │  CS / sector z-scores
                └──────────┬─────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   _build_unified     │  horizontal stack →
                │                      │  + breadth, RS-vs-SPY,
                │                      │    VIX term structure,
                │                      │    sector one-hots
                └──────────┬───────────┘
                           │
                           ▼
                  unified.parquet
```

---

## Module layout (post-v6 refactor)

During v6, `feature_engineer.py` was split from a single ~3,000-line file
into an orchestrator (~2,300 lines) plus 8 helper modules under
`src/v6/preprocessing/`:

```
constants.py          shared thresholds, SHARED_REGIME_PREFIXES,
                      MIN_TICKER_LENGTH, sector ID map
sectors.py            sector label resolution + fallback rules
nyse_calendar.py      trading-day calendar utilities
splits.py             corporate action / split handling
gaps.py               discontinuity gate, lifecycle segmentation
features.py           per-ticker feature computation
                      (candlestick, volatility, trend, volume, temporal)
normalization.py      per-ticker rolling z-score + per-ticker deltas
regime_features.py    shared regime feature computation
                      (kept in tree but partially unused — see notes)
worker.py             per-ticker dispatch orchestration
```

The orchestrator (`feature_engineer.py`) imports from these and exposes
the same public surface as before. Verified bit-identical output via
`df.equals` on `unified.parquet` immediately after the split.

When adding new features:
- Per-ticker features → `features.py` (`_add_*_features` family)
- Per-ticker post-normalization deltas → `normalization.py` (extend
  `DELTA_FEATURES_MEDIUM` or `DELTA_FEATURES_LONG`)
- Shared regime features → `regime_features.py` (and add the prefix to
  `constants.SHARED_REGIME_PREFIXES`)
- New filter rules → wherever the relevant existing logic lives
  (`gaps.py` for time-series sanity gates, etc.)

The `SHARED_REGIME_PREFIXES` constant is imported by both the auditor and
the trainer. **Adding a new shared regime feature family means appending
its prefix to this constant once** — both auditor categorization and
trainer column classification pick it up automatically. v6 had a
classifier bug caused by hardcoding `("breadth_", "vix_term_")` in two
places; the shared constant fixes that.

---

## Stage-by-stage walkthrough

### Stage 1: `_load_all_data`

Reads every CSV under `raw/day_data/*.csv.gz`, concatenates into one long
DataFrame. Columns: `timestamp`, `ticker`, OHLCV.

This is the only stage that touches disk for raw data; everything downstream
operates in memory or against the loaded frame.

### Stage 1.5: `_split_lifecycle_segments`

The most consequential pre-filter step.

A symbol that trades, stops for an extended period, then resumes is not a
single continuous price series. The most common cases:

- Bankruptcy + emergence (CIT: filed Nov 2009, emerged Dec 2009, ~28-day gap
  but legally a different entity)
- Exchange transfer with long interruption
- Symbol reuse by an unrelated entity after dormancy

Treating these as one continuous ticker would have `_handle_gaps` forward-fill
across the gap, then produce a spurious "jump" on the relist day. That fake
move was the root cause of the 2088%-return outliers seen in run 1.

**Detection.** Convert each bar's timestamp to a trading-day index against a
single global NYSE calendar, then per-ticker compute the gap to the previous
bar. Gaps of `GAP_SPLIT_TRADING_DAYS = 10` or more trigger segmentation.
Gaps in `[GAP_SUSPICIOUS_TRADING_DAYS, GAP_SPLIT_TRADING_DAYS) = [5, 10)` are
logged for manual review but not split — that band is the "ambiguous" zone
where halts and short corporate actions both occur.

**Naming.** Multi-segment tickers get `{ticker}.{N}` (1-indexed): `CIT.1`,
`CIT.2`. Single-segment tickers keep their raw symbol — no churn on the
~99%+ of the universe that didn't have a long gap. The `.N` suffix
deliberately cannot collide with Polygon class-share symbols like `BRK.B`
because those are loaded before segmentation, and `.` here is always
followed by an integer.

**Source-symbol mapping.** `self.segment_to_source` maps each segment name
back to the original Polygon symbol. Used by every stage that needs to query
an external API (splits, sectors, ticker events) — those caches are keyed by
source symbol, not segment name. `self.segment_end_dates` similarly maps to
the segment's last calendar date, used by `_apply_split_adjustments` to scope
splits correctly.

### Stage 2: `_filter_tickers_by_liquidity`

Three gates, applied as a single Polars filter:

1. **Average daily dollar volume ≥ $10M.** Ensures the model is training on
   stocks whose sizes are realistic to fill in production.
2. **Trading history ≥ 504 bars.** Two years minimum, after which the
   normalization warmup leaves enough usable post-cutoff data.
3. **Lifetime-median close ≥ $5.** The penny-stock floor. Dollar volume
   measures trading activity; price floor measures whether per-share price
   is meaningful enough that fixed-tick frictions don't dominate. A
   $0.30 stock trading 50M shares/day clears the dollar-volume floor while
   still being uninvestable — at sub-$5 prices the env's spread/slippage
   model breaks down.

Why median (not mean / not min): mean is sensitive to brief pumps; min is
too strict (would exclude a stock that briefly dipped during a market crash).
Median captures where the ticker "lived" most of its history.

Regime tickers (SPY, VIXY, VIXM, etc.) bypass the filter. They're context,
not trading targets, and a few of them legitimately don't meet liquidity
thresholds throughout history.

### Stage 3: `_fetch_sector_labels`

Polygon API call (cached to `raw/sectors/sectors.parquet`). Hits
`get_ticker_details` first, falls back to `list_tickers` for delisted tickers
where the details endpoint returns 404.

Operates on **source symbols** — both `CIT.1` and `CIT.2` look up `CIT` and
inherit its sector. Defensible: companies don't usually change SIC codes
across a bankruptcy. The edge case where a delisted symbol gets reused by an
entity in a different sector is a known weakness; rare enough that we don't
build for it.

### Stage 4: `_filter_by_ticker_details`

Keeps only `CS` (common stock), `ADRC` (American Depositary Receipts), and
`OS` (ordinary shares). Drops:

- Warrants (`W` suffix)
- SPAC units (`U` suffix)
- Preferred stock (`p` infix patterns like `JPMpA`)
- Class-share variants we're not training on

This filter lives **after** segmentation by design. When run on the raw
universe, it would slow down segmentation by ~10× because `_split_lifecycle_segments`
operates on every symbol that has a CSV file — including the 30,000+
warrants/units/preferreds we'd then drop. The trade-off is a noisier
segmentation log; the segments declared on dropped symbols never reach
feature engineering and are dropped between Stage 4 and Stage 7.

### Stage 5: `_fetch_ticker_events` + `_build_ticker_aliases`

Polygon's ticker-events endpoint (cached to `raw/ticker_events/`). Detects
historical symbol-rename events: `FB → META`, etc. Polygon labels this
endpoint "experimental" and coverage for old tickers is uneven.

The first alias-build pass runs **without** suspect filtering — needed because
`_fetch_splits` (next stage) wants to fetch splits for both the current and
previous symbols, and at this point the splits cache doesn't exist yet so
suspects (where the previous symbol's splits postdate the rename) can't be
detected.

### Stage 6: `_fetch_splits`

Polygon API for stock splits. Cached to `raw/splits/splits.parquet`. After
the fetch, `_build_ticker_aliases` runs again — this time *with* suspect
filtering. An "alias" is suspect if the previous symbol has splits dated
*after* the rename event, indicating the symbol was reused by an unrelated
entity rather than a true rename.

### Stage 7: Per-ticker dispatch (`_process_ticker` in workers)

The hot path. One `ProcessPoolExecutor` worker per segment. The pipeline
inside each worker:

1. **`_apply_split_adjustments`** — multiply prices by Polygon's
   `historical_adjustment_factor`, divide volume by the same factor.
   Lifecycle-aware: filters splits to `execution_date <= segment_end_date`
   so a CIT.2 reverse split doesn't bleed backward into CIT.1's adjustment.
2. **`_handle_gaps`** — left-join onto a continuous NYSE calendar within
   the segment, forward-fill prices and zero-fill volume on any remaining
   nulls. After segmentation these are guaranteed to be ≤ 10 days, so
   forward-fill is now a defensible operation.
3. **DISCONTINUITY GATE** — compute `np.diff(np.log(close))` over the full
   segment. If any single-day `|log_return|` exceeds
   `DISCONTINUITY_LOG_THRESHOLD = log(2.0) ≈ 0.6931` (a 2× up or 50% down
   single-day move), skip the segment entirely. Returns
   `SKIP:{ticker}:discontinuity (...)` to the orchestrator. This catches
   unadjusted corporate actions that Polygon's splits endpoint doesn't have
   records for (WB.1, WELL, SDRL.1, ICON.1, etc.) — the residual category
   after `_apply_split_adjustments` does its work.
4. **Feature stack:** `_calculate_vwap`, `_add_candlestick_features`,
   `_add_temporal_patterns`, `_add_volatility_features`, `_add_trend_features`,
   `_add_volume_features`. Pure functions, no shared state.
5. **`_drop_unnecessary_features`** — strips intermediates, keeps only
   what's ML-ready.
6. **`_normalize_data`** — rolling 252-day z-score per feature
   (`_rolling_zscore_normalize_vectorized`), clipped to ±3. Operates on the
   full timeline including pre-cutoff warmup, so the rolling-window leadup
   nulls land in the discarded warmup region rather than in actual training
   data. The cutoff (default 2004-12-13) gets enforced later in
   `_build_unified`.

   After z-scoring, the same function computes per-ticker level deltas on
   a hand-picked subset of features (delta-of-z-scored-level convention,
   matching the regime delta machinery). Two delta horizons:

   - **Medium (delta_20)** on 5 features (v6 run 4a):
     `ema_close_ratio_20`, `ema_20_60_ratio`, `adx_5_20_ratio`,
     `volatility_5_20_ratio`, `volume_5_20_ratio`.
   - **Long (delta_60)** on 2 long-window features (v6 run 4c):
     `ema_close_ratio_252`, `volatility_60_252_ratio`.

   Deltas inherit the same scale as their base features (no separate
   normalization needed). The first `delta_window` rows are filled with
   zero via `fill_null(0.0)` and live in the discarded warmup region.
7. **`_prefix_columns`** — every feature column gets prefixed with
   `{TICKER}_` for the unified frame. Ticker name here is the **segment
   name**, not the source symbol.
8. **MIN_TICKER_LENGTH GATE** — `MIN_TICKER_LENGTH = 564`
   (`NORMALIZATION_WINDOW + WINDOW_SIZE + 252`). Ensures every saved segment
   has at least one full episode's worth of post-warmup data. Anything
   shorter is dropped with `SKIP:{ticker}:{n} rows < {min}`.
9. Write `tickers/{segment_name}.parquet`.

The orchestrator collects results, tallies skip/error counts per category,
and reports a summary at end of dispatch:

```
Per-ticker pipeline complete: 2950 succeeded, 612 skipped, 0 errors
  Skipped [discontinuity]: 487  (sample: AHM.2, AWH.1, BAS, BCEI, ...)
  Skipped [{rowcount}]:    125  (sample: ...)
```

### Stage 8: `_cross_sectional_normalization`

For each timestamp, computes z-scores of a feature subset across all
tradable tickers (CS z-score) and within each sector (sector z-score).
Adds `_cs_zscore` and `_sector_zscore` suffixed columns alongside the
per-ticker rolling z-scores from Stage 7.

### Stage 9: `_build_unified`

Horizontal stack of every per-segment parquet onto one timestamp axis
(left-joined onto the NYSE calendar). Plus shared regime features
computed at this stage by functions in `regime_features.py`:

- **Breadth**: percentage of tradable tickers above moving averages, at
  multiple horizons (5d, 20d, 60d, 200d) plus deltas
- **VIX term structure**: `log(VIXY/VIXM)` and rolling-window variants
  (5d, 20d, 60d) plus deltas
- **RS-vs-SPY**: per-ticker relative strength against SPY at multiple
  horizons (5d, 20d, 60d, 252d) plus `delta_20` and `delta_60` on the
  long-horizon level

`regime_features.py` also contains computation for 5 macro regime
feature families (yield curve, credit spread, size factor, growth factor,
sector rotation) added in v6 run 3a. **These are NOT called in the v6
capstone configuration** — run 3a was rejected during MC and the call
sites in `_build_unified` were reverted. The functions remain in the
module as opt-in plumbing if a future run wants to revisit macro regime
features (e.g. v7's allocation problem might value them differently from
v6's single-ticker timing). All shared regime feature prefixes,
including the rejected 3a families, are listed in
`constants.SHARED_REGIME_PREFIXES`.

`_resolve_close_col` is the helper that handles lifecycle-segmented
regime tickers transparently (HYG.2, QQQ.2 — the .1 segments got dropped
by MIN_TICKER_LENGTH). Each regime-feature computation routes through
this helper rather than hardcoding the segment suffix.

Sector one-hot encoding gets added per-ticker (`{TICKER}_sector_0` …
`{TICKER}_sector_{N_SECTORS-1}`). Each ticker gets the full block of
zero-or-one columns, redundant for any single ticker but useful for
cross-sectional aggregations downstream.

The 0-fill at the end is **not** applied to `_close` columns. Closes
intentionally retain null in non-tradable rows (pre-IPO, post-delisting,
inter-segment). Other features get `fill_null(0.0)`, which is
semantically valid because z-scored features have 0.0 = neutral signal.

Finally, `_drop_rows_before_timestamp(CUTOFF_TIMESTAMP)` discards the
warmup rows. The cutoff math: 60-day upstream feature window + 252-day
rolling z-score warmup = ~312 trading days, so the cutoff lands at
2004-12-13 (~324 trading days after raw data starts).

---

## Cross-cutting concerns

### Source ticker vs segment name

A persistent distinction throughout the pipeline. `CIT.1` and `CIT.2` are
**segment names**. `CIT` is the **source ticker** (the literal Polygon symbol).

| Used for                              | Which name |
|---------------------------------------|------------|
| Polygon API calls (splits, sector, events) | source |
| Splits / sector / events caches            | source |
| Per-segment parquet files                  | segment |
| Unified-parquet column prefixes            | segment |
| Trainer's ticker pool                      | segment |
| Audit reports                              | segment |

The mapping lives on the engineer instance:

- `self.segment_to_source: Dict[str, str]` — segment → source
- `self.segment_end_dates: Dict[str, date]` — segment → last date

For single-segment tickers, both maps are populated trivially (segment name
== source symbol).

### Vendor caveats — what's compensating for what

This pipeline does substantial work that wouldn't be necessary if we had a
data vendor with cleaner historical-corporate-actions coverage. Worth being
explicit about which compensations are vendor-specific (i.e. would shrink or
disappear with a vendor switch) versus universal (i.e. correct preparation
regardless of source).

**Vendor-specific (Polygon):**

- **DISCONTINUITY GATE.** Filters out tickers with single-day moves > 100%
  on adjusted prices. If `_apply_split_adjustments` had complete coverage,
  the only remaining critical breaches would be genuine penny-stock
  moonshots — which are already caught by the price floor. Most of what
  this gate catches is *splits Polygon doesn't have records for*. A vendor
  with stronger historical splits coverage (e.g. Databento) would
  significantly reduce this filter's catch.
- **Manual ticker alias handling.** Polygon's ticker-events endpoint is
  experimental and uneven for older symbols. We work around this with
  manual alias overrides and suspect detection. A vendor with a robust
  symbology service (Databento again) would handle this natively.

**Universal:**

- **Lifecycle segmentation.** Bankruptcy + relist is a discontinuous price
  series regardless of vendor. Treating CIT.1 and CIT.2 as separate is the
  *correct* representation, not a workaround.
- **Penny-stock floor.** NAKD's +8000% day is real and faithfully reported
  by any vendor. We exclude penny stocks because the env's microstructure
  assumptions don't hold there, not because any data is wrong.
- **Liquidity, history, type, sector filters.** Universal data preparation.
- **Lifecycle gap threshold (10 days).** Universal.

If you're considering a vendor switch, the discontinuity gate's catch count
is the cleanest measure of how much the current data quality is costing —
that filter directly counts how many tickers Polygon's splits endpoint is
missing. Currently around 500/3500 ≈ 14% of the post-filter universe, which
is meaningful but not catastrophic.

### Known data limitations

Things that are wrong or imperfect in `unified.parquet` that we accept rather
than fix. Documented here so future-me doesn't waste a debugging session
re-discovering them.

**Suspect 50–100% single-day moves (~1,100 events).**

The discontinuity gate filters at 100% (`log(2.0)`). The audit reports a
"suspicious" bucket at 50% (`log(1.5)`) which is *not* filtered. Inspection
of the suspect bucket showed disproportionate clustering at exact 50% / 100%
boundaries — the mathematical signatures of 2:1 forward splits and 1:2
reverse splits — meaning Polygon's splits cache is missing some splits even
on otherwise-fine tickers.

The clearest example: GOOG's 2014-04-03 Class C creation (effectively a 2:1
split) is not in Polygon's splits endpoint. We verified this by hitting
the API directly. The endpoint returns only the 2022 20:1 stock dividend
for GOOG, missing the much more consequential 2014 event.

We're choosing to live with this rather than tighten the gate to 50%, because:

- Tightening would exclude ~600 additional tickers, most of which are
  clean stocks with one unrelated weird day
- Probabilistic impact on training is bounded: ~0.5-2% of episodes per
  affected ticker would touch the bad day
- The agent has to handle noisy data eventually anyway; flash crashes
  and fat-finger events do happen in real markets

If a future training run shows the model is exploiting these single-day
artifacts at scale (Train Max returns >200% across many tickers), revisit:
either tighten the gate, add manual overrides for high-importance names
(GOOG specifically), or build the algorithmic split-detection logic
described in path B of the vendor-caveats discussion.

**Sector labels for lifecycle-split segments.**

Both `CIT.1` and `CIT.2` inherit the same SIC code from the source-ticker
sector cache. Defensible for CIT (financial services pre and post BK) but
breaks if a delisted symbol gets reused by an entity in a different
sector. Edge case, not building for it.

**Polygon's experimental ticker-events endpoint.**

Coverage of historical symbol renames is uneven for older tickers. We
detect and reject obvious suspects (where the previous symbol has splits
postdating the rename, indicating reuse rather than rename) but cannot
detect cases where the events endpoint is silently incomplete. Manual
overrides via `TICKER_ALIASES` in `config.py` cover known important cases.

**Cross-sectional / sector z-scores are highly correlated.**

By design — same input feature, slightly different normalization scope.
Audit reports `cs_zscore <-> sector_zscore` correlations of 0.98+ for
many features. Informational, not a bug. Could prune one of each pair
if input dimensionality becomes a constraint, but currently the cost is
negligible.

**Long-horizon trend features drift across regimes.**

Audit's temporal-drift check flags tens of thousands of long-window
features (`adx_20_60_ratio`, `ema_20_60_ratio`, `log_return_60`) as
drifting across yearly windows. Expected — long-horizon trend signals
inherently shift through 2008, 2020, and other regime breaks. Would only
be a real concern if short-horizon features (`log_return_5`, `adx_14`)
showed up as drifted, which they don't.

**Long-horizon features (252d window) are effectively absent for
short-lifecycle tickers.**

The v6 4c features (`log_return_252`, `ema_close_ratio_252`,
`volatility_60_252_ratio`, and their `delta_60` variants) need 252 bars
of feature-relevant history before they produce non-trivial values, on
top of the existing 252-day rolling z-score warmup. A ticker with
~500-700 post-warmup bars will have these features carrying meaningful
signal for only the last ~250 bars of its lifecycle; before that the
rolling z-score sees ~zero variance in the underlying long window and
the z-scored value sits near 0 (= neutral signal, harmless but
uninformative).

For lifecycle-segmented regime tickers (HYG.2, QQQ.2), the segment
boundary creates the same effect — long-horizon features need ~252
bars *after* the segment start before they produce meaningful signal.

Not a bug — the alternative (NaN-padded long-horizon features) would
break the network. The current zero-fill behavior is the correct trade-
off, but the practical effect is "long-horizon features are absent for
short-lifecycle tickers / segment beginnings." Downstream code (trainer,
agent) handles z-score 0 as neutral, so no special-casing needed.

---

## Filtering decision tree

A ticker passes through these gates in order. Each gate catches a different
failure mode; they're complementary, not redundant.

```
Raw symbol from CSV
    │
    ├─ has multi-day gap ≥ 10 days? ──► split into segments
    │
    ▼
Per-segment from here on
    │
    ├─ avg dollar volume < $10M?  ──► drop  (insufficient liquidity)
    ├─ trading days < 504?         ──► drop  (insufficient history)
    ├─ median close < $5?          ──► drop  (penny stock)
    │
    ├─ ticker type ∉ {CS,ADRC,OS}? ──► drop  (warrant/unit/preferred)
    │
    ├─ inside _process_ticker:
    │  ├─ critical discontinuity?  ──► drop  (unadjusted split / data bug)
    │  └─ post-feature length < 564? ──► drop (insufficient post-warmup data)
    │
    ▼
Saved as tickers/{segment_name}.parquet
    │
    ▼
Joined into unified.parquet
```

If a ticker is showing up where you don't expect it, walk this tree
top-to-bottom.

---

## On-disk artifacts

Generated by the pipeline, organized under `data/`:

```
raw/
├── day_data/                          # source — one CSV per ticker
├── splits/splits.parquet              # cached Polygon splits
├── sectors/sectors.parquet            # cached SIC labels
└── ticker_events/                     # cached Polygon events

preprocessed/v6/
├── tickers/
│   └── {SEGMENT_NAME}.parquet         # per-segment, post-feature, post-norm
└── unified/
    └── unified.parquet                # the trainer's actual input
```

All `raw/*` caches are populated incrementally. Only re-fetch if you delete
them or call the relevant fetch methods directly. The pipeline doesn't
auto-invalidate them — assume they're valid unless something specific
changed.

`unified.parquet` schema, in column order: `timestamp`, then for each
segment `{SEGMENT}_close`, `{SEGMENT}_open`, … (per-ticker market features
including 4a deltas and 4c long-history features), then
`{SEGMENT}_sector_0` … `{SEGMENT}_sector_10` (one-hots), then shared
regime features (`breadth_*`, `vix_term_structure_*`, `rs_spy_*`,
`rs_spy_*_delta_*`), then temporal features. The trainer uses column-
name patterns (suffixes for per-ticker, prefixes from
`SHARED_REGIME_PREFIXES` for regime) to classify columns into market /
regime / temporal / close groups.

unified.parquet is currently ~210k columns (~3000 tickers × ~37
per-ticker features + ~110 shared regime / temporal / sector columns).

---

## Verification — what the auditor checks

The auditor (`auditor.py`) is read-only and runs after preprocessing. Its
job is to verify the pipeline's output, not to filter or modify. After all
the filters in the pipeline, the auditor's discontinuity check should
report ~0 critical breaches. If it doesn't, something in the filter chain
isn't catching what it should.

Checks, in order:

1. **`_check_data_shape_and_size`** — frame dimensions, time range,
   total ticker count, downcast analysis. Sanity baseline.
2. **`_check_missing_and_invalid`** — null / NaN / Inf counts. `_close`
   columns whose nulls are entirely a contiguous prefix and/or suffix
   (i.e. pre-IPO + post-delisting only) are suppressed from the warning
   list — those are intentional, not bugs. The survivorship check covers
   them separately. Interior nulls in `_close` would still be flagged;
   those would indicate a real data corruption.
3. **`_check_statistics`** — high-mean, low-variance, high-skew, clipping
   saturation, temporal drift. The drift check often flags large numbers
   of long-horizon features (`adx_20_60_ratio`, `ema_20_60_ratio`,
   `log_return_60`) — that's expected, those features inherently shift
   across regime changes (2008, 2020). Worth checking if short-horizon
   features show up as drifted.
4. **`_check_survivorship_coverage`** — split into "active" and "likely
   delisted" buckets. Real-world equity universes over 22 years see
   30–50% delisting rates; numbers in that range are normal.
5. **`_check_price_discontinuities`** — single-day log-return breaches.
   Suspicious bucket (≥ 50% move) and critical bucket (≥ 100% move).
   The critical count should be near zero post-fix; that's the canary
   for whether the discontinuity gate caught everything it should have.
6. **`_check_regime_multicollinearity`** — sanity-checks the shared
   regime features for redundant pairs (some are expected, like
   `vix_term_structure_5d` and `_20d` being highly correlated).
7. **`_check_for_multicollinearity`** — per-ticker feature-pair
   correlations. CS z-score and sector z-score are highly correlated by
   design (same input, slightly different normalization scope); that's
   informational, not a bug.

---

## Operational notes

### When to rerun

Full preprocessing rerun required when:

- Filter thresholds change (`min_dollar_volume`, `min_history_days`,
  `min_median_price`, `GAP_SPLIT_TRADING_DAYS`, `DISCONTINUITY_LOG_THRESHOLD`)
- Feature definitions change (any of the `_add_*_features` functions)
- Cutoff date changes
- Sector mapping changes (`_sic_to_sector`)
- New raw CSVs added

API-call cache rebuild only (`raw/splits/`, `raw/sectors/`, `raw/ticker_events/`):

- Polygon corrects historical data
- New tickers added that need fresh sector / events lookups
- Suspect-alias detection logic changes

### Wall-clock expectations

Approximate timings on a 16-thread machine, ~3,500-segment universe:

- `_load_all_data`: ~1–2 minutes (depends on disk)
- `_split_lifecycle_segments`: ~30 seconds
- `_filter_tickers_by_liquidity`: ~10 seconds
- API-cached steps (sectors, events, splits): seconds if cached, minutes if not
- Per-ticker dispatch: ~20–40 minutes (the bulk of the time, parallelizable)
- `_cross_sectional_normalization`: ~5 minutes
- `_build_unified`: ~5–10 minutes

Total: ~40–60 minutes for a full rerun on a fresh API-call cache. ~30 minutes
when caches are populated.

### What to watch in logs

- **`Lifecycle-splitting N ticker(s)`** — how many CIT-style splits
  happened. Eyeball the list; anything unexpected (a mega-cap unexpectedly
  splitting) is worth investigating.
- **`Liquidity filter:` summary line** — per-gate breakdown. Useful for
  tuning thresholds without multiple reruns.
- **`Skipped [discontinuity]: N`** — how many tickers got excluded for
  unadjusted-split issues. This number is the canary for vendor data
  quality; if it suddenly grows, something changed upstream.
- **End-of-dispatch summary** — `done`, `skipped`, `errors`. Errors should
  always be 0; investigate immediately if not.

---

## Future work

Tracked in TODO comments throughout the code. The most consequential
open items:

- **Vendor evaluation (Databento migration)** — see "Vendor caveats"
  above. Pre-staked as a v7+ data quality improvement. Cleaner historical
  corporate actions would recover the ~14% of the universe currently lost
  to the discontinuity gate. Most useful paired with v7+ work since v7's
  small basket doesn't need it; bigger v8+ universes would.

- **Universe-selection survivorship bias** — discontinuity gate,
  liquidity filter, MIN_TICKER_LENGTH, and ticker-events filtering all
  run once on the full timeline before walk-forward splits, so each fold
  trains on a universe pre-selected for full-timeline survival. Fixing
  per-fold is expensive (9× preprocessing). v7's hand-picked basket
  sidesteps the problem for free. Documented in detail in `v7_handoff.md`.

- **Macro regime feature redesign** — `regime_features.py` contains
  implementations of yield curve, credit spread, size factor, growth
  factor, and sector rotation feature families added in v6 run 3a but
  not currently called from `_build_unified` (the run rejected during
  MC for single-ticker timing). v7's allocation problem may value these
  differently — the call sites in `_build_unified` are commented out, so
  reactivating them is a one-line change. New regime feature families
  would need their prefix added to `constants.SHARED_REGIME_PREFIXES`.

- **Underutilized regime tickers** — currently the shared regime feature
  set only effectively uses SPY + VIXY + VIXM + (in the rejected 3a code)
  TLT/SHY/HYG/LQD/IWM/QQQ/XL* out of ~25 configured regime tickers.
  Dollar regime (UUP), commodities (GLD, USO), international RS (EFA,
  EEM, EWJ), additional rate signals — all reasonable additions to a
  future regime feature pass that would pull more value from already-
  loaded regime tickers.

**Recently completed (v6):**

- ✅ **File-size refactor of `feature_engineer.py`** — done in v6.
  Split from one 3079-line file into 8 helper modules + 2298-line
  orchestrator under `src/v6/preprocessing/`. Verified bit-identical
  output via `df.equals`.
- ✅ **Long-horizon per-ticker features** — done in v6 run 4c.
  `log_return_120/252`, `ema_close_ratio_120/252`,
  `volatility_60_252_ratio`, with `delta_60` on the long-window levels.
  Lives in `features.py` and `normalization.py`.
- ✅ **Per-ticker medium-horizon deltas** — done in v6 run 4a. Five
  `delta_20` features in `normalization.py`.
- ✅ **`SHARED_REGIME_PREFIXES` constant** — single source of truth for
  shared regime feature prefixes, imported by both auditor and trainer.
  Fixed a silent classifier bug in v6.
- ✅ **`_resolve_close_col` helper** — transparent handling of
  lifecycle-segmented regime tickers (HYG.2, QQQ.2) in regime feature
  computation.