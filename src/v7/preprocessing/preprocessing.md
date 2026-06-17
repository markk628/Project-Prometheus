# Preprocessing pipeline — design doc

Reference for the data-preprocessing layer that turns raw Polygon daily bars
into the single `unified.parquet` consumed by the SAC trainer. Covers every
module in `src/v7/preprocessing/` plus the relevant `config.py` / `constants.py`
knobs.

---

## 1. Purpose & output

The pipeline ingests ~22 years of raw daily OHLCV CSVs for the full US-equity
universe (plus ~28 regime ETFs), engineers a causal, leakage-free feature set,
normalizes it at three different scopes, and emits two artifact tiers:

- **Per-ticker parquets** — `data/preprocessed/v7/tickers/{ticker}.parquet`,
  one per surviving ticker/lifecycle-segment. Intermediate; overwritten twice.
- **Unified parquet** — `data/preprocessed/v7/unified/unified.parquet`. The
  single training input: every tradable ticker horizontally stacked on a shared
  timestamp index, with cross-ticker regime features bolted on.

There is **no train/valid/test split** at this layer. The trainer owns
walk-forward fold boundaries at runtime.

---

## 2. Entry point & orchestration

`preprocessor.py` is what actually runs:

```
DataPreprocessor.preprocess_data()
  ├─ DataFeatureEngineer.feature_engineer_and_save_tickers(24)   # build
  └─ DataAuditor.audit()                                          # QA
```

`POLARS_MAX_THREADS = "1"` is set at module import in both `preprocessor.py`
and `feature_engineer.py`. Parallelism is owned by the `ProcessPoolExecutor`
(one worker per ticker); forcing Polars single-threaded inside each worker
prevents thread oversubscription.

The `24` passed to `feature_engineer_and_save_tickers` is `batch_size` — how
many tickers are submitted to the pool per batch (a memory/back-pressure knob),
**not** worker count. Worker count is `max_workers` (defaults to `os.cpu_count()`).

---

## 3. Module map

| Module | Responsibility |
| --- | --- |
| `preprocessor.py` | Top-level entry; runs feature engineer then auditor. |
| `feature_engineer.py` | Orchestrator. Load, lifecycle-split, filter universe, fetch metadata (sectors/types/aliases/splits), dispatch workers, cross-sectional normalize, build unified file. |
| `worker.py` | `_process_ticker` — the per-ticker pipeline run inside each pool worker. Top-level/picklable, no shared state. |
| `splits.py` | `_apply_split_adjustments` — split-adjust OHLCV via Polygon `historical_adjustment_factor`, with lifecycle-segment-aware filtering. |
| `gaps.py` | `_handle_gaps` — left-join bars onto the NYSE trading-day grid, forward-fill prices through halts, zero-fill volume. |
| `nyse_calendar.py` | `_build_nyse_valid_days` — NYSE trading-day grid via `pandas_market_calendars`. |
| `features.py` | Per-ticker feature computations (VWAP, candlestick, temporal, volatility, trend, volume) via TA-Lib + Polars. |
| `normalization.py` | Causal rolling z-score (`shift(1)` + rolling, winsorize, tanh), per-ticker delta features, ticker-column prefixing. |
| `regime_features.py` | Shared macro-regime families (yield curve, credit spread, size, growth, sector rotation). |
| `sectors.py` | SIC→12-sector taxonomy + ETF sector overrides. |
| `constants.py` | Pipeline thresholds + shared-regime column-prefix source of truth. |
| `config.py` | Global config: tickers, paths, windows, hyperparameters, fee model. |
| `auditor.py` | `DataAuditor` — post-build QA over `unified.parquet` and per-ticker parquets. |

---

## 4. End-to-end data flow

`feature_engineer_and_save_tickers` runs nine numbered steps; steps 7–9 contain
the three normalization passes.

1. **Load** all daily CSVs into one Polars frame (`_load_all_data`). Timestamps
   truncated to date (midnight UTC) so raw-load and gap-fill paths produce
   joinable timestamps.
2. **Lifecycle-split** (`_split_lifecycle_segments`) — relabel the `ticker`
   column so each continuous lifetime is its own series (see §6.2).
3. **Liquidity filter** (`_filter_tickers_by_liquidity`) — avg dollar volume,
   history length, median price floor. Regime tickers always retained.
4. **Sector + type metadata** (`_fetch_sector_labels`) — Polygon SIC → sector,
   ticker type/name; cached to parquet. Then `_filter_by_ticker_details` keeps
   only tradable types `{CS, ADRC, OS}` (regime ETFs exempt).
5. **Ticker events / aliases** (`_fetch_ticker_events`, `_build_ticker_aliases`)
   — discover symbol renames (FB→META) so historical data can be stitched.
6. **Splits** (`_fetch_splits`) — fetch + cache Polygon splits (including alias
   symbols), then rebuild aliases with *suspect* filtering (drop aliases whose
   previous symbol has splits dated after the rename → likely ticker reuse).
7. **Per-ticker workers** — partition by ticker, resolve aliases, dispatch
   `_process_ticker` to the pool in batches. Pass 1 of normalization happens
   here. Writes per-ticker parquets.
8. **Cross-sectional normalization** (`_cross_sectional_normalization`) — pass 2.
   Reload all parquets, z-score the held-back feature subset globally and per
   sector. Overwrites parquets.
9. **Unified build** (`_build_unified`) — pass 3 (cross-ticker regime features).
   Stack, add regime features, fill nulls, prune regime columns, trim warmup,
   write `unified.parquet`.

Then `DataAuditor.audit()` QAs the result (see §8).

---

## 5. Stage detail

### 5.1 Per-ticker worker pipeline (`worker._process_ticker`)

Runs once per ticker/segment in a pool worker. Operates on the **full timeline**
(no trimming) so rolling-warmup zeros land in the pre-cutoff period. Order:

```
_apply_split_adjustments     (splits.py)
_handle_gaps                 (gaps.py)
── DISCONTINUITY GATE ──      (constants.py)   bail if max |daily log-ret| ≥ log 2
_calculate_vwap              (features.py)
_add_candlestick_features    (features.py)
_add_temporal_patterns       (features.py)
_add_volatility_features     (features.py)
_add_trend_features          (features.py)
_add_volume_features         (features.py)
_drop_unnecessary_features   (features.py)     drop raw OHLCV (keep close)
_normalize_data              (normalization.py) rolling z-score + deltas
_prefix_columns              (normalization.py) {feat} → {ticker}_{feat}
── MIN_TICKER_LENGTH GATE ──  (constants.py)    skip if < 564 bars
```

Workers return `"{ticker}"` on success, `"SKIP:{ticker}:{reason}"`, or
`"ERROR:{ticker}:{msg}"`; the orchestrator tallies a per-reason skip summary.

**Split adjustment** (`splits.py`): for each bar, `join_asof(strategy="forward")`
on `bar_date + 1 day` finds the first split with `execution_date` strictly after
the bar; price columns are multiplied by `historical_adjustment_factor`, volume
divided by it (preserves price × volume). For lifecycle segments, splits with
`execution_date > segment_end_date` are filtered out *before* the join so a
later entity's splits don't leak into an earlier lifetime.

**Gap handling** (`gaps.py`): left-join onto the NYSE grid, drop leading rows
before the first valid close, forward-fill prices, zero-fill volume/transactions,
reconstruct a 20:00-UTC timestamp. Forward-fill is only ever applied *within* a
segment — long delist/relist gaps were already split out upstream.

### 5.2 Per-ticker features (`features.py`)

All use only current-bar OHLC and `shift(1)` of prior close — no look-ahead.

- **VWAP**: typical price `(high + low + close) / 3` (daily proxy; no intraday
  volume distribution). Used by `price_vwap_distance`, then dropped.
- **Candlestick**: `overnight_gap` = log(open/prev_close), `intraday_return` =
  log(close/open), `upper_wick_ratio`, `lower_wick_ratio`. (`close_location` was
  removed — >0.96 corr with `price_vwap_distance`.)
- **Temporal** (shared across tickers, not prefixed): `day_sin/cos`,
  `month_sin/cos`, `quarter_sin/cos`.
- **Volatility**: `log_return_{5,20,60,120,252}`, `volatility_5_20_ratio`,
  `volatility_20_60_ratio`, `volatility_60_252_ratio` (log-ratios of rolling
  STDDEV).
- **Trend**: `ema_close_ratio_{5,20,60,120,252}`, `ema_5_20_ratio`,
  `ema_20_60_ratio`, `adx_5_20_ratio`, `adx_20_60_ratio`, `adx_14`.
- **Volume**: `volume_ema_ratio_20`, `volume_5_20_ratio`, `volume_20_60_ratio`,
  `price_vwap_distance`, `avg_trade_size_ratio`, `range_per_trade_ratio`,
  `trade_intensity_ratio`, `volume_price_corr_20` (20-day rolling corr of volume
  vs `price_roc`; `price_roc` is a temp column, dropped).

### 5.3 Normalization pass 1 — per-ticker rolling z-score (`normalization.py`)

`_rolling_zscore_normalize_vectorized` is the causal core used throughout the
codebase. At bar `t` the lookback is `[t-window, t-1]` (achieved via `shift(1)`
+ `rolling(min_samples=window)`), so valid/test rows never see their own bar.
Steps: rolling mean/std + rolling 1st/99th quantiles for winsorization → clip →
z-score → `tanh` squash → warmup rows (NaN stats) set to `0.0`. Window =
`NORMALIZATION_WINDOW = 252`. Replaced an older stride-tricks version that OOM'd;
this is O(T·F) memory.

`_normalize_data` rolling-z-scores one fixed list (`features_to_scale`): the
trend/vol/volume **ratio** features, the candlestick wicks, and the long-horizon
levels (`ema_close_ratio_120/252`, `log_return_120/252`, `volatility_60_252_ratio`).

A specific subset is **intentionally left raw** here, to be cross-sectionally
normalized in pass 2:
`log_return_5/20/60`, `overnight_gap`, `intraday_return`, `price_vwap_distance`,
`volume_price_corr_20`, `adx_14`.

Then **per-ticker deltas** (change-in-z-scored-level over W bars):
- MEDIUM `delta_20` on `ema_close_ratio_20`, `ema_20_60_ratio`, `adx_5_20_ratio`,
  `volatility_5_20_ratio`, `volume_5_20_ratio`.
- LONG `delta_60` on `ema_close_ratio_252`, `volatility_60_252_ratio`.

Finally `_prefix_columns` renames every feature `{feat}` → `{ticker}_{feat}`,
leaving `timestamp` and the shared temporal columns untouched.

### 5.4 Normalization pass 2 — cross-sectional (`_cross_sectional_normalization`)

Reloads all per-ticker parquets. Computes over **tradables only** (regime ETFs
excluded so e.g. XLK doesn't double-count its tech constituents).

For the held-back `CS_FEATURES` (the raw subset from §5.3):
1. **Per-ticker rolling z-score** is applied first, overwriting the base column
   (`volume_price_corr_20` excepted via `CS_ROLLING_SKIP`, since it's already
   bounded `[-1, 1]`).
2. **Per-timestamp cross-sectional z-scores** are then computed on those values
   (`np.nanmean/nanstd` across tickers, `tanh` squash), at two scopes:
   - `{ticker}_{feat}_cs_zscore` — across the whole tradable universe.
   - `{ticker}_{feat}_sector_zscore` — within the ticker's SIC sector peer group
     (sectors with <2 members skipped).

Net effect: each CS feature contributes its rolling-z-scored base **plus** a
global and a sector cross-sectional column. Parquets are overwritten in place.

### 5.5 Normalization pass 3 — unified build (`_build_unified`)

Horizontally stacks all per-ticker frames via left-join off the longest series
(shorter-history tickers get nulls before inception). Shared temporal columns +
`timestamp` are kept from the first frame only. Then adds the cross-ticker
features (§7), with these phases:

1. **Stack** + **sector one-hot** (`{ticker}_sector_0..11`, tradables only).
2. **Breadth** (over tradables), **VIX term structure** (VIXY/VIXM), **RS vs SPY**
   (tradables + V7 basket), the **five macro families**, then **regime momentum
   deltas**.
3. **Null fill**: feature columns → `0.0` (neutral in z-score space). `_close`
   columns are **excluded** — a null close means "not trading" (pre-IPO,
   post-delist, between segments). Done *after* regime features so null-aware
   aggregations work.
4. **Prune** regime-ticker columns, **except** the V7 basket and `SPY_close`.
5. **Trim warmup** (`_drop_rows_before_timestamp`, cutoff `2004-12-13`) — the
   *only* row trim, deferred to the very end so every pass saw full history.
6. **Write** `unified.parquet`.

---

## 6. Load-bearing invariants & design decisions

### 6.1 Causality / leakage safety
Every rolling statistic is `shift(1)`-then-rolling, so bar `t` is normalized
using strictly `[t-window, t-1]`. Cross-sectional stats use only same-timestamp
data from other tickers. These two properties together are the no-look-ahead
guarantee; any new feature must preserve them.

### 6.2 Lifecycle segmentation (the 2088%-return fix)
A symbol that trades, stops for ≥ `GAP_SPLIT_TRADING_DAYS` (10) trading days,
then resumes is split into `{ticker}.1`, `{ticker}.2`, … Each segment is
first-class for the rest of the pipeline. This prevents the forward-fill
flat-line + spurious relist jump + blended cross-lifetime liquidity stats that
produced pathological training-episode returns. Gaps in `[5, 10)` are logged as
"suspicious" but **not** split.

Two shared maps make it work:
- `segment_to_source` — segment name → source Polygon symbol, for API lookups
  (splits/sectors are keyed by source symbol).
- `segment_end_dates` — segment name → last date, so split adjustment can drop
  splits belonging to a later lifetime.

The `.N` naming is deliberate: the unified parquet flattens to `{ticker}_{feat}`
and the trainer parses the ticker via a single split on `_`, so a `.`-suffixed
segment name survives that parse (and can't collide with class-share symbols
like `BRK.B`, which are source symbols resolved before segmentation).

### 6.3 Warmup & cutoff
Longest causal chain is a 60-day upstream window followed by 252-day rolling
z-score ≈ 312 trading days of warmup. `CUTOFF_TIMESTAMP = 2004-12-13` sits ~324
trading days after the raw start, so post-cutoff rows have no zero-padded
features. Because normalization runs on the full timeline and the trim is last,
warmup zeros land before the cutoff, not in training data.

### 6.4 Regime tickers & the V7 basket
~28 regime ETFs are loaded as feature inputs (they drive breadth, VIX term, RS,
and the macro families) but are **not** trained on or traded — their per-ticker
columns are pruned from `unified.parquet`. `SPY_close` is kept as a benchmark
reference.

**V7 change:** `V7_BASKET = ['SPY', 'TLT', 'GLD', 'USO', 'UUP']` are regime
tickers that *are* trained on (the "regime = context only" rule is broken for
exactly these five). They're exempt from the prune and get `rs_spy_*` features
computed. `action[i]` in the v7 env maps to `V7_BASKET[i]` — order is locked.

### 6.5 Two-scope normalization rationale
Per-ticker rolling z-score answers "is this unusual vs the ticker's own
history." Cross-sectional z-score answers "is this unusual vs the universe right
now." Return/gap-type features get the CS treatment because their ticker-specific
scale makes cross-sectional comparison strictly more informative; they're held
raw through pass 1 specifically so pass 2 can normalize them with all tickers
visible.

---

## 7. Feature taxonomy in `unified.parquet`

### Per-tradable-ticker columns (`{ticker}_*`)
- **Rolling-z-scored levels & ratios** — trend, volatility, volume features and
  long-horizon levels (from pass 1).
- **Per-ticker deltas** — `*_delta_20` (medium) and `*_delta_60` (long).
- **Cross-sectional z-scores** — `{feat}_cs_zscore` (global) and
  `{feat}_sector_zscore` (intra-sector) for each CS feature.
- **Rolling-z-scored CS bases** — `log_return_5/20/60`, `overnight_gap`,
  `intraday_return`, `price_vwap_distance`, `volume_price_corr_20`, `adx_14`.
- **Relative strength** — `rs_spy_{5,20,60,252}d`, plus `rs_spy_252d_delta_{20,60}`.
- **Sector one-hot** — `sector_0` … `sector_11`.
- **`{ticker}_close`** — preserved (needed by the trainer; not 0-filled).

### Shared cross-ticker columns (one each, not ticker-prefixed)
- **Temporal** — `day_sin/cos`, `month_sin/cos`, `quarter_sin/cos`.
- **Breadth** — `breadth_{5,20,60,200}d`; deltas `breadth_{5,20,60}d_delta_{1,5,20}`
  (short bases) and `breadth_200d_delta_{20,60}` (long base).
- **VIX term structure** — `vix_term_structure_{5,20,60}d`; deltas
  `*_delta_{1,5,20}`.
- **Macro regime** (from `regime_features.py`, each a log-ratio triple
  short-5d / long-60d / `60d_delta_20`):
  - `yield_curve_*` = log(TLT/SHY)
  - `credit_spread_*` = log(HYG/LQD)
  - `size_factor_*` = log(IWM/SPY)
  - `growth_factor_*` = log(QQQ/SPY)
- **Sector rotation** — `sector_dispersion_{60,200}d`, `sector_topbottom_{60,200}d`
  (cross-sectional std and max−min of 11 XL* cumulative returns), plus
  `sector_dispersion_200d_delta_20`, `sector_topbottom_200d_delta_20`.
- **`SPY_close`** — benchmark reference only.

Window families: `BREADTH_WINDOWS = [5,20,60,200]`, `RS_WINDOWS = [5,20,60,252]`,
`VIX_TERM_WINDOWS = [5,20,60]`; `SHORT_DELTA_WINDOWS = [1,5,20]`,
`MEDIUM_DELTA_WINDOWS = [20,60]`.

> **Construction note:** breadth and RS-vs-SPY are built on the `log_return_5`
> series (`rolling_sum(log_return_5, w)`), i.e. a sum of *overlapping* 5-day
> returns — not clean `log(close_t / close_{t-w})`. Harmless for breadth (sign
> only) and washed out for RS (z-scored), but a different convention from
> `sector_rotation`, which uses clean close ratios. Docstrings calling these
> "N-day cumulative return" overstate the precision. See §9.

---

## 8. Audit (`auditor.py`)

`DataAuditor.audit()` runs, in order:

1. `_load_unified`.
2. `_check_data_shape_and_size` — shape, time range, ticker count, and a
   feature-family breakdown (per-ticker market / sector one-hot / CS global /
   CS sector / RS / breadth / VIX / macro / deltas).
3. `find_downcastable_columns(convert=True)` — **rewrites** `unified.parquet`
   downcasting f64→f32 where error < 1e-6. (Note: this mutates the artifact;
   audit is not purely read-only.)
4. `_check_missing_and_invalid` — null/NaN/Inf counts. Benign `_close` nulls
   (edge-only = pre-IPO/post-delist, via `_classify_close_nulls`) are suppressed;
   interior close nulls are still flagged.
5. `_check_statistics` — mean (`|mean|>1`), low variance (`<1e-8`, sector one-hot
   exempt), skew (`|skew|>3`), tanh-saturation (`>1%` of values `|x|>0.99`),
   and year-over-year temporal drift.
6. `_check_survivorship_coverage` — % of tickers whose data ends >30d before the
   dataset end; warns if <5% delisted (survivorship-bias red flag).
7. `_check_price_discontinuities` — per-ticker single-day log-return scan;
   critical ≥ log 2, suspicious ≥ log 1.5. Should read ~0 critical post-worker
   gate; surfaces residual unadjusted corporate actions.
8. `_check_regime_multicollinearity` — corr among regime-only columns (RS
   sampled), flags `|corr| ≥ 0.90`.
9. `_check_for_multicollinearity` — per-ticker feature pairs across a 50-ticker
   sample, flags `|corr| ≥ 0.95`.

Shared-regime prefixes (`constants.py`: `BREADTH_VIX_PREFIXES`,
`MACRO_REGIME_PREFIXES`, union `SHARED_REGIME_PREFIXES`) are the single source of
truth the auditor and trainer both use to classify columns.

---

## 9. Known issues / stale documentation

- **`REGIME_TICKERS` TODO in `config.py` is stale.** It claims ~25 tickers are
  "dead weight, processed then pruned with zero contribution." Most are now
  consumed by the v6 macro features (TLT/SHY, HYG/LQD, IWM/SPY, QQQ/SPY, all 11
  XL*, VIXY/VIXM, SPY). After accounting for those plus the V7 basket
  (UUP/GLD/USO are trained on directly), the only regime tickers still genuinely
  processed-then-pruned are **MDY, IEF, EFA, EEM, EWJ** (5, not 25). The "v5.1
  design pass" the comment proposes is largely already done.
- **Breadth/RS construction** (see §7 note) — built on `rolling_sum(log_return_5)`
  rather than clean close ratios, inconsistent with `sector_rotation` and with
  the docstrings. Worth confirming intentional before adjusting any window logic.
- **Module-level mutable lists in `features.py`** (`temporal_features`,
  `volatility_features`, …) are populated in place; only `temporal_features` is
  actually read (by `_prefix_columns`). Author-flagged tech debt; safe under the
  process-pool model since each worker gets its own copy.
- **Stale docstrings** — `_drop_rows_before_timestamp` references a "pre-covid
  era" cutoff (actual cutoff is 2004), and the per-ticker docstring mentions
  `_drop_rows_before_timestamp` being "deferred to the end of `_build_unified`"
  (correct, but the naming can read as if it runs per ticker).

---

## 10. Configuration quick reference

| Symbol | Value | Source | Meaning |
| --- | --- | --- | --- |
| `CUTOFF_TIMESTAMP` | `2004-12-13` | config | Warmup trim boundary (post-warmup, clean week start). |
| `WINDOW_SIZE` | `60` | config | Trainer observation window (feeds `MIN_TICKER_LENGTH`). |
| `NORMALIZATION_WINDOW` | `252` | constants | Rolling z-score lookback (~1 trading year). |
| `MIN_TICKER_LENGTH` | `564` | constants | `252 + 60 + 252`; min bars to yield one episode. |
| `GAP_SPLIT_TRADING_DAYS` | `10` | constants | ≥ this gap → lifecycle split. |
| `GAP_SUSPICIOUS_TRADING_DAYS` | `5` | constants | `[5,10)` gaps logged, not split. |
| `DISCONTINUITY_LOG_THRESHOLD` | `0.6931` (log 2) | constants | Single-day move that bails a ticker. |
| `min_dollar_volume` | `$10M` | feature_engineer | Avg daily dollar-volume floor. |
| `min_history_days` | `504` (~2y) | feature_engineer | Min trading days. |
| `min_median_price` | `$5.00` | feature_engineer | Penny-stock floor (lifetime-median close). |
| `TRADABLE_TYPES` | `{CS, ADRC, OS}` | feature_engineer | Kept security types (regime ETFs exempt). |
| `V7_BASKET` | `[SPY, TLT, GLD, USO, UUP]` | config | Regime tickers that are also trained on; index = action index. |
| `DATA_START/END_DATE` | `2000-01-01` / `2026-03-04` | config | Raw data span. |

---

*Generated as a digest of `src/v7/preprocessing/` + `config.py`/`constants.py`.
The trainer / `DailyEnvironment` / `TickerData` / SAC layer that consumes
`unified.parquet` is documented separately.*