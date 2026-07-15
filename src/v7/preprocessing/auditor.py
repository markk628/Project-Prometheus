import polars as pl
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.config.config import DATA_DIR
from src.utils.logger import Logger
from src.utils.utils import save_to_parquet

from .constants import (
    BREADTH_VIX_PREFIXES,
    MACRO_REGIME_PREFIXES,
    SHARED_REGIME_PREFIXES,
)


class DataAuditor:
    def __init__(
        self,
        logger: Logger,
        unified_path: str = None,
        tickers_dir: str = None,
    ):
        self.data_dir = f"{DATA_DIR}/preprocessed/v7"
        self.unified_path = f"{self.data_dir}/unified/unified.parquet" if unified_path is None else unified_path
        self.tickers_dir = f"{self.data_dir}/tickers" if tickers_dir is None else tickers_dir
        self.logger = logger

    # =========================
    # Utility
    # =========================
    def _log_list(self, name: str, values: list[str], max_items: int = 200):
        if values:
            self.logger.warning(f"{name}: {len(values)} columns")
            self.logger.warning(f"Sample: {values[:max_items]}")
        else:
            self.logger.info(f"{name}: OK")

    def _get_feature_cols(self, df: pl.DataFrame) -> list[str]:
        """Get feature columns, excluding metadata, temporal, and close columns."""
        EXCLUDE = {
            "timestamp",
            "day_sin", "day_cos",
            "month_sin", "month_cos",
            "quarter_sin", "quarter_cos",
        }
        return [
            c for c in df.columns
            if c not in EXCLUDE and not c.endswith("_close")
        ]

    # =========================
    # Data Loading
    # =========================
    def _load_unified(self) -> pl.DataFrame:
        self.logger.info(f"Loading unified data: {self.unified_path}")
        df = pl.read_parquet(self.unified_path)
        self.logger.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns, "
                        f"{df.estimated_size('gb'):.2f} GB")
        return df

    # =========================
    # Shape / Size
    # =========================
    def _check_data_shape_and_size(self, df: pl.DataFrame):
        self.logger.info("Checking data shape and size...")
        self.logger.info(f"Shape: {df.shape} | Size: {df.estimated_size('gb'):.2f} GB")

        if "timestamp" in df.columns:
            ts = df["timestamp"]
            self.logger.info(f"Time range: {ts.min()} → {ts.max()}")
            n_days = df.select(pl.col("timestamp").dt.date().n_unique()).item()
            self.logger.info(f"Trading days: {n_days:,}")

        # Count tickers by looking for _close columns
        close_cols = [c for c in df.columns if c.endswith("_close")]
        self.logger.info(f"Tickers: {len(close_cols)}")

        # Count feature types
        feature_cols = self._get_feature_cols(df)

        # Sector one-hot: {ticker}_sector_0, _sector_1, ... _sector_11
        # The suffix is _sector_ followed by digits. Must exclude _sector_zscore.
        sector_onehot_re = re.compile(r"_sector_\d+$")
        sector_onehot_cols = [c for c in feature_cols if sector_onehot_re.search(c)]

        # CS z-scores (global + sector scopes)
        cs_global_cols = [c for c in feature_cols if c.endswith("_cs_zscore")]
        cs_sector_cols = [c for c in feature_cols if c.endswith("_sector_zscore")]

        # Relative strength vs SPY
        rs_cols = [c for c in feature_cols if "_rs_spy_" in c]

        # Shared regime features
        breadth_cols = [c for c in feature_cols if c.startswith("breadth_") and "_delta_" not in c]
        vix_cols = [c for c in feature_cols if c.startswith("vix_term_") and "_delta_" not in c]
        macro_cols = [
            c for c in feature_cols
            if c.startswith(MACRO_REGIME_PREFIXES) and "_delta_" not in c
        ]

        # Regime momentum deltas
        delta_cols = [c for c in feature_cols if "_delta_" in c]

        # Per-ticker market features (everything else on a ticker prefix)
        per_ticker_market = [
            c for c in feature_cols
            if c not in sector_onehot_cols
            and c not in cs_global_cols
            and c not in cs_sector_cols
            and "_rs_spy_" not in c
            and not c.startswith(SHARED_REGIME_PREFIXES)
            and "_delta_" not in c
        ]

        self.logger.info(f"Feature columns: {len(feature_cols)} total")
        self.logger.info(f"  Per-ticker market: {len(per_ticker_market)}")
        self.logger.info(f"  Sector one-hot: {len(sector_onehot_cols)}")
        self.logger.info(f"  CS z-scores (global): {len(cs_global_cols)}")
        self.logger.info(f"  CS z-scores (sector): {len(cs_sector_cols)}")
        self.logger.info(f"  RS vs SPY: {len(rs_cols)}")
        self.logger.info(f"  Breadth: {len(breadth_cols)}")
        self.logger.info(f"  VIX term structure: {len(vix_cols)}")
        self.logger.info(f"  Macro regime: {len(macro_cols)}")
        self.logger.info(f"  Regime deltas: {len(delta_cols)}")

    # =========================
    # Missing / Invalid
    # =========================
    def _check_missing_and_invalid(self, df: pl.DataFrame):
        self.logger.info("Checking missing/invalid values...")

        # Everything except timestamp is a numeric float column
        check_cols = [c for c in df.columns if c != "timestamp"]

        self.logger.info(f"  Computing null counts across {len(df.columns)} columns...")
        null_counts = df.select([
            pl.col(c).null_count().alias(c) for c in df.columns
        ]).row(0, named=True)

        self.logger.info(f"  Computing NaN counts across {len(check_cols)} columns...")
        nan_counts = df.select([
            pl.col(c).is_nan().sum().alias(c) for c in check_cols
        ]).row(0, named=True)

        self.logger.info(f"  Computing Inf counts across {len(check_cols)} columns...")
        inf_counts = df.select([
            pl.col(c).is_infinite().sum().alias(c) for c in check_cols
        ]).row(0, named=True)

        # _close columns intentionally hold null in non-tradable rows
        # (pre-IPO, post-delisting, between lifecycle segments). The
        # survivorship check already reports those. Filter them out of
        # the null warning so the report shows only *unexpected* nulls
        # — i.e. closes with nulls in the interior of their valid range,
        # which would indicate a real data-quality issue.
        close_cols = [c for c in df.columns if c.endswith("_close")]
        benign_close_nulls = self._classify_close_nulls(df, close_cols)
        n_benign = sum(1 for ok in benign_close_nulls.values() if ok)

        null_cols = [
            f"{c} ({n:,})"
            for c, n in null_counts.items()
            if n and n > 0 and not benign_close_nulls.get(c, False)
        ]
        nan_cols = [
            f"{c} ({n:,})" for c, n in nan_counts.items() if n and n > 0
        ]
        inf_cols = [
            f"{c} ({n:,})" for c, n in inf_counts.items() if n and n > 0
        ]

        if n_benign > 0:
            self.logger.info(
                f"  Suppressed {n_benign} _close column(s) whose nulls are "
                f"all pre-IPO/post-delisting (covered by survivorship check)."
            )

        self._log_list("Null", null_cols)
        self._log_list("NaN", nan_cols)
        self._log_list("Inf", inf_cols)

    def _classify_close_nulls(
        self, df: pl.DataFrame, close_cols: List[str]
    ) -> Dict[str, bool]:
        """
        For each ``{ticker}_close`` column, decide whether its nulls are
        "benign" — consist solely of a contiguous prefix (pre-IPO) and/or
        a contiguous suffix (post-delisting). Interior nulls indicate a
        real data-quality issue and should still be flagged.

        Returns a dict mapping column -> True (benign, suppress) /
        False (interior nulls present, keep flagging).

        Implementation: a column with only edge nulls satisfies
        ``total_nulls == leading_nulls + trailing_nulls``. Computed for
        every close column in a single Polars pass via cumulative-min
        over the ``is_null`` mask (which stays 1 only while inside an
        unbroken null run from the edge).
        """
        if not close_cols:
            return {}

        n_rows = df.height
        exprs = []
        for c in close_cols:
            is_null = pl.col(c).is_null().cast(pl.Int8)
            exprs.append(is_null.sum().alias(f"{c}__total"))
            exprs.append(is_null.cum_min().sum().alias(f"{c}__lead"))
            exprs.append(is_null.cum_min(reverse=True).sum().alias(f"{c}__trail"))

        row = df.select(exprs).row(0, named=True)

        result: Dict[str, bool] = {}
        for c in close_cols:
            total = row[f"{c}__total"] or 0
            if total == 0:
                # No nulls at all — already absent from the warning list.
                result[c] = True
                continue
            lead  = row[f"{c}__lead"] or 0
            trail = row[f"{c}__trail"] or 0
            # Entirely-null column: don't suppress, that's a real bug.
            if lead >= n_rows:
                result[c] = False
                continue
            result[c] = (total == lead + trail)
        return result

    # =========================
    # Core Statistics
    # =========================
    def _check_statistics(self, df: pl.DataFrame):
        self.logger.info("Checking statistics...")

        feature_cols = self._get_feature_cols(df)

        # =========================
        # 1. Core stats (vectorized)
        # =========================
        self.logger.info("Computing mean / variance...")

        exclude_set = set(df.columns) - set(feature_cols)
        stats = df.select([
            pl.all().exclude(exclude_set).mean().name.suffix("__mean"),
            pl.all().exclude(exclude_set).var().name.suffix("__var"),
        ])

        stats_row = stats.row(0, named=True)

        # Sector one-hot columns are constant-per-ticker by design (11 zeros
        # and 1 one per row). Their variance is intentionally low — not a bug.
        # Skip them in the variance check to avoid drowning real issues in noise.
        sector_onehot_re = re.compile(r"_sector_\d+$")

        bad_mean = []
        low_var = []

        for c in feature_cols:
            mean = stats_row.get(f"{c}__mean")
            var  = stats_row.get(f"{c}__var")

            if mean is not None and abs(mean) > 1:
                bad_mean.append(f"{c} (mean={mean:.4f})")

            if var is not None and var < 1e-8 and not sector_onehot_re.search(c):
                low_var.append(c)

        # =========================
        # 2. Skew
        # =========================
        self.logger.info("Checking skew...")

        # Compute skew for all feature columns in one pass
        skew_row = df.select([
            pl.col(c).skew().alias(c) for c in feature_cols
        ]).row(0, named=True)

        skewed = [
            f"{c} (skew={s:.2f})"
            for c, s in skew_row.items()
            if s is not None and abs(s) > 3
        ]

        # =========================
        # 3. Clipping saturation (tanh output bounds)
        # =========================
        self.logger.info("Checking clipping saturation...")

        # tanh-normalized features are bounded to [-1, 1]
        # Check for saturation at the bounds — vectorized across all columns
        clip_row = df.select([
            (pl.col(c).abs() > 0.99).mean().alias(c) for c in feature_cols
        ]).row(0, named=True)

        clipped_cols = [
            f"{c} ({ratio:.2%})"
            for c, ratio in clip_row.items()
            if ratio is not None and ratio > 0.01
        ]

        # =========================
        # 4. Temporal drift (yearly)
        # =========================
        self.logger.info("Checking temporal drift (yearly windows)...")

        drift_cols = self._check_temporal_drift(df, feature_cols)

        # =========================
        # REPORT
        # =========================
        self.logger.info("=== STATISTICS REPORT ===")

        self._log_list("High mean (|mean| > 1)", bad_mean)
        self._log_list("Low variance", low_var)
        self._log_list("High skew (|skew| > 3)", skewed)
        self._log_list("Clipping saturation (>1% at bounds)", clipped_cols)
        self._log_list("Temporal drift", drift_cols)

    # =========================
    # Temporal Drift
    # =========================
    def _check_temporal_drift(self, df: pl.DataFrame, feature_cols: list[str]) -> list[str]:
        """
        Check distribution drift between adjacent yearly windows.

        Instead of comparing train vs valid (which no longer exist as
        separate splits), compare each year's feature distributions
        against the previous year. This reveals when and where features
        shift, which is more informative for walk-forward validation.
        """
        if "timestamp" not in df.columns:
            return []

        df = df.with_columns(pl.col("timestamp").dt.year().alias("_year"))
        years = sorted(df.select("_year").unique().to_series().to_list())

        if len(years) < 2:
            return []

        drift_results = []
        eps = 1e-6

        # Skip sector one-hot columns — they're constant per ticker so
        # drift is meaningless. Use regex to match only _sector_\d+$
        # (the one-hot columns) and NOT _sector_zscore (which is a
        # legitimate CS feature worth checking).
        sector_onehot_re = re.compile(r"_sector_\d+$")
        check_cols = [
            c for c in feature_cols
            if not sector_onehot_re.search(c)
        ]

        # Precompute per-year stats
        year_stats = {}
        for year in years:
            year_df = df.filter(pl.col("_year") == year)
            if len(year_df) < 10:
                continue
            means = year_df.select([pl.col(c).mean().alias(c) for c in check_cols]).row(0, named=True)
            stds = year_df.select([pl.col(c).std().alias(c) for c in check_cols]).row(0, named=True)
            year_stats[year] = (means, stds)

        # Compare adjacent years
        drift_counts = defaultdict(int)

        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]

            if prev_year not in year_stats or curr_year not in year_stats:
                continue

            prev_means, prev_stds = year_stats[prev_year]
            curr_means, curr_stds = year_stats[curr_year]

            for c in check_cols:
                try:
                    mean_shift = abs((curr_means[c] - prev_means[c]) / (prev_stds[c] + eps))
                    std_shift = abs(curr_stds[c] - prev_stds[c])

                    if mean_shift > 0.5 or std_shift > 0.5:
                        drift_counts[c] += 1
                except Exception:
                    continue

        # Report features that drift in multiple year transitions
        for c, count in sorted(drift_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:
                drift_results.append(f"{c} (drifted in {count}/{len(years)-1} transitions)")

        return drift_results

    # =========================
    # Multicollinearity
    # =========================
    def _check_for_multicollinearity(self):
        self.logger.info("Checking multicollinearity...")

        pair_counts = defaultdict(int)
        pair_corr_sum = defaultdict(float)

        tickers_dir = Path(self.tickers_dir)
        ticker_files = sorted(tickers_dir.glob("*.parquet"))

        if not ticker_files:
            self.logger.warning(f"No ticker parquet files found in {tickers_dir}")
            return

        # Sample up to 50 tickers for efficiency (multicollinearity patterns
        # are consistent across tickers since features are the same)
        sample_files = ticker_files[:50] if len(ticker_files) > 50 else ticker_files

        self.logger.info(f"Checking multicollinearity across {len(sample_files)} tickers...")

        EXCLUDE = {
            "timestamp",
            "day_sin", "day_cos",
            "month_sin", "month_cos",
            "quarter_sin", "quarter_cos",
        }

        for file in sample_files:
            ticker = file.stem

            try:
                df = pl.read_parquet(file)
                drop_cols = [c for c in df.columns if c in EXCLUDE or c.endswith("_close")]
                df = df.drop(drop_cols)

                # Drop any remaining non-numeric columns
                numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
                if len(numeric_cols) < 2:
                    continue

                data = df.select(numeric_cols).to_numpy()

                if data.shape[0] < 100:
                    continue

                corr = np.corrcoef(data, rowvar=False)
                cols = numeric_cols

                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        val = corr[i, j]
                        if np.isnan(val):
                            continue
                        if abs(val) >= 0.95:
                            f1 = cols[i].replace(f"{ticker}_", "")
                            f2 = cols[j].replace(f"{ticker}_", "")
                            pair = tuple(sorted([f1, f2]))

                            pair_counts[pair] += 1
                            pair_corr_sum[pair] += abs(val)

            except Exception as e:
                self.logger.warning(f"Failed to check {ticker}: {e}")
                continue

        if not pair_counts:
            self.logger.info("No highly correlated feature pairs found (|corr| >= 0.95)")
            return

        self.logger.info("Most Common Highly Correlated Feature Pairs:")

        for (f1, f2), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            avg = pair_corr_sum[(f1, f2)] / count
            self.logger.info(f"  {f1} <-> {f2} | count={count}/{len(sample_files)} tickers | avg_corr={avg:.4f}")

    # =========================
    # Survivorship Coverage
    # =========================
    def _check_survivorship_coverage(self, df: pl.DataFrame):
        """
        Check how many tickers in the dataset are likely delisted
        (their data ends well before the most recent date).

        This verifies that the training data includes dead companies
        and isn't suffering from survivorship bias.
        """
        self.logger.info("Checking survivorship coverage...")

        if "timestamp" not in df.columns:
            self.logger.warning("No timestamp column — skipping survivorship check")
            return

        close_cols = [c for c in df.columns if c.endswith("_close")]
        data_end = df["timestamp"].max()

        # Compute the last valid timestamp per ticker in a single vectorized pass.
        # For each close column, mask the timestamp where close is null or <= 0,
        # then take the max. `when/then/otherwise` with null lets max ignore invalid rows.
        last_valid_exprs = [
            pl.when((pl.col(col).is_not_null()) & (pl.col(col) > 0))
            .then(pl.col("timestamp"))
            .otherwise(None)
            .max()
            .alias(col)
            for col in close_cols
        ]
        last_valid_row = df.select(last_valid_exprs).row(0, named=True)

        active = 0
        delisted = 0
        delisted_tickers = []

        for col, last_valid in last_valid_row.items():
            ticker = col.replace("_close", "")

            if last_valid is None:
                delisted += 1
                delisted_tickers.append(f"{ticker} (no valid data)")
                continue

            # If the ticker's data ends 30+ days before the dataset end, likely delisted
            days_before_end = (data_end - last_valid).total_seconds() / 86400
            if days_before_end > 30:
                delisted += 1
                delisted_tickers.append(f"{ticker} (last: {last_valid.date()})")
            else:
                active += 1

        total = active + delisted
        self.logger.info(f"Survivorship coverage: {total} tickers total")
        self.logger.info(f"  Active (data near end): {active} ({active/total:.1%})")
        self.logger.info(f"  Likely delisted: {delisted} ({delisted/total:.1%})")

        if delisted_tickers:
            sample = delisted_tickers[:20]
            self.logger.info(f"  Sample delisted: {sample}")

        if delisted / total < 0.05:
            self.logger.warning(
                "WARNING: Less than 5% delisted tickers — possible survivorship bias. "
                "Verify that your raw data includes historical tickers that were delisted."
            )

    # =========================
    # Memory Efficiency
    # =========================
    def find_downcastable_columns(
        self,
        df: pl.DataFrame,
        tol: float = 1e-6,
        convert: bool = False,
    ) -> pl.DataFrame:
        f32_cols = []
        f16_cols = []
        keep_f64 = []

        for col in df.columns:
            if df[col].dtype == pl.Float64:
                col_f64 = df[col].to_numpy()

                col_f32 = col_f64.astype(np.float32).astype(np.float64)
                err32 = np.max(np.abs(col_f64 - col_f32))

                if err32 < tol:
                    col_f16 = col_f64.astype(np.float16).astype(np.float64)
                    err16 = np.max(np.abs(col_f64 - col_f16))

                    if err16 < tol:
                        f16_cols.append(col)
                    else:
                        f32_cols.append(col)
                else:
                    keep_f64.append((col, err32))

        self.logger.info(f"Downcast analysis:")
        self.logger.info(f"  Convert to Float32 (safe at Float16 tol): {len(f16_cols)}")
        self.logger.info(f"  Convert to Float32: {len(f32_cols)}")
        self.logger.info(f"  Keep Float64: {len(keep_f64)}")

        if (f16_cols or f32_cols) and convert:
            before = df.estimated_size("gb")

            if f16_cols:
                df = df.with_columns([pl.col(c).cast(pl.Float32) for c in f16_cols])

            if f32_cols:
                df = df.with_columns([pl.col(c).cast(pl.Float32) for c in f32_cols])

            after = df.estimated_size("gb")

            save_to_parquet(df, self.unified_path)

            self.logger.info(f"Downcasting complete: {before:.2f}GB → {after:.2f}GB")

        return df

    # =========================
    # Regime Multicollinearity
    # =========================
    def _check_regime_multicollinearity(self, df: pl.DataFrame):
        """
        Check multicollinearity among regime features that only exist
        in the unified file (not in per-ticker parquets).

        This covers: breadth, VIX term structure, regime momentum deltas,
        RS vs SPY (sampled), and sector one-hot columns.
        """
        self.logger.info("Checking regime feature multicollinearity...")

        # Identify regime-only columns
        regime_cols = [
            c for c in df.columns
            if c.startswith(SHARED_REGIME_PREFIXES)
            or "_rs_spy_" in c
            or "_delta_" in c
        ]

        if not regime_cols:
            self.logger.info("No regime columns found — skipping")
            return

        # RS vs SPY columns are per-ticker — sample a few to keep the matrix manageable
        rs_cols = [c for c in regime_cols if "_rs_spy_" in c]
        non_rs_cols = [c for c in regime_cols if "_rs_spy_" not in c]

        # Sample up to 10 tickers' RS columns
        rs_tickers_seen = set()
        rs_sampled = []
        for c in rs_cols:
            ticker = c.split("_rs_spy_")[0]
            if ticker not in rs_tickers_seen and len(rs_tickers_seen) < 10:
                rs_tickers_seen.add(ticker)
                rs_sampled.append(c)

        check_cols = non_rs_cols + rs_sampled

        if len(check_cols) < 2:
            self.logger.info("Too few regime columns to check")
            return

        self.logger.info(f"Checking {len(check_cols)} regime columns "
                        f"({len(non_rs_cols)} shared + {len(rs_sampled)} RS samples)...")

        data = df.select(check_cols).to_numpy()

        # Drop rows with NaN for clean correlation
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]

        if data.shape[0] < 100:
            self.logger.warning("Too few valid rows for regime correlation check")
            return

        corr = np.corrcoef(data, rowvar=False)

        pairs = []
        for i in range(len(check_cols)):
            for j in range(i + 1, len(check_cols)):
                val = corr[i, j]
                if np.isnan(val):
                    continue
                if abs(val) >= 0.90:
                    pairs.append((check_cols[i], check_cols[j], val))

        if not pairs:
            self.logger.info("No highly correlated regime feature pairs found (|corr| >= 0.90)")
            return

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        self.logger.info(f"Highly Correlated Regime Feature Pairs (|corr| >= 0.90):")
        for f1, f2, val in pairs[:30]:  # cap output
            self.logger.info(f"  {f1} <-> {f2} | corr={val:.4f}")

    # =========================
    # Per-ticker price discontinuities
    # =========================
    def _check_price_discontinuities(
        self,
        df: pl.DataFrame,
        critical_threshold: float = 0.6931,   # log(2.0)  — 2x up or 50% down
        suspicious_threshold: float = 0.4055, # log(1.5)  — 50% up or 33% down
        top_n: int = 30,
    ):
        """
        Scan every ticker's {ticker}_close series for single-day moves large
        enough to suggest a data artifact rather than an organic price move.

        Typical causes of critical breaches:
          - Unadjusted forward splits (a 2:1 shows as ~50% drop)
          - Unadjusted reverse splits (a 1:10 shows as ~10x jump)
          - Delisting-day garbage prices
          - Post-bankruptcy relist with the same ticker
          - Bad ticks / data entry errors in the source
          - Symbol reuse across merged/acquired entities

        Uses log returns so that 2x up and 50% down have equal magnitude.
        Legitimate earnings moves (≈15–40%) on small-caps will show up as
        "suspicious" but not "critical" — the critical bucket is the one
        to manually inspect against known corporate-action history before
        training on the affected tickers.
        """
        self.logger.info("Checking for per-ticker price discontinuities...")

        close_cols = [c for c in df.columns if c.endswith("_close")]
        if not close_cols or "timestamp" not in df.columns:
            self.logger.warning(
                "Price discontinuity check: missing close columns or timestamp — skipping"
            )
            return

        # Unpivot all close columns to long form, drop nulls/zeros, then
        # compute within-ticker day-to-day log returns in a single lazy
        # pass. `shift().over("ticker")` ensures the lag is per-ticker so
        # one ticker's first valid day never references another's last.
        long = (
            df.lazy()
            .select(["timestamp"] + close_cols)
            .unpivot(
                index="timestamp",
                on=close_cols,
                variable_name="close_col",
                value_name="close",
            )
            .with_columns(
                pl.col("close_col").str.replace(r"_close$", "").alias("ticker")
            )
            .drop("close_col")
            .filter(pl.col("close").is_not_null() & (pl.col("close") > 0))
            .sort(["ticker", "timestamp"])
            .with_columns(
                pl.col("close").shift(1).over("ticker").alias("prev_close")
            )
            .filter(pl.col("prev_close").is_not_null() & (pl.col("prev_close") > 0))
            .with_columns(
                (pl.col("close") / pl.col("prev_close")).log().alias("log_ret")
            )
            .filter(pl.col("log_ret").abs() >= suspicious_threshold)
            .with_columns(
                ((pl.col("close") / pl.col("prev_close")) - 1.0).alias("pct_move")
            )
            .select(["ticker", "timestamp", "prev_close", "close", "pct_move", "log_ret"])
        )
        breaches = long.collect()

        n_breach_total = breaches.height
        n_critical = breaches.filter(
            pl.col("log_ret").abs() >= critical_threshold
        ).height

        self.logger.info(
            f"Scanned {len(close_cols)} tickers. "
            f"Suspicious day-moves (|log_ret| ≥ {suspicious_threshold:.4f}, "
            f"≈±{np.expm1(suspicious_threshold)*100:.0f}%): {n_breach_total:,}"
        )
        self.logger.info(
            f"  Of which CRITICAL (|log_ret| ≥ {critical_threshold:.4f}, "
            f"≈±{np.expm1(critical_threshold)*100:.0f}%): {n_critical:,}"
        )

        if n_breach_total == 0:
            self.logger.info("Price discontinuities: OK")
            return

        # Per-ticker rollup: how many breaches, how many critical, biggest move
        per_ticker = (
            breaches
            .group_by("ticker")
            .agg([
                pl.len().alias("n_suspicious"),
                (pl.col("log_ret").abs() >= critical_threshold)
                    .sum()
                    .alias("n_critical"),
                pl.col("pct_move").abs().max().alias("max_abs_pct"),
            ])
            .sort(["n_critical", "n_suspicious", "max_abs_pct"], descending=True)
        )

        worst_tickers = per_ticker.filter(pl.col("n_critical") > 0).head(top_n)
        if worst_tickers.height > 0:
            self.logger.warning(
                f"Top {worst_tickers.height} tickers by CRITICAL breach count "
                f"(inspect corporate-action history before training):"
            )
            for row in worst_tickers.iter_rows(named=True):
                self.logger.warning(
                    f"  {row['ticker']:<10} | critical: {row['n_critical']:<3} | "
                    f"suspicious: {row['n_suspicious']:<4} | "
                    f"max single-day: {row['max_abs_pct']:+.1%}"
                )

        # Top-N worst individual events globally — the single most extreme
        # day-moves across the whole universe, with ticker + date + prices.
        top_events = (
            breaches
            .with_columns(pl.col("log_ret").abs().alias("abs_log_ret"))
            .sort("abs_log_ret", descending=True)
            .head(top_n)
        )
        if top_events.height > 0:
            self.logger.warning(
                f"Top {top_events.height} worst single-day events across all tickers:"
            )
            for row in top_events.iter_rows(named=True):
                ts = row["timestamp"]
                date_str = ts.date().isoformat() if ts is not None else "?"
                self.logger.warning(
                    f"  {row['ticker']:<10} {date_str} | "
                    f"${row['prev_close']:>10.4f} → ${row['close']:>10.4f} | "
                    f"move: {row['pct_move']:+.1%}"
                )

        # Final summary
        n_affected = per_ticker.filter(pl.col("n_critical") > 0).height
        pct_affected = n_affected / len(close_cols)
        if n_affected > 0:
            self.logger.warning(
                f"SUMMARY: {n_affected} of {len(close_cols)} tickers "
                f"({pct_affected:.1%}) have at least one critical single-day "
                f"discontinuity. These are the most likely source of "
                f"pathological training-episode returns."
            )
        else:
            self.logger.info("No critical per-ticker price discontinuities found.")

    # =========================
    # Run
    # =========================
    def audit(self):
        self.logger.info("Starting audit...")

        df = self._load_unified()

        self._check_data_shape_and_size(df)

        df = self.find_downcastable_columns(df, tol=1e-6, convert=True)

        self._check_missing_and_invalid(df)

        self._check_statistics(df)

        self._check_survivorship_coverage(df)

        self._check_price_discontinuities(df)

        self._check_regime_multicollinearity(df)

        del df

        self._check_for_multicollinearity()

        self.logger.info("Audit complete.")


def main():
    auditor = DataAuditor(Logger())
    auditor.audit()


if __name__ == '__main__':
    main()