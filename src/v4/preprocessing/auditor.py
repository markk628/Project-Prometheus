import polars as pl
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from src.config.config import DATA_DIR
from src.utils.logger import Logger
from src.utils.utils import save_to_parquet


class DataAuditor:
    def __init__(self, logger: Logger, train_path: str=None, valid_path: str=None, test_path: str=None):
        self.data_dir = f"{DATA_DIR}/preprocessed/v4/"
        self.train_path = f"{self.data_dir}/unified/unified_train.parquet" if train_path is None else train_path
        self.valid_path = f"{self.data_dir}/unified/unified_valid.parquet" if valid_path is None else valid_path
        self.test_path = f"{self.data_dir}/unified/unified_test.parquet" if test_path is None else test_path
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

    # =========================
    # Data Loading
    # =========================
    def _fetch_unified_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.logger.info("Fetching data...")
        self.logger.info(f"Train: {self.train_path}")
        self.logger.info(f"Valid: {self.valid_path}")
        self.logger.info(f"Test: {self.test_path}")
        train = pl.read_parquet(self.train_path)
        valid = pl.read_parquet(self.valid_path)
        test  = pl.read_parquet(self.test_path)
        self.logger.info("Data fetched.")
        return train, valid, test

    # =========================
    # Shape / Size
    # =========================
    def _check_data_shape_and_size(self, train, valid, test):
        self.logger.info("Checking data shape and size...")

        sizes = {
            "Train": train.estimated_size("gb"),
            "Valid": valid.estimated_size("gb"),
            "Test":  test.estimated_size("gb"),
        }

        for name, df in zip(["Train", "Valid", "Test"], [train, valid, test]):
            self.logger.info(f"{name}: Shape {df.shape} | Size {sizes[name]:.2f} GB")

        total_rows = train.shape[0] + valid.shape[0] + test.shape[0]
        total_size = sum(sizes.values())

        self.logger.info(f"Total: Shape ({total_rows}, {train.shape[1]}) | Size {total_size:.2f} GB")

    # =========================
    # Missing / Invalid
    # =========================
    def _check_missing_and_invalid(self, df: pl.DataFrame, name: str):
        self.logger.info(f"Checking missing/invalid values: {name}...")

        null_cols = []
        nan_cols = []
        inf_cols = []

        for c in df.columns:
            try:
                if df.select(pl.col(c).has_nulls()).item():
                    null_cols.append(c)

                if df.select(pl.col(c).is_nan().any()).item():
                    nan_cols.append(c)

                if df.select(pl.col(c).is_infinite().any()).item():
                    inf_cols.append(c)

            except Exception:
                continue

        self._log_list(f"{name} - Null", null_cols)
        self._log_list(f"{name} - NaN", nan_cols)
        self._log_list(f"{name} - Inf", inf_cols)

    # =========================
    # Core Statistics (CLIPPING AWARE)
    # =========================
    def _check_statistics(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame):
        self.logger.info("Checking statistics...")

        EXCLUDE = {
            "timestamp",
            "minute_sin", "minute_cos",
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            "minutes_since_open", "minutes_to_close"
        }

        feature_cols = [
            c for c in train.columns
            if c not in EXCLUDE and not c.endswith("_close")
        ]

        # =========================
        # 1. Core stats (vectorized)
        # =========================
        self.logger.info("Computing mean / variance...")

        stats = train.select([
            pl.all().exclude(EXCLUDE).mean().name.suffix("__mean"),
            pl.all().exclude(EXCLUDE).var().name.suffix("__var"),
        ])

        stats_row = stats.row(0, named=True)

        bad_mean = []
        low_var = []

        for c in feature_cols:
            mean = stats_row.get(f"{c}__mean")
            var  = stats_row.get(f"{c}__var")

            if mean is not None and abs(mean) > 1:
                bad_mean.append(c)

            if var is not None and var < 1e-8:
                low_var.append(c)

        # =========================
        # 2. Skew (adjusted for clipping)
        # =========================
        self.logger.info("Checking skew...")

        skewed = []
        for c in feature_cols:
            try:
                skew = train.select(pl.col(c).skew()).item()
                if skew is not None and abs(skew) > 3:  # adjusted threshold
                    skewed.append(c)
            except Exception:
                continue

        # =========================
        # 3. CLIPPING SATURATION (vectorized)
        # =========================
        self.logger.info("Checking clipping saturation...")

        clipped_expr = [
            (((pl.col(c) == 5) | (pl.col(c) == -5)).mean()).alias(c)
            for c in feature_cols
        ]

        clipped_df = train.select(clipped_expr)
        clipped_row = clipped_df.row(0, named=True)

        clipped_cols = [
            f"{c} ({ratio:.2%})"
            for c, ratio in clipped_row.items()
            if ratio is not None and ratio > 0.01
        ]

        # =========================
        # 4. Drift (most important now)
        # =========================
        self.logger.info("Checking distribution drift...")

        drift_cols = []

        def detect_drift(df_a, df_b, name):
            means_a = df_a.select(pl.all().exclude(EXCLUDE).mean())
            means_b = df_b.select(pl.all().exclude(EXCLUDE).mean())

            std_a = df_a.select(pl.all().exclude(EXCLUDE).std())
            std_b = df_b.select(pl.all().exclude(EXCLUDE).std())

            row_a = means_a.row(0, named=True)
            row_b = means_b.row(0, named=True)
            std_a = std_a.row(0, named=True)
            std_b = std_b.row(0, named=True)
            eps = 1e-6

            for c in feature_cols:
                try:
                    if abs((row_a[c] - row_b[c]) / (std_a[c] + eps)) > 0.5:
                        drift_cols.append(f"{c} (mean {name})")

                    if abs(std_a[c] - std_b[c]) > 0.5:
                        drift_cols.append(f"{c} (std {name})")

                except Exception:
                    continue

        detect_drift(train, valid, "valid")
        detect_drift(train, test, "test")

        # =========================
        # REPORT
        # =========================
        self.logger.info("=== STATISTICS REPORT ===")

        self._log_list("High mean (|mean| > 1)", bad_mean)
        self._log_list("Low variance", low_var)
        self._log_list("High skew (|skew| > 3)", skewed)
        self._log_list("Clipping saturation (>1%)", clipped_cols)
        self._log_list("Distribution drift", drift_cols)

    # =========================
    # Multicollinearity
    # =========================
    def _check_for_multicollinearity(self):
        self.logger.info("Checking multicollinearity...")

        pair_counts = defaultdict(int)
        pair_corr_sum = defaultdict(float)

        for ticker_path in Path(self.data_dir).iterdir():
            if not ticker_path.is_dir():
                continue

            ticker = ticker_path.name
            if "unified" in ticker:
                continue

            file = ticker_path / f"{ticker}_train.parquet"
            if not file.exists():
                continue

            df = pl.read_parquet(file).drop([
                f"{ticker}_close", "timestamp",
                "minute_sin", "minute_cos",
                "hour_sin", "hour_cos",
                "day_sin", "day_cos",
                "minutes_since_open", "minutes_to_close"
            ])

            data = df.to_numpy()
            corr = np.corrcoef(data, rowvar=False)

            cols = df.columns

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr[i, j]
                    if abs(val) >= 0.95:
                        f1 = cols[i].replace(f"{ticker}_", "")
                        f2 = cols[j].replace(f"{ticker}_", "")
                        pair = tuple(sorted([f1, f2]))

                        pair_counts[pair] += 1
                        pair_corr_sum[pair] += abs(val)

        self.logger.info("Most Common Highly Correlated Feature Pairs:")

        for (f1, f2), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            avg = pair_corr_sum[(f1, f2)] / count
            self.logger.info(f"{f1} <-> {f2} | count={count} | avg_corr={avg:.4f}")
            
    # =========================
    # Memory Efficiency 
    # =========================
    def find_downcastable_columns(
        self, 
        train: pl.DataFrame, 
        valid: pl.DataFrame, 
        test: pl.DataFrame, 
        tol=1e-6, 
        convert: bool=False, 
        ask: bool=False
    ):
        f32_cols = []
        f16_cols = []
        keep_f64 = []

        for col in train.columns:
            if train[col].dtype == pl.Float64:
                col_f64 = train[col].to_numpy()

                # Test float32
                col_f32 = col_f64.astype(np.float32).astype(np.float64)
                err32 = np.max(np.abs(col_f64 - col_f32))

                if err32 < tol:
                    # Test float16
                    col_f16 = col_f64.astype(np.float16).astype(np.float64)
                    err16 = np.max(np.abs(col_f64 - col_f16))

                    if err16 < tol:
                        f16_cols.append(col)
                    else:
                        f32_cols.append(col)
                else:
                    keep_f64.append((col, err32))

        self.logger.info(f"Convert to Float16: {len(f16_cols)}")
        self.logger.info(f"Convert to Float32: {len(f32_cols)}")
        self.logger.info(f"Keep Float64: {len(keep_f64)}")

        if (f16_cols or f32_cols) and convert:
            do_convert = True

            if ask:
                while True:
                    user_input = input("Convert float columns to smaller types? (Y/N): ").strip().lower()
                    if user_input in ["y", "yes"]:
                        do_convert = True
                        break
                    elif user_input in ["n", "no"]:
                        do_convert = False
                        break
                    else:
                        print("Please enter Y or N.")

            if do_convert:
                before_train = train.estimated_size("gb")
                before_valid = valid.estimated_size("gb")
                before_test  = test.estimated_size("gb")
                if f16_cols:
                    train = train.with_columns([pl.col(c).cast(pl.Float16) for c in f16_cols])
                    valid = valid.with_columns([pl.col(c).cast(pl.Float16) for c in f16_cols])
                    test  = test.with_columns([pl.col(c).cast(pl.Float16) for c in f16_cols])

                if f32_cols:
                    train = train.with_columns([pl.col(c).cast(pl.Float32) for c in f32_cols])
                    valid = valid.with_columns([pl.col(c).cast(pl.Float32) for c in f32_cols])
                    test  = test.with_columns([pl.col(c).cast(pl.Float32) for c in f32_cols])
                after_train = train.estimated_size("gb")
                after_valid = valid.estimated_size("gb")
                after_test  = test.estimated_size("gb")
                
                save_to_parquet(train, self.train_path)
                save_to_parquet(valid, self.valid_path)
                save_to_parquet(test, self.test_path)
                
                self.logger.info("Float downcasting complete.")
                self.logger.info(f"Train memory: {before_train:.2f}GB → {after_train:.2f}GB")
                self.logger.info(f"Valid memory: {before_valid:.2f}GB → {after_valid:.2f}GB")
                self.logger.info(f"Test memory:  {before_test:.2f}GB → {after_test:.2f}GB")
                self.logger.info(f"Total memory:  {before_train + before_valid + before_test:.2f}GB → {after_train + after_valid + after_test:.2f}GB")
            else:
                self.logger.info("Conversion skipped.")

        return train, valid, test

    # =========================
    # Run
    # =========================
    def audit(self):
        self.logger.info("Starting audit...")

        train, valid, test = self._fetch_unified_data()

        self._check_data_shape_and_size(train, valid, test)
        
        train, valid, test = self.find_downcastable_columns(
            train, valid, test, tol=1e-6, convert=True, ask=False
        )

        self._check_missing_and_invalid(train, "Train")
        self._check_missing_and_invalid(valid, "Valid")
        self._check_missing_and_invalid(test, "Test")

        self._check_statistics(train, valid, test)

        del train, valid, test

        self._check_for_multicollinearity()

        self.logger.info("Audit complete.")
        
def main():
    data_dir = f"{DATA_DIR}/preprocessed/v4/"
    train_latent = f"{data_dir}/unified_latent/unified_latent_train_v2.parquet"
    valid_latent = f"{data_dir}/unified_latent/unified_latent_valid_v2.parquet"
    test_latent = f"{data_dir}/unified_latent/unified_latent_test_v2.parquet"
    auditor = DataAuditor(Logger(), train_latent, valid_latent, test_latent)
    # auditor = DataAuditor(Logger())
    auditor.audit()
    
if __name__ == '__main__':
    main()