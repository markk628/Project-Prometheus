from massive import RESTClient
import polars as pl

from src.config.config import DATA_DIR, MASSIVE_APIKEY, TICKERS
from src.utils.logger import Logger
from src.utils.utils import create_directory, save_to_parquet

logger = Logger()
client = RESTClient(MASSIVE_APIKEY)

base_dir = f"{DATA_DIR}/raw/splits"
create_directory(base_dir)

for ticker in TICKERS:
    logger.info(f'Retrieving splits for {ticker}...')
    rows = []

    for s in client.list_stocks_splits(
        ticker=ticker,
        limit=5000,
        sort="execution_date.asc", # Sort ascending (IMPORTANT for join_asof later)
    ):
        rows.append({
            "ticker": ticker,
            "execution_date": s.execution_date,
            "adjustment_type": s.adjustment_type,
            "split_from": s.split_from,
            "split_to": s.split_to,
            "effective_ratio": (s.split_to / s.split_from) if s.split_from else None,
            "historical_adjustment_factor": s.historical_adjustment_factor,
        })

    if not rows:
        logger.info(f"No splits found for {ticker}")
        continue

    # Convert to Polars DataFrame
    df = pl.DataFrame(rows)

    # Ensure proper types
    df = df.with_columns([
        pl.col("execution_date").str.strptime(pl.Date, strict=False),
        pl.col("adjustment_type").cast(pl.Categorical),
        pl.col("split_from").cast(pl.Float32),
        pl.col("split_to").cast(pl.Float32),
        pl.col("effective_ratio").cast(pl.Float32),
        pl.col("historical_adjustment_factor").cast(pl.Float64),
    ])

    file_path = f"{base_dir}/{ticker}_splits.parquet"
    save_to_parquet(df, file_path, logger=logger)