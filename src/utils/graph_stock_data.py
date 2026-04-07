import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datetime import datetime, timezone

from src.config.config import DATA_DIR
from src.utils.utils import load_stock_data


def main():
    ticker = 'TSLA'
    path = f'{DATA_DIR}/preprocessed/v4/tickers/{ticker}.parquet'

    start = datetime(2020, 3, 9, 13, 30, 0, tzinfo=timezone.utc)
    end   = datetime(2020, 3, 13, 19, 59, 0, tzinfo=timezone.utc)

    data = load_stock_data(path, start, end)

    data = (
        data
        .with_row_index(name='t_index')
        .with_columns(pl.col('timestamp').dt.date().alias('date'))
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for (date,), day_data in data.group_by('date', maintain_order=True):
        ax.plot(day_data['t_index'].to_numpy(), day_data[f'{ticker}_close'].to_numpy(), alpha=0.6)

    day_starts = (
        data
        .group_by('date', maintain_order=True)
        .agg(pl.col('t_index').first())
        .sort('date')
    )

    ax.set_xticks(day_starts['t_index'].to_numpy())
    ax.set_xticklabels(day_starts['date'].to_list(), rotation=45)
    ax.set_title(ticker)
    ax.set_ylabel('Close Price')
    ax.set_xlabel('Trading Time (compressed)')
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()