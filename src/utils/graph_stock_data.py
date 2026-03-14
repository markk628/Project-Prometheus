import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.config import DATA_DIR
from src.utils.utils import load_stock_data

def main():
    ticker = 'TSLA'
    train_data_dir = f'{DATA_DIR}/preprocessed/v2/{ticker}/{ticker}_train.csv'
    start_timestamp = '2022-06-07 09:30:00-4:00'
    end_timestamp = '2022-06-13 15:59:00-4:00'

    data = load_stock_data(train_data_dir, start_timestamp, end_timestamp)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data = data.sort_values('timestamp').reset_index(drop=True)
    data['t_index'] = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10, 6))

    for date, day_data in data.groupby(data['timestamp'].dt.date):
        ax.plot(day_data['t_index'], day_data['close'], alpha=0.6)

    day_starts = data.groupby(data['timestamp'].dt.date)['t_index'].first()
    ax.set_xticks(day_starts.values)
    ax.set_xticklabels(day_starts.index, rotation=45)
    ax.set_title(ticker)
    ax.set_ylabel("Close Price")
    ax.set_xlabel("Trading Time (compressed)")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate() 
    fig.tight_layout()

    plt.show()
    plt.close(fig)
    
if __name__ == '__main__':
    main()