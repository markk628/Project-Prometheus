import json
from massive import RESTClient

from src.utils.logger import Logger
from src.utils.utils import create_directory
from src.config.config import (
    DATA_DIR, 
    DATA_TIMESPAN, 
    DATA_START_DATE, 
    DATA_END_DATE,
    MASSIVE_APIKEY, 
    TICKERS
)

def save_raw_market_data_as_json(tickers: list[str]=TICKERS, 
                                 multiplier: int=1,
                                 timespan: str=DATA_TIMESPAN,
                                 start: str=DATA_START_DATE,
                                 end: str=DATA_END_DATE,
                                 adjusted: str='true',
                                 sort: str='asc',
                                 logger: Logger=None) -> None:
    """
    Retrieve aggregated historical OHLC (Open, High, Low, Close) and volume data for specified stock tickers
    
    :param tickers: List of ticker symbols
    :type tickers: list[str]
    :param multiplier: The size of the timespan multiplier
    :type multiplier: int
    :param timespan: The size of the time window
    :type timespan: str
    :param start: The start of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp
    :type start: str
    :param end: The end of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp
    :type end: str
    :param adjusted: Whether or not the results are adjusted for splits
    :type adjusted: str
    :param sort: Sort the results by timestamp
    :type sort: str
    """
    if logger:
        logger.info('Retrieving data from API...')
    base_dir = f"{DATA_DIR}/raw/{multiplier}_{timespan}"
    create_directory(base_dir)
    client = RESTClient(MASSIVE_APIKEY)
    for ticker in tickers:
        if logger:
            logger.info(f'Retrieving {ticker} data...')
        aggregates = []
        for aggregate in client.list_aggs(
            ticker,
            multiplier,
            timespan,
            start,
            end,
            adjusted=adjusted,
            sort=sort
        ):
            aggregates.append({
                'timestamp': aggregate.timestamp, 
                'open': aggregate.open, 
                'high': aggregate.high,
                'low': aggregate.low,
                'close': aggregate.close,
                'transactions': aggregate.transactions,
                'volume': aggregate.volume,
                'vwap': aggregate.vwap
            })
        file_path = f'{base_dir}/{ticker}.json'
        with open (file_path, 'w') as file:
            json.dump(aggregates, file, indent=4)

def main():
    save_raw_market_data_as_json(logger=Logger())
    
if __name__ == '__main__':
    main()