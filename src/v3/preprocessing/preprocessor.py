import json
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import talib as ta
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

from src.config.config import DATA_DIR, CUTOFF_TIMESTAMP, TICKERS
from src.utils.database_manager import DatabaseManager
from src.utils.logger import Logger
from src.utils.utils import create_directory, save_to_csv


class DataPreprocessor:
    def __init__(
        self, 
        data_dir: str=DATA_DIR,
        timestamp: str=CUTOFF_TIMESTAMP,
        tickers: List[str]=TICKERS,
        database_manager: DatabaseManager=None,
        logger: Optional[Logger]=None
    ):
        self.data_dir = data_dir
        self.timestamp = timestamp
        self.tickers = tickers
        self.database_manager = database_manager
        self.logger = logger

    def _get_data(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve raw historical market data for a single ticker.

        This function acts as the data ingestion layer for the pipeline.
        It supports loading from either cached JSON files (for fast iteration)
        or a database backend (for larger-scale or production use).

        The returned data is unprocessed and reflects the original market feed,
        making it suitable as the starting point for all downstream
        preprocessing and feature engineering steps.

        :param ticker: Stock ticker symbol to retrieve data for
        :type ticker: str
        :return: Raw OHLCV market data indexed by timestamp
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info(f'Retrieving {ticker} data...')

        if self.database_manager:
            df = self.database_manager.get_1min_market_data_df(ticker)
            df = df.drop(['symbol'], axis=1)
            return df
        raw_data_path = self.data_dir / 'raw'
        with open(f'{raw_data_path}/{ticker}.json', 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df

    def _handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing timestamps and market data gaps.

        Financial time series often contain missing minutes due to
        exchange pauses, data outages, or low-liquidity periods.
        This method enforces a continuous 1-minute time index so that
        time-based indicators and rolling features behave correctly.

        - Forward-fills price-based fields to maintain price continuity
        - Sets volume and transaction counts to zero when no trades occur
        - Restricts data to active trading hours and weekdays

        This step is critical for ensuring temporal consistency and
        preventing downstream indicators from misinterpreting gaps
        as extreme market events.

        :param df: Raw market data with possible missing timestamps
        :type df: pd.DataFrame
        :return: Gap-free, time-aligned market data
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info("Handling gaps...")

        df = df.set_index("timestamp")
        df = df.tz_convert("America/New_York")
        df = df.sort_index()

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=df.index.min().date(),
            end_date=df.index.max().date()
        )

        session_frames = []

        for session_date, row in schedule.iterrows():

            market_open = row["market_open"].tz_convert("America/New_York")
            market_close = row["market_close"].tz_convert("America/New_York")

            session_raw = df.loc[
                (df.index >= market_open) &
                (df.index < market_close)
            ]

            if session_raw.empty:
                continue

            full_index = pd.date_range(
                start=market_open,
                end=market_close - pd.Timedelta(minutes=1),
                freq="1min",
                tz="America/New_York"
            )

            session = session_raw.reindex(full_index)

            price_cols = ["open", "high", "low", "close", "vwap"]
            volume_cols = ["volume", "transactions"]

            session[price_cols] = session[price_cols].ffill()
            session[volume_cols] = session[volume_cols].fillna(0)

            if session[price_cols].iloc[0].isna().any():
                continue

            session_frames.append(session)

        if not session_frames:
            raise ValueError("No valid sessions after gap handling.")

        df_clean = pd.concat(session_frames)

        return df_clean.reset_index().rename(columns={"index": "timestamp"})

    def _add_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical time-based market patterns using sine and cosine transforms.

        Intraday markets exhibit strong periodic behavior driven by
        human activity, institutional schedules, and market microstructure
        (e.g., open/close volatility, lunch-hour slowdowns).

        By encoding time components (minute, hour, day, month, quarter)
        as cyclical features, the model can learn recurring temporal
        patterns without artificial discontinuities (e.g., 23:59 → 00:00).

        Precalculus FTW bois 😤📐

        :param df: Market data containing timestamps
        :type df: pd.DataFrame
        :return: Data enriched with cyclical time features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding temporal patterns...')
            
        timestamp = df['timestamp']
        
        minute = timestamp.dt.minute
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        
        hour = timestamp.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        day = timestamp.dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 5)
        df['day_cos'] = np.cos(2 * np.pi * day / 5)
        
        df['minutes_since_open'] = df.groupby(timestamp.dt.date).cumcount()
        df['minutes_to_close'] = (
            df.groupby(timestamp.dt.date)
            .cumcount(ascending=False)
        )
        
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info('Adding volatility features...')
            
        close = df['close']
        
        df['log_return_1'] = np.log(close / close.shift(1))
        
        df['volatility_15m'] = ta.STDDEV(close, timeperiod=15)
        df['volatility_15m'] = np.log1p(df['volatility_15m'])

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info('Adding trend features...')
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        df['ema_ratio_5'] = close / ta.EMA(close, timeperiod=5)
        df['ema_ratio_15'] = close / ta.EMA(close, timeperiod=15)
        
        df['adx_20'] = ta.ADX(high, low, close, timeperiod=20)
        df['adx_20'] = df['adx_20'] / 100.0

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info('Adding momentum features...')
            
        close = df['close']
        
        df['roc_10'] = ta.ROC(close, timeperiod=10)
        df['roc_10'] = df['roc_10'].clip(-3, 3)

        df['cum_return_15'] = close / close.shift(15) - 1

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info('Adding volume features...')
            
        close = df['close']
        volume = df['volume']
        vwap = df['vwap']
        
        df['volume_log'] = np.log1p(df['volume']) 
        df['volume_ratio_15m'] = volume / ta.EMA(volume, timeperiod=15)
        df['volume_ratio_15m'] = np.log1p(df['volume_ratio_15m'])
        
        price_roc = ta.ROC(close, timeperiod=1)
        df['volume_price_corr_15m'] = volume.rolling(15).corr(price_roc)
        df['volume_price_corr_15m'] = df['volume_price_corr_15m'].fillna(0.0)
        
        df['price_vwap_distance'] = (close - vwap) / vwap
        
        return df

    def _drop_unnecessary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logger:
            self.logger.info('Dropping unnecessary features...')
            
        # df = df.drop(['open', 'high', 'low', 'transactions', 'volume', 'vwap'], axis=1)
        df = df.drop(['open', 'high', 'low'], axis=1)
         
        return df

    def _drop_rows_before_timestamp(self, df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
        """
        Remove early data to ensure relevant historical context.

        This function trims the dataset to exclude rows where
        features may be unreliable or incomplete (rows during pre-covid).

        :param df: Feature-engineered data
        :type df: pd.DataFrame
        :param timestamp: Earliest timestamp to retain
        :type timestamp: str
        :return: Cleaned dataset with valid feature history
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info(f'Dropping rows before {timestamp}...')
        
        cutoff_timestamp = pd.to_datetime(timestamp)
        return df[df['timestamp'] >= cutoff_timestamp]
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def clip_first_and_last_day(df: pd.DataFrame) -> pd.DataFrame:
            df['date'] = df['timestamp'].dt.date
            valid_dates = df['date'].unique()
            
            if len(valid_dates) > 2:
                df = df[df['date'].isin(valid_dates[1:-1])]
            
            return df.drop(columns=['date'])
            
        if self.logger:
            self.logger.info('Splitting data...')
            
        TRAIN_RATIO = 0.7
        VALID_RATIO = 0.15

        df_length = len(df)
        train_idx = int(df_length * TRAIN_RATIO)
        valid_idx = int(df_length * (TRAIN_RATIO + VALID_RATIO))
        
        df_train = df[:train_idx].reset_index(drop=True)
        df_valid = df[train_idx:valid_idx].reset_index(drop=True)
        df_test  = df[valid_idx:].reset_index(drop=True)

        df_train = clip_first_and_last_day(df_train)
        df_valid = clip_first_and_last_day(df_valid)
        df_test = clip_first_and_last_day(df_test)
        
        return df_train, df_valid, df_test

    def _normalize_data(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features_to_scale = [
            'transactions', 'volume', 'vwap',
            'minutes_since_open', 'minutes_to_close',               # _add_temporal_patterns 
            'log_return_1', 'volatility_15m',                       # _add_volatility_features
            'adx_20',                                               # _add_trend_features 
            'roc_10', 'cum_return_15',                              # _add_momentum_features
            'volume_log', 'volume_ratio_15m', 'price_vwap_distance' # _add_volume_features 
        ]
        
        if self.logger:
            self.logger.info(f'Normalizing features {features_to_scale}...')
            
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(df_train[features_to_scale])
        X_valid_scaled = scaler.transform(df_valid[features_to_scale])
        X_test_scaled  = scaler.transform(df_test[features_to_scale])

        df_train = df_train.copy()
        df_valid = df_valid.copy()
        df_test  = df_test.copy()

        df_train[features_to_scale] = X_train_scaled
        df_valid[features_to_scale] = X_valid_scaled
        df_test[features_to_scale]  = X_test_scaled
        
        for feature in features_to_scale:
            df_train[feature] = df_train[feature].clip(-5, 5)
            df_valid[feature] = df_valid[feature].clip(-5, 5)
            df_test[feature] = df_test[feature].clip(-5, 5)
        
        return df_train, df_valid, df_test

                
    def _preprocess_and_save_data(self, ticker: str, timestamp: str) -> None:
        """
        Run the full feature engineering pipeline for a single ticker.

        :param ticker: Stock ticker symbol to process
        :type ticker: str
        :param timestamp: Cutoff timestamp for valid feature history
        :type timestamp: str
        """
        if self.logger:
            self.logger.info(f'Preprocessing {ticker} data')
        
        df = self._get_data(ticker)
        df = self._handle_gaps(df)
        df = self._add_temporal_patterns(df)
        df = self._add_volatility_features(df)
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volume_features(df)
        df = self._drop_unnecessary_features(df)
        df = self._drop_rows_before_timestamp(df, timestamp)
        
        df_train, df_valid, df_test = self._split_data(df)
        df_train, df_valid, df_test = self._normalize_data(df_train, df_valid, df_test)
        
        base_dir = self.data_dir / 'preprocessed' / 'v3' / ticker
        create_directory(base_dir)
        save_to_csv(df_train, f'{base_dir}/{ticker}_train.csv', index=False)
        save_to_csv(df_valid, f'{base_dir}/{ticker}_valid.csv', index=False)
        save_to_csv(df_test, f'{base_dir}/{ticker}_test.csv', index=False)

    def preprocess_and_save_tickers(self) -> None:
        """
        Run the full feature engineering pipeline for specified tickers.
        
        :param tickers: Stock ticker symbols to process
        :type tickers: List[str]
        """
        for ticker in self.tickers:
            self._preprocess_and_save_data(ticker, self.timestamp)
        
        if self.logger:
            self.logger.info('Preprocessing complete. Godspeed brotherman')

def main():
    data_preprocessor = DataPreprocessor(logger=Logger())
    data_preprocessor.preprocess_and_save_tickers()
    
if __name__ == '__main__':
    main()