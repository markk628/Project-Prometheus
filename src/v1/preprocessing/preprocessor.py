import json
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import talib as ta
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
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add standard technical analysis indicators.

        Technical indicators transform raw OHLCV data into signals
        that capture momentum, trend strength, volatility, and
        buying/selling pressure.

        These indicators provide the model with higher-level abstractions
        of market behavior that are difficult to infer directly from
        raw prices alone, especially in noisy intraday data.

        :param df: Market data to enrich with technical indicators
        :type df: pd.DataFrame
        :return: Data augmented with technical indicator features
        :rtype: pd.DataFrame
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicators = {
            'stochrsi_k_14_1min,stochrsi_d_14_1min': (ta.STOCHRSI, {'real': close, 'timeperiod': 14}),
            'rsi_14_1min': (ta.RSI, {'real': close, 'timeperiod': 14}),
            'macd_12_26_9_1min,macd_signal_12_26_9_1min,macd_hist_12_26_9_1min': (ta.MACD, {'real': close, 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            'roc_10_1min': (ta.ROC, {'real': close, 'timeperiod': 10}),
            'plusdi_20_1min': (ta.PLUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'minusdi_20_1min': (ta.MINUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'adx_20_1min': (ta.ADX, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'cci_20_1min': (ta.CCI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'ema_3_1min': (ta.EMA, {'real': close, 'timeperiod': 3}),
            'ema_9_1min': (ta.EMA, {'real': close, 'timeperiod': 9}),
            'ema_21_1min': (ta.EMA, {'real': close, 'timeperiod': 21}),
            'obv_1min': (ta.OBV, {'close': close, 'volume': volume}),
            'mfi_14_1min': (ta.MFI, {'high': high, 'low': low, 'close': close, 'volume': volume, 'timeperiod': 14}),
            'bband_upper_20_1min,bband_middle_20_1min,bband_lower_20_1min': (ta.BBANDS, {'real': close, 'timeperiod': 20}),
            'atr_14_1min': (ta.ATR, {'high': high, 'low': low, 'close': close, 'timeperiod': 14})
        }

        if self.logger:
            self.logger.info('Adding technical indicators...')
            
        for out_cols, (func, params) in indicators.items():
            if out_cols == 'obv_1min':
                results = func(*params.values())
            else:
                results = func(**params)
            if isinstance(results, tuple):
                for i, col_name in enumerate(out_cols.split(',')):
                    df[col_name] = results[i]
            else:
                df[out_cols] = results
        return df

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
        # probably out of scope for intraday trading
        # month = timestamp.dt.month
        # df['month_sin'] = np.sin(2 * np.pi * month / 12)
        # df['month_cos'] = np.cos(2 * np.pi * month / 12)
        # quarter = timestamp.dt.quarter
        # df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
        # df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        return df

    def _add_last_significant_change(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Add a feature measuring time since the last significant price movement.

        This feature captures market "memory" by indicating how long the
        price has remained relatively stable since the last large move.
        It helps the model distinguish between consolidation phases
        and high-momentum regimes.

        :param df: Market data to analyze
        :type df: pd.DataFrame
        :param threshold: Fixed percentage change defining a significant move
        :type threshold: float
        :return: Data with time-since-last-significant-move feature
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding last significant change...')
            
        price_change_pct = df['close'].pct_change().abs()
        significant_change = (price_change_pct > threshold).astype(int)
        group_changes = (significant_change != significant_change.shift()).cumsum()
        df['time_since_last_significant_change'] = df.groupby(group_changes).cumcount() + 1
        return df

    def _add_lagged_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Add lagged versions of core market variables.

        Lagged features provide the model with short-term historical context,
        enabling it to learn temporal dependencies and short-horizon dynamics
        without relying solely on recurrent architectures.

        This is especially important for intraday trading, where recent
        price and volume behavior often dominates decision-making.

        :param df: Market data to augment
        :type df: pd.DataFrame
        :param lags: Number of past minutes to include for each feature
        :type lags: List[int]
        :return: Data enriched with lagged features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding lagged features...')
            
        for lag in lags:
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def _add_windowed_statistics(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Add rolling window statistics for price and volume.

        Rolling means and standard deviations summarize recent market
        behavior over multiple time horizons, allowing the model to
        compare current conditions against short- and medium-term norms.

        These features help capture local trend strength, volatility
        expansion/contraction, and abnormal trading activity.

        :param df: Market data to analyze
        :type df: pd.DataFrame
        :param windows: Rolling window sizes (in minutes)
        :type windows: List[int]
        :return: Data with rolling statistical features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding windowed statistics...')
        
        for window in windows:
            for col in ['close', 'volume']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        return df

    def _add_price_differences_and_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price change and return-based features.

        These features directly encode short-term price movement,
        direction, and magnitude, which are fundamental signals
        for momentum and mean-reversion strategies.

        Log returns are included for numerical stability and to
        better reflect proportional price changes.

        :param df: Market data to augment
        :type df: pd.DataFrame
        :return: Data enriched with price difference and return features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding price differences and related features...')
            
        df['close_diff_1'] = df['close'].diff(1)
        df['close_pct_change_1'] = df['close'].pct_change(1)
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_range'] = df['close'] - df['open']
        return df

    def _add_volatility_measures(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Add rolling volatility estimates based on log returns.

        Volatility is a key state variable in trading.
        Measuring it over multiple time horizons allows the model
        to adapt its behavior during calm versus turbulent periods,
        which is especially important for risk-sensitive policies.

        :param df: Market data to analyze
        :type df: pd.DataFrame
        :param windows: Rolling window sizes for volatility estimation
        :type windows: List[int]
        :return: Data with volatility features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding volatility measures...')
        
        for window in windows:
            df[f'log_return_rolling_std_{window}'] = df['log_return_1'].rolling(window=window).std()
        return df

    def _add_OHLC_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ratio-based OHLC features.

        Ratios between OHLC values normalize price relationships
        and help the model detect candle structure patterns such as
        range expansion, directional bias, and intrabar dominance.

        These features are scale-invariant and generalize well
        across assets with different price levels.

        :param df: Market data to augment
        :type df: pd.DataFrame
        :return: Data with OHLC ratio features
        :rtype: pd.DataFrame
        """
        if self.logger:
            self.logger.info('Adding OHLC ratios...')
        
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
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
            self.logger.info('Dropping rows...')
        
        cutoff_timestamp = pd.to_datetime(timestamp).tz_localize('UTC')
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

    # def _normalize_data(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     # stats = df_train.describe() # TODO after setting up a working env use describe() to determine which features to normalize
    #     for col in df_train.columns:
    #         if col not in TEMPORAL_FEATURES:
                
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
        # TODO keep it simple until environment is set up
        # df = self._add_technical_indicators(df)
        # df = self._add_temporal_patterns(df)
        # df = self._add_last_significant_change(df, threshold=0.001)
        # df = self._add_lagged_features(df, [1, 5, 10])
        # df = self._add_windowed_statistics(df, [5, 15, 30])
        # df = self._add_price_differences_and_returns(df)
        # df = self._add_volatility_measures(df, [15, 30])
        # df = self._add_OHLC_ratios(df)
        df = self._add_temporal_patterns(df)
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_log'] = np.log1p(df['volume']) # TODO volume_log_norm = (volume_log - volume_log.mean()) / volume_log.std() instead of scaling use this instead MAKE SURE THERE IS NO DATA LEAK
        df['price_vwap_distance'] = (df["close"] - df["vwap"]) / df["vwap"]
        df = df.drop(['open', 'high', 'low', 'transactions', 'volume', 'vwap'], axis=1) 
        
        df = self._drop_rows_before_timestamp(df, timestamp)
        df_train, df_valid, df_test = self._split_data(df)
        # df_train, df_valid, df_test = self._normalize_data(df_train, df_valid, df_test)
        base_dir = self.data_dir / 'preprocessed' / 'v1' / ticker
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