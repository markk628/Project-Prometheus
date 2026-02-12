import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

from src.utils.logger import Logger

def create_directory(directory_path: Union[str, Path], logger: Logger=None) -> None:
    """
    Creates a directory if it does not already exist
    
    :param directory_path: Path of the directory to create
    :type directory_path: Union[str, Path]
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        if logger:
            logger.info(f"Directory created: {directory_path}")
        
def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = True, logger: Logger=None) -> None:
    """
    Saves a DataFrame to a CSV file
    
    :param df: DataFrame to save
    :type df: pd.DataFrame
    :param file_path: Path where the file will be saved
    :type file_path: Union[str, Path]
    :param index: Whether to include the index
    :type index: bool
    """
    create_directory(os.path.dirname(file_path))
    df.to_csv(file_path, index=index, encoding='utf-8-sig')
    if logger:
        logger.info(f"File saved: {file_path}")

def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file
    
    :param file_path: Path of the csv file to load
    :type file_path: Union[str, Path]
    :return: Loaded DataFrame
    :rtype: DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, encoding='utf-8-sig')

def format_duration(seconds: float):
    """
    Converts a duration in seconds into a human-readable string using the most appropriate time units
    
    :param seconds: Duration in seconds
    :type seconds: float
    :return: Human-readable duration
    :rtype: str
    """
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

    return ' '.join(parts)

def load_stock_data(data_path: str, start_timestamp: Optional[pd.Timestamp]=None) -> Tuple[pd.DataFrame, datetime, datetime]:
    """
    Get saved csv data and filter to regular market hours
    
    :param data_path: Path to data
    :type data_path: str
    :param start_timestamp: Starting timestamp of data. If none the data's default start date will be used
    :type start_timestamp: Optional[pd.Timestamp]
    :return: Data, data's starting and ending timestamps
    :rtype: Tuple[pd.DataFrame, datetime, datetime]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)

    if start_timestamp:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if start_timestamp.tz is not None:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(start_timestamp.tz)
        else:
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= start_timestamp]
        
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df, start_date, end_date