import os
import pandas as pd
import polars as pl
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
        
def save_to_csv(df: Union[pd.DataFrame, pl.DataFrame], file_path: Union[str, Path], index: bool = True, logger: Logger=None) -> None:
    """
    Saves a DataFrame to a CSV file
    
    :param df: DataFrame to save
    :type df: DataFrame
    :param file_path: Path where the file will be saved
    :type file_path: Union[str, Path]
    :param index: Whether to include the index
    :type index: bool
    """
    create_directory(os.path.dirname(file_path))
    if isinstance(df, pd.DataFrame):
        df.to_csv(file_path, index=index, encoding='utf-8-sig')
    elif isinstance(df, pl.DataFrame):
        df.write_csv(file_path, include_bom=True)
    else:
        if logger:
            logger.error(f"Did not implement a way to handle saving object of type {type(df)} as csv.")
    if logger:
        logger.info(f"File saved: {file_path}")
        
def save_to_parquet(df: Union[pd.DataFrame, pl.DataFrame], file_path: Union[str, Path], index: bool = True, logger: Logger=None) -> None:
    """
    Saves a DataFrame to a CSV file
    
    :param df: DataFrame to save
    :type df: DataFrame
    :param file_path: Path where the file will be saved
    :type file_path: Union[str, Path]
    :param index: Whether to include the index
    :type index: bool
    """
    create_directory(os.path.dirname(file_path))
    if isinstance(df, pd.DataFrame):
        df.to_parquet(file_path, index=index)
    elif isinstance(df, pl.DataFrame):
        df.write_parquet(file_path)
    else:
        if logger:
            logger.error(f"Did not implement a way to handle saving object of type {type(df)} as parquet.")
    if logger:
        logger.info(f"File saved: {file_path}")

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

def load_stock_data(data_path: str, start_timestamp: Optional[Union[str, datetime]]=None, end_timestamp: Optional[Union[str, datetime]]=None) -> Union[pl.DataFrame, pd.DataFrame]:
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
        raise FileNotFoundError(f"Where's the file, Lebowski: {data_path}")
    
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        df = pl.read_parquet(data_path)
        df = df.with_columns(
            pl.col('timestamp').cast(pl.Datetime('us', 'UTC'))
        )

    if start_timestamp:
        if isinstance(start_timestamp, str):
            df = df[df['timestamp'] >= start_timestamp]
        else:
            df = df.filter(
                pl.col('timestamp') >= start_timestamp
            )
    
    if end_timestamp:
        if isinstance(end_timestamp, str):
            df = df[df['timestamp'] < end_timestamp]
        else:
            df = df.filter(
                pl.col('timestamp') <= end_timestamp
            )
    return df.sort('timestamp')