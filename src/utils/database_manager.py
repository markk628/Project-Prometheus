import json
import pandas as pd
import psycopg2
from contextlib import contextmanager
from datetime import timezone
from functools import wraps
from psycopg2 import pool
from psycopg2.extras import execute_values
from psycopg2.sql import Identifier, Literal, SQL
from typing import List, Optional

from src.utils.logger import Logger
from src.config.config import DATA_DIR, TICKERS, DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD


class DatabaseManager:
    def __init__(
        self,
        host: str = DATABASE_HOST,
        port: int = DATABASE_PORT,
        database: str = DATABASE_NAME,
        user: str = DATABASE_USER,
        password: str = DATABASE_PASSWORD,
        logger: Optional[Logger] = None
    ):
        """
        Initialize DatabaseManager with connection parameters and an optional logger.

        :param host: Database host address
        :type host: str
        :param port: Database port number
        :type port: int
        :param database: Database name
        :type database: str
        :param user: Database username
        :type user: str
        :param password: Database password
        :type password: str
        :param logger: Optional logger instance for recording database operations
        :type logger: Optional[Logger]
        """
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.logger = logger
        self.connection_pool = None
        self.MARKET_DATA_RAW_SCHEMA = "market_data_raw"
        self.MARKET_DATA_FEATURES_SCHEMA = "market_data_features"
        self.MARKET_DATA_BARS_1MIN = "bars_1m"
        self.MODELS_SCHEMA = "models"
        self.BACKTESTS_SCHEMA = "backtests"
        self.TRADES_SCHEMA = "trades"
        self.LOGS_SCHEMA = "logs"
        self._initialize_pool()
        
    def _initialize_pool(self) -> None:
        """
        Initialize a PostgreSQL connection pool to efficiently manage multiple connections.
        """
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1, 
                maxconn=10,
                **self.config
            )
            if self.logger:
                self.logger.info(
                    f'Database connection pool initialized: {self.config["host"]}:{self.config["port"]}/{self.config["database"]}'
                )
        except pool.PoolError as e:
            if self.logger:
                self.logger.error(f'Failed to initialize database connection pool: {e}')
            raise
        except psycopg2.OperationalError as e:
            if self.logger:
                self.logger.error(f'Database connection failed: {e}')
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f'Unexpected database error: {e}')
            raise
        
    @contextmanager
    def _connect(self):
        """
        Provide a database connection from the connection pool with context management.

        :yield: Active database connection
        :rtype: psycopg2.extensions.connection
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except pool.PoolError as e:
            if self.logger:
                self.logger.error(f'Failed to obtain connection from pool: {e}')
            raise
        except psycopg2.OperationalError as e:
            if self.logger:
                self.logger.error(f'Database connection failed: {e}')
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f'Unexpected database error: {e}')
            raise
        finally:
            if connection:
                try:
                    self.connection_pool.putconn(connection)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f'Failed to return connection to pool: {e}')
                        
    def db_exception_handler(method):
        """
        Decorator for handling database exceptions and logging them.

        :param method: Method to wrap with exception handling
        :type method: callable
        :return: Wrapped method with exception handling
        :rtype: callable
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                with self._connect() as connection:
                    with connection.cursor() as cursor:
                        return method(self, cursor, connection, *args, **kwargs)
            except (psycopg2.OperationalError,
                    psycopg2.ProgrammingError,
                    psycopg2.IntegrityError,
                    psycopg2.DataError,
                    psycopg2.InterfaceError,
                    psycopg2.DatabaseError,
                    Exception) as e:
                if self.logger:
                    self.logger.error(f'Database error ({type(e).__name__}): {e}')
                raise
        return wrapper
    
    @db_exception_handler
    def execute(self, cursor, connection, query: str, params=None, fetch: str = None) -> Optional[pd.DataFrame]:
        """
        Execute a single SQL statement, optionally fetching results as a pandas DataFrame.

        :param cursor: Active database cursor
        :type cursor: psycopg2.extensions.cursor
        :param connection: Active database connection
        :type connection: psycopg2.extensions.connection
        :param query: SQL query string to execute
        :type query: str
        :param params: Optional query parameters
        :type params: Optional[tuple]
        :param fetch: Fetch mode: 'one' for single row, 'all' for all rows
        :type fetch: Optional[str]
        :return: Query results as a pandas DataFrame if fetch is specified, otherwise None
        :rtype: Optional[pd.DataFrame]
        """
        cursor.execute(query, params)
        connection.commit()
        if fetch:
            records = cursor.fetchone() if fetch == 'one' else cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            if fetch == 'one':
                return pd.DataFrame([records], columns=column_names)
            else:
                return pd.DataFrame(records, columns=column_names)
        
    @db_exception_handler
    def execute_many(self, cursor, connection, query: str, params=None) -> None:
        """
        Execute multiple SQL statements efficiently in a batch.

        :param cursor: Active database cursor
        :type cursor: psycopg2.extensions.cursor
        :param connection: Active database connection
        :type connection: psycopg2.extensions.connection
        :param query: SQL query string to execute
        :type query: str
        :param params: List of parameter tuples for batch execution
        :type params: Optional[List[tuple]]
        """
        cursor.executemany(query, params)
        connection.commit()
    
    @db_exception_handler
    def execute_values(self, cursor, connection, query: str, params=None) -> None:
        """
        Perform bulk insertion using psycopg2's execute_values for maximum efficiency.

        :param cursor: Active database cursor
        :type cursor: psycopg2.extensions.cursor
        :param connection: Active database connection
        :type connection: psycopg2.extensions.connection
        :param query: SQL query string to execute
        :type query: str
        :param params: List of tuples containing values to insert
        :type params: List[tuple]
        """
        execute_values(cursor, query.as_string(connection), params)
        connection.commit()
    
    def create_schemas(self) -> None:
        """
        Create schemas for tables
        """
        schemas = [
            self.MARKET_DATA_RAW_SCHEMA, 
            self.MARKET_DATA_FEATURES_SCHEMA, 
            self.MODELS_SCHEMA, 
            self.BACKTESTS_SCHEMA, 
            self.TRADES_SCHEMA, 
            self.LOGS_SCHEMA
        ]
        create_schemas_query = """
            CREATE SCHEMA IF NOT EXISTS {};
        """
        
        if self.logger:
            self.logger.info(f'Creating schemas...')
        for schema in schemas:
            query = SQL(create_schemas_query).format(Identifier(schema))
            self.execute(query)
        
    def create_1min_market_data_table(self) -> None:
        """
        Create a table for 1 minute market data and convert it to TimescaleDB hypertables.
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS {} (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                transactions INTEGER,
                volume BIGINT,
                vwap DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            );
        """
        create_hypertable_query = """
            SELECT create_hypertable({}, 'timestamp', partitioning_column => 'symbol', number_partitions => 8, if_not_exists => TRUE);
        """
        table = f'{self.MARKET_DATA_RAW_SCHEMA}.{self.MARKET_DATA_BARS_1MIN}'
        identifier = Identifier(self.MARKET_DATA_RAW_SCHEMA, self.MARKET_DATA_BARS_1MIN)
        query = SQL(create_table_query).format(identifier)
        hypertable_query = SQL(create_hypertable_query).format(Literal(table))
        
        if self.logger:
            self.logger.info(f'Creating {table} table...')
        self.execute(query)
        self.execute(hypertable_query)
               
    def insert_1min_market_data(self, tickers: List[str]=TICKERS) -> None:
        """
        Insert 1 minute JSON market data into 1 minute market data table.

        :param tickers: List of ticker symbols
        :type tickers: List[str]
        """
        insert_data_query = """
            INSERT INTO {} (symbol, timestamp, open, high, low, close, transactions, volume, vwap) 
            VALUES %s ON CONFLICT (symbol, timestamp) DO NOTHING;
        """
        query = SQL(insert_data_query).format(Identifier(self.MARKET_DATA_RAW_SCHEMA, self.MARKET_DATA_BARS_1MIN))
        
        for ticker in tickers:
            if self.logger:
                self.logger.info(f'Inserting 1 minute data for {ticker} into {self.MARKET_DATA_RAW_SCHEMA}.{self.MARKET_DATA_BARS_1MIN}...')
            filepath = f'{DATA_DIR}/raw/{ticker}.json'
            with open(filepath, 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            data = df.to_dict(orient='records')
            formatted_data = [(
                    ticker,
                    item["timestamp"].to_pydatetime(),
                    item["open"],
                    item["high"],
                    item["low"],
                    item["close"],
                    item["transactions"],
                    item["volume"],
                    item["vwap"]
                ) for item in data
            ]
            self.execute_values(query, formatted_data)
            
    def get_1min_market_data_df(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve all rows from a specific ticker as a pandas DataFrame.

        :param ticker: Ticker symbol to filter by
        :type ticker: str
        :return: DataFrame containing all rows of the ticker 
        :rtype: pd.DataFrame
        """
        query = SQL("SELECT * FROM {} WHERE symbol = %s ORDER BY timestamp").format(Identifier(self.MARKET_DATA_RAW_SCHEMA, self.MARKET_DATA_BARS_1MIN))
        return self.execute(query, params=(ticker,), fetch='all')
    
    # TODO save 1 day data, try to make above method work for data of any timeframe
    
    # TODO save feature engineered data
    
    def set_up_database(self) -> None:
        self.create_schemas()
        self.create_1min_market_data_table()
        self.insert_1min_market_data()
    
    def close(self) -> None:
        if self.connection_pool:
            self.connection_pool.closeall()
        
def main():
    database_manager = DatabaseManager(logger=Logger())
    df = database_manager.get_1min_market_data_df('TSLA')
    print(df.head())
    
if __name__ == '__main__':
    main()
