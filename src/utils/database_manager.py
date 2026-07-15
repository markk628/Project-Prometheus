import gzip
import pandas as pd
import psycopg2
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from psycopg2 import pool
from psycopg2.extras import execute_values
from psycopg2.sql import Composable, Identifier, SQL
from typing import Optional, Union

from src.utils.logger import Logger
from src.config.config import (
    DATA_DIR, 
    DATABASE_HOST, 
    DATABASE_PORT, 
    DATABASE_NAME, 
    DATABASE_USER, 
    DATABASE_PASSWORD
)

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
            except (Exception) as e:
                if self.logger:
                    self.logger.error(f'Database error ({type(e).__name__}): {e}')
                raise
        return wrapper
    
    @db_exception_handler
    def execute(self, cursor, connection, query: Union[str, Composable], params=None, fetch: str=None, autocommit: bool=False) -> Optional[pd.DataFrame]:
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
        if autocommit:
            connection.autocommit = True
        cursor.execute(query, params)
        if autocommit:
            connection.autocommit = False
        else:
            connection.commit()
        if fetch:
            records = cursor.fetchone() if fetch == 'one' else cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            if fetch == 'one':
                return pd.DataFrame([records], columns=column_names)
            else:
                return pd.DataFrame(records, columns=column_names)
        
    @db_exception_handler
    def execute_many(self, cursor, connection, query: Union[str, Composable], params=None) -> None:
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
    def execute_values(self, cursor, connection, query: Union[str, Composable], params=None) -> None:
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
        
    @db_exception_handler
    def copy_from_gzip_csv(self, cursor, connection, file_path: str, schema: str, table: str) -> None:
        """
        Fast ingest of Polygon CSV.GZ using PostgreSQL COPY.
        """
        sql = SQL("""
            CREATE TEMP TABLE _stage (
                ticker TEXT,
                volume DOUBLE PRECISION,
                open DOUBLE PRECISION,
                close DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                window_start BIGINT,
                transactions INTEGER
            ) ON COMMIT DROP;
        """)
        cursor.execute(sql)

        copy_sql = """
            COPY _stage (ticker, volume, open, close, high, low, window_start, transactions)
            FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        """
        with gzip.open(file_path, "rt") as f:
            cursor.copy_expert(copy_sql, f)

        cursor.execute("SELECT COUNT(*) FROM _stage WHERE ticker IS NULL OR window_start IS NULL")
        dropped = cursor.fetchone()[0]
        if dropped and self.logger:
            self.logger.warning(f"{Path(file_path).name}: dropping {dropped} rows with null ticker/window_start")

        insert_sql = SQL("""
            INSERT INTO {} (ticker, volume, open, close, high, low, window_start, transactions, timestamp)
            SELECT
                ticker,
                CAST(volume AS BIGINT),
                open,
                close,
                high,
                low,
                window_start,
                transactions,
                to_timestamp(window_start / 1000000000.0)
            FROM _stage
            WHERE ticker IS NOT NULL
            AND window_start IS NOT NULL
            ON CONFLICT DO NOTHING;
        """).format(Identifier(schema, table))
        cursor.execute(insert_sql)
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
        if self.logger:
            self.logger.info('Schemas created successfully.')
        
    def create_raw_market_data_table(self, schema: str, table: str, chunk_interval: str) -> None:
        """
        Create a table for raw market data and convert it to a TimescaleDB hypertable.
        """

        create_table_query = """
            CREATE TABLE IF NOT EXISTS {} (
                ticker TEXT NOT NULL,
                volume BIGINT,
                open DOUBLE PRECISION,
                close DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                window_start BIGINT NOT NULL,
                transactions INTEGER,
                timestamp TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (ticker, timestamp)
            );
        """

        identifier = Identifier(schema, table)

        create_table_sql = SQL(create_table_query).format(identifier)

        hypertable_sql = SQL("""
            SELECT create_hypertable(%s, 'timestamp',
                chunk_time_interval => INTERVAL %s,
                if_not_exists => TRUE);
        """)

        table_name = f"{schema}.{table}"

        if self.logger:
            self.logger.info(f"Creating {table_name} table...")

        self.execute(create_table_sql)
        self.execute(hypertable_sql, (table_name, chunk_interval))

        if self.logger:
            self.logger.info(f"{table_name} table created successfully.")
    
    def insert_market_data(self, schema: str, table: str, data_dir: Path) -> None:
        """
        Insert compressed CSV data into a TimescaleDB table.
        """

        table_name = f"{schema}.{table}"

        if self.logger:
            self.logger.info(f"Inserting data into {table_name}...")

        for year in sorted(data_dir.iterdir()):
            if not year.is_dir():
                continue

            for month in sorted(year.iterdir()):
                if not month.is_dir():
                    continue

                for file in sorted(month.glob("*.csv.gz")):
                    self.copy_from_gzip_csv(str(file), schema, table)

                    if self.logger:
                        size_mb = file.stat().st_size / 1e6
                        self.logger.info(f"Loaded {file.name} ({size_mb:.1f} MB)")

        if self.logger:
            self.logger.info(f"{table_name} insert completed successfully.")
            
    def enable_compression(self, schema: str, table: str) -> None:
        """
        Enable TimescaleDB compression for a table.
        """

        identifier = Identifier(schema, table)

        compression_sql = SQL("""
            ALTER TABLE {} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'ticker',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
        """).format(identifier)

        table_name = f"{schema}.{table}"

        if self.logger:
            self.logger.info(f"Enabling compression for {table_name}...")

        self.execute(compression_sql)

        if self.logger:
            self.logger.info(f"Compression enabled for {table_name}.")
    
    def apply_compression_policy(self, schema: str, table: str, interval: str) -> None:
        """
        Add automatic compression policy.
        """

        policy_sql = SQL("SELECT add_compression_policy(%s, INTERVAL %s, if_not_exists => TRUE);")

        table_name = f"{schema}.{table}"

        if self.logger:
            self.logger.info(f"Adding compression policy ({interval}) to {table_name}...")

        self.execute(policy_sql, (table_name, interval))

        if self.logger:
            self.logger.info("Compression policy added successfully.")
            
    def analyze_table(self, schema: str, table: str) -> None:
        identifier = Identifier(schema, table)
        sql = SQL("ANALYZE {};").format(identifier)
        
        table_name = f"{schema}.{table}"

        if self.logger:
            self.logger.info(f"Analyzing {table_name}...")

        self.execute(sql)

        if self.logger:
            self.logger.info(f"{table_name} analyzed successfully.")
            
    @contextmanager
    def bulk_ingest_optimizations(self):
        """
        Context manager that applies session-level PostgreSQL optimizations for
        bulk ingestion, then restores defaults on exit.

        This sets:
          - synchronous_commit = off  : flushes WAL asynchronously (~200ms lag),
                                        giving the same throughput benefit without
                                        risking data loss beyond a crash window.
          - work_mem               : more memory per sort/hash for large COPYs.
          - maintenance_work_mem   : speeds up index builds post-ingest.

        Usage:
            with self.bulk_ingest_optimizations():
                self.insert_market_data(...)
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if self.logger:
                    self.logger.info('Applying bulk ingest session optimizations...')
                cursor.execute("SET LOCAL synchronous_commit = off;")
                cursor.execute("SET LOCAL work_mem = '256MB';")
                cursor.execute("SET LOCAL maintenance_work_mem = '512MB';")
                connection.commit()
            try:
                yield
            finally:
                with connection.cursor() as cursor:
                    cursor.execute("RESET synchronous_commit;")
                    cursor.execute("RESET work_mem;")
                    cursor.execute("RESET maintenance_work_mem;")
                    connection.commit()
                if self.logger:
                    self.logger.info('Bulk ingest session optimizations reset.')
            
    # def create_time_index(self, schema: str, table: str) -> None:
    #     index_sql = SQL("""
    #         CREATE INDEX IF NOT EXISTS {} 
    #         ON {} (ticker, timestamp DESC);
    #     """).format(
    #         Identifier(f"{table}_timestamp_idx"),
    #         Identifier(schema, table)
    #     )

    #     if self.logger:
    #         self.logger.info(f"Creating timestamp index on {schema}.{table}...")

    #     self.execute(index_sql)
        
    #     if self.logger:
    #         self.logger.info('Timestamp index created successfully.')
            
    def create_continuous_aggregates(self, schema: str, source_table: str) -> None:
        """
        Create common continuous aggregates (5m, 15m, 30m, 1h, 1d)
        from a base hypertable containing minute data.
        """

        aggregates = {
            # "5m": "5 minutes",
            # "15m": "15 minutes",
            # "30m": "30 minutes",
            "1h": "1 hour",
            "1d": "1 day"
        }

        source_table_name = f"{schema}.{source_table}"

        for suffix, interval in aggregates.items():

            view_name = f"{source_table}_{suffix}"
            full_view_name = f"{schema}.{view_name}"

            query = f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {full_view_name}
                WITH (
                    timescaledb.continuous,
                    timescaledb.materialized_only = false
                ) AS
                SELECT
                    ticker,
                    time_bucket('{interval}', timestamp) AS bucket,
                    first(open, timestamp) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, timestamp) AS close,
                    sum(volume) AS volume,
                    sum(transactions) AS transactions
                FROM {source_table_name}
                GROUP BY ticker, bucket;
            """

            if self.logger:
                self.logger.info(f"Creating continuous aggregate {full_view_name}...")

            self.execute(query, autocommit=True)

            if self.logger:
                self.logger.info(f"{full_view_name} created successfully.")
                
    def add_continuous_aggregate_policy(self, schema: str, table: str, interval: str = "1 hour") -> None:
        """
        Adds refresh policies to continuous aggregates.
        """

        aggregates = ["5m", "15m", "30m", "1h", "1d"]

        for suffix in aggregates:

            view_name = f"{schema}.{table}_{suffix}"

            query = f"""
            SELECT add_continuous_aggregate_policy(
                '{view_name}',
                start_offset => INTERVAL '30 days',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '{interval}'
            );
            """

            if self.logger:
                self.logger.info(f"Adding refresh policy to {view_name}...")

            self.execute(query)
            
        if self.logger:
            self.logger.info("Successfully added refresh policies")
            
    def enable_aggregate_compression(self, schema: str, table: str) -> None:
        """
        Enable compression on continuous aggregates.
        """

        aggregates = ["5m", "15m", "30m", "1h", "1d"]

        for suffix in aggregates:

            identifier = Identifier(schema, f"{table}_{suffix}")

            query = """
                ALTER MATERIALIZED VIEW {} SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'ticker',
                    timescaledb.compress_orderby = 'bucket DESC'
                );
            """

            sql = SQL(query).format(identifier)

            if self.logger:
                self.logger.info(
                    f"Enabling compression on {schema}.{table}_{suffix}"
                )

            self.execute(sql)
            
    def add_aggregate_compression_policy(self, schema: str, table: str) -> None:
        """
        Add compression policies to continuous aggregates.
        """

        policies = {
            "5m": "30 days",
            "15m": "30 days",
            "30m": "30 days",
            "1h": "60 days",
            "1d": "180 days"
        }

        for suffix, interval in policies.items():

            view_name = f"{schema}.{table}_{suffix}"

            query = f"""
                SELECT add_compression_policy(
                    '{view_name}',
                    INTERVAL '{interval}',
                    if_not_exists => TRUE
                );
            """

            if self.logger:
                self.logger.info(
                    f"Adding compression policy ({interval}) to {view_name}"
                )

            self.execute(query)
    
    def set_up_database(self) -> None:
        self.create_schemas()
        self.create_raw_market_data_table(
            schema=self.MARKET_DATA_RAW_SCHEMA,
            table=self.MARKET_DATA_BARS_1MIN,
            chunk_interval="1 day"
        )
        # self.create_time_index(
        #     schema=self.MARKET_DATA_RAW_SCHEMA,
        #     table=self.MARKET_DATA_BARS_1MIN
        # )
        
    def insert_data(self) -> None:
        # with self.bulk_ingest_optimizations():
        #     self.insert_market_data(
        #         schema=self.MARKET_DATA_RAW_SCHEMA,
        #         table=self.MARKET_DATA_BARS_1MIN,
        #         data_dir=Path(f'{DATA_DIR}/raw/minute_data')
        #     )
        # self.enable_compression(
        #     schema=self.MARKET_DATA_RAW_SCHEMA,
        #     table=self.MARKET_DATA_BARS_1MIN
        # )
        # self.apply_compression_policy(
        #     schema=self.MARKET_DATA_RAW_SCHEMA,
        #     table=self.MARKET_DATA_BARS_1MIN,
        #     interval='30 days'
        # )
        self.create_continuous_aggregates(
            schema=self.MARKET_DATA_RAW_SCHEMA,
            source_table=self.MARKET_DATA_BARS_1MIN
        )
        self.add_continuous_aggregate_policy(
            schema=self.MARKET_DATA_RAW_SCHEMA,
            table=self.MARKET_DATA_BARS_1MIN
        )
        self.enable_aggregate_compression(
            schema=self.MARKET_DATA_RAW_SCHEMA,
            table=self.MARKET_DATA_BARS_1MIN
        )
        self.add_aggregate_compression_policy(
            schema=self.MARKET_DATA_RAW_SCHEMA,
            table=self.MARKET_DATA_BARS_1MIN
        )
    
    def close(self) -> None:
        if self.connection_pool:
            self.connection_pool.closeall()
        
def main():
    database_manager = DatabaseManager(logger=Logger())
    try:
        # database_manager.set_up_database()
        database_manager.insert_data()
    finally:
        database_manager.close()
    
if __name__ == '__main__':
    main()