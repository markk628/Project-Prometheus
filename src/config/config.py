import os
import logging
import torch
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# from src.config.api_keys import MASSIVE_APIKEY
# from src.config.database_keys import *

# Keys
load_dotenv()
MASSIVE_APIKEY = os.getenv('MASSIVE_APIKEY')
DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_PORT = os.getenv('DATABASE_PORT')
DATABASE_NAME = os.getenv('DATABASE_NAME')
DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD')

# Directory paths config
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # Project root dir
DATA_DIR = ROOT_DIR / "data"                                                       # Data dir
LOGS_DIR = ROOT_DIR / "logs"                                                       # Logs dir
TRAINING_LOGS_DIR = LOGS_DIR / "training"                                          # Training logs dir
TRADING_LOGS_DIR = LOGS_DIR / "trading"                                            # Trading logs dir
MODELS_DIR = ROOT_DIR / "models"                                                   # Models dir
RESULTS_DIR = ROOT_DIR / "results"                                                 # Results dir

# Stock tickers config
# TICKERS = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
TICKERS = ['TSLA']

# Data config
DATA_TIMESPAN = "minute"          # Data timespan
DATA_START_DATE = '1430899200000' # 2015-05-06 4:00 AM UTC
DATA_END_DATE = '1746573780000'   # 2025-05-06 11:23 PM UTC

# Preprocessing config
CUTOFF_TIMESTAMP = '2021-05-06 08:00:00' 

# Trading env config
WINDOW_SIZE = 20
INITIAL_BALANCE = 10000.0       # Initial trading balance
MAX_TRADING_UNITS = 10          
TRANSACTION_FEE_PERCENT = 0.001 # Trade transcation fee percentage

# Model hyperparameters config
HIDDEN_DIM = 256            # Hidden dim size
LEARNING_RATE_ACTOR = 3e-4  # Actor NN learning rate
LEARNING_RATE_CRITIC = 3e-4 # Crtic NN learning rate
LEARNING_RATE_ALPHA = 3e-4  # Optimizer learning rate
ALPHA_INIT = 0.2            # Entropy temperature (controls how random the policy is)
GAMMA = 0.99                # Discount factor (higher = cares more about long term rewards)
TAU = 0.005                 # Controls how soft the target network is updated
REPLAY_BUFFER_SIZE = 300000 # Replay buffer's max size (increase/decrease based on ram size)
TARGET_UPDATE_INTERVAL = 1

# Training config
BATCH_SIZE = 256
NUM_EPISODES = 1000
VALIDATION_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50

MINUTES_PER_TRADING_DAY = 390   # 9:30 AM to 3:59 PM EST (regular market)
MINUTES_PER_EXTENDED_DAY = 959  # 4:00 AM to 7:59 PM EST (full extended hours)
MINUTES_PER_ENHANCED_DAY = 449  # 8:30 AM to 3:59 PM EST (1hr pre + regular)

# Evaluation config
ANNUAL_RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backtest config
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-01-01" 