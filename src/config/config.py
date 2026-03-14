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
MASSIVE_AWSKEY = os.getenv('MASSIVE_AWSKEY')
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
TICKERS = [

    # ===== Core Market Structure =====
    "SPY", "VOO", "IVV", "VTI", "QQQ",
    "IWM",          # Small caps
    "MDY",          # Mid caps

    # ===== Sector ETFs =====
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",

    # ===== International =====
    "EFA",          # Developed ex-US
    "EEM",          # Emerging markets
    "FXI",          # China
    "EWJ",          # Japan

    # ===== Rates & Bonds =====
    "TLT",          # Long duration
    "IEF",          # 7-10 year
    "SHY",          # Short duration
    "LQD",          # Investment grade credit
    "HYG",          # High yield credit

    # ===== Volatility =====
    "VIXY",         # VIX ETF proxy
    "UVXY",         # Leveraged vol (stress regime signal)

    # ===== Dollar =====
    "UUP",          # Dollar index ETF

    # ===== Commodities =====
    "GLD",          # Gold
    "SLV",          # Silver
    "USO",          # Oil
    "DBA",          # Agriculture

    # ===== Defense / Industrial =====
    "ITA",

    # ===== Real Assets =====
    "VNQ",          # REIT ETF
    "AMT", "PLD",

    # ===== Financial Sensitivity =====
    "JPM", "BAC", "GS", "KKR", "BRK.B",

    # ===== Energy / Materials Leaders =====
    "XOM", "CVX", "COP", "FCX", "NEM",
    
    # ===== Semiconductor Supply Chain =====
    "TSM", "ASML", "AMAT", "LRCX", "KLAC", "AMKR", "ASX", "SOXX", "SMH",

    # ===== Mega Cap Tech =====
    "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", ["FB", "META"], "TSLA", "AMD",

    # ===== Defensive Staples / Healthcare =====
    "PG", "KO", "WMT", "JNJ", "UNH",

    # ===== Emerging Tech / High Beta =====
    "ARKK", "MRNA", "SHOP", "NET", "ROKU", "TWLO"
]
TICKERS = ['TSLA']

# Data config
DATA_TIMESPAN = "minute"
DATA_START_DATE = '2000-01-01'
DATA_END_DATE = '2026-03-04'

# Preprocessing config
CUTOFF_TIMESTAMP = '2019-12-31 15:59:00-4:00' 

# Trading env config
WINDOW_SIZE = 60
INITIAL_BALANCE = 10000.0       # Initial trading balance
MAX_TRADING_UNITS = 10          
# Regularly check https://alpaca.markets/support/regulatory-fees for updated fees
SEC_FEE = 0                 # per SEC_FEE_PRINCIPAL of principal (sells only) - this fee is rounded up to the nearest penny
SEC_FEE_PRINCIPAL = 1000000
TAF_FEE = 0.000166          # per share (sells only) — this fee is applied on a per-trade basis, rounded up to the nearest penny, 
TAF_FEE_CAP = 8.30          # and capped at $8.30
CAT_FEE = 0.0000265         # charged per trade
SPREAD = 0.02               # currently for minute data 0.05 or 0.10 for daily data
SLIPPAGE = 0.0005           # currently for minute data 0.001 for daily data
MULTIDAY_MINUTE_EPISODE_DAYS = 5

# Model hyperparameters config
HIDDEN_DIM = 256            # Hidden dim size
LEARNING_RATE_ACTOR = 3e-4  # Actor NN learning rate
LEARNING_RATE_CRITIC = 3e-4 # Crtic NN learning rate
LEARNING_RATE_ALPHA = 3e-4  # Optimizer learning rate
ALPHA_INIT = 0.2            # Entropy temperature (controls how random the policy is)
GAMMA = 0.99                # Discount factor increase for longer episodes (higher = cares more about long term rewards)
GAMMA_MULTIDAY_MINUTE = 0.995 # 0.999
TAU = 0.005                 # Controls how soft the target network is updated
REPLAY_BUFFER_SIZE = 1000000 # Replay buffer's max size (increase/decrease based on ram size)
TARGET_UPDATE_INTERVAL = 1
SEED = 42

# Training config
BATCH_SIZE = 256
BATCH_SIZE_MULTIDAY_MINUTE = 256 # 512
NUM_EPISODES = 1000
VALID_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50

# Evaluation config
ANNUAL_RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backtest config
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-01-01" 