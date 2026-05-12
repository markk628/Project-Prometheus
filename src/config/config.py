import os
from dotenv import load_dotenv
from pathlib import Path

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
PREPROCESSING_LOGS_DIR = LOGS_DIR / "preprocessing"                                      # Preprocessing logs dir
MODELS_DIR = ROOT_DIR / "models"                                                   # Models dir
RESULTS_DIR = ROOT_DIR / "results"                                                 # Results dir

# Stock tickers config
#
# TODO(v5.1): audit regime ticker utilization + design new regime features.
#
# Post-Option B, regime tickers are not trained on or traded, and breadth
# is restricted to tradables only (to avoid double-counting XLK with its
# tech constituents, etc.). As a consequence, most of the tickers in the
# list below currently contribute ZERO signal to the unified parquet —
# feature engineering processes their per-ticker parquets, includes them
# in the stacked frame, and then Option B's final prune drops them out
# without any shared regime feature having referenced their columns.
#
# Current actual utilization:
#   SPY  -> drives rs_spy_* for every tradable (active, used)
#   VIXY -> half of vix_term_structure_* via log(VIXY/VIXM) (active, used)
#   VIXM -> other half of vix_term_structure_* (active, used)
#   SPY_close -> preserved as a benchmark reference column only
#
#   Everyone else (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC,
#   TLT, IEF, SHY, HYG, LQD, UUP, GLD, USO, EFA, EEM, EWJ, QQQ, IWM, MDY)
#   -> currently dead weight. Processed through feature engineering but
#   no feature references their closes. Dropped by the Option B prune.
#
# This wasn't the intent — Option B was about "don't train on these",
# not "contribute nothing". The fix is to design shared regime features
# that explicitly use these ETFs' closes, analogous to how VIX term
# structure uses VIXY/VIXM. Candidates:
#
#   Sector rotation     -> rolling rank/correlation across the 11 XL* ETFs
#   Yield curve         -> TLT/SHY or TLT/IEF ratio
#   Credit spread       -> HYG/LQD ratio (risk-on vs risk-off)
#   Dollar regime       -> UUP return/trend
#   Commodities regime  -> GLD, USO returns as regime signals
#   International       -> EFA, EEM, EWJ vs SPY relative strength
#   Size factor         -> IWM/SPY, MDY/SPY
#   Growth factor       -> QQQ/SPY
#
# Potential addition: crypto regime.
#   Adding e.g. BITO to this list alone does NOTHING by itself. Would need
#   a corresponding shared feature computation such as:
#     btc_spy_correlation_60d = rolling_corr(BITO_log_return_1, SPY_log_return_1, 60)
#   Caveats: (1) BITO launched Oct 2021 → ~17 of 21 training years will
#   have this feature as 0, higher zero-padding than XLC. (2) spot BTC
#   via CoinGecko/Kraken (back to ~2014) is a cleaner alternative but
#   requires data plumbing outside Polygon. (3) BTC-equity coupling is
#   non-stationary across sub-periods — validate on fold years containing
#   regime shifts (2021 euphoria, 2022 tightening, 2024 ETF adoption).
#
# Approach: bundle this into one coherent v5.1 design pass rather than
# adding features one at a time. Ship the current baseline first to see
# what the SPY+VIX regime signal alone produces.
REGIME_TICKERS = [
    # ===== Broad Market =====
    "SPY",          # S&P 500
    "QQQ",          # Nasdaq 100
    "IWM",          # Russell 2000 (small cap)
    "MDY",          # S&P 400 (mid cap)

    # ===== Sector ETFs (all 11 GICS sectors) =====
    "XLK",          # Technology
    "XLF",          # Financials
    "XLE",          # Energy
    "XLV",          # Healthcare
    "XLI",          # Industrials
    "XLY",          # Consumer Discretionary
    "XLP",          # Consumer Staples
    "XLU",          # Utilities
    "XLB",          # Materials
    "XLRE",         # Real Estate
    "XLC",          # Communication Services

    # ===== Fixed Income =====
    "TLT",          # 20+ year treasuries
    "IEF",          # 7-10 year treasuries
    "SHY",          # 1-3 year treasuries
    "HYG",          # High yield corporate
    "LQD",          # Investment grade corporate

    # ===== Volatility =====
    "VIXY",         # VIX short-term futures
    "VIXM",         # VIX mid-term futures

    # ===== Dollar =====
    "UUP",          # US Dollar Index

    # ===== Commodities =====
    "GLD",          # Gold
    "USO",          # Oil

    # ===== International =====
    "EFA",          # Developed markets ex-US
    "EEM",          # Emerging markets
    "EWJ",          # Japan (yen carry trade / BoJ divergence signal)
]

# Data config
DATA_TIMESPAN = "minute"
DATA_START_DATE = '2000-01-01'
DATA_END_DATE = '2026-03-04'

# Preprocessing config
#
# Warmup math: the longest causal dependency chain is a 60-day upstream
# window (e.g. ema_20_60_ratio, log_return_60) followed by a 252-day
# rolling z-score normalization. That's ~312 trading days of raw history
# needed before any such feature produces non-zero values — so the cutoff
# must be at least 312 trading days after the raw data start, or a
# meaningful chunk of post-cutoff rows still come out as zeros (which
# was happening with the previous 2004-09-11 cutoff: ~50 bars past cutoff
# still had zeros for ema_20_60_ratio / adx_20_60_ratio / log_return_60,
# because raw data starts 2003-09-10 and 312 trading days lands in late
# November 2004).
#
# 2004-12-13 (Monday) is ~324 trading days after 2003-09-10 — comfortably
# past the 60+252 warmup and lands on a clean week start. Costs ~3 months
# of training data vs. 2004-09-11, negligible against the 22-year window.
CUTOFF_TIMESTAMP = '2004-12-13 00:00:00+00:00'

# Trading env config
WINDOW_SIZE = 60
INITIAL_BALANCE = 10000.0       # Initial trading balance
MAX_TRADING_UNITS = 30          # Legacy (minute-level, ticker-specific). Kept for back-compat.
MAX_POSITION_FRACTION = 1.0     # Max target position as fraction of initial_balance.
                                # 1.0 = fully invested. Used by DailyEnvironment's
                                # dollar-exposure action space (ticker-agnostic).
# Regularly check https://alpaca.markets/support/regulatory-fees for updated fees
SEC_FEE = 20.60 / 1000000   # per SEC_FEE_PRINCIPAL of principal (sells only) - this fee is rounded up to the nearest penny
SEC_FEE_PRINCIPAL = 1000000
TAF_FEE = 0.000166          # per share (sells only) — this fee is applied on a per-trade basis, rounded up to the nearest penny, 
TAF_FEE_CAP = 8.30          # and capped at $8.30
CAT_FEE = 0.0000265         # charged per share
# Execution cost model (applied on every buy and sell inside
# DailyEnvironment._calculate_execution_price). Values are tuned for
# daily bars, where executions are assumed to happen at the close with
# typical daily-frequency slippage rather than at touch.
#
# Previous defaults (SPREAD=$0.02, SLIPPAGE=0.0005=5bps) were calibrated
# for minute-level execution on a liquid single ticker and materially
# underestimated costs for the 3,341-ticker daily universe. At $100 a
# share the old model charged ~6 bps per side; the new values charge
# ~12-15 bps, closer to realistic MOC/VWAP execution across mixed
# liquidity. If baseline results show cost-sensitivity problems, tune
# further — especially per-ticker-scaled spread for low-priced names.
SPREAD = 0.05               # $ per share absolute. ~$0.01 for mega-caps,
                            #   $0.05 mid-caps, up to $0.20 small-caps.
                            #   $0.05 is a universe-weighted midpoint.
SLIPPAGE = 0.001            # Fraction of price (10 bps). Captures
                            #   execution drift from daily close prints.
MULTIDAY_MINUTE_EPISODE_DAYS = 5

# Model hyperparameters config
HIDDEN_DIM = 128            # Hidden dim size
LEARNING_RATE_ACTOR = 3e-4  # Actor NN learning rate
LEARNING_RATE_CRITIC = 4.5e-4 # 3e-4 # Crtic NN learning rate
LEARNING_RATE_ALPHA = 1e-5  # Optimizer learning rate (decrease to increase exploration (meaning alpha will reach 0 slower))
                            # Run 4 change: 1e-4 → 1e-5. Run 3 showed alpha
                            # collapsing aggressively (final ~0.005) even at
                            # 1e-4 — better long-horizon regime features made
                            # the actor confident faster, accelerating entropy
                            # collapse. Combined with the UPDATE_RATIO bump
                            # below (1 → 4), keeping LR at 1e-4 would put alpha
                            # near zero by fold 4. 1e-5 holds exploration alive
                            # through later folds without re-introducing the
                            # over-exploration that the alpha mechanism is
                            # designed to prevent.
ALPHA_INIT = 0.2            # Entropy temperature (controls how random the policy is, increase to increase exploration) 
GAMMA = 0.99                # Discount factor increase for longer episodes (higher = cares more about long term rewards)
GAMMA_MULTIDAY_MINUTE = 0.995 # 0.999
TAU = 0.005                 # Controls how soft the target network is updated
REPLAY_BUFFER_SIZE = 1750000 # Replay buffer's max size (increase/decrease based on ram size)
TARGET_UPDATE_INTERVAL = 1
UPDATE_RATIO = 2            # Gradient updates per env step (UTD ratio).
                            # Run 4 change: 1 → 4. Run 3 loss curves showed
                            # the model still actively learning at fold 9
                            # (Q-value rising, critic loss declining) when
                            # training ended — i.e. the bottleneck is gradient
                            # updates per transition, not transitions
                            # themselves. Higher UTD lets the agent extract
                            # more from each environment step before the next
                            # one arrives. Cost: ~4× wall-clock training time.
                            # Note: target soft-update stays at 1 per env step
                            # (not per gradient step) — TAU is calibrated for
                            # the lower frequency.
SEED = 42

# Training config
BATCH_SIZE = 256
BATCH_SIZE_MULTIDAY_MINUTE = 512 
NUM_EPISODES = 200
VALID_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50

# Evaluation config
ANNUAL_RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252

# Backtest config
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-01-01"