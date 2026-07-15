import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from src.config.config import DATA_DIR, MODELS_DIR, INITIAL_BALANCE, SEC_FEE, SEC_FEE_PRINCIPAL, TAF_FEE, TAF_FEE_CAP, CAT_FEE, SPREAD, SLIPPAGE, WINDOW_SIZE
from src.v1.environment.environment import Environment
from src.v1.model.agent import Agent
from src.utils.logger import Logger
from src.utils.utils import create_directory, save_to_csv, load_stock_data

class Backtester:
    def __init__(
        self,
        agent: Agent,
        test_data: pd.DataFrame,
        window_size: int=WINDOW_SIZE,
        logger: Optional[Logger]=None,
        initial_balance: float=INITIAL_BALANCE,
        sec_fee: float=SEC_FEE,
        sec_fee_principal: float=SEC_FEE_PRINCIPAL,
        taf_fee: float=TAF_FEE,
        taf_fee_cap: float=TAF_FEE_CAP,
        cat_fee: float=CAT_FEE,
        spread: float=SPREAD,
        slippage: float=SLIPPAGE,
    ):
        self.agent = agent
        self.test_data = test_data
        self.window_size = window_size
        self.logger = logger
        self.initial_balance = initial_balance
        self.sec_fee = sec_fee
        self.sec_fee_principal = sec_fee_principal
        self.taf_fee = taf_fee
        self.taf_fee_cap = taf_fee_cap
        self.cat_fee = cat_fee
        self.spread = spread
        self.slippage = slippage
        
        self.env = Environment(
            data=self.test_data,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            logger=self.logger
        )
        
        self.results = {
            "balances": [],
            "returns": [],
            "return_pcts": [],
            "rewards": [],
        }
        
    def run_backtest(self) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("Backtesting...")
        
        for i in range(len(self.env.market_open_idx) - 1):
            state = self.env.reset()
            done = False
            
            while not done:  
                action = self.agent.select_action(state, validate=True)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                
            self.results["balances"].append(info["balance"])
            self.results["returns"].append(info["net_return"])
            self.results["return_pcts"].append(info["net_return_pct"])
            self.results["rewards"].append(reward)
                
            
        self.results["balances"] = np.array(self.results["balances"])
        self.results["returns"] = np.array(self.results["returns"])
        self.results["return_pcts"] = np.array(self.results["return_pcts"])
        self.results["rewards"] = np.array(self.results["rewards"])
        
        self.calculate_metrics()
        
        if self.logger:
            self.logger.info(f"\nBacktest Complete\
                               \nMean Final Balance: {self.results['balances'].mean():.2f}\
                               \nMean Return: ${self.results['returns'].mean():.2f}\
                               \nMean Return PCT: {self.results['return_pcts'].mean():.2%}\
                               \nMean Reward: {self.results['rewards'].mean():.2f}\
                               \nBest Return PCT: {self.results['return_pcts'].max():.2%}\
                               \nWorst Return PCT: {self.results['return_pcts'].min():.2%}\
                               \nSharpe Ratio {self.results['metrics']['sharpe_ratio']}\
                               \nMax Drawdown {self.results['metrics']['max_drawdown']:.2%}\
                               \nWin Rate {self.results['metrics']['win_rate']:.2%}")
        
        return self.results
        
    def calculate_metrics(self) -> Dict[str, float]:
        daily_returns = pd.Series(self.results["return_pcts"])

        if daily_returns.empty:
            return {}

        # Mean return per day
        mean_return = daily_returns.mean()

        # Volatility per day
        volatility = daily_returns.std(ddof=1)

        # Sharpe ratio (per-day)
        sharpe_ratio = (
            mean_return / volatility
            if volatility > 0 else 0.0
        )

        # Max drawdown (on compounded equity)
        # equity_curve = (1 + daily_returns).cumprod()
        # rolling_max = equity_curve.cummax()
        # drawdown = (equity_curve - rolling_max) / rolling_max
        # max_drawdown = drawdown.min()
        max_drawdown = daily_returns.min()

        # Win rate
        win_rate = (daily_returns > 0).mean()

        metrics = {
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }

        self.results["metrics"] = metrics
        return metrics

    
    def save_results(self, filepath: str) -> None:
        save_data = {
            "initial_balance": self.initial_balance,
            "mean_balance": f"{self.results['balances'].mean():.2f}",
            "mean_return": f"{self.results['returns'].mean():.2f}",
            "mean_return_pct": f"{self.results['return_pcts'].mean() * 100:.2f}",
            "metrics": self.results["metrics"],
        }
        
        df = pd.DataFrame(save_data)
        create_directory(filepath)
        save_to_csv(df, filepath)
            
        if self.logger:
            self.logger.info(f"Backtest results saved to {filepath}")
            
def main():
    logger = Logger()
    
    ticker = 'TSLA'
    test_data_dir = f'{DATA_DIR}/preprocessed/v1/{ticker}/{ticker}_test.csv'
    test_data = load_stock_data(test_data_dir)
    
    action_dim = 1
    env = Environment(data=test_data)
    action_dim = env.action_space.shape[0]
    portfolio_dim = env.observation_space['portfolio_state'].shape[0]
    agent = Agent(
        action_dim=action_dim,
        input_shape=(env.window_size, env.feature_dim),
        portfolio_state_len=portfolio_dim,
        logger=logger
    )
    model_path = f"{MODELS_DIR}/v1/final_sac_model_20260219_223246"
    agent.load_model(model_path)
    
    backtester = Backtester(
        agent=agent,
        test_data=test_data,
        logger=logger
    )
    
    backtester.run_backtest()

if __name__ == '__main__':
    main()