import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SEED
)
from src.v4.environment.environment import Environment
from src.v4.model.agent import Agent
from src.utils.logger import Logger
from src.utils.utils import create_directory, load_stock_data


class Backtester:
    def __init__(
        self,
        agent: Agent,
        test_env: Environment,
        logger: Optional[Logger] = None,
    ):
        self.agent = agent
        self.env = test_env
        self.logger = logger

        self.episode_results = []

    def run_backtest(self) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("Backtesting...")

        self.agent.actor.eval()

        # Run sequentially through all possible episode start points
        # for i, start_idx in enumerate(self.env.market_open_idx):
        for i, start_idx in enumerate(self.env.market_open_idx[::5]):
            self.env.current_step = start_idx
            state = self.env.reset()
            done = False
            episode_reward = 0
            actions = []

            while not done:
                action = self.agent.select_action(state, validate=True)
                action_value = action[0] if isinstance(action, np.ndarray) else action
                action_value = self.env.mask_action(action_value)
                next_state, reward, done, info = self.env.step(action_value)
                state = next_state
                episode_reward += reward
                actions.append(action_value)

            episode_start_price = self.env.prices[self.env.episode_start]
            episode_end_price = self.env.prices[self.env.episode_end]
            price_change_pct = (episode_end_price - episode_start_price) / episode_start_price

            self.episode_results.append({
                "episode": i,
                "start_date": self.env.timestamps[self.env.episode_start],
                "end_date": self.env.timestamps[self.env.episode_end],
                "start_price": episode_start_price,
                "end_price": episode_end_price,
                "price_change_pct": price_change_pct * 100,
                "balance": info["balance"],
                "portfolio_value": info["portfolio_value"],
                "net_return": info["net_return"],
                "net_return_pct": info["net_return_pct"] * 100,
                "gross_return_pct": info["gross_return_pct"] * 100,
                "fee_impact": (info["gross_return_pct"] - info["net_return_pct"]) * 100,
                "reward": episode_reward,
                "trade_count": info["trade_execution_count"],
                "turnover_ratio": info["turnover_ratio"] * 100,
                "avg_hold_time": info["avg_hold_time"],
                "sharpe_ratio": info["sharpe_ratio"],
                "max_drawdown": info["max_drawdown"] * 100,
                "win_rate": info["win_rate"] * 100,
                "completed_trades": info["completed_trades"],
            })

            if self.logger:
                r = self.episode_results[-1]
                beat_bh = "BEAT B&H" if r["net_return_pct"] > r["price_change_pct"] else ""
                self.logger.info(
                    f"\nBacktest EP: {i + 1}/{len(self.env.market_open_idx)}"
                    f"\n  {r['start_date']} → {r['end_date']}"
                    f"\n  Price: ${r['start_price']:.2f} → ${r['end_price']:.2f} ({r['price_change_pct']:+.2f}%)"
                    f"\n  Net Return: {r['net_return_pct']:+.2f}% {beat_bh}"
                    f"\n  Trades: {r['trade_count']} | Win Rate: {r['win_rate']:.1f}%"
                    f"\n  Sharpe: {r['sharpe_ratio']:.4f} | Drawdown: {r['max_drawdown']:.2f}%"
                )

        self.results = self._aggregate_results()
        self._log_summary()
        return self.results

    def _aggregate_results(self) -> Dict[str, Any]:
        returns = np.array([r["net_return_pct"] for r in self.episode_results])
        price_changes = np.array([r["price_change_pct"] for r in self.episode_results])
        sharpes = np.array([r["sharpe_ratio"] for r in self.episode_results])
        drawdowns = np.array([r["max_drawdown"] for r in self.episode_results])
        win_rates = np.array([r["win_rate"] for r in self.episode_results])
        trade_counts = np.array([r["trade_count"] for r in self.episode_results])
        rewards = np.array([r["reward"] for r in self.episode_results])

        # Annualized Sharpe from per-episode returns
        # Each episode is 5 trading days, so ~50 episodes per year
        episodes_per_year = 252 / 5
        mean_ep_return = returns.mean() / 100  # back to decimal
        std_ep_return = returns.std() / 100
        annualized_sharpe = (
            (mean_ep_return / (std_ep_return + 1e-9)) * np.sqrt(episodes_per_year)
        )

        # Beat buy-and-hold rate
        beat_bh = np.sum(returns > price_changes) / len(returns)

        # Cumulative return (compounded)
        cumulative_return = np.prod(1 + returns / 100) - 1

        # Max drawdown across episodes (on cumulative equity)
        equity_curve = np.cumprod(1 + returns / 100)
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdown_curve = (equity_curve - rolling_max) / rolling_max
        max_cumulative_drawdown = drawdown_curve.min() * 100

        return {
            "episodes": self.episode_results,
            "returns": returns,
            "price_changes": price_changes,
            "metrics": {
                "num_episodes": len(returns),
                "mean_return": returns.mean(),
                "median_return": np.median(returns),
                "std_return": returns.std(),
                "max_return": returns.max(),
                "min_return": returns.min(),
                "positive_pct": np.mean(returns > 0) * 100,
                "annualized_sharpe": annualized_sharpe,
                "mean_episode_sharpe": sharpes.mean(),
                "max_cumulative_drawdown": max_cumulative_drawdown,
                "mean_episode_drawdown": drawdowns.mean(),
                "mean_win_rate": win_rates.mean(),
                "mean_trade_count": trade_counts.mean(),
                "cumulative_return": cumulative_return * 100,
                "beat_buy_hold_pct": beat_bh * 100,
                "mean_bh_return": price_changes.mean(),
                "mean_reward": rewards.mean(),
            },
        }

    def _log_summary(self):
        if not self.logger:
            return

        m = self.results["metrics"]
        self.logger.info(
            f"\n{'='*60}"
            f"\nBACKTEST SUMMARY ({m['num_episodes']} episodes)"
            f"\n{'='*60}"
            f"\n  Cumulative Return:     {m['cumulative_return']:+.2f}%"
            f"\n  Mean Episode Return:   {m['mean_return']:+.2f}%"
            f"\n  Median Episode Return: {m['median_return']:+.2f}%"
            f"\n  Std Episode Return:    {m['std_return']:.2f}%"
            f"\n  Best / Worst:          {m['max_return']:+.2f}% / {m['min_return']:+.2f}%"
            f"\n  Positive Episodes:     {m['positive_pct']:.1f}%"
            f"\n{'─'*60}"
            f"\n  Annualized Sharpe:     {m['annualized_sharpe']:.2f}"
            f"\n  Mean Episode Sharpe:   {m['mean_episode_sharpe']:.4f}"
            f"\n  Max Cumul. Drawdown:   {m['max_cumulative_drawdown']:.2f}%"
            f"\n  Mean Episode Drawdown: {m['mean_episode_drawdown']:.2f}%"
            f"\n  Mean Win Rate:         {m['mean_win_rate']:.1f}%"
            f"\n  Mean Trade Count:      {m['mean_trade_count']:.0f}"
            f"\n{'─'*60}"
            f"\n  Beat Buy & Hold:       {m['beat_buy_hold_pct']:.1f}%"
            f"\n  Mean B&H Return:       {m['mean_bh_return']:+.2f}%"
            f"\n  Mean Model Return:     {m['mean_return']:+.2f}%"
            f"\n  Mean Reward:           {m['mean_reward']:.2f}"
            f"\n{'='*60}"
        )

    def plot_results(self, save_dir: Optional[str] = None):
        if not self.episode_results:
            return

        save_dir = Path(save_dir) if save_dir else Path(RESULTS_DIR) / "v4" / "backtest"
        create_directory(save_dir)

        returns = self.results["returns"]
        price_changes = self.results["price_changes"]
        m = self.results["metrics"]

        # 1. Returns vs Buy & Hold
        fig, ax = plt.subplots(figsize=(12, 6))
        episodes = range(len(returns))
        ax.bar(episodes, returns, alpha=0.6, color=["green" if r > 0 else "red" for r in returns], label="Model Return")
        ax.plot(episodes, price_changes, color="gray", linewidth=1, marker=".", markersize=3, label="Buy & Hold", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"Backtest Returns vs Buy & Hold ({m['num_episodes']} episodes)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        stats_text = (
            f"Model Mean: {m['mean_return']:+.2f}%\n"
            f"B&H Mean: {m['mean_bh_return']:+.2f}%\n"
            f"Beat B&H: {m['beat_buy_hold_pct']:.1f}%\n"
            f"Ann. Sharpe: {m['annualized_sharpe']:.2f}\n"
            f"Cumul. Return: {m['cumulative_return']:+.2f}%"
        )
        fig.text(0.82, 0.85, stats_text, fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        fig.tight_layout(rect=[0, 0, 0.80, 1])
        plt.savefig(save_dir / "returns_vs_bh.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Cumulative equity curve
        equity = np.cumprod(1 + returns / 100)
        bh_equity = np.cumprod(1 + price_changes / 100)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity, label="Model", color="blue", linewidth=1.5)
        ax.plot(bh_equity, label="Buy & Hold", color="gray", linewidth=1.5, linestyle="--")
        ax.axhline(y=1.0, color="black", linewidth=0.5)
        ax.set_title("Cumulative Equity Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Equity (1.0 = initial)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(save_dir / "equity_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Drawdown curve
        rolling_max = np.maximum.accumulate(equity)
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.4, color="red")
        ax.set_title(f"Cumulative Drawdown (Max: {m['max_cumulative_drawdown']:.2f}%)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        plt.savefig(save_dir / "drawdown.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Win rate and trade count per episode
        win_rates = np.array([r["win_rate"] for r in self.episode_results])
        trade_counts = np.array([r["trade_count"] for r in self.episode_results])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.bar(episodes, win_rates, alpha=0.6, color="blue")
        ax1.axhline(y=50, color="black", linewidth=0.5, linestyle="--")
        ax1.set_title(f"Win Rate per Episode (Mean: {m['mean_win_rate']:.1f}%)")
        ax1.set_ylabel("Win Rate (%)")
        ax1.grid(True, alpha=0.3)

        ax2.bar(episodes, trade_counts, alpha=0.6, color="purple")
        ax2.set_title(f"Trade Count per Episode (Mean: {m['mean_trade_count']:.0f})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Trade Count")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "winrate_trades.png", dpi=300, bbox_inches="tight")
        plt.close()

        if self.logger:
            self.logger.info(f"Backtest plots saved to {save_dir}")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    logger = Logger()

    ticker = "TSLA"
    test_data_dir = f"{DATA_DIR}/preprocessed/v4/unified_latent/unified_latent_test_v2.parquet"
    test_data = load_stock_data(test_data_dir)

    def prepare_df(df: pl.DataFrame) -> pl.DataFrame:
        ticker_close = f"{ticker}_close"
        cols_to_keep = [
            c for c in df.columns
            if not (c.endswith("_close") and c not in ("minutes_to_close", ticker_close))
            and not c.startswith("pca_")
        ]
        return (
            df
            .select(cols_to_keep)
            .rename({ticker_close: "close"})
            .with_row_index("index")
        )

    test_data = prepare_df(test_data)
    test_env = Environment(data=test_data)

    action_dim = test_env.action_space.shape[0]
    portfolio_dim = test_env.PORTFOLIO_STATE_DIM
    temporal_dim = test_env.n_temporal

    agent = Agent(
        market_data=test_env.data,
        action_dim=action_dim,
        input_shape=(test_env.window_size, test_env.feature_dim),
        portfolio_state_len=portfolio_dim,
        temporal_state_len=temporal_dim,
        n_temporal=temporal_dim,
        target_entropy=-0.5,
    )

    model_path = f"{MODELS_DIR}/v4/checkpoint_ep400_sac_model_20260404_111847"
    agent.load_model(model_path)

    backtester = Backtester(
        agent=agent,
        test_env=test_env,
        logger=logger,
    )

    backtester.run_backtest()
    backtester.plot_results()


if __name__ == "__main__":
    main()