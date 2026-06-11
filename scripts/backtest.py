import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Import our previously built data pipeline
from features_engineering import FeatureEngineer


class StrategyBacktester:
    """
    Executes a financial strategy based on ML probabilities, normalizes daily
    investments, and evaluates performance (PnL, Max Drawdown) against a benchmark.
    """

    def __init__(
        self, model_path: str, signal_path: str, save_dir: str = "results/backtest"
    ) -> None:
        self.model_path = model_path
        self.signal_path = signal_path
        self.save_dir = save_dir

        self.features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bb_high",
            "bb_low",
            "rsi",
            "macd",
            "macd_signal",
        ]

        # Strategy Thresholds
        self.long_threshold = 0.515
        self.short_threshold = 0.485

        # Risk Management Parameter
        self.stop_loss = -0.05

        os.makedirs(self.save_dir, exist_ok=True)

    def prepare_full_timeline(self, engineer: FeatureEngineer) -> pd.DataFrame:
        """
        Combines the OOF Train signal with newly generated predictions for the Test set.
        """
        logger.info("Generating predictions for the unseen 2017 Test Set...")
        try:
            # 1. Get the pre-split DataFrames
            train_df, test_df = engineer.split_train_test()

            # 2. Load the trained ultimate pipeline
            pipeline = joblib.load(self.model_path)

            # 3. Predict on Test Set
            # Find the index for Class '2' (Upward movement) to extract buy probability
            class_up_index = np.where(pipeline.classes_ == 2)[0][0]
            test_probs = pipeline.predict_proba(test_df[self.features])[
                :, class_up_index
            ]

            test_signal = test_df[["date", "Name", "target_return"]].copy()
            test_signal["signal_prob"] = test_probs
            test_signal["dataset"] = "Test"

            # 4. Load the OOF Train Signal and format it
            train_signal = pd.read_csv(self.signal_path)
            train_signal["date"] = pd.to_datetime(train_signal["date"])
            # Merge target_return back into train_signal from the original train_df
            train_signal = pd.merge(
                train_signal,
                train_df[["date", "Name", "target_return"]],
                on=["date", "Name"],
                how="left",
            )
            train_signal["dataset"] = "Train"

            # 5. Concatenate to form the full timeline
            full_signal = pd.concat([train_signal, test_signal], ignore_index=True)
            full_signal = full_signal.sort_values(by=["date", "Name"]).dropna(
                subset=["target_return"]
            )

            logger.success(
                f"Timeline prepared. Total trading days evaluated: {full_signal['date'].nunique()}"
            )
            return full_signal

        except Exception as e:
            logger.error(f"Failed to prepare full timeline: {e}")
            raise

    def execute_strategy(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts probabilities into discrete trades (Long/Short), applies a Stop-Loss,
        and calculates normalized daily portfolio returns.
        """
        logger.info(
            f"Executing Long/Short Strategy (Long > {self.long_threshold}, Short < {self.short_threshold})..."
        )
        try:
            # 1. Define Positions
            signal_df["position"] = 0
            signal_df.loc[
                signal_df["signal_prob"] > self.long_threshold, "position"
            ] = 1
            signal_df.loc[
                signal_df["signal_prob"] < self.short_threshold, "position"
            ] = -1

            # 2. Calculate individual raw trade return
            signal_df["trade_return"] = (
                signal_df["position"] * signal_df["target_return"]
            )

            # THE STOP-LOSS GUARDRAIL
            # If any single trade loses more than 5%, the broker liquidates it instantly.
            stop_loss_mask = signal_df["trade_return"] < self.stop_loss
            num_stops_hit = stop_loss_mask.sum()
            logger.warning(
                f"Risk Management: Stop-Loss (-5%) triggered on {num_stops_hit} individual trades."
            )

            # Cap the maximum loss at -5%
            signal_df.loc[stop_loss_mask, "trade_return"] = self.stop_loss

            # 3. Normalize daily investment: $1 spread equally among all active trades
            active_trades = signal_df[signal_df["position"] != 0]

            daily_portfolio = (
                active_trades.groupby(["date", "dataset"])["trade_return"]
                .mean()
                .reset_index()
            )
            daily_portfolio.rename(
                columns={"trade_return": "strategy_return"}, inplace=True
            )

            # Calculate Strategy Compounding Equity and PnL
            daily_portfolio["strategy_equity"] = (
                1 + daily_portfolio["strategy_return"]
            ).cumprod()
            daily_portfolio["strategy_cum_pnl"] = daily_portfolio["strategy_equity"] - 1

            return daily_portfolio

        except Exception as e:
            logger.error(f"Failed to execute strategy: {e}")
            raise

    def process_benchmark(
        self, engineer: FeatureEngineer, strategy_dates: pd.Series
    ) -> pd.DataFrame:
        """
        Calculates the S&P 500 cumulative compounding PnL over the exact same dates for a 1:1 comparison.
        """
        logger.info("Processing S&P 500 Benchmark returns...")
        try:
            benchmark = engineer.index_df.copy()
            benchmark.rename(columns={"Date": "date"}, inplace=True)

            # Align the target return logic (D+1 to D+2)
            benchmark["benchmark_return"] = benchmark["close"].pct_change().shift(-2)

            # Filter to only the dates where our strategy traded
            benchmark = benchmark[benchmark["date"].isin(strategy_dates)].copy()

            # Calculate Benchmark Compounding Equity and PnL
            benchmark["benchmark_equity"] = (
                1 + benchmark["benchmark_return"]
            ).cumprod()
            benchmark["benchmark_cum_pnl"] = benchmark["benchmark_equity"] - 1

            return benchmark[["date", "benchmark_cum_pnl", "benchmark_equity"]]

        except Exception as e:
            logger.error(f"Failed to process benchmark: {e}")
            raise

    def calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculates Maximum Drawdown from a $1.00 base Equity Curve."""
        running_max = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - running_max) / running_max
        return drawdown.min()

    def plot_and_evaluate(
        self, portfolio: pd.DataFrame, benchmark: pd.DataFrame
    ) -> None:
        """
        Calculates final financial metrics, saves raw results, generates a Markdown report,
        and plots the strategy vs. benchmark.
        """
        logger.info("Calculating final financial metrics and generating artifacts...")
        try:
            # Merge strategy and benchmark for plotting and saving
            merged = pd.merge(portfolio, benchmark, on="date", how="left").ffill()

            # Find the exact date the Train set ended and Test set began
            test_start_date = portfolio[portfolio["dataset"] == "Test"]["date"].min()

            # Final Metrics
            strat_final_pnl = merged["strategy_cum_pnl"].iloc[-1]
            bench_final_pnl = merged["benchmark_cum_pnl"].iloc[-1]

            # Calculate drawdown using the compounding equity curves
            strat_max_dd = self.calculate_max_drawdown(merged["strategy_equity"])
            bench_max_dd = self.calculate_max_drawdown(merged["benchmark_equity"])

            logger.success(f"--- BACKTEST RESULTS ---")
            logger.success(f"Strategy Final PnL: {strat_final_pnl:.2%}")
            logger.success(f"Benchmark Final PnL: {bench_final_pnl:.2%}")
            logger.success(f"Strategy Maximum Drawdown: {strat_max_dd:.2%}")
            logger.success(f"Benchmark Maximum Drawdown: {bench_max_dd:.2%}")

            # Save results.csv
            csv_path = os.path.join(self.save_dir, "results.csv")
            merged.to_csv(csv_path, index=False)
            logger.info(f"Raw daily performance data saved to {csv_path}")

            # Generate report.md
            report_path = os.path.join(self.save_dir, "report.md")
            with open(report_path, "w") as f:
                f.write(f"# Quantitative Strategy Backtest Report\n\n")
                f.write(f"## Performance Metrics\n")
                f.write(f"- **Strategy Final PnL:** {strat_final_pnl:.2%}\n")
                f.write(f"- **Benchmark (S&P 500) Final PnL:** {bench_final_pnl:.2%}\n")
                f.write(f"- **Strategy Maximum Drawdown:** {strat_max_dd:.2%}\n")
                f.write(f"- **Benchmark Maximum Drawdown:** {bench_max_dd:.2%}\n\n")
                f.write(f"## Execution Parameters\n")
                f.write(f"- **Long Threshold:** > {self.long_threshold}\n")
                f.write(f"- **Short Threshold:** < {self.short_threshold}\n")
                f.write(f"- **Risk Management (Stop-Loss):** {self.stop_loss:.0%}\n")
                f.write(f"- **Total Trading Days Evaluated:** {len(merged)}\n")
            logger.info(f"Execution summary report saved to {report_path}")

            # Rendering the Plot
            plt.figure(figsize=(14, 7))
            plt.plot(
                merged["date"],
                merged["strategy_cum_pnl"],
                label=f"ML Strategy",
                color="blue",
                lw=2,
            )
            plt.plot(
                merged["date"],
                merged["benchmark_cum_pnl"],
                label="S&P 500 Benchmark",
                color="gray",
                alpha=0.6,
                lw=2,
            )

            # Vertical Train/Test Split Line
            if pd.notnull(test_start_date):
                plt.axvline(
                    x=test_start_date,
                    color="red",
                    linestyle="--",
                    lw=2,
                    label="Train / Test Split",
                )
                plt.text(
                    test_start_date,
                    plt.ylim()[1] * 0.9,
                    "  Test Set (Unseen Future) ->",
                    color="red",
                )

            plt.title(
                "Algorithmic Strategy vs. S&P 500 Benchmark (Normalized $1/Day, Compounding)"
            )
            plt.xlabel("Date")
            plt.ylabel("Cumulative PnL")
            plt.legend(loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            plot_path = os.path.join(self.save_dir, "strategy.png")
            plt.savefig(plot_path)
            plt.close()
            logger.success(f"Strategy plot successfully saved to {plot_path}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


if __name__ == "__main__":
    try:
        # 1. Initialize data engine to get raw prices and targets
        engineer = FeatureEngineer(
            index_data_path="data/HistoricalPrices.csv",
            constituents_data_path="data/all_stocks_5yr.csv",
        )
        engineer.load_data()
        engineer.define_target()
        engineer.apply_technical_indicators()

        # 2. Initialize and run Backtester
        backtester = StrategyBacktester(
            model_path="results/selected-model/selected_model.pkl",
            signal_path="results/signal/signal.csv",
        )

        full_signal_df = backtester.prepare_full_timeline(engineer)
        portfolio_df = backtester.execute_strategy(full_signal_df)
        benchmark_df = backtester.process_benchmark(engineer, portfolio_df["date"])

        backtester.plot_and_evaluate(portfolio_df, benchmark_df)

    except Exception as main_e:
        logger.critical(f"Backtesting pipeline failed: {main_e}")
