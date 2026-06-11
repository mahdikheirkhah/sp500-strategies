import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

# Import our previously built data pipeline
from features_engineering import FeatureEngineer


class CrossValidator:
    """
    Implements strict time-series cross-validation to prevent data leakage
    and evaluate model performance across chronological market regimes.
    """

    def __init__(self, n_splits: int = 10) -> None:
        """
        Initializes the CrossValidator.

        Args:
            n_splits (int): The number of cross-validation folds. Must be >= 10.

        Returns:
            None
        """
        self.n_splits: int = n_splits
        self.unique_dates: np.ndarray = np.array([])
        self.date_folds: list[tuple[np.ndarray, np.ndarray]] = []

    def generate_splits(self, df: pd.DataFrame) -> None:
        """
        Generates Time Series splits based strictly on unique dates to ensure
        assets from the same day are never split between train and validation sets.

        Args:
            df (pd.DataFrame): The training dataset.

        Returns:
            None: Modifies the `date_folds` instance variable in-place.
        """
        logger.info(f"Generating {self.n_splits} Time-Series folds grouped by date...")
        try:
            # 1. Extract and sort purely unique dates to prevent same-day leakage
            self.unique_dates = np.sort(df["date"].unique())
            total_days = len(self.unique_dates)

            # 2. Ensure initial train size > 2 years (approx 504 trading days)
            # We calculate the test_size so the leftover starting train set is exactly 504 days
            min_train_days = 504
            test_size = (total_days - min_train_days) // self.n_splits

            logger.info(
                f"Total trading days: {total_days}. Calculated test size per fold: {test_size} days."
            )

            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)

            # 3. Generate the splits based on the unique date array indices
            for train_idx, val_idx in tscv.split(self.unique_dates):
                train_dates = self.unique_dates[train_idx]
                val_dates = self.unique_dates[val_idx]
                self.date_folds.append((train_dates, val_dates))

            logger.info(
                f"Successfully generated {len(self.date_folds)} folds. First fold train size: {len(self.date_folds[0][0])} days (> 2 years)."
            )

        except Exception as e:
            logger.error(f"Failed to generate Time Series splits: {e}")
            raise

    def plot_cv(self, save_dir: str = "results/cross-validation") -> None:
        """
        Generates and saves a visualization of the cross-validation strategy.

        Args:
            save_dir (str): The directory path to save the generated plot.

        Returns:
            None: Saves the plot to disk.
        """
        logger.info(f"Plotting cross-validation strategy to {save_dir}...")
        try:
            # Ensure the output directory exists
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "Time_series_split.png")

            fig, ax = plt.subplots(figsize=(12, 6))

            # We plot the indices of the dates for visual simplicity
            for i, (train_dates, val_dates) in enumerate(self.date_folds):
                # Map the actual dates back to their original 0-N integer index for the X-axis
                train_idx = np.where(np.isin(self.unique_dates, train_dates))[0]
                val_idx = np.where(np.isin(self.unique_dates, val_dates))[0]

                # Plot Train as blue, Validation as orange
                ax.scatter(
                    train_idx, [i + 1] * len(train_idx), c="blue", marker="_", lw=8
                )
                ax.scatter(
                    val_idx, [i + 1] * len(val_idx), c="orange", marker="_", lw=8
                )

            ax.set_title("Standard Time Series Split (Grouping by Date)")
            ax.set_xlabel("Trading Days (Chronological Index)")
            ax.set_ylabel("CV Fold")
            ax.set_yticks(range(1, self.n_splits + 1))

            # Custom legend
            from matplotlib.lines import Line2D

            custom_lines = [
                Line2D([0], [0], color="blue", lw=4),
                Line2D([0], [0], color="orange", lw=4),
            ]
            ax.legend(custom_lines, ["Train Set", "Validation Set"], loc="upper left")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logger.success(f"Plot successfully saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to plot cross-validation: {e}")
            raise


if __name__ == "__main__":
    try:
        # Step 1: Run the feature engineering pipeline to get the strictly formatted data
        engineer = FeatureEngineer(
            index_data_path="data/HistoricalPrices.csv",
            constituents_data_path="data/all_stocks_5yr.csv",
        )
        engineer.load_data()
        engineer.define_target()
        engineer.apply_technical_indicators()
        train_df, test_df = engineer.split_train_test()

        # Step 2: Pass the Train DataFrame to our new CrossValidator
        cv_engine = CrossValidator(n_splits=10)
        cv_engine.generate_splits(train_df)
        cv_engine.plot_cv()

    except Exception as main_e:
        logger.critical(f"Model Selection pipeline failed: {main_e}")
