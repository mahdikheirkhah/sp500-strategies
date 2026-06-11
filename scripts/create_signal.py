import os
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import clone

# Import our previously built data pipeline
from features_engineering import FeatureEngineer
from model_selection import CrossValidator


class SignalGenerator:
    """
    Generates an Out-of-Fold (OOF) trading signal to guarantee that the
    probabilities are entirely out-of-sample and untainted by future data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cv_folds: list[tuple[np.ndarray, np.ndarray]],
        model_path: str,
    ) -> None:
        """
        Initializes the SignalGenerator.
        """
        self.df = df.reset_index(drop=True)
        self.date_folds = cv_folds
        self.model_path = model_path

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
        self.target = "target"

        # We must apply the exact same target shift we used during training
        # -1 (Down) -> 0,  0 (Flat) -> 1,  1 (Up) -> 2
        self.df[self.target] = self.df[self.target] + 1

        self.oof_predictions: list[pd.DataFrame] = []

    def generate_oof_signal(self) -> None:
        """
        Loops through the expanding time-series window, trains a cloned (blank)
        model on the historical fold, and predicts strictly on the unseen validation fold.
        """
        logger.info(f"Loading winning pipeline architecture from {self.model_path}...")
        try:
            fitted_best_model = joblib.load(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model. Did you run gridsearch.py? Error: {e}")
            raise

        logger.info("Initiating Out-of-Fold (OOF) Expanding Window predictions...")

        try:
            for fold, (train_dates, val_dates) in enumerate(self.date_folds):
                # CRITICAL: We clone the model to wipe its memory.
                # This ensures it learns purely from this specific fold's history.
                fold_model = clone(fitted_best_model)

                # Filter data for this specific fold
                train_idx = self.df["date"].isin(train_dates)
                val_idx = self.df["date"].isin(val_dates)

                X_train = self.df.loc[train_idx, self.features]
                y_train = self.df.loc[train_idx, self.target]

                X_val = self.df.loc[val_idx, self.features]

                # 1. Train the blank model on the historical block
                fold_model.fit(X_train, y_train)

                # 2. Predict on the unseen validation block
                probs = fold_model.predict_proba(X_val)

                # 3. Extract the probability of the asset going UP (Target Class '2')
                # We use np.where to dynamically find the correct column index for class '2'
                # to prevent bugs if Scikit-Learn reorders the columns internally.
                class_up_index = np.where(fold_model.classes_ == 2)[0][0]
                buy_probs = probs[:, class_up_index]

                # 4. Store the results
                fold_preds = self.df.loc[val_idx, ["date", "Name"]].copy()
                fold_preds["signal_prob"] = buy_probs

                self.oof_predictions.append(fold_preds)
                logger.info(
                    f"Fold {fold+1} complete. Generated {len(fold_preds)} unbiased predictions."
                )

        except Exception as e:
            logger.error(f"Failed during OOF generation: {e}")
            raise

    def save_signal(self, save_dir: str = "results/signal") -> pd.DataFrame:
        """
        Stitches the OOF predictions together into a single continuous DataFrame,
        formats it with a MultiIndex, and saves it to disk.
        """
        logger.info("Stitching OOF predictions into a continuous historical signal...")
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Stack all the blocks vertically
            final_signal_df = pd.concat(self.oof_predictions, ignore_index=True)

            # Sort chronologically, then alphabetically by ticker
            final_signal_df = final_signal_df.sort_values(by=["date", "Name"])

            # Fulfill the project requirement: A double-indexed DataFrame (Date, Ticker)
            final_signal_df.set_index(["date", "Name"], inplace=True)

            save_path = os.path.join(save_dir, "signal.csv")
            final_signal_df.to_csv(save_path)

            logger.success(
                f"Pure OOF Signal successfully saved to {save_path}. Total Shape: {final_signal_df.shape}"
            )
            return final_signal_df

        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            raise


if __name__ == "__main__":
    try:
        # 1. Reproduce the exact dataset structure
        engineer = FeatureEngineer(
            index_data_path="data/HistoricalPrices.csv",
            constituents_data_path="data/all_stocks_5yr.csv",
        )
        engineer.load_data()
        engineer.define_target()
        engineer.apply_technical_indicators()
        train_df, _ = engineer.split_train_test()

        # 2. Reproduce the exact time-series folds
        cv_engine = CrossValidator(n_splits=10)
        cv_engine.generate_splits(train_df)

        # 3. Generate the Out-of-Fold signal
        generator = SignalGenerator(
            df=train_df,
            cv_folds=cv_engine.date_folds,
            model_path="results/selected-model/selected_model.pkl",
        )
        generator.generate_oof_signal()
        final_signal = generator.save_signal()

        # Display a quick preview to verify the MultiIndex format
        print("\n--- Signal DataFrame Preview ---")
        print(final_signal.head())

    except Exception as main_e:
        logger.critical(f"Signal Generation pipeline failed: {main_e}")
