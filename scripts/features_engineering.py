import pandas as pd
import numpy as np
from loguru import logger

# Import the new decoupled class
from indicators import TechnicalIndicators


class FeatureEngineer:
    """
    Handles data loading, technical indicator generation, target definition, 
    and splitting for S&P 500 historical data while strictly preventing look-ahead bias.
    """

    def __init__(self, index_data_path: str, constituents_data_path: str) -> None:
        """
        Initializes the FeatureEngineer with file paths for the required datasets.

        Args:
            index_data_path (str): The file path to the SP500 index OHLC data (HistoricalData.csv).
            constituents_data_path (str): The file path to the constituents OHLCV data (all_stocks_5yr.csv).

        Returns:
            None: Constructor does not return any value.
        """
        self.index_path: str = index_data_path
        self.constituents_path: str = constituents_data_path
        self.index_df: pd.DataFrame = pd.DataFrame()
        self.constituents_df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> None:
        """
        Loads the raw CSV files into Pandas DataFrames, standardizes column names, 
        cleans string pricing, and formats the date columns.
        """
        logger.info("Initiating data load sequence...")
        
        try:
            # 1. Load Constituents
            self.constituents_df = pd.read_csv(self.constituents_path)
            self.constituents_df['date'] = pd.to_datetime(self.constituents_df['date'], format='%Y-%m-%d')
            self.constituents_df = self.constituents_df.sort_values(by=['Name', 'date']).reset_index(drop=True)
            logger.info(f"Constituents data loaded successfully. Shape: {self.constituents_df.shape}")

            # 2. Load Benchmark (S&P 500)
            self.index_df = pd.read_csv(self.index_path)
            
            # --- ARCHITECTURAL FIX: Clean the Benchmark Data Here ---
            # Dynamically find the close column (handles " Close/Last", "Close", etc.) and standardize to 'close'
            for col in self.index_df.columns:
                if 'close' in col.lower():
                    self.index_df.rename(columns={col: 'close'}, inplace=True)
                    break
                    
            # If the price loaded as a string (e.g., "$2050.50"), convert it to a clean float
            if 'close' in self.index_df.columns and self.index_df['close'].dtype == 'object':
                self.index_df['close'] = self.index_df['close'].astype(str).str.replace('$', '', regex=False)
                self.index_df['close'] = self.index_df['close'].str.replace(',', '', regex=False).astype(float)
            # --------------------------------------------------------

            self.index_df['Date'] = pd.to_datetime(self.index_df['Date'], format='%m/%d/%y')
            self.index_df = self.index_df.sort_values(by=['Date']).reset_index(drop=True)
            logger.info(f"S&P 500 Index benchmark data loaded successfully. Shape: {self.index_df.shape}")

        except FileNotFoundError as e:
            logger.error(f"Failed to locate data files. Ensure they are in the correct directory: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected column during data load: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}")
            raise

    def define_target(self) -> None:
        """
        Constructs the predictive target: sign(return(D+1, D+2)) while preventing cross-asset contamination.

        Args:
            None

        Returns:
            None: Modifies the `constituents_df` in-place by adding 'target_return' and 'target' columns.
        """
        logger.info("Computing target returns strictly avoiding look-ahead bias...")
        
        try:
            grouped_close = self.constituents_df.groupby('Name')['close']
            
            self.constituents_df['target_return'] = grouped_close.pct_change().shift(-2)
            self.constituents_df['target'] = np.sign(self.constituents_df['target_return'])
            
            initial_len = len(self.constituents_df)
            self.constituents_df = self.constituents_df.dropna(subset=['target']).reset_index(drop=True)
            
            logger.info(f"Target defined successfully. Dropped {initial_len - len(self.constituents_df)} boundary rows with NaN targets.")
            
        except KeyError as e:
            logger.error(f"Missing column required for target calculation: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to compute target returns: {e}")
            raise

    def apply_technical_indicators(self) -> None:
        """
        Delegates technical indicator generation to the TechnicalIndicators class.

        Args:
            None

        Returns:
            None: Modifies the `constituents_df` in-place.
        """
        logger.info("Initializing technical indicator computation engine...")
        try:
            indicator_engine = TechnicalIndicators(group_column='Name', price_column='close')
            self.constituents_df = indicator_engine.apply_indicators(self.constituents_df)
            logger.info(f"Indicators successfully integrated. New DataFrame shape: {self.constituents_df.shape}")
        except Exception as e:
            logger.error(f"Failed to integrate technical indicators: {e}")
            raise

    def split_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and testing sets based on the year 2017.

        Args:
            None

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the (train_df, test_df).
        """
        logger.info("Splitting data into Train (< 2017) and Test (>= 2017) sets...")
        
        try:
            train_df = self.constituents_df[self.constituents_df['date'].dt.year < 2017].copy()
            test_df = self.constituents_df[self.constituents_df['date'].dt.year >= 2017].copy()
            
            assert train_df['date'].max() < test_df['date'].min(), "CRITICAL: Train and Test sets overlap chronologically!"
            
            logger.info(f"Split complete. Train set size: {len(train_df)}, Test set size: {len(test_df)}.")
            return train_df, test_df
            
        except AssertionError as e:
            logger.error(f"Data Leakage Detected during split: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise

    def log_column_types(self) -> None:
        """
        Logs the data types of all columns in the DataFrame to verify pipeline integrity.

        Args:
            None

        Returns:
            None: Outputs data types to the logger.
        """
        logger.info("Verifying final data column types...")
        try:
            types_series = self.constituents_df.dtypes
            types_str = "\n".join([f"  - {col}: {dtype}" for col, dtype in types_series.items()])
            logger.info(f"Constituents DataFrame Types:\n{types_str}")
        except Exception as e:
            logger.error(f"Failed to log column types: {e}")
            raise


if __name__ == "__main__":
    try:
        # NOTE: Ensure you are running this from the root directory so Python finds `scripts.indicators`
        engineer = FeatureEngineer(
            index_data_path="data/HistoricalPrices.csv", 
            constituents_data_path="data/all_stocks_5yr.csv"
        )
        engineer.load_data()
        engineer.define_target()
        
        # New Step: Generate Technical Indicators via the decoupled class
        engineer.apply_technical_indicators()
        
        train_data, test_data = engineer.split_train_test()
        engineer.log_column_types()
        
        logger.success("Feature Engineering Phase (Data Prep, Target, Indicators) completed successfully.")
        
    except Exception as main_e:
        logger.critical(f"Pipeline execution failed: {main_e}")