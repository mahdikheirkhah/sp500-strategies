import pandas as pd
from loguru import logger
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD


class TechnicalIndicators:
    """
    Computes financial technical indicators (Bollinger Bands, RSI, MACD) 
    while strictly preventing cross-asset data leakage by grouping calculations.
    """

    def __init__(self, group_column: str = 'Name', price_column: str = 'close') -> None:
        """
        Initializes the TechnicalIndicators class with the necessary column mappings.

        Args:
            group_column (str): The column name representing the asset ticker.
            price_column (str): The column name representing the asset's closing price.

        Returns:
            None: Constructor does not return any value.
        """
        self.group_col: str = group_column
        self.price_col: str = price_column

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Bollinger Bands, RSI, and MACD grouped strictly by the asset ticker.

        Args:
            df (pd.DataFrame): The raw input DataFrame containing price data.

        Returns:
            pd.DataFrame: A new DataFrame with the appended technical indicator columns.
        """
        logger.info("Applying technical indicators: Bollinger Bands, RSI, MACD...")
        
        try:
            # Temporary lists to hold the calculated series
            bb_high_list, bb_low_list = [], []
            rsi_list = []
            macd_list, macd_signal_list = [], []

            grouped = df.groupby(self.group_col)

            for name, group in grouped:
                prices = group[self.price_col]
                
                # 1. Bollinger Bands (20-day window, 2 standard deviations)
                bb = BollingerBands(close=prices, window=20, window_dev=2)
                bb_high_list.append(bb.bollinger_hband())
                bb_low_list.append(bb.bollinger_lband())
                
                # 2. RSI (14-day window)
                rsi_ind = RSIIndicator(close=prices, window=14)
                rsi_list.append(rsi_ind.rsi())
                
                # 3. MACD (Fast=12, Slow=26, Signal=9)
                macd_ind = MACD(close=prices, window_slow=26, window_fast=12, window_sign=9)
                macd_list.append(macd_ind.macd())
                macd_signal_list.append(macd_ind.macd_signal())

            # Copy DataFrame to avoid Pandas SettingWithCopy warnings
            result_df = df.copy()
            
            logger.info("Concatenating calculated indicators to the main DataFrame...")
            result_df['bb_high'] = pd.concat(bb_high_list)
            result_df['bb_low'] = pd.concat(bb_low_list)
            result_df['rsi'] = pd.concat(rsi_list)
            result_df['macd'] = pd.concat(macd_list)
            result_df['macd_signal'] = pd.concat(macd_signal_list)

            # Drop NaN values caused by indicator lookback periods (e.g., first 26 days for MACD)
            initial_len = len(result_df)
            result_df = result_df.dropna().reset_index(drop=True)
            logger.info(f"Dropped {initial_len - len(result_df)} warmup rows containing NaN indicator values.")

            return result_df

        except KeyError as e:
            logger.error(f"Missing required column for indicator calculation: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during indicator calculation: {e}")
            raise