# S&P 500 Strategies: Learning Objectives & Core Concepts

## 1. Market Data Biases & Properties

### Survivorship Bias: The Hidden Illusion
In the project instructions, it notes that the dataset assumes the S&P 500 composition remains static over 5 years. In reality, the S&P 500 is constantly changing. Companies that perform poorly, go bankrupt, or get acquired are removed, and new successful companies are added. 
* **The Concept:** Survivorship Bias occurs when a dataset only includes the companies that "survived" until the end of the time period.
* **The Danger:** If a model trains on this data, it never learns what a failing company looks like (because bankruptcies like Enron or Lehman Brothers were scrubbed from the dataset).
* **The Result:** The backtest will look incredibly profitable because it essentially "picks winning stocks" using future knowledge—it already knows they survived the next 5 years. 
* *Note:* For this project, we accept this bias as a limitation of the simplified dataset, but in the real world, quants pay millions for "point-in-time" datasets that include delisted stocks to prevent this exact illusion.

### Stationarity: Prices vs. Returns
Machine learning models are fundamentally pattern recognizers. To find a pattern, the statistical properties of the data—specifically the mean and variance—must remain relatively constant over time. This property is called **Stationarity**.
* **The Problem with Prices:** Raw stock prices are highly non-stationary. If a stock price is $50 in 2017 and $300 in 2023, a model trained on 2017 prices suffers from *concept drift* and will have no idea how to interpret the 2023 prices.
* **The Solution:** Quants convert prices into returns (percentage changes). Whether a stock goes from $50 to $51 or $300 to $306, the return is +2%. The distribution of daily returns stays centered around 0% regardless of the decade, giving the ML model a stable target to learn from.

---

## 2. Time-Series Integrity

### The "Time Machine" Problem: Data Leakage
In traditional machine learning, the order of data doesn't matter. But in algorithmic trading, time has a strict, unbreakable one-way arrow.
* **The Concept:** Data Leakage occurs when a model inadvertently gains access to information during training that it would not actually have at the moment it makes a prediction in the real world. 
* **The Danger:** If leakage occurs, the model will look like a genius during backtesting, achieving impossible metrics. But the moment it is deployed with real money, it will fail catastrophically because the "future" data it relied on is no longer there.

### Target Shifting & The Timeline of a Trade
To prevent leakage, we must shift the target based on the strict timeline of how a quantitative trade is actually executed:
1. **Day D (11:59 PM):** The market is closed. Features (RSI, MACD) are calculated using data up to the close of Day D. The model looks at these features and generates a signal (e.g., "Buy AAPL").
2. **Day D+1 (Market Open):** The stock is purchased and the position is held.
3. **Day D+2 (Market Close):** The stock is sold.
* **The Mechanical Solution:** The success of the decision made on Day D is measured *only* by the price change between D+1 and D+2. Therefore, in our DataFrame, we must shift the target column upward so that `return(D+1, D+2)` physically sits on the same horizontal row as the features for `Day D`. 

---

## 3. Strategy Execution & Position Sizing

### Holding Periods
* **1-Day Holding:** For the basic evaluation of this project, we force the model into a strict 1-day holding period (Buy at D+1, Sell at D+2). By doing this for every stock, every single day, we measure the "pure" predictive power of the model for that exact 24-hour window.
* *Real-world context:* Hedge funds don't always sell the next day; they may hold a position for weeks to avoid transaction fees. However, for mathematical backtesting, treating every trade as an isolated 1-day event is the cleanest method.

### The Signal Pipeline
There is a strict pipeline that converts a mathematical prediction into real money:
1. **Phase A: The Raw ML Signal (Probability)**
   * The machine learning model does not output actions (-1, 0, or 1). It outputs a *probability* (e.g., AAPL has a 65% probability of a positive return).
2. **Phase B: The Strategy (Action)**
   * Business logic converts the probability into an action:
     * *Binary Long-Only:* If Probability > 50%, Action = 1 (Buy). Otherwise, 0.
     * *Ternary (Long/Short):* If Probability > 60%, Action = 1 (Buy). If Probability < 40%, Action = -1 (Short Sell).
     * *Stock Picking:* Action = 1 for the top 10 stocks, Action = -1 for the bottom 10 stocks.
3. **Phase C: Position Sizing (How much?)**
   * Deciding how much capital to allocate to the action:
     * *Equal Weighting (Project Standard):* Invest exactly $1 per day per stock that triggers a signal.
     * *Proportional Weighting (Advanced):* Size the bet based on confidence (e.g., invest $2 if 90% sure, or $0.50 if only 55% sure).