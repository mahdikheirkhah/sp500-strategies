# S&P 500 Machine Learning Trading Strategy

## Overview

This repository contains a quantitative finance project aimed at developing a machine learning-based trading strategy to consistently outperform the S&P 500 index. Built as a rigorous data science pipeline, the system extracts signals from technical indicators, validates models using time-series specific cross-validation, and backtests financial performance.

### Global Approach
The architecture follows a strict pipeline to prevent data leakage and ensure statistical robustness:
1. **Feature Engineering:** Calculation of technical indicators (Bollinger Bands, RSI, MACD) on S&P 500 constituent OHLCV data without look-ahead bias. The target is defined as `sign(return(D+1, D+2))`.
2. **Cross-Validation:** Implementation of Time Series Split / Blocking Time Series Split to handle the non-stationary nature of financial data and identify distinct market regimes.
3. **Machine Learning Pipeline:** A complete model training phase (Imputation, Scaling, Dimensionality Reduction, and ML Model) utilizing Out-of-Fold (OOF) predictions to generate untainted trading signals.
4. **Strategy Backtesting:** Conversion of ML probability signals into actionable financial positions (e.g., Long/Short, Stock Picking), evaluated via strict PnL and Maximum Drawdown metrics against the S&P 500 benchmark.

### Constraints & Simplifications
To isolate the predictive power of the machine learning models and simplify the backtesting mechanics, this project assumes the following constraints:
* **Strict 1-Day Holding Period:** The strategy executes exactly one action per day per asset. If a buy signal is generated based on Day D's data, the asset is purchased on Day D+1 and strictly sold on Day D+2 to measure the exact 24-hour predictive validity.
* **Fixed Position Sizing (Equal Weighting):** Portfolio allocation does not scale with model confidence. The strategy assumes a fixed investment of exactly **$1 per day** into each stock that triggers an active signal.
* **Static Index Composition:** The dataset assumes the S&P 500 constituents remain constant over the 5-year period. In reality, index composition frequently changes; ignoring delisted companies introduces *Survivorship Bias* into the backtest results.

## Repository Structure

```text
project/
├── data/
│   ├── HistoricalData.csv          # SP500 index OHLC data
│   └── all_stocks_5yr.csv          # SP500 constituents OHLCV data
├── requirements.txt                # Pip fallback dependencies
├── pyproject.toml                  # Poetry dependency management
├── README.md                       # Project documentation
├── CONTRIBUTING.md                 # Code quality and PR guidelines
├── results/
│   ├── cross-validation/           # CV metrics, plots, and feature importance
│   ├── selected-model/             # Pickled model, hyperparameters, and ML signal
│   └── strategy/                   # Markdown report, PnL plots, and backtest results
└── scripts/
    ├── features_engineering.py     # Data processing and indicator generation
    ├── model_selection.py          # Time-series cross-validation logic
    ├── gridsearch.py               # Hyperparameter tuning and model training
    ├── create_signal.py            # OOF prediction generation
    └── strategy.py                 # Financial strategy conversion and backtesting

```

## Environment & Prerequisites

This project enforces strict environment constraints to ensure reproducibility.

* **Python:** Standard Python 3.12+ is required. *(Note for Windows users: Ensure standard Python is prioritized in your PATH and Windows App Execution Aliases for `python` are disabled to avoid conflicts with MSYS2 environments).*
* **Dependency Management:** We use [Poetry](https://python-poetry.org/) for resolving dependencies and maintaining a locked environment.
* **Git Configuration:** Ensure your local Git identity is configured properly before committing (check `git config --global user.name` and `user.email`).

## Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository_url>
cd <repository_directory>

```


2. **Initialize the environment:**
Using Poetry (Recommended):
```bash
poetry install

```


*Alternatively, using pip:*
```bash
pip install -r requirements.txt

```


3. **Data Placement:**
Ensure `HistoricalData.csv` and `all_stocks_5yr.csv` are placed inside the `data/` directory.

## How to Run

To execute the full quantitative pipeline, run the scripts in the following sequence from the root directory:

**1. Data Processing & Feature Engineering**
Processes the raw OHLCV data, shifts targets to prevent leakage, and calculates indicators.

```bash
poetry run python scripts/features_engineering.py

```

**2. Model Selection & Cross-Validation**
Generates the cross-validation folds and visualizations.

```bash
poetry run python scripts/model_selection.py

```

**3. Pipeline Training & Grid Search**
Trains the ML models, logs metrics, and saves the best pipeline to `results/selected-model/`.

```bash
poetry run python scripts/gridsearch.py

```

**4. Signal Generation**
Produces the out-of-fold machine learning signals and saves them as a double-indexed CSV.

```bash
poetry run python scripts/create_signal.py

```

**5. Strategy Backtesting**
Converts the generated ML signal into a simulated financial portfolio, calculating PnL and generating plots.

```bash
poetry run python scripts/strategy.py

```

## Logging & Code Quality

This project adheres to clean code standards and OOP principles. Standard `print()` statements are prohibited; all application flow, warnings, and errors are handled via **Loguru**. Before submitting any Pull Requests, ensure code is formatted with Black (`poetry run black .`).

## Author

**Mohammad Mahdi Kheirkhah** Student at grit:lab, Åland

Email: `mohammad.kheirkhah@gritlab.ax`
