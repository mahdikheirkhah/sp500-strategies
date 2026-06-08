# Contributing to S&P 500 Strategies Project

Thank you for your interest in contributing to our quantitative finance and machine learning project! To ensure a smooth collaboration, maintain a high standard of code quality, and guarantee the statistical integrity of our trading strategies, we require all contributors to strictly adhere to the following guidelines.

## 1. Development Workflow (Branching & CI/CD)

* **Branching Strategy:** Never commit or push directly to the `main` branch. Whether you are fixing a bug, adding a feature, or refactoring, you must create a dedicated branch (e.g., `feature/bollinger-bands`, `fix/target-shift-leakage`, `experiment/xgboost-gridsearch`).
* **CI/CD Checks:** Your branch must successfully pass all automated CI/CD pipelines before it can be merged. This includes formatting checks, linting, and automated tests.
* **Merging:** Once your pipeline passes, open a Pull Request (PR) to `main` and request a code review.

## 2. Dependency Management & Formatting

* **Poetry:** We use **Poetry** for package dependency resolving and environment management. Make sure you install dependencies using `poetry install` (ensure `loguru`, `pandas`, and `scikit-learn` are added to your dependencies).
* **Black Formatter:** We enforce a uniform code style. Before committing your code, you must format your files using Black:
```bash
poetry run black .

```



## 3. Architecture & Paradigm

* **Object-Oriented Programming (OOP):** All code should be structured using OOP principles. Encapsulate related logic (e.g., feature engineering blocks, cross-validation generators, backtesting engines) within well-defined classes.

## 4. Coding Standards & Naming Conventions

* **Variable Naming:** Use clear, descriptive variable names. Ensure your naming conventions follow standard Python Regex rules (e.g., `^[a-z_][a-z0-9_]*$` for snake_case variables and functions, `^[A-Z][a-zA-Z0-9]*$` for PascalCase classes).
* **Logging over Printing:** **Never use the standard `print()` statement.** We use **Loguru** for logging to eliminate boilerplate configuration and maintain clean code. Always use it to record application flow, debugging information, and errors.
```python
from loguru import logger

logger.info("Successfully engineered MACD and RSI features.")
logger.warning("High correlation detected between features; consider dimensionality reduction.")
logger.error("Failed to concatenate OOF predictions; index mismatch detected.")

```



## 5. Function & Method Design

* **Single Responsibility Principle:** Make your functions and methods as reusable as possible. Each function/method should serve **exactly one purpose**. Break large, monolithic functions into smaller, modular pieces.
* **Type Hinting:** You must explicitly declare the data types for all arguments and the return type for every function and method.
```python
def compute_bollinger_bands(price_series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:

```


* **Documentation (Docstrings):** Every function and method must include a comprehensive docstring that clearly explains:
1. The goal and behavior of the function.
2. The types and descriptions of the input parameters.
3. The type and description of the output/return value.



## 6. Error Handling

* **Mandatory Try/Except:** Use `try` and `except` blocks thoroughly. You must include exception handling in **each function and method**.
* **Granular Handling:** Ensure that every distinct logic flow or block within your methods is wrapped in appropriate error handling to catch specific exceptions (e.g., `KeyError` for missing tickers, `ValueError` for dimension mismatches) and log them accordingly using Loguru.

## 7. Quantitative Research & Data Integrity

* **Zero Data Leakage:** Ensure absolutely no future information leaks into the training features. When constructing target variables, explicitly verify that Day $D$ features only use data available up to 23:59 PM on Day $D$, and predict the return strictly between $D+1$ and $D+2$.
* **Reproducibility:** Machine learning models and cross-validation splits must be fully reproducible. Always set and document `random_state` seeds for any stochastic processes (like standard Scalers, ML models, or splitters).

## 8. Testing

* **Test-Driven Collaboration:** If you write a new function or method, you are required to write the corresponding test code.
* **Flow Coverage:** Your tests must account for different logic flows and edge cases inside the method, including testing the `except` blocks by triggering known errors. Test edge cases specific to financial data, such as `NaN` values, missing dates, and untraded assets.