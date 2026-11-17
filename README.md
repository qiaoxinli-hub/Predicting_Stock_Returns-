# Blog Post 9: Predicting Stock Returns with Gradient Boosting

This project revisits the stock-selection exercise conducted in class and develops an out-of-sample stock-return prediction and portfolio evaluation workflow using XGBoost, a gradient-boosted decision-tree method.

The project addresses two questions:

1. Did financial indicators help student teams build better portfolios than intuition alone?
2. Can improved feature engineering and model tuning make gradient boosting useful for forecasting future stock returns and constructing an investable portfolio?

The central lesson is that fitting historical data is not the same as forecasting unseen returns, and better predictive metrics do not necessarily translate into better investment performance.


## 1. The classroom portfolio experiment

In the class exercise, 45 teams of two constructed two portfolios, each starting with $100,000, for a two-month investment period:

- **Intuition-based portfolio:** stocks were selected using each team's judgment.
- **Indicator-based portfolio:** stocks were selected using financial measures such as the Sharpe ratio, Jensen's alpha, and correlations among stocks.

The indicators capture different aspects of portfolio choice. The Sharpe ratio summarizes return relative to volatility, Jensen's alpha measures performance relative to a CAPM benchmark, and correlations help assess diversification.

The class analysis found that the indicator-based portfolio had a higher average final value by approximately $1,909 (about 1.89%). The two portfolio outcomes were moderately positively correlated (approximately 0.40). However, the paired two-sided t-test was not significant at the 5% level (p approximately 0.0999). A paired bootstrap with 10,000 replications gave an approximate 95% confidence interval of [-$298, $4,119] and a two-sided p-value of approximately 0.0908.

These results do not establish a statistically reliable advantage for the indicator-based strategy. The exercise motivates the machine-learning analysis, but it does not demonstrate that financial indicators or machine learning can consistently outperform intuition or a market benchmark.

The classroom analysis reads `data/Full_Results copy.xlsx`, using the `portfolio1` and `portfolio2` columns. The R script calculates summary statistics, paired differences, paired tests, bootstrap uncertainty, and plots programmatically.

## 2. The original XGBoost experiment

The instructor's example uses daily market data for a diversified universe of 95 large U.S. companies, together with SPY as an S&P 500 proxy. The original R script retrieves prices through `tidyquant::tq_get()` from Yahoo Finance, starting on January 1, 2022.

The original model uses historical price-based predictors, including:

- Approximate one-, three-, and six-month momentum measures.
- A lagged one-month volatility measure.
- A short-term five-day return measure.
- Stock returns in excess of SPY returns.

The prediction target is the return from the next trading day's adjusted close to the adjusted close 20 trading sessions ahead:

\[
r_{t+1,t+20}=\frac{P_{t+20}}{P_{t+1}}-1.
\]

The original time split trains on observations dated before January 1, 2024 and evaluates observations dated on or after that date. The XGBoost regression model uses 300 trees, maximum tree depth 5, learning rate 0.05, and minimum child size 10.

For portfolio construction, stocks are ranked by predicted return. The ten highest-ranked stocks are selected and weighted in proportion to positive predictions. If all ten predictions are non-positive, the portfolio falls back to equal weights. The instructor's workflow also compares the strategy with SPY.

## 3. Why stock returns are difficult to predict

Stock-return prediction is difficult for several reasons:

1. **Low signal-to-noise ratio.** Daily returns contain substantial unpredictable variation, so any forecasting signal may be small relative to noise.
2. **Changing market conditions.** Relationships between past prices and future returns can weaken or reverse across market regimes.
3. **Competition and information efficiency.** Public information is quickly incorporated into prices, making persistent opportunities difficult to identify.
4. **Non-stationarity.** Volatility, correlations, investor behavior, and macroeconomic conditions change over time.
5. **Overfitting.** Flexible models can fit historical patterns that do not generalize to unseen observations, especially after repeated feature and hyperparameter experiments.
6. **Prediction is not profit.** Forecast accuracy alone does not account for turnover, transaction costs, liquidity, concentration, or investment risk.

For these reasons, this project emphasizes chronological out-of-sample evaluation and examines both predictive quality and realized portfolio performance.

## 4. How the model is improved

The improved workflow retains the original investment universe, forward-return target, and January 1, 2024 test start date. This makes the comparison with the supplied baseline more consistent. The main changes are feature engineering, leakage control, and hyperparameter tuning.

### Feature engineering

The improved model replaces the legacy price-ratio definitions with standard backward-looking return calculations and adds signals from multiple horizons:

- One-, three-, and six-month stock momentum.
- Twenty-day realized volatility.
- Five-day short-term reversal.
- Stock returns in excess of SPY returns.
- One- and three-month SPY momentum and SPY 20-day volatility.
- Twenty-day average dollar volume, an Amihud-style illiquidity proxy, and a 20-day high-low range.

An optional `fundamentals_point_in_time.csv` file can add log market capitalization and book-to-market. It must contain `symbol`, `available_date`, `market_cap`, and `book_equity`. The `available_date` must be when the information became public, not merely the fiscal-period end date. If the file is absent or has no usable pre-test observations, the model omits these fundamental features and uses price/volume features only.

### Leakage control and tuning

The final test period starts on January 1, 2024. Because the target uses future prices over a 20-trading-day horizon, the improved workflow removes training observations whose target window crosses a train/test or validation boundary.

Hyperparameters are selected using three chronological, purged validation windows within the pre-test sample:

- July 1, 2023 to September 1, 2023.
- September 1, 2023 to November 1, 2023.
- November 1, 2023 to January 1, 2024.

The tuning grid contains 16 combinations over tree depth (2 or 3), learning rate (0.03 or 0.05), minimum child weight (10 or 20), and L2 regularization (5 or 20). L1 regularization, gamma, row subsampling, and column subsampling are also specified. Candidates are assessed using validation-period rank information, RMSE, and returns from the same top-ten portfolio rule. The selected configuration is then fitted on the full purged pre-test training sample.

The baseline in the improved script is re-estimated using the instructor's legacy feature construction and model settings on the same leakage-controlled training sample. This provides a more comparable baseline than fitting the two models on different date ranges.

## 5. Out-of-sample evaluation

The project compares three investment approaches:

- **Original baseline XGBoost:** the instructor's feature construction and model settings.
- **Improved XGBoost:** expanded and corrected price/market features with purged walk-forward tuning.
- **SPY buy-and-hold:** a passive S&P 500 proxy benchmark.

### Predictive performance

The improved workflow writes out-of-sample predictive metrics for the two XGBoost models:

- **RMSE:** penalizes large forecast errors.
- **MAE:** measures average absolute forecast error.
- **Directional accuracy:** the share of predictions with the correct return sign.
- **Mean daily rank information coefficient:** the average daily Spearman correlation between predicted and realized returns across stocks.

### Economic performance

The portfolio simulation forms a signal on the first available prediction date of each month and executes selected holdings at the following SPY trading day's close. It selects the ten highest predicted-return stocks and weights them in proportion to positive predictions, falling back to equal weights when all selected predictions are non-positive. Holdings are evaluated over the common interval to the next scheduled rebalance date.

The backtest compares both XGBoost portfolios with SPY over aligned holding periods. It deducts a modeled transaction cost of **10 basis points per dollar of turnover** and reports:

- Total return and ending growth of one dollar.
- Annualized Sharpe ratio, calculated from monthly returns using \(\sqrt{12}\).
- CAPM-style annualized alpha and beta relative to SPY.
- Maximum drawdown.

The transaction-cost rate is a modeling assumption, not a quoted brokerage fee. The backtest is therefore an educational simulation rather than a live-trading estimate.

**No out-of-sample metric values are hard-coded in this README.** Numerical results should be taken from the CSV outputs generated by the R script after it is run. A more complicated model should not be described as an improvement unless its held-out test results support that conclusion.

## 6. Data sources and limitations

| Input | Source / use |
|---|---|
| Historical stock prices | Yahoo Finance, retrieved through `tidyquant::tq_get()` |
| Market benchmark | SPY adjusted prices retrieved through `tidyquant` |
| Classroom portfolio outcomes | `data/Full_Results copy.xlsx` |
| Optional fundamentals | User-supplied point-in-time `fundamentals_point_in_time.csv` |

The stock-price script downloads data at runtime rather than reading a fixed price snapshot. The end date and downloaded observations may therefore change between runs as new trading dates become available or the data provider updates its history. For strict replication, save and version the downloaded price snapshot and record the retrieval date.

The stock universe consists of current tickers, not a point-in-time record of historical index constituents. This creates a **survivorship-bias limitation**: companies that no longer exist or were not constituents during earlier periods may be absent. Optional fundamental features should only be used when their historical availability dates are reliable.

## 7. Project organization

The following is the recommended local project layout. Keep the file names and paths consistent with the commands below, or update the commands to match your actual folders.

```text
Predicting_Stock_Returns/
├── data/
│   └── Full_Results copy.xlsx
├── UPenn_XGBoostStockPrediction copy(2).R
├── Improved Model.R
├── stockmarketgameresultsFall2026.R
├── Blog_Post_9_Predicting_Stock_Returns.qmd
├── frontpage.png
├── improved_xgb_results/                 # created when the improved model runs
│   ├── hyperparameter_tuning.csv
│   ├── predictive_metrics.csv
│   ├── monthly_strategy_returns.csv
│   ├── economic_metrics.csv
│   ├── latest_improved_portfolio.csv
│   ├── oos_cumulative_returns.png
│   └── transaction_cost_comparison.png
└── README.md
```

The improved script creates `improved_xgb_results/` automatically. The CSV tables and figures are generated programmatically and should not be manually edited.

## 8. Reproducing the analysis

### Requirements

Use R and install the packages used by the analysis:

```r
install.packages(c(
  "tidyverse",
  "tidyquant",
  "tidymodels",
  "lubridate",
  "slider",
  "xgboost",
  "readxl",
  "knitr",
  "scales"
))
```

The stock-price download requires an internet connection. The Quarto post also requires the Quarto command-line tool.

### Run the classroom experiment analysis

Place `Full_Results copy.xlsx` in the `data/` folder and run:

```r
source("stockmarketgameresultsFall2026.R")
```

If the script contains a computer-specific absolute path, replace it with a project-relative path, for example:

```r
readxl::read_excel(file.path("data", "Full_Results copy.xlsx"))
```

### Run the improved model

From the repository root, run:

```r
source("Improved Model.R")
```

This downloads the 95-stock universe and SPY price history, constructs features, tunes and fits the models, evaluates the test period, and writes CSV and PNG outputs to `improved_xgb_results/`.

### Render the Quarto post

After the class data and model results are available, render the post:

```sh
quarto render Blog_Post_9_Predicting_Stock_Returns.qmd
```

Use the generated CSV files to populate result tables in the post. Do not enter model metrics before they have been generated and checked.
