# Methodology (for the paper and thesis defense)

**Research question.** Can machine-learning models predict future Philippine retail rice prices accurately, and does adding
rice demand and inflation improve forecasts compared with historical prices alone?

## 1. Data
| Dataset | Frequency | Coverage | Used as |
|---|---|---|---|
| PSA retail price, regular milled rice, national | monthly | 1990-01 to 2021-12 | target and price-history features |
| PSA supply utilization, rice (net food disposable) | annual | 1990-2022 | previous-year demand growth |
| Inflation, headline (World Bank column) | annual | 1990-2025 | previous-year inflation |

Quality findings (see `outputs/reports/`): the price file has no missing or duplicate months, but all 12 months of 2021 are the
same value (36.23) and were treated as a placeholder and excluded. Three of the four inflation/CPI columns repeat one value in
86-94% of years and were not used. The price series has one classification and one location, so no per-classification model exists.

## 2. Frequency alignment
Price is monthly; demand and inflation are annual. The monthly model is kept, and each month receives the **previous calendar
year's** annual inflation and demand growth. This avoids (a) repeating one annual number as if it were monthly information and
(b) using a figure that is only published after the year ends. Demand enters as year-over-year growth, not level, because the
level trends with price and would only add a spurious correlation.

## 3. Forecasting problem
- Target: price in PHP/kg at month t+h, h in {1, 3, 6, 12}.
- Direct strategy: a separate model per horizon, trained to predict log(price[t+h] / price[t]). Predictions are converted back
  to PHP/kg before scoring. No predicted value is ever used as an input, so predictor availability is never an issue.
- Why change instead of level: prices trend upward and a Random Forest cannot predict above the highest price it has seen.

## 4. Features (all computed from information available at month t)
Price history (config A): last three monthly log changes, 3/6/12-month average change, price relative to its trailing
12-month average, 12-month change, and sine/cosine of the target calendar month.
Config B adds previous-year inflation and previous-year demand growth.

## 5. Validation design
Chronological split on the forecast origin. Training rows must have targets realised by 2012-12; validation uses origins
after 2012-12 with targets through 2016-12; the test window is origins after 2016-12 (targets 2017-2020). For the test
evaluation the models are refit on train + validation. All models are scored on identical rows. Hyperparameters (Random Forest
grid of depth, leaf size, feature fraction) and model selection use validation only. Standardisation for Linear Regression is
fit inside the training data.

## 6. Models
Naive (last observed price); seasonal naive (same month last year; equals naive at h = 12); Linear Regression; Random Forest.
An ablation Random Forest predicting the price level shows the cost of ignoring the trend.

## 7. Evaluation and decision rule
MAE and RMSE in PHP/kg, R-squared on price levels (inflated by trend) and R-squared on the predicted change (stricter), plus the
ratio of RMSE to the naive RMSE. An ML model is recommended for a horizon only if it beats naive on validation **and** on test.
Config B vs A is compared with RMSE and a paired Wilcoxon test on monthly absolute errors.

## 8. Forecast and uncertainty
Genuine forecasts are issued from the last trustworthy month (2020-12). Intervals are empirical quantiles (10/90 and 5/95) of
the recommended model's out-of-sample test errors, applied multiplicatively to the point forecast.

## 9. Limitations to state explicitly
One national series, a short test window containing the 2018 spike, autocorrelated errors, annual predictors with about 30
distinct values, a placeholder price year, forecasts for a period that has already passed, and no modelling of supply shocks,
trade policy or weather.
