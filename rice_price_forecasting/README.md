# Philippine rice price forecasting (monthly, PHP/kg)

Compares naive baselines, Linear Regression and Random Forest for predicting national regular-milled-rice retail prices 1, 3,
6 and 12 months ahead, and tests whether previous-year inflation and rice-demand growth improve the forecasts.

## Install and run
```bash
pip install -r requirements.txt          # Python 3.10+
bash run_all.sh                          # about 1.5 minutes
```
Or run the scripts one by one: `01_inspect_data.py`, `02_prepare_data.py`, `03_eda.py`, `04_train_evaluate.py`.
The three CSVs live in `data/`. Splits, horizons and the placeholder-year rule are set in `common.py`.

## What each script produces (all under `outputs/` and `models/`)
| Script | Output |
|---|---|
| 01_inspect_data.py | `reports/dataset_inspection_report.md`, `reports/data_compatibility_report.md` |
| 02_prepare_data.py | `integrated_rice_price_dataset.csv`, `reports/cleaning_log.md` (rows kept/removed at each step) |
| 03_eda.py | `figures/eda1..6_*.png`, `reports/eda_notes.md` |
| 04_train_evaluate.py | `metrics_all.csv`, `reports/evaluation_report.md`, `model_recommendation.csv`, `feature_config_comparison.csv`, `rf_feature_importance.csv`, `future_price_forecast.csv`, `figures/eval1..5_*.png`, `models/best_models.joblib`, `models/model_metadata.json` |

## Key assumptions
- 2021 prices (12 identical values) are a placeholder and are excluded; the usable series ends 2020-12.
- Only the World Bank inflation column is used; the other inflation/CPI columns are repeated filler values.
- Each month uses the previous year's annual inflation and demand growth (no leakage, no fake monthly values).
- An ML model is recommended only if it beats naive on both validation and test.

## Using the saved models
```python
import joblib
art = joblib.load("models/best_models.joblib")
h = art["final_models"][1]                       # 1-month horizon
print(h["recommended"])                          # model recommended for this horizon
ch = h["ml_challenger"]                          # best ML model by validation (fitted on all data through 2020-12)
# prediction = last_price * exp(ch["fitted"].predict(X)[0]), where X holds the columns in ch["features"]
```
See `METHODOLOGY.md` for the paper-ready method description and `outputs/reports/evaluation_report.md` for results and limitations.
