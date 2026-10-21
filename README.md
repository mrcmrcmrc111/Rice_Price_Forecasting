# 🌾 Rice Price Forecasting — Philippines

> **Monthly retail price prediction (PHP/kg) using naive baselines, Linear Regression, and Random Forest, for horizons of 1 to 60 months ahead.**

Forecasts national regular-milled-rice retail prices from historical price data, rice demand, and World Bank headline inflation.  The project tests whether macroeconomic covariates (previous-year inflation, demand growth) improve over naive baselines and reports results for every horizon on held-out test data.

---

## 📂 Repository layout

```
rice_price_forecasting/
├── data/
│   ├── preprocessed_rice_retailprice.csv   # Monthly retail prices (PHP/kg)
│   ├── preprocessed_rice_demand.csv        # Annual rice demand (Mt)
│   └── preprocessed_inflation.csv          # World Bank headline inflation
├── 01_inspect_data.py      # Data quality & compatibility checks
├── 02_prepare_data.py      # Cleaning, integration → outputs/integrated_rice_price_dataset.csv
├── 03_eda.py               # Exploratory analysis → outputs/figures/, outputs/reports/
├── 04_train_evaluate.py    # Model training, evaluation & forecasting
├── common.py               # Shared config (paths, horizons, split dates, plot helpers)
├── METHODOLOGY.md          # Paper-ready method description
├── requirements.txt
└── run_all.sh              # Runs all four scripts in order (~1.5 min)
```

> `outputs/` and `models/` are **generated** — they are not tracked in git. Run the pipeline to create them.

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/mrcmrcmrc111/Rice_Price_Forecasting.git
cd Rice_Price_Forecasting
pip install -r rice_price_forecasting/requirements.txt   # Python 3.10+
```

### 2. Run the full pipeline

```bash
bash rice_price_forecasting/run_all.sh        # ~1.5 minutes
```

Or run each step individually:

```bash
python rice_price_forecasting/01_inspect_data.py
python rice_price_forecasting/02_prepare_data.py
python rice_price_forecasting/03_eda.py
python rice_price_forecasting/04_train_evaluate.py
```

### 3. Results

| Path | Contents |
|---|---|
| `outputs/metrics_all.csv` | MAE / RMSE / MAPE for every model × horizon |
| `outputs/model_recommendation.csv` | Recommended model per horizon |
| `outputs/future_price_forecast.csv` | Out-of-sample price forecasts |
| `outputs/figures/` | EDA and evaluation plots |
| `outputs/reports/evaluation_report.md` | Full narrative results & limitations |
| `models/best_models.joblib` | Serialised fitted models |

---

## 🔑 Key design decisions

| Decision | Detail |
|---|---|
| **Placeholder exclusion** | 2021 prices (12 identical values) are treated as a placeholder and excluded; usable series ends 2020-12 |
| **Inflation feature** | Only the World Bank headline inflation column is used; other CPI columns are repeated filler |
| **No leakage** | Each row uses the *previous year's* annual inflation and demand growth |
| **Model selection rule** | An ML model is recommended only if it beats the naive baseline on **both** validation and test sets |
| **Horizons** | 1 – 60 months ahead; train / validation / test split is chronological on the forecast origin |

---

## 🧠 Loading the saved models

```python
import joblib

art = joblib.load("models/best_models.joblib")
h   = art["final_models"][1]          # horizon = 1 month ahead
print(h["recommended"])               # e.g. "random_forest"

ch  = h["ml_challenger"]              # best ML model (fitted through 2020-12)
# prediction = last_price * exp(ch["fitted"].predict(X)[0])
# where X contains the columns listed in ch["features"]
```

See [`METHODOLOGY.md`](rice_price_forecasting/METHODOLOGY.md) for the full mathematical description.

---

## 📦 Dependencies

```
pandas>=2.0   numpy>=1.24   scikit-learn>=1.3   scipy>=1.10   matplotlib>=3.7   joblib>=1.3
```

---

## 📄 License

This project is released for academic and research purposes.
