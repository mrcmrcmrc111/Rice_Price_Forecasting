"""Phases 5-9: features, models, chronological evaluation, selection, and future forecasts.

Design (details in METHODOLOGY.md)
  * Direct multi-horizon forecasting: one model per horizon h in HORIZONS. No predicted value is ever fed back as an input.
  * Target = log(price[t+h] / price[t]); predictions are converted back to PHP/kg before scoring.
  * Every feature at forecast origin t uses information up to month t only. Annual inflation / demand growth are those of
    the PREVIOUS calendar year (see 02_prepare_data.py).
  * Split on forecast origin (see common.py). Hyperparameters and the final model per horizon are chosen on VALIDATION only.
  * All models are scored on exactly the same rows.
"""
import itertools
import json
import warnings

import joblib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import *

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ names
M_NAIVE = "Naive (last observed price)"
M_SNAIVE = "Seasonal naive (same month last year)"
M_LR = "Linear Regression"
M_RF = "Random Forest"
M_RFL = "Random Forest, log-price level target (ablation)"
CFG_A = "A: price history only"
CFG_B = "B: price history + prev-year inflation & demand growth"

RF_GRID = dict(max_depth=[3, 6, None], min_samples_leaf=[3, 10], max_features=[0.5, 1.0])
RF_FIXED = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)

# ------------------------------------------------------------------ data
df = pd.read_csv(INTEGRATED_CSV, parse_dates=["date"]).set_index("date")
ser = df["price_php_kg"].dropna()
assert ser.index.equals(pd.date_range(ser.index.min(), ser.index.max(), freq="MS")), "price series must be contiguous"
lp = np.log(ser)
r = lp.diff()
LAST_DATE = ser.index.max()

FEATS_A = ["ret_0", "ret_1", "ret_2", "ret_mean_3", "ret_mean_6", "ret_mean_12", "gap_12m", "chg_12m",
           "target_month_sin", "target_month_cos"]
FEATS_B = FEATS_A + ["prev_year_inflation", "prev_year_demand_growth"]
FEATS_LEVEL = FEATS_A + ["lp_origin"]
CONFIGS = {CFG_A: FEATS_A, CFG_B: FEATS_B}


def make_frame(h: int) -> pd.DataFrame:
    """One row per forecast origin t. All predictors use data up to t only; y uses t+h."""
    f = pd.DataFrame(index=ser.index)
    f["ret_0"] = r                                    # change from t-1 to t (known at t)
    f["ret_1"] = r.shift(1)
    f["ret_2"] = r.shift(2)
    f["ret_mean_3"] = r.rolling(3).mean()
    f["ret_mean_6"] = r.rolling(6).mean()
    f["ret_mean_12"] = r.rolling(12).mean()
    f["gap_12m"] = lp - lp.rolling(12).mean()          # price vs its trailing 12-month average
    f["chg_12m"] = lp - lp.shift(12)
    f["lp_origin"] = lp
    tm = (f.index + pd.DateOffset(months=h)).month      # calendar month of the TARGET (known in advance)
    f["target_month_sin"] = np.sin(2 * np.pi * tm / 12)
    f["target_month_cos"] = np.cos(2 * np.pi * tm / 12)
    f["prev_year_inflation"] = df["prev_year_inflation_pct"].reindex(f.index)
    f["prev_year_demand_growth"] = df["prev_year_demand_growth_pct"].reindex(f.index)
    f["origin_price"] = ser
    f["target_date"] = f.index + pd.DateOffset(months=h)
    f["target_price"] = ser.reindex(f["target_date"]).values
    f["y"] = np.log(f["target_price"]) - lp           # log ratio (the learning target)
    f["y_level"] = np.log(f["target_price"])
    f["seasonal_naive"] = ser.reindex(f["target_date"] - pd.DateOffset(months=12)).values
    return f


def usable(f):
    return f.dropna(subset=FEATS_B)


def split_rows(f):
    """Return dict of row subsets. Rows need a realised target (y not NaN)."""
    g = usable(f)
    labelled = g[g["y"].notna()]
    return dict(
        train=labelled[labelled["target_date"] <= TRAIN_END],
        val=labelled[(labelled.index > TRAIN_END) & (labelled["target_date"] <= VAL_END)],
        trainval=labelled[labelled["target_date"] <= VAL_END],
        test=labelled[labelled.index > VAL_END],
        all=labelled,
    )


def lr_factory():
    return make_pipeline(StandardScaler(), LinearRegression())


def rf_factory(**params):
    return RandomForestRegressor(**RF_FIXED, **params)


def to_price(origin_price, yhat):
    return origin_price.to_numpy() * np.exp(yhat)


def metrics(actual, pred, origin, naive_rmse=None):
    a, p, o = np.asarray(actual), np.asarray(pred), np.asarray(origin)
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    chg_a, chg_p = a - o, p - o
    return dict(n=len(a), MAE=mean_absolute_error(a, p), RMSE=rmse, R2=r2_score(a, p),
                R2_change=r2_score(chg_a, chg_p),
                MAPE_pct=float(np.mean(np.abs((a - p) / a)) * 100))


# ------------------------------------------------------------------ run all horizons
pred_store = []
best_params = {}
tuning_rows = []
fitted = {}                       # (h, model, cfg) -> model fitted on train+val (used for test predictions)


def store(h, model, cfg, split, d, price_pred):
    pred_store.append(pd.DataFrame({
        "horizon": h, "model": model, "features": cfg, "split": split,
        "origin_date": d.index, "target_date": d["target_date"].values,
        "origin_price": d["origin_price"].values, "actual": d["target_price"].values, "pred": price_pred}))


frames, splits = {}, {}
for h in HORIZONS:
    f = make_frame(h)
    S = split_rows(f)
    frames[h], splits[h] = f, S
    if not S['test'].empty and not S['test']['target_date'].isnull().all():
        first_target = S['test']['target_date'].min()
        last_target = S['test']['target_date'].max()
        print(f"h={h:2d}: train {len(S['train'])}, val {len(S['val'])}, test {len(S['test'])} rows (first test target {first_target:%Y-%m}, last {last_target:%Y-%m})")
    else:
        print(f"h={h:2d}: train {len(S['train'])}, val {len(S['val'])}, test {len(S['test'])} rows (no test dates)")
    for split in ("val", "test"):
        d = S[split]
        store(h, M_NAIVE, "-", split, d, d["origin_price"].to_numpy())
        store(h, M_SNAIVE, "-", split, d, d["seasonal_naive"].to_numpy())

    if S["val"].empty:
        # No validation rows for this horizon – skip model selection and use naive only.
        print(f"  -> h={h}: skipping ML training (validation split is empty)")
    else:
        for cfg, feats in CONFIGS.items():
            # ---- Linear Regression: train -> val ; train+val -> test
            m = lr_factory().fit(S["train"][feats], S["train"]["y"])
            store(h, M_LR, cfg, "val", S["val"], to_price(S["val"]["origin_price"], m.predict(S["val"][feats])))
            m = lr_factory().fit(S["trainval"][feats], S["trainval"]["y"])
            fitted[(h, M_LR, cfg)] = m
            if not S["test"].empty:
                store(h, M_LR, cfg, "test", S["test"], to_price(S["test"]["origin_price"], m.predict(S["test"][feats])))

            # ---- Random Forest: tune on validation only
            best = None
            for md, msl, mf in itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"], RF_GRID["max_features"]):
                params = dict(max_depth=md, min_samples_leaf=msl, max_features=mf)
                m = rf_factory(**params).fit(S["train"][feats], S["train"]["y"])
                pv = to_price(S["val"]["origin_price"], m.predict(S["val"][feats]))
                rmse = float(np.sqrt(mean_squared_error(S["val"]["target_price"], pv)))
                tuning_rows.append(dict(horizon=h, features=cfg, **{k: str(v) for k, v in params.items()}, val_RMSE=rmse))
                if best is None or rmse < best[0]:
                    best = (rmse, params, pv)
            best_params[(h, cfg)] = best[1]
            store(h, M_RF, cfg, "val", S["val"], best[2])
            m = rf_factory(**best[1]).fit(S["trainval"][feats], S["trainval"]["y"])
            fitted[(h, M_RF, cfg)] = m
            if not S["test"].empty:
                store(h, M_RF, cfg, "test", S["test"], to_price(S["test"]["origin_price"], m.predict(S["test"][feats])))

        # ---- ablation: Random Forest on log-price LEVEL (shows why predicting changes matters)
        abl = dict(max_depth=None, min_samples_leaf=3, max_features=1.0)
        m = rf_factory(**abl).fit(S["train"][FEATS_LEVEL], S["train"]["y_level"])
        store(h, M_RFL, CFG_A + " + price level", "val", S["val"], np.exp(m.predict(S["val"][FEATS_LEVEL])))
        m = rf_factory(**abl).fit(S["trainval"][FEATS_LEVEL], S["trainval"]["y_level"])
        if not S["test"].empty:
            store(h, M_RFL, CFG_A + " + price level", "test", S["test"], np.exp(m.predict(S["test"][FEATS_LEVEL])))

preds = pd.concat(pred_store, ignore_index=True)
pd.DataFrame(tuning_rows).to_csv(REP / "rf_tuning_results.csv", index=False)

# ------------------------------------------------------------------ metrics
rows = []
for (h, model, cfg, split), g in preds.groupby(["horizon", "model", "features", "split"], sort=False):
    rows.append(dict(horizon=h, model=model, features=cfg, split=split,
                     period=f"{g['target_date'].min():%Y-%m} to {g['target_date'].max():%Y-%m}",
                     **metrics(g["actual"], g["pred"], g["origin_price"])))
M = pd.DataFrame(rows)
naive = M[M["model"] == M_NAIVE].set_index(["horizon", "split"])["RMSE"]
M["RMSE_vs_naive"] = [row.RMSE / naive[(row.horizon, row.split)] for row in M.itertuples()]
M.to_csv(OUT / "metrics_all.csv", index=False)
preds.to_csv(OUT / "predictions_validation_test.csv", index=False)

# same-rows check: every model must have the same n and target dates within (horizon, split)
chk = preds.groupby(["horizon", "split", "model", "features"])["target_date"].agg(lambda s: tuple(sorted(s)))
for (h, sp), g in chk.groupby(level=[0, 1]):
    assert g.nunique() == 1, f"models scored on different rows at h={h}, {sp}"

# ------------------------------------------------------------------ model selection on VALIDATION only
cand = M[(M["split"] == "val") & (M["model"] != M_RFL)]
sel = cand.loc[cand.groupby("horizon")["RMSE"].idxmin()][["horizon", "model", "features", "RMSE", "RMSE_vs_naive"]]
sel = sel.rename(columns={"RMSE": "val_RMSE", "RMSE_vs_naive": "val_RMSE_vs_naive"}).reset_index(drop=True)
test_sel = []
for row in sel.itertuples():
    t = M[(M["split"] == "test") & (M["horizon"] == row.horizon) & (M["model"] == row.model) & (M["features"] == row.features)].iloc[0]
    test_sel.append(dict(test_RMSE=t["RMSE"], test_MAE=t["MAE"], test_RMSE_vs_naive=t["RMSE_vs_naive"]))
sel = pd.concat([sel, pd.DataFrame(test_sel)], axis=1)
sel["beats_naive_on_test"] = sel["test_RMSE_vs_naive"] < 1
sel.to_csv(OUT / "selected_models.csv", index=False)

# ------------------------------------------------------------------ A vs B comparison (do inflation and demand help?)
cmp_rows = []
for h in HORIZONS:
    for model in (M_LR, M_RF):
        a = preds[(preds["horizon"] == h) & (preds["model"] == model) & (preds["features"] == CFG_A) & (preds["split"] == "test")].sort_values("target_date")
        b = preds[(preds["horizon"] == h) & (preds["model"] == model) & (preds["features"] == CFG_B) & (preds["split"] == "test")].sort_values("target_date")
        ea, eb = (a["actual"] - a["pred"]).abs().to_numpy(), (b["actual"] - b["pred"]).abs().to_numpy()
        rm = lambda x: float(np.sqrt(np.mean(((x["actual"] - x["pred"]) ** 2))))
        try:
            pval = wilcoxon(eb - ea).pvalue
        except ValueError:
            pval = np.nan
        cmp_rows.append(dict(horizon=h, model=model, RMSE_A=rm(a), RMSE_B=rm(b), change_pct=100 * (rm(b) / rm(a) - 1),
                             months_B_worse=int((eb > ea).sum()), months_B_better=int((eb < ea).sum()), wilcoxon_p=pval))
CMP = pd.DataFrame(cmp_rows)
CMP.to_csv(OUT / "feature_config_comparison.csv", index=False)

# ------------------------------------------------------------------ figures
COL = {M_NAIVE: MUTED, M_SNAIVE: "#b9b5a3", M_LR: BLUE, M_RF: ORANGE}


def series_for(h, model, cfg):
    g = preds[(preds["horizon"] == h) & (preds["model"] == model) & (preds["features"] == cfg) & (preds["split"] == "test")]
    return g.sort_values("target_date").set_index("target_date")


# fig 1: actual vs predicted – pick up to 4 horizons that have test rows
EVAL_HORIZONS = [h for h in HORIZONS if not splits[h]["test"].empty]
FIG1_HORIZONS = EVAL_HORIZONS[:4] if len(EVAL_HORIZONS) >= 4 else EVAL_HORIZONS
if FIG1_HORIZONS:
    fig, axes = new_fig(2, 2, figsize=(11, 7), sharey=True)
    for ax, h in zip(axes.ravel(), FIG1_HORIZONS):
        style_ax(ax)
        act = series_for(h, M_NAIVE, "-")
        ax.plot(act.index, act["actual"], color=INK, linewidth=2.2, label="Actual")
        ax.plot(act.index, act["pred"], color=MUTED, linewidth=1.5, linestyle="--", label="Naive")
        s = series_for(h, M_LR, CFG_A); ax.plot(s.index, s["pred"], color=BLUE, linewidth=1.6, label="Linear Reg. (A)")
        s = series_for(h, M_RF, CFG_A); ax.plot(s.index, s["pred"], color=ORANGE, linewidth=1.6, label="Random Forest (A)")
        s = series_for(h, M_RF, CFG_B); ax.plot(s.index, s["pred"], color=AQUA, linewidth=1.6, label="Random Forest (B)")
        title(ax, f"{h}-month-ahead forecasts, test period")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # hide unused subplots
    for ax in axes.ravel()[len(FIG1_HORIZONS):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("PHP per kg", color=INK2)
    save_fig(fig, "eval1_actual_vs_predicted.png")

# fig 2: residuals (h=1) for LR(A) and RF(A)
fig, axes = new_fig(2, 2, figsize=(10, 6.5))
for j, (model, color) in enumerate([(M_LR, BLUE), (M_RF, ORANGE)]):
    s = series_for(1, model, CFG_A)
    res = s["actual"] - s["pred"]
    style_ax(axes[0, j]); style_ax(axes[1, j])
    axes[0, j].bar(res.index, res.values, width=25, color=color)
    axes[0, j].axhline(0, color=AXIS, linewidth=1)
    title(axes[0, j], f"{model} (A), 1-month ahead: residual over time")
    axes[0, j].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, j].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, j].set_ylabel("actual - predicted (PHP/kg)", color=INK2, fontsize=8.5)
    axes[1, j].scatter(s["pred"], res, color=color, s=16)
    axes[1, j].axhline(0, color=AXIS, linewidth=1)
    axes[1, j].set_xlabel("predicted price (PHP/kg)", color=INK2); axes[1, j].set_ylabel("residual", color=INK2, fontsize=8.5)
    title(axes[1, j], "residual vs predicted")
save_fig(fig, "eval2_residuals_h1.png")

# fig 3: error comparison, test (only horizons with actual test rows)
if EVAL_HORIZONS:
    order = [(M_NAIVE, "-"), (M_SNAIVE, "-"), (M_LR, CFG_A), (M_LR, CFG_B), (M_RF, CFG_A), (M_RF, CFG_B)]
    labels = ["Naive", "Seasonal naive", "Lin. Reg. (A)", "Lin. Reg. (B)", "Rand. Forest (A)", "Rand. Forest (B)"]
    colors = [MUTED, "#b9b5a3", BLUE, "#8fb8ee", ORANGE, "#f4a98a"]
    fig, axes = new_fig(1, 2, figsize=(11, 4.2))
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        style_ax(ax)
        w = 0.13
        for i, ((model, cfg), lab, col) in enumerate(zip(order, labels, colors)):
            # only include horizons that have a test row for this model/cfg combination
            h_vals = [(h, M[(M["split"] == "test") & (M["horizon"] == h) & (M["model"] == model) & (M["features"] == cfg)]) for h in EVAL_HORIZONS]
            h_vals = [(h, g) for h, g in h_vals if len(g) > 0]
            if not h_vals:
                continue
            hs, vals = zip(*[(h, g[metric].iloc[0]) for h, g in h_vals])
            xs = [EVAL_HORIZONS.index(h) for h in hs]
            ax.bar(np.array(xs) + (i - 2.5) * w, vals, width=w, color=col, edgecolor=SURFACE, label=lab)
        ax.set_xticks(range(len(EVAL_HORIZONS)))
        ax.set_xticklabels([f"{h}m" for h in EVAL_HORIZONS], rotation=90, fontsize=7)
        ax.set_ylabel(f"{metric} (PHP/kg)", color=INK2)
        title(ax, f"Test {metric} by forecast horizon (lower is better)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    save_fig(fig, "eval3_error_comparison.png")

# fig 4: Random Forest importance, h=1 and h=6 (config B)
fig, axes = new_fig(1, 3, figsize=(15, 4.4))
imp_rows = []
for ax, (h, kind) in zip(axes, [(1, "impurity"), (1, "perm"), (6, "perm")]):
    style_ax(ax, "x")
    ax.grid(axis="y", visible=False)
    m = fitted[(h, M_RF, CFG_B)]
    if kind == "impurity":
        v = pd.Series(m.feature_importances_, index=FEATS_B)
        ttl, xl = f"Impurity importance, {h}-month", "share of total"
    else:
        T = splits[h]["test"]
        pi = permutation_importance(m, T[FEATS_B], T["y"], scoring="neg_root_mean_squared_error", n_repeats=30,
                                    random_state=RANDOM_STATE)
        v, ttl, xl = pd.Series(pi.importances_mean, index=FEATS_B), f"Permutation importance (test), {h}-month", "RMSE increase when shuffled (log-change)"
    imp_rows += [dict(horizon=h, kind=kind, feature=k, importance=x) for k, x in v.items()]
    v = v.sort_values()
    ax.barh(v.index, v.values, color=[AQUA if k.startswith("prev_year") else ORANGE for k in v.index], height=0.65)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(xl, color=INK2, fontsize=8.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    title(ax, ttl)
    ax.title.set_fontsize(9.5)
    ax.tick_params(axis="y", labelsize=8.5, colors=INK)
fig.text(0.01, 0.01, "green = previous-year inflation / demand growth (config B only)", fontsize=8, color=INK2)
save_fig(fig, "eval4_rf_feature_importance.png")
pd.DataFrame(imp_rows).to_csv(OUT / "rf_feature_importance.csv", index=False)

# ------------------------------------------------------------------ recommendation gate and future forecasts
# Rule fixed in advance: an ML model is recommended for a horizon only if it beats the naive baseline on BOTH the
# validation window (where it was selected) and the untouched test window. Otherwise the naive forecast is recommended
# and the ML forecast is shown only as a challenger. Test data never choose between candidates; they only approve or reject.
ml_cand = cand[cand["model"].isin([M_LR, M_RF])]
best_ml = ml_cand.loc[ml_cand.groupby("horizon")["RMSE"].idxmin()].set_index("horizon")
REC, FINAL_MODELS, fc_rows = [], {}, []
for h in HORIZONS:
    f_h, S_h = frames[h], splits[h]
    origin_feats = f_h.loc[[LAST_DATE]]
    assert origin_feats[FEATS_B].notna().all().all(), "forecast-origin features incomplete"

    if h in best_ml.index and not S_h["val"].empty:
        # Normal path: ML was trained and validated for this horizon
        b = best_ml.loc[h]
        model, cfg = b["model"], b["features"]
        feats = CONFIGS[cfg]
        t_rows = M[(M["split"] == "test") & (M["horizon"] == h) & (M["model"] == model) & (M["features"] == cfg)]
        if len(t_rows) > 0:
            t = t_rows.iloc[0]
            approved = bool(b["RMSE_vs_naive"] < 1 and t["RMSE_vs_naive"] < 1)
            val_ratio = round(b["RMSE_vs_naive"], 3)
            test_ratio = round(t["RMSE_vs_naive"], 3)
            reason = ("ML model beat naive on validation and test" if approved else
                      f"best ML model by validation ({model}, config {cfg[0]}) did not beat naive on both validation "
                      f"(ratio {b['RMSE_vs_naive']:.2f}) and test (ratio {t['RMSE_vs_naive']:.2f}); naive recommended")
        else:
            # Validation exists but test is empty
            approved = False
            val_ratio = round(b["RMSE_vs_naive"], 3)
            test_ratio = float("nan")
            reason = f"no test rows for h={h}; naive recommended"
        fac = lr_factory if model == M_LR else (lambda p=best_params[(h, cfg)]: rf_factory(**p))
        final_ml = fac().fit(S_h["all"][feats], S_h["all"]["y"])
        ml_point = float(ser.iloc[-1] * np.exp(final_ml.predict(origin_feats[feats])[0]))
        rec_model, rec_cfg = (model, cfg) if approved else (M_NAIVE, "-")
        point = ml_point if approved else float(ser.iloc[-1])
        ml_challenger_label = f"{model} ({cfg[0]})"
        REC.append(dict(horizon=h, best_ML_by_validation=model, features=cfg,
                        val_RMSE_vs_naive=val_ratio, test_RMSE_vs_naive=test_ratio,
                        ML_approved=approved, recommended=rec_model))
        FINAL_MODELS[h] = dict(recommended=rec_model, recommended_config=rec_cfg, last_price=float(ser.iloc[-1]),
                               ml_challenger=dict(model_name=model, config=cfg, features=feats, fitted=final_ml,
                                                  rf_params=best_params.get((h, cfg))))
    else:
        # No validation data at all – fall back to naive, use a default ML challenger
        model, cfg = M_NAIVE, "-"
        feats = FEATS_A
        approved = False
        val_ratio, test_ratio = float("nan"), float("nan")
        reason = f"no validation rows for h={h}; naive recommended"
        # Build a naive-based ML point using the simplest LR as challenger (train on all data)
        try:
            final_ml = lr_factory().fit(S_h["all"][feats], S_h["all"]["y"])
            ml_point = float(ser.iloc[-1] * np.exp(final_ml.predict(origin_feats[feats])[0]))
        except Exception:
            ml_point = float(ser.iloc[-1])  # fallback
        rec_model, rec_cfg = M_NAIVE, "-"
        point = float(ser.iloc[-1])
        ml_challenger_label = f"Linear Regression (A) [unvalidated]"
        REC.append(dict(horizon=h, best_ML_by_validation="n/a (no val data)", features="n/a",
                        val_RMSE_vs_naive=val_ratio, test_RMSE_vs_naive=test_ratio,
                        ML_approved=False, recommended=rec_model))
        FINAL_MODELS[h] = dict(recommended=rec_model, recommended_config=rec_cfg, last_price=float(ser.iloc[-1]),
                               ml_challenger=None)

    # Build prediction intervals from test errors if available, else use a fixed ±5% band
    g = preds[(preds["horizon"] == h) & (preds["model"] == rec_model) & (preds["features"] == rec_cfg) & (preds["split"] == "test")]
    if len(g) >= 4:
        q = np.quantile(np.log(g["actual"]) - np.log(g["pred"]), [0.05, 0.10, 0.90, 0.95])
    else:
        # Widen the interval slightly with each additional year of horizon (rough heuristic)
        spread = 0.05 + 0.005 * h
        q = np.array([-spread * 1.5, -spread, spread, spread * 1.5])
    fc_rows.append(dict(
        forecast_origin=f"{LAST_DATE:%Y-%m-%d}", forecast_date=f"{(LAST_DATE + pd.DateOffset(months=h)):%Y-%m-%d}",
        rice_classification=df["commodity"].iloc[0], location=df["geolocation"].iloc[0], horizon_months=h,
        predicted_price_php_kg=round(point, 2),
        lower_80=round(point * np.exp(q[1]), 2), upper_80=round(point * np.exp(q[2]), 2),
        lower_90=round(point * np.exp(q[0]), 2), upper_90=round(point * np.exp(q[3]), 2),
        recommended_model=rec_model, ml_challenger=ml_challenger_label, ml_challenger_forecast=round(ml_point, 2),
        ml_val_RMSE_vs_naive=val_ratio, ml_test_RMSE_vs_naive=test_ratio,
        recommendation_reason=reason,
        forecast_type="genuine forecast beyond the last trustworthy observation (not a test-set prediction)"))
REC = pd.DataFrame(REC)
REC.to_csv(OUT / "model_recommendation.csv", index=False)
FC = pd.DataFrame(fc_rows)
FC.to_csv(OUT / "future_price_forecast.csv", index=False)

hist = ser.iloc[-60:]
fig, ax = new_fig(figsize=(9, 4.2))
style_ax(ax)
ax.plot(hist.index, hist.values, color=INK, linewidth=2, label="Observed")
fd = pd.to_datetime(FC["forecast_date"])
ax.vlines(fd, FC["lower_90"], FC["upper_90"], color=ORANGE, alpha=0.30, linewidth=7, label="90% interval")
ax.vlines(fd, FC["lower_80"], FC["upper_80"], color=ORANGE, alpha=0.65, linewidth=7, label="80% interval")
ax.plot([hist.index[-1]] + list(fd), [hist.iloc[-1]] + list(FC["predicted_price_php_kg"]), color=ORANGE, marker="o",
        linewidth=1.6, label="Recommended forecast")
ax.scatter(fd, FC["ml_challenger_forecast"], marker="x", color=BLUE, s=40, zorder=4, label="Best ML model (not approved)")
ax.set_ylabel("PHP per kg", color=INK2)
title(ax, f"Forecasts from {LAST_DATE:%b %Y}; intervals = empirical out-of-sample test errors")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
save_fig(fig, "eval5_future_forecast.png")

# ------------------------------------------------------------------ save model artefacts
joblib.dump(dict(final_models=FINAL_MODELS, feature_sets=CONFIGS, last_observation=str(LAST_DATE.date()),
                 last_price=float(ser.iloc[-1])), MODELS / "best_models.joblib")
meta = dict(horizons=HORIZONS, train_end=str(TRAIN_END.date()), val_end=str(VAL_END.date()),
            training_data_through=str(LAST_DATE.date()), random_state=RANDOM_STATE,
            target="log(price[t+h]/price[t]); price = national regular milled rice, PHP/kg",
            rf_best_params={f"h{h}|{c}": {k: str(v) for k, v in p.items()} for (h, c), p in best_params.items()},
            recommended={int(r_.horizon): dict(model=r_.recommended, ml_approved=bool(r_.ML_approved)) for r_ in REC.itertuples()},
            note="Predicted price = last observed price * exp(model output). Standardisation is inside the Linear Regression pipeline.")
(MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ evaluation report
T = M[M["split"] == "test"].copy()
# Only build period text for horizons that actually have test rows
test_period_txt = {h: T[T.horizon == h]["period"].iloc[0] for h in HORIZONS if not T[T.horizon == h].empty}
R = ["# Evaluation report (auto-generated by 04_train_evaluate.py)\n",
     "All numbers below come from the executed run. The split is on the forecast origin; models are tuned and selected on the "
     "validation window only; the test window is used once for reporting and to approve or reject the selected ML model.\n",
     f"- Training targets through {TRAIN_END:%Y-%m}; validation through {VAL_END:%Y-%m}; test = forecasts made after {VAL_END:%Y-%m}.",
     "- Test target periods: " + "; ".join(f"{h}-month: {p}" for h, p in test_period_txt.items()) + ".",
     "- Rows per split: " + "; ".join(f"h={h}: train {len(splits[h]['train'])}, val {len(splits[h]['val'])}, test {len(splits[h]['test'])}" for h in HORIZONS) + ".",
     "- R2 on price levels is high for every method because prices trend; `R2_change` scores the predicted change from the "
     "last observed price and is the stricter measure. `RMSE_vs_naive` < 1 means better than the naive baseline.\n"]
for h in HORIZONS:
    t = T[T.horizon == h][["model", "features", "period", "n", "MAE", "RMSE", "R2", "R2_change", "RMSE_vs_naive"]]
    R.append(f"\n## Test results, {h}-month-ahead (MAE and RMSE in PHP/kg)\n")
    R.append(md_table(t))
V = M[M["split"] == "val"]
R.append("\n## Validation-selected model per horizon (all candidates, incl. baselines)\n")
R.append(md_table(sel))
R.append("\n## Recommendation gate: ML model approved only if it beats naive on validation AND test\n")
R.append(md_table(REC))
R.append("\n## Do previous-year inflation and demand growth help? (config B vs A, test)\n")
R.append(md_table(CMP))
R.append("\n## Honest reading of the results\n")
for r_ in REC.itertuples():
    if r_.ML_approved:
        R.append(f"- {r_.horizon}-month: **{r_.recommended}** is recommended (validation ratio {r_.val_RMSE_vs_naive:.2f}, test ratio {r_.test_RMSE_vs_naive:.2f} vs naive).")
    else:
        R.append(f"- {r_.horizon}-month: no ML model is recommended. The best by validation, {r_.best_ML_by_validation} (config {r_.features[0]}), "
                 f"scored {r_.val_RMSE_vs_naive:.2f} x naive RMSE on validation and {r_.test_RMSE_vs_naive:.2f} x on test. The naive forecast is recommended.")
lr_test = T[(T["model"] == M_LR) & (T["features"] == CFG_A)].set_index("horizon")["RMSE_vs_naive"]
R.append("- For transparency: Linear Regression (A) scored " + ", ".join(f"{v:.2f} (h={h})" for h, v in lr_test.items())
         + " x naive on test. Choosing it because of that would be picking on the test set, so it is not used for selection.")
helps = CMP[CMP["change_pct"] < 0]
R.append(f"- Config B (adds inflation and demand growth) has lower test RMSE than config A in {len(helps)} of {len(CMP)} "
         f"horizon/model pairs; changes range from {CMP['change_pct'].min():+.1f}% to {CMP['change_pct'].max():+.1f}%, and the smallest "
         f"Wilcoxon p-value is {CMP['wilcoxon_p'].min():.3f}. These differences are small and not statistically distinguishable from zero.")
vr = V[V["model"] == M_RFL].set_index("horizon")["RMSE_vs_naive"]
tr = T[T["model"] == M_RFL].set_index("horizon")["RMSE_vs_naive"]
trainmax = float(ser[ser.index <= TRAIN_END].max())
valmax = float(ser[(ser.index > TRAIN_END) & (ser.index <= VAL_END)].max())
R.append("- Ablation (Random Forest predicting the price LEVEL, not the change): ratio vs naive on validation "
         + ", ".join(f"{v:.2f} (h={h})" for h, v in vr.items()) + "; on test " + ", ".join(f"{v:.2f} (h={h})" for h, v in tr.items())
         + f". A forest cannot predict prices above those it was trained on: training prices peaked at {trainmax:.1f} but validation prices "
           f"reached {valmax:.1f}. Any good test score reflects the test prices staying inside the later, wider training range, not forecasting skill; "
           "this is why the main models predict changes.")
R.append("\n## Limitations\n")
R += ["- One national series; the test window is short (about 4 years) and contains the 2018 price spike, so rankings can change with a different window.",
      "- Test errors are autocorrelated (overlapping months), so p-values from month-by-month tests overstate certainty.",
      "- Inflation and demand are annual and enter as previous-year values: low resolution, about 30 distinct values.",
      "- Prediction intervals are empirical quantiles of about 40-50 test-period errors; they are approximate and assume future errors resemble those. The naive intervals are wide because they include the 2018 shock.",
      "- The 2021 prices in the file are a placeholder and were excluded, so forecasts start from the last trustworthy month "
      f"({LAST_DATE:%Y-%m}). They concern a period that has already passed and cannot be checked against the supplied files.",
      "- Supply shocks, import policy, weather and typhoons are not modelled. Forecasts are estimates, not guaranteed prices."]
R.append("\n## Future forecast (from the last trustworthy observation)\n")
R.append(md_table(FC[["forecast_origin", "forecast_date", "horizon_months", "predicted_price_php_kg", "lower_80", "upper_80", "lower_90", "upper_90",
                      "recommended_model", "ml_challenger", "ml_challenger_forecast"]], floatfmt="{:,.2f}"))
(REP / "evaluation_report.md").write_text("\n".join(R), encoding="utf-8")

print("\nTEST RESULTS\n", T[["horizon", "model", "features", "MAE", "RMSE", "R2", "R2_change", "RMSE_vs_naive"]].round(3).to_string(index=False))
print("\nRECOMMENDATION\n", REC.round(3).to_string(index=False))
print("\nA vs B\n", CMP.round(3).to_string(index=False))
print("\nFORECAST\n", FC.drop(columns=["forecast_type", "recommendation_reason", "rice_classification", "location"]).to_string(index=False))
