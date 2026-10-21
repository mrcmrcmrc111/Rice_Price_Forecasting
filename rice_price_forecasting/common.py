"""Shared configuration and helpers for the rice price forecasting project."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REP = OUT / "reports"
MODELS = ROOT / "models"
for _d in (DATA, OUT, FIG, REP, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

PRICE_FILE = DATA / "preprocessed_rice_retailprice.csv"
DEMAND_FILE = DATA / "preprocessed_rice_demand.csv"
INFLATION_FILE = DATA / "preprocessed_inflation.csv"
INTEGRATED_CSV = OUT / "integrated_rice_price_dataset.csv"

# ----------------------------------------------------------------------------
# Modelling configuration (edit here, then rerun the scripts in order)
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
HORIZONS = list(range(1, 61))                  # months ahead (1‑through‑60 months forecast)
# Chronological split on the FORECAST ORIGIN (the last month whose price is known when the forecast is made).
#   train      : rows whose target month is <= TRAIN_END
#   validation : origins after TRAIN_END and target <= VAL_END   (hyperparameter tuning and model selection)
#   test       : origins after VAL_END (targets 2017-01 .. 2020-12); never used for tuning or selection
# For the test evaluation the models are refit on train + validation rows (target <= VAL_END).
# This guarantees no model is trained on a price that had not yet happened when a forecast was made.
TRAIN_END = pd.Timestamp("2012-12-01")
VAL_END = pd.Timestamp("2016-12-01")
# A year in which all 12 monthly prices are identical is treated as a placeholder, not data.
EXCLUDE_FLAT_PRICE_YEARS = True
INFLATION_COL = "headline_inflation_worldbank_pct"
DEMAND_COL = "ut total net food disposable"

# ----------------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"


def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def new_fig(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, color=INK, fontweight="bold")


def md_table(df: pd.DataFrame, floatfmt="{:,.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
