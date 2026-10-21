"""Phase 3: clean the three datasets and integrate them into one monthly dataset.

Output: outputs/integrated_rice_price_dataset.csv and outputs/reports/cleaning_log.csv
Every step records rows before/after so nothing is dropped silently.
"""
import numpy as np
import pandas as pd

from common import *

log = []


def record(step, before, after, note=""):
    log.append(dict(step=step, rows_before=before, rows_after=after, rows_removed=before - after, note=note))
    print(f"[{step}] {before} -> {after}  {note}")


# ------------------------------------------------------------------ price (monthly, primary)
price = pd.read_csv(PRICE_FILE)
n0 = len(price)
price.columns = [c.strip().lower() for c in price.columns]
price["date"] = pd.to_datetime(price["date"], format="%Y-%m-%d", errors="raise")
price["retail_price"] = pd.to_numeric(price["retail_price"], errors="raise")
record("price: load and parse", n0, len(price), "date -> datetime, retail_price -> float")

n = len(price)
price = price.drop_duplicates()
record("price: drop exact duplicate rows", n, len(price))

dup_keys = price.duplicated(["geolocation", "commodity", "date"]).sum()
assert dup_keys == 0, f"{dup_keys} duplicated (geolocation, commodity, date) keys - decide how to aggregate"

n = len(price)
price = price[price["retail_price"] > 0]
record("price: drop non-positive prices", n, len(price))

# one location, one classification -> single national series (checked in 01_inspect_data.py)
assert price["geolocation"].nunique() == 1 and price["commodity"].nunique() == 1
meta_loc, meta_com = price["geolocation"].iloc[0], price["commodity"].iloc[0]

monthly = (price.sort_values("date").set_index("date")["retail_price"].rename("price_php_kg_raw").to_frame())
full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
missing_months = len(full_idx.difference(monthly.index))
monthly = monthly.reindex(full_idx)
monthly.index.name = "date"
record("price: reindex to complete monthly calendar", len(price), len(monthly),
       f"{missing_months} missing months (not interpolated)")

# placeholder years: all 12 months identical
yr = monthly["price_php_kg_raw"].groupby(monthly.index.year)
flat_years = [int(y) for y, g in yr if g.notna().sum() == 12 and g.nunique() == 1]
monthly["flat_year_flag"] = monthly.index.year.isin(flat_years)
monthly["price_php_kg"] = monthly["price_php_kg_raw"].where(~monthly["flat_year_flag"]) if EXCLUDE_FLAT_PRICE_YEARS \
    else monthly["price_php_kg_raw"]
n_flag = int(monthly["flat_year_flag"].sum())
record("price: flag placeholder years", len(monthly), len(monthly) - (n_flag if EXCLUDE_FLAT_PRICE_YEARS else 0),
       f"years {flat_years}: {n_flag} months set to missing (target series ends {monthly['price_php_kg'].last_valid_index():%Y-%m})"
       if EXCLUDE_FLAT_PRICE_YEARS else f"years {flat_years} flagged but kept")

# ------------------------------------------------------------------ demand (annual)
demand = pd.read_csv(DEMAND_FILE)
demand.columns = [c.strip().lower() for c in demand.columns]
n = len(demand)
demand = demand[demand["commodity"].str.strip().str.lower() == "rice"]
record("demand: keep rice rows only", n, len(demand), "corn rows dropped (out of scope)")
assert demand["year"].is_unique
demand = demand.sort_values("year").set_index("year")
dcol = DEMAND_COL.lower()
annual = pd.DataFrame(index=demand.index)
annual["demand_kmt"] = pd.to_numeric(demand[dcol], errors="raise")
annual["demand_growth_pct"] = annual["demand_kmt"].pct_change() * 100

# ------------------------------------------------------------------ inflation (annual)
infl = pd.read_csv(INFLATION_FILE)
infl.columns = [c.strip().lower() for c in infl.columns]
assert infl["year"].is_unique
infl = infl.set_index("year").sort_index()
annual["inflation_wb_pct"] = pd.to_numeric(infl[INFLATION_COL], errors="raise")
record("inflation: select usable column", len(infl), len(infl),
       f"kept '{INFLATION_COL}'; other inflation/CPI columns are repeated filler values")

# ------------------------------------------------------------------ integrate: previous-year annual values -> months
# A month in calendar year Y is given the annual figures for year Y-1, which are known before Y starts.
lagged = annual.rename(columns={
    "demand_kmt": "prev_year_demand_kmt",
    "demand_growth_pct": "prev_year_demand_growth_pct",
    "inflation_wb_pct": "prev_year_inflation_pct"})
lagged.index = lagged.index + 1              # value for year Y-1 is attached to year Y
lagged.index.name = "year"

df = monthly.reset_index()
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
n = len(df)
df = df.merge(lagged.reset_index(), on="year", how="left", validate="m:1")
assert len(df) == n, "join changed the number of monthly rows"
record("integrate: attach previous-year demand and inflation", n, len(df), "many-to-one join; row count unchanged")

df.insert(1, "geolocation", meta_loc)
df.insert(2, "commodity", meta_com)
df = df.sort_values("date").reset_index(drop=True)
assert df["date"].is_monotonic_increasing and df["date"].is_unique

cols = ["date", "year", "month", "geolocation", "commodity", "price_php_kg_raw", "flat_year_flag", "price_php_kg",
        "prev_year_inflation_pct", "prev_year_demand_kmt", "prev_year_demand_growth_pct"]
df = df[cols]
df.to_csv(INTEGRATED_CSV, index=False, date_format="%Y-%m-%d")

usable = df.dropna(subset=["price_php_kg"])
record("final: usable monthly target observations", len(df), len(usable),
       f"{usable['date'].min():%Y-%m} to {usable['date'].max():%Y-%m}")
print("\nMissing values in the integrated dataset:\n", df.isna().sum().to_string())
print(f"\nSaved {INTEGRATED_CSV}")

pd.DataFrame(log).to_csv(REP / "cleaning_log.csv", index=False)
(REP / "cleaning_log.md").write_text("# Cleaning log\n\n" + md_table(pd.DataFrame(log)), encoding="utf-8")
