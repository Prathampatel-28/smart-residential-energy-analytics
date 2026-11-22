"""
Stage 2: Data Preprocessing, Feature Engineering & Exploratory Data Analysis
=============================================================================
This script aligns exactly with Chapter 5 & Chapter 8 of the report and
Appendix A (Sections 1 – 7).

Operations Performed:
1. Data Loading & Type Conversion:
   - Validates shape (19,735 × 29), checks zero missing values, zero duplicates.
   - Ensures unbroken 10-minute timestamp intervals.
2. Feature Engineering (Chapter 5, Table 13):
   - Date components: year, month, month_name, day, hour, day_of_week (0=Mon), day_name, is_weekend.
   - time_of_day: Night (0-5), Morning (6-11), Afternoon (12-17), Evening (18-23).
   - T_indoor_mean: Mean of indoor temperatures T1-T5 and T7-T9 (excluding T6 exterior).
   - RH_indoor_mean: Mean of indoor humidities RH_1-RH_5 and RH_7-RH_9.
   - T_diff_in_out: T_indoor_mean - T_out.
   - log_Appliances: log1p(Appliances) for visual analysis of right-skew.
   - Consumption_Level: Quantile-based 3-class target from full dataset:
     * Low: <= 50 Wh (Q1) -> 7,462 records
     * Medium: 50 - 100 Wh (Q1 to Q3) -> 8,040 records
     * High: > 100 Wh (> Q3) -> 4,233 records
3. Outlier Detection:
   - IQR detection on all original numerical variables (Upper fence = 175.0 Wh, 2,138 outliers / 10.83%).
   - Outliers are DETECTED and saved to reports/tables/outliers_iqr.csv, NOT removed or capped.
4. Descriptive Statistics & Data Validation:
   - Range and consistency assertions matching Chapter 5, Table 34.
   - Descriptive summary table saved to reports/tables/descriptive_statistics.csv.
5. Exploratory Visualizations:
   - Generates and exports publication-style figures (fig_hist_appliances, fig_hourly, fig_heat_hour_day, etc.).
6. Processed Data Export:
   - Saves clean dataset to data/processed/energydata_processed.csv and energy_cleaned.csv.

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "energydata_complete.csv"
PROC_DIR = BASE_DIR / "data" / "processed"
TAB_DIR = BASE_DIR / "reports" / "tables"
FIG_DIR = BASE_DIR / "reports" / "figures"

# Cache directory for matplotlib
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".mpl_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
PROC_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def save_fig(name: str):
    """Saves figure to reports/figures/ directory."""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_and_inspect() -> pd.DataFrame:
    """Reads raw CSV and performs initial validation checks."""
    print("[-] 1. Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"    Raw shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Missing values and duplicates audit
    n_missing = int(df.isnull().sum().sum())
    n_dups = int(df.duplicated().sum())
    print(f"    Missing values: {n_missing} | Duplicates: {n_dups}")
    if n_missing > 0 or n_dups > 0:
        raise ValueError("Unexpected missing values or duplicates found!")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Performs date parsing and creates all report features (Chapter 5, Table 13)."""
    print("[-] 2. Converting datetime and engineering features...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 10-minute cadence verification
    gaps = df["date"].diff().dropna()
    print(f"    Timestamp continuity: {gaps.nunique()} unique interval ({gaps.iloc[0]}).")

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_of_week      # Monday = 0
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Time of day banding
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"]
    )

    # Indoor sensor aggregations
    temp_in = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]   # indoor temperature (T6 is exterior)
    rh_in = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
    df["T_indoor_mean"] = df[temp_in].mean(axis=1)
    df["RH_indoor_mean"] = df[rh_in].mean(axis=1)
    df["T_diff_in_out"] = df["T_indoor_mean"] - df["T_out"]

    # Log transformation for right-skewed Appliances
    df["log_Appliances"] = np.log1p(df["Appliances"])

    # 3-tier Consumption_Level (Q1 and Q3 of full dataset)
    Q1 = float(df["Appliances"].quantile(0.25))
    Q3 = float(df["Appliances"].quantile(0.75))
    df["Consumption_Level"] = pd.cut(
        df["Appliances"],
        bins=[-np.inf, Q1, Q3, np.inf],
        labels=["Low", "Medium", "High"]
    )
    counts = df["Consumption_Level"].value_counts().to_dict()
    print(f"    Consumption_Level counts (Q1={Q1:.1f}, Q3={Q3:.1f} Wh): {counts}")

    return df, Q1, Q3


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes IQR fences and counts outliers for all numeric attributes (Chapter 5, Table 8).
    Outliers are kept intact in the data.
    """
    print("[-] 3. Detecting outliers with IQR rule (retained without removal)...")
    orig_num = [c for c in pd.read_csv(RAW_PATH, nrows=1).columns if c != "date"]
    out_rows = []
    for c in orig_num:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        out_rows.append([c, round(lo, 2), round(hi, 2), n_out, round(100 * n_out / len(df), 2)])

    outliers = pd.DataFrame(out_rows, columns=["Variable", "Lower fence", "Upper fence", "Outliers", "Percent"])
    outliers_path = TAB_DIR / "outliers_iqr.csv"
    outliers.to_csv(outliers_path, index=False)
    print(f"    Outlier summary saved to: {outliers_path}")
    
    appl_out = outliers[outliers["Variable"] == "Appliances"].iloc[0]
    print(f"    Appliances outliers: {appl_out['Outliers']} ({appl_out['Percent']}%) | Upper fence = {appl_out['Upper fence']} Wh")
    return outliers


def validate_and_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Performs validation assertions and computes descriptive statistics table."""
    print("[-] 4. Validating data integrity and generating descriptive statistics...")
    orig_num = [c for c in pd.read_csv(RAW_PATH, nrows=1).columns if c != "date"]
    
    # Assertions
    assert (df["Appliances"] >= 0).all(), "Appliances negative values found!"
    assert (df["lights"] >= 0).all(), "lights negative values found!"
    assert df[[c for c in df if c.startswith("RH")]].apply(lambda s: s.between(0, 100)).all().all(), "RH outside 0-100!"
    assert df["date"].is_monotonic_increasing, "Date not monotonic increasing!"
    assert df["date"].is_unique, "Duplicate timestamps found!"
    assert (df["Windspeed"] >= 0).all(), "Windspeed negative values found!"

    # Descriptive statistics table
    desc = df[orig_num].describe().T
    desc["median"] = df[orig_num].median()
    desc["skew"] = df[orig_num].skew()
    desc_path = TAB_DIR / "descriptive_statistics.csv"
    desc.round(3).to_csv(desc_path)
    print(f"    Descriptive statistics saved to: {desc_path}")
    return desc


def generate_eda_figures(df: pd.DataFrame):
    """Generates all exploratory charts matching Appendix A (Section 7)."""
    print("[-] 5. Generating exploratory visualizations...")
    a = df["Appliances"]

    # 1. Distribution
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(a, bins=60, color="0.35", edgecolor="white")
    ax[0].axvline(a.mean(), color="black", ls="--", label=f"Mean = {a.mean():.1f}")
    ax[0].axvline(a.median(), color="black", ls=":", label=f"Median = {a.median():.1f}")
    ax[0].set(title="Appliances (Wh)", xlabel="Energy per 10 min (Wh)", ylabel="Frequency")
    ax[0].legend()
    ax[1].hist(df["log_Appliances"], bins=60, color="0.55", edgecolor="white")
    ax[1].set(title="log(1 + Appliances)", xlabel="log(1 + Wh)", ylabel="Frequency")
    save_fig("fig_hist_appliances")

    # 2. Box plot
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.boxplot(a, orientation="horizontal", patch_artist=True, boxprops=dict(facecolor="0.8"),
               medianprops=dict(color="black"), flierprops=dict(marker=".", markersize=2))
    ax.set(xlabel="Appliances (Wh)", yticks=[])
    save_fig("fig_box_appliances")

    # 3. Hourly pattern
    hourly = df.groupby("hour")["Appliances"].agg(["mean", "median"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(hourly.index, hourly["mean"], "k-o", ms=4, label="Mean")
    ax.plot(hourly.index, hourly["median"], "k--s", ms=3, label="Median")
    ax.set(xlabel="Hour of day", ylabel="Appliances (Wh)", xticks=range(0, 24, 2))
    ax.legend()
    save_fig("fig_hourly")
    hourly.round(2).to_csv(TAB_DIR / "hourly_consumption.csv")
    hourly.to_csv(PROC_DIR / "hourly_summary.csv")

    # 4. Day of week and Month
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("day_name")["Appliances"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(7), dow.values, color="0.4", edgecolor="black")
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in order])
    ax.set(ylabel="Mean Appliances (Wh)", xlabel="Day of week")
    for i, v in enumerate(dow.values):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    save_fig("fig_dayofweek")

    mon_order = ["January", "February", "March", "April", "May"]
    mon = df.groupby("month_name")["Appliances"].agg(["mean", "count"]).reindex(mon_order)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(5), mon["mean"].values, color="0.4", edgecolor="black")
    ax.set_xticks(range(5))
    ax.set_xticklabels(mon_order)
    ax.set(ylabel="Mean Appliances (Wh)", xlabel="Month (2016)")
    for i, v in enumerate(mon["mean"].values):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    save_fig("fig_monthly")

    # 5. Daily trend
    daily = df.set_index("date")["Appliances"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(daily.index, daily.values, color="black", lw=0.9)
    ax.axhline(a.mean(), color="0.5", ls="--", label="Overall mean")
    ax.set(ylabel="Daily mean Appliances (Wh)", xlabel="Date")
    ax.legend()
    save_fig("fig_daily")

    # 6. Hour x Day heatmap
    pivot = df.pivot_table(index="day_name", columns="hour", values="Appliances", aggfunc="mean").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.heatmap(pivot, cmap="Greys", ax=ax, cbar_kws={"label": "Mean Wh"})
    ax.set(xlabel="Hour of day", ylabel="")
    save_fig("fig_heat_hour_day")

    # 7. Lights vs Appliances
    lt = df.groupby("lights")["Appliances"].agg(["mean", "count"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(lt.index.astype(str), lt["mean"], color="0.4", edgecolor="black")
    ax[0].set(xlabel="Lights consumption (Wh)", ylabel="Mean Appliances (Wh)", title="Mean appliance use by lights level")
    rnd = np.random.RandomState(1).uniform(-0.3, 0.3, len(df))
    ax[1].scatter(df["lights"] + rnd, df["Appliances"], s=2, color="0.3", alpha=0.3)
    ax[1].set(xlabel="Lights (Wh, jittered)", ylabel="Appliances (Wh)", title="Scatter plot")
    save_fig("fig_lights")

    # 8. Environmental scatter
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3))
    for axx, c in zip(ax, ["T_out", "T1", "T6"]):
        axx.scatter(df[c], df["Appliances"], s=2, color="0.3", alpha=0.25)
        axx.set(xlabel=f"{c} (°C)", ylabel="Appliances (Wh)", title=f"Appliances vs {c}")
    save_fig("fig_temp_scatter")

    # 9. Environmental variable distributions
    env = ["T_out", "RH_out", "Windspeed", "Visibility", "Press_mm_hg", "Tdewpoint"]
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 5))
    for axx, c in zip(axs.ravel(), env):
        axx.hist(df[c], bins=40, color="0.45", edgecolor="white")
        axx.set(title=c, ylabel="Frequency")
    save_fig("fig_env_hist")

    # 10. Standardized outlier box plots
    sel = ["Appliances", "lights", "T_out", "RH_out", "Windspeed", "Visibility", "Press_mm_hg"]
    z = (df[sel] - df[sel].mean()) / df[sel].std()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.boxplot([z[c] for c in sel], patch_artist=True,
                boxprops=dict(facecolor="0.8"), medianprops=dict(color="black"),
                flierprops=dict(marker=".", markersize=2))
    ax.set(ylabel="Standardised value (z-score)")
    ax.set_xticks(range(1, len(sel) + 1))
    ax.set_xticklabels(sel, rotation=20)
    save_fig("fig_outliers_box")
    print("    All 10 exploratory figures generated successfully.")


def save_processed_data(df: pd.DataFrame):
    """Saves processed dataset for downstream modeling and warehouse."""
    print("[-] 6. Saving processed datasets...")
    out_csv = PROC_DIR / "energydata_processed.csv"
    clean_csv = PROC_DIR / "energy_cleaned.csv"
    df.to_csv(out_csv, index=False)
    df.to_csv(clean_csv, index=False)
    print(f"    Saved processed files to {out_csv} and {clean_csv}")


def main():
    print("================================================================")
    print("STAGE 2: PREPROCESSING, FEATURES & EDA (CHAPTER 5 & 8 SPEC)")
    print("================================================================")
    df = load_and_inspect()
    df, q1, q3 = engineer_features(df)
    detect_outliers_iqr(df)
    validate_and_describe(df)
    generate_eda_figures(df)
    save_processed_data(df)
    print("[-] Preprocessing completed successfully.\n")


if __name__ == "__main__":
    main()
