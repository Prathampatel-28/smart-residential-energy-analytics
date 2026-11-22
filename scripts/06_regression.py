"""
Stage 4: Regression Analysis & Correlation Modeling (Chapters 9 & 10)
=====================================================================
This script implements the exact regression and correlation pipelines
specified in Chapters 9 & 10 of the report and Appendix A (Sections 8 & 9).

Models Implemented:
1. Simple Linear Regression:
   - Strongest non-random predictor: lights (Pearson r = 0.1973).
   - Fitted equation: Appliances = 88.53 + 2.47 * lights.
   - Test R² = 0.0490 (random 80/20 split, random_state=42).
2. Multiple Linear Regression:
   - Features: Exactly 28 predictors (excluding Appliances, rv1, rv2).
   - Scaler: StandardScaler placed strictly inside Pipeline (no leakage).
   - Split 1 (Random 80/20, random_state=42):
     * Test MAE = 52.58 Wh, Test RMSE = 91.11 Wh, Test R² = 0.1705.
   - Split 2 (Chronological: first 15,788 rows train, last 3,947 rows test):
     * Test MAE = 52.47 Wh, Test RMSE = 86.42 Wh, Test R² = 0.0989.
   - Baseline: Mean-only prediction on chronological test set.

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_CSV = BASE_DIR / "data" / "processed" / "energy_cleaned.csv"
RAW_CSV = BASE_DIR / "data" / "raw" / "energydata_complete.csv"
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
RANDOM_STATE = 42


def save_fig(name: str):
    """Saves figure to reports/figures/."""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def metrics(y_true, y_pred) -> dict:
    """Computes standard regression evaluation metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred))
    }


def run_correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Computes Pearson and Spearman correlation matrices and identifies
    strongest predictors of Appliances (Chapter 9, Tables 21 & 22).
    """
    print("[-] 1. Running Correlation Analysis (Chapter 9)...")
    orig_num = [c for c in pd.read_csv(RAW_CSV, nrows=1).columns if c != "date"]
    
    pear = df[orig_num].corr(method="pearson")
    spear = df[orig_num].corr(method="spearman")

    cmp_ = pd.DataFrame({"Pearson": pear["Appliances"], "Spearman": spear["Appliances"]}).drop("Appliances")
    cmp_["abs_P"] = cmp_["Pearson"].abs()
    cmp_ = cmp_.sort_values("abs_P", ascending=False).drop(columns="abs_P")

    cmp_.round(4).to_csv(TAB_DIR / "corr_with_appliances.csv")
    pear.round(3).to_csv(TAB_DIR / "pearson_matrix.csv")
    spear.round(3).to_csv(TAB_DIR / "spearman_matrix.csv")

    r_lights = float(pear.loc["lights", "Appliances"])
    print(f"    Strongest non-random correlation: lights (Pearson r = {r_lights:.4f})")

    # Heatmaps
    heat_cols = [
        "Appliances", "lights", "T1", "T2", "T3", "T4", "T5", "T6", "T_out",
        "RH_1", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"
    ]
    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap="Greys", vmin=-1, vmax=1,
                annot_kws={"size": 7}, linewidths=0.3, ax=ax, square=True)
    save_fig("fig_corr_heatmap")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(cmp_))
    w = 0.4
    ax.bar(x - w / 2, cmp_["Pearson"], w, color="0.25", label="Pearson")
    ax.bar(x + w / 2, cmp_["Spearman"], w, color="0.7", edgecolor="black", label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_.index, rotation=90, fontsize=7)
    ax.set(ylabel="Correlation with Appliances")
    ax.legend()
    save_fig("fig_corr_bar")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(pear, cmap="Greys", vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": 0.6})
    save_fig("fig_corr_full")

    return cmp_, orig_num


def run_simple_linear_regression(df: pd.DataFrame) -> dict:
    """
    Fits Simple Linear Regression: Appliances ~ lights.
    Matches Chapter 10, Table 24.
    """
    print("[-] 2. Fitting Simple Linear Regression (Appliances ~ lights)...")
    Xs = df[["lights"]]
    y = df["Appliances"]

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=RANDOM_STATE)
    slr = LinearRegression().fit(Xtr, ytr)

    coef = float(slr.coef_[0])
    intercept = float(slr.intercept_)
    pred_te = slr.predict(Xte)
    m_test = metrics(yte, pred_te)

    print(f"    Fitted model: Appliances = {intercept:.2f} + {coef:.2f} * lights")
    print(f"    Test R² = {m_test['R2']:.4f} | Test RMSE = {m_test['RMSE']:.2f} Wh")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(Xte["lights"], yte, s=3, color="0.4", alpha=0.35, label="Test data")
    xx = np.linspace(Xs["lights"].min(), Xs["lights"].max(), 100).reshape(-1, 1)
    ax.plot(xx, slr.predict(pd.DataFrame(xx, columns=["lights"])), "k-", lw=2, label="Fitted line")
    ax.set(xlabel="lights (Wh)", ylabel="Appliances (Wh)")
    ax.legend()
    save_fig("fig_slr")

    return {
        "coef": coef,
        "intercept": intercept,
        "test_metrics": m_test
    }


def run_multiple_linear_regression(df: pd.DataFrame, orig_num: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fits Multiple Linear Regression on 28 features (excluding rv1, rv2)
    using StandardScaler inside a Pipeline. Evaluates Random 80/20 and
    Chronological splits plus mean-only baseline. Matches Chapter 10, Table 25.
    """
    print("[-] 3. Fitting Multiple Linear Regression (28 Features, Random & Chrono Splits)...")
    
    # 28 predictors: 25 raw sensor features + hour, day_of_week, is_weekend
    features = [c for c in orig_num if c not in ["Appliances", "rv1", "rv2"]] + ["hour", "day_of_week", "is_weekend"]
    print(f"    Number of features: {len(features)}")
    
    X = df[features]
    y = df["Appliances"]

    def fit_mlr(Xtr, Xte, ytr, yte):
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
        return model, metrics(ytr, model.predict(Xtr)), metrics(yte, model.predict(Xte))

    # (a) Random 80/20 split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    mlr_rand, m_rand_tr, m_rand_te = fit_mlr(Xtr, Xte, ytr, yte)
    pred_rand = mlr_rand.predict(Xte)

    # (b) Chronological split: first 80% (15,788 rows train), last 20% (3,947 rows test)
    cut = int(len(df) * 0.8)
    mlr_chrono, m_chrono_tr, m_chrono_te = fit_mlr(X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:])
    pred_chrono = mlr_chrono.predict(X.iloc[cut:])

    # Baseline: mean of training set applied to test set
    base_chrono = np.full(len(df) - cut, y.iloc[:cut].mean())
    m_base_chrono = metrics(y.iloc[cut:], base_chrono)

    # Compile comparison table matching Table 25
    comp_rows = [
        {
            "Split / Dataset": "Random 80/20 - Training",
            "MAE (Wh)": round(m_rand_tr["MAE"], 2),
            "RMSE (Wh)": round(m_rand_tr["RMSE"], 2),
            "R² Score": round(m_rand_tr["R2"], 4)
        },
        {
            "Split / Dataset": "Random 80/20 - Testing",
            "MAE (Wh)": round(m_rand_te["MAE"], 2),
            "RMSE (Wh)": round(m_rand_te["RMSE"], 2),
            "R² Score": round(m_rand_te["R2"], 4)
        },
        {
            "Split / Dataset": "Chronological - Training (First 80%)",
            "MAE (Wh)": round(m_chrono_tr["MAE"], 2),
            "RMSE (Wh)": round(m_chrono_tr["RMSE"], 2),
            "R² Score": round(m_chrono_tr["R2"], 4)
        },
        {
            "Split / Dataset": "Chronological - Testing (Last 20%)",
            "MAE (Wh)": round(m_chrono_te["MAE"], 2),
            "RMSE (Wh)": round(m_chrono_te["RMSE"], 2),
            "R² Score": round(m_chrono_te["R2"], 4)
        },
        {
            "Split / Dataset": "Chronological Baseline (Mean Only)",
            "MAE (Wh)": round(m_base_chrono["MAE"], 2),
            "RMSE (Wh)": round(m_base_chrono["RMSE"], 2),
            "R² Score": round(m_base_chrono["R2"], 4)
        }
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_path = TAB_DIR / "regression_model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[-] Model Evaluation Summary saved to: {comp_path}")
    print(comp_df.to_string(index=False))

    # Save standardized coefficients (Table 26)
    coefs = pd.Series(mlr_rand.named_steps["linearregression"].coef_, index=features)
    coef_sorted = coefs.sort_values(key=abs, ascending=False)
    coef_path = TAB_DIR / "mlr_coefficients.csv"
    coef_sorted.round(3).to_csv(coef_path)
    print(f"    Coefficients saved to: {coef_path}")

    # Visualizations: Actual vs Predicted & Residual Plot
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    ax[0].scatter(yte, pred_rand, s=3, color="0.35", alpha=0.35)
    lim = [0, max(yte.max(), pred_rand.max())]
    ax[0].plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax[0].set(xlabel="Actual (Wh)", ylabel="Predicted (Wh)", title="Actual vs predicted")
    ax[0].legend()

    resid = yte - pred_rand
    ax[1].scatter(pred_rand, resid, s=3, color="0.35", alpha=0.35)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set(xlabel="Predicted (Wh)", ylabel="Residual (Wh)", title="Residual plot")
    save_fig("fig_mlr_eval")

    # Time series prediction plot (3 days of test set)
    seg = slice(cut, cut + 144 * 3)
    pc = mlr_chrono.predict(X.iloc[seg])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(df["date"].iloc[seg], y.iloc[seg], "k-", lw=1, label="Actual")
    ax.plot(df["date"].iloc[seg], pc, color="0.5", lw=1.2, label="Predicted")
    ax.set(ylabel="Appliances (Wh)")
    ax.legend()
    save_fig("fig_mlr_timeseries")

    return comp_df, coef_sorted


def main():
    print("================================================================")
    print("STAGE 4: REGRESSION & CORRELATION ANALYSIS (CHAPTERS 9 & 10)")
    print("================================================================")
    df = pd.read_csv(PROC_CSV)
    df["date"] = pd.to_datetime(df["date"])

    cmp_, orig_num = run_correlation_analysis(df)
    run_simple_linear_regression(df)
    run_multiple_linear_regression(df, orig_num)
    print("[-] Regression analysis completed successfully.\n")


if __name__ == "__main__":
    main()
