"""
Stage 6: Decision Tree Analysis (Chapter 12 Specification)
===========================================================
This script implements the exact classification and regression decision trees
specified in Chapter 12 and Appendix A (Section 11).

Models Implemented:
1. Classification Tree (Consumption_Level: Low / Medium / High):
   - Algorithm: CART DecisionTreeClassifier (criterion='gini').
   - Hyperparameters: max_depth=5, min_samples_leaf=50, random_state=42.
   - Evaluation on Stratified 80/20 Split:
     * Test Accuracy = 59.1% (vs 40.7% majority baseline).
   - Chronological Split (first 15,788 train, last 3,947 test):
     * Test Accuracy = 54.4%.
   - Artifacts: Confusion matrix, Classification report, Feature importances.
2. Regression Tree (Continuous Appliances Wh):
   - Algorithm: DecisionTreeRegressor.
   - Hyperparameters: max_depth=6, min_samples_leaf=50, random_state=42.
   - Evaluation on Random 80/20 Split:
     * Test MAE = 48.36 Wh, Test RMSE = 89.50 Wh, Test R² = 0.1995.
   - Chronological Split (first 15,788 train, last 3,947 test):
     * Test RMSE = 139.80 Wh, Test R² = -1.3579 (honest reporting of seasonal drift).

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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

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
    """Computes regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred))
    }


def run_classification_tree(df: pd.DataFrame, tree_features: list[str]) -> dict:
    """
    Fits 3-class classification tree on Consumption_Level (Low, Medium, High).
    Matches Chapter 12, Tables 32 & 33.
    """
    print("[-] 1. Training Classification Tree (Gini, max_depth=5, min_samples_leaf=50)...")
    Xt = df[tree_features]
    yc = df["Consumption_Level"].astype(str)

    # Stratified 80/20 Split
    Xtr, Xte, ytr, yte = train_test_split(Xt, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc)
    clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE).fit(Xtr, ytr)
    
    pred_te = clf.predict(Xte)
    pred_tr = clf.predict(Xtr)

    acc_te = float(accuracy_score(yte, pred_te))
    acc_tr = float(accuracy_score(ytr, pred_tr))
    baseline = float(yte.value_counts(normalize=True).max())

    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(yte, pred_te, labels=labels)
    cr = classification_report(yte, pred_te, labels=labels, output_dict=True)

    print(f"    Random Split Test Accuracy: {acc_te * 100:.1f}% (vs {baseline * 100:.1f}% majority baseline)")

    # Chronological Split: first 80% (15,788) train, last 20% (3,947) test
    cut = int(len(df) * 0.8)
    clf_t = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE).fit(Xt.iloc[:cut], yc.iloc[:cut])
    pred_chrono = clf_t.predict(Xt.iloc[cut:])
    acc_chrono = float(accuracy_score(yc.iloc[cut:], pred_chrono))
    print(f"    Chronological Split Test Accuracy: {acc_chrono * 100:.1f}%")

    # Feature importances
    imp = pd.Series(clf.feature_importances_, index=tree_features).sort_values(ascending=False)
    imp_path = TAB_DIR / "tree_feature_importance.csv"
    imp.round(4).to_csv(imp_path)
    print(f"    Feature importances saved to: {imp_path}")

    # Save classification report
    cr_df = pd.DataFrame(cr).T
    cr_path = TAB_DIR / "classification_report.csv"
    cr_df.round(4).to_csv(cr_path)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.imshow(cm, cmap="Greys")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="Actual class")
    ax.grid(False)
    save_fig("fig_confusion")

    # Plot Shallow Tree (depth=3)
    shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=RANDOM_STATE).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_tree(shallow, feature_names=tree_features, class_names=list(shallow.classes_),
              filled=False, rounded=True, fontsize=7, ax=ax, impurity=True, proportion=False)
    save_fig("fig_tree")

    # Plot Feature Importance Barh
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    imp.head(10)[::-1].plot.barh(ax=ax, color="0.4", edgecolor="black")
    ax.set(xlabel="Gini importance")
    save_fig("fig_tree_importance")

    return {
        "acc_test": acc_te,
        "acc_chrono": acc_chrono,
        "baseline": baseline,
        "imp": imp
    }


def run_regression_tree(df: pd.DataFrame, tree_features: list[str]) -> pd.DataFrame:
    """
    Fits Regression Tree for continuous Appliances (max_depth=6, min_samples_leaf=50).
    Matches Chapter 12, Table 35.
    """
    print("[-] 2. Training Regression Tree (max_depth=6, min_samples_leaf=50)...")
    Xt = df[tree_features]
    yr = df["Appliances"]

    # Random 80/20 Split
    Xtr, Xte, ytr, yte = train_test_split(Xt, yr, test_size=0.2, random_state=RANDOM_STATE)
    reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE).fit(Xtr, ytr)

    m_rand_tr = metrics(ytr, reg.predict(Xtr))
    m_rand_te = metrics(yte, reg.predict(Xte))
    print(f"    Random Split Test: R² = {m_rand_te['R2']:.4f} | RMSE = {m_rand_te['RMSE']:.2f} Wh")

    # Chronological Split
    cut = int(len(df) * 0.8)
    reg_t = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE).fit(Xt.iloc[:cut], yr.iloc[:cut])
    m_chrono_te = metrics(yr.iloc[cut:], reg_t.predict(Xt.iloc[cut:]))
    print(f"    Chronological Split Test: R² = {m_chrono_te['R2']:.4f} | RMSE = {m_chrono_te['RMSE']:.2f} Wh")

    # Tree comparison table
    tree_metrics_df = pd.DataFrame([
        {
            "Model / Split": "Regression Tree - Random 80/20 Train",
            "MAE (Wh)": round(m_rand_tr["MAE"], 2),
            "RMSE (Wh)": round(m_rand_tr["RMSE"], 2),
            "R² Score": round(m_rand_tr["R2"], 4)
        },
        {
            "Model / Split": "Regression Tree - Random 80/20 Test",
            "MAE (Wh)": round(m_rand_te["MAE"], 2),
            "RMSE (Wh)": round(m_rand_te["RMSE"], 2),
            "R² Score": round(m_rand_te["R2"], 4)
        },
        {
            "Model / Split": "Regression Tree - Chronological Test",
            "MAE (Wh)": round(m_chrono_te["MAE"], 2),
            "RMSE (Wh)": round(m_chrono_te["RMSE"], 2),
            "R² Score": round(m_chrono_te["R2"], 4)
        }
    ])
    metrics_path = TAB_DIR / "decision_tree_metrics.csv"
    tree_metrics_df.to_csv(metrics_path, index=False)
    print(f"    Regression tree metrics saved to: {metrics_path}")
    print(tree_metrics_df.to_string(index=False))

    return tree_metrics_df


def main():
    print("================================================================")
    print("STAGE 6: DECISION TREE ANALYSIS (CHAPTER 12 SPECIFICATION)")
    print("================================================================")
    df = pd.read_csv(PROC_CSV)
    df["date"] = pd.to_datetime(df["date"])

    orig_num = [c for c in pd.read_csv(RAW_CSV, nrows=1).columns if c != "date"]
    tree_features = [c for c in orig_num if c not in ["Appliances", "rv1", "rv2"]] + ["hour", "day_of_week", "is_weekend"]

    run_classification_tree(df, tree_features)
    run_regression_tree(df, tree_features)
    print("[-] Decision tree analysis completed successfully.\n")


if __name__ == "__main__":
    main()
