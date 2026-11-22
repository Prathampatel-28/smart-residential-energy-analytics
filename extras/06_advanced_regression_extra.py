"""
Stage 5: Regression Modeling & Diagnostic Evaluation
=====================================================
This script implements comprehensive regression modeling for appliance energy
consumption (continuous target: Appliances) and logistic regression for peak
classification (binary target: high_usage).

Key Methodological Safeguards:
1. Strict Chronological Split:
   - Training period: Jan 11, 2016 to Apr 30, 2016 (15,882 observations, ~80.5%)
   - Test period: May 01, 2016 to May 27, 2016 (3,853 observations, ~19.5%)
   - NEVER use random shuffling to avoid temporal look-ahead leakage.
2. Leakage Prevention:
   - Scalers and encoders fit solely on the training set.
3. Models Evaluated:
   - Simple Linear Regression (Outdoor Temp T_out benchmark)
   - Multiple Linear Regression (Full 30-feature vector)
   - Polynomial Regression (Degree 2 vs 3 tuned on validation block)
   - Ridge & Lasso (Alpha tuned via TimeSeriesSplit CV)
   - Scratch Gradient Descent Linear Regression (NumPy with loss history plot)
   - Logistic Regression (high_usage classification: ROC-AUC, Precision, Recall)
4. Honest Reporting:
   - Explicitly explains the low-to-moderate R2 expected in 10-min smart-meter data.

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "energy_cleaned.csv"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
REPORTS_FIGURES = BASE_DIR / "reports" / "figures"

# Matplotlib cache directory
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".mpl_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)


class ScratchGradientDescentRegressor:
    """
    Ordinary Least Squares Linear Regression implemented from scratch
    using Batch Gradient Descent with vectorized NumPy operations.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1500, tolerance: float = 1e-7):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        # Prepend bias column of ones
        X_b = np.c_[np.ones((m, 1)), X]
        self.weights = np.zeros(n + 1)
        self.loss_history = []

        prev_loss = float("inf")
        for i in range(self.n_iterations):
            predictions = X_b.dot(self.weights)
            errors = predictions - y
            # Mean Squared Error loss
            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            # Gradient vector: (2/m) * X_b^T * errors
            gradients = (2.0 / m) * X_b.T.dot(errors)
            self.weights -= self.lr * gradients

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b.dot(self.weights)


def prepare_features_and_splits(df: pd.DataFrame):
    """
    Splits data chronologically and prepares standardized feature matrices.
    """
    print("[-] Chronological Train-Test Split (Jan-Apr 2016 Train, May 2016 Test)...")
    
    # Feature columns
    indoor_temp = [f"T{i}" for i in range(1, 10)]
    indoor_rh = [f"RH_{i}" for i in range(1, 10)]
    weather_cols = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    occupancy_cols = ["lights"]
    time_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]

    feature_cols = indoor_temp + indoor_rh + weather_cols + occupancy_cols + time_cols
    print(f"    Feature vector dimension: {len(feature_cols)} predictors.")

    # Time-based splitting
    train_mask = df["datetime"] < pd.Timestamp("2016-05-01")
    test_mask = df["datetime"] >= pd.Timestamp("2016-05-01")

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    print(f"    Train set: {len(df_train)} rows ({df_train['datetime'].min()} to {df_train['datetime'].max()})")
    print(f"    Test set:  {len(df_test)} rows ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    X_train_raw = df_train[feature_cols].values
    X_test_raw = df_test[feature_cols].values

    y_train = df_train["Appliances"].values
    y_test = df_test["Appliances"].values

    y_train_high = df_train["high_usage"].values
    y_test_high = df_test["high_usage"].values

    # Fit scaler strictly on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return (
        feature_cols,
        df_train, df_test,
        X_train_raw, X_test_raw,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        y_train_high, y_test_high
    )


def evaluate_regression_models(
    feature_cols: list[str],
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    X_train_raw: np.ndarray, X_test_raw: np.ndarray,
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Trains and compares all regression architectures."""
    model_results = []
    predictions_dict = {}

    # -------------------------------------------------------------
    # 1. Simple Linear Regression (Baseline: Outdoor Temp T_out)
    # -------------------------------------------------------------
    print("[-] 1. Training Simple Linear Regression (Predictor: T_out)...")
    idx_tout = feature_cols.index("T_out")
    X_tr_tout = X_train_raw[:, [idx_tout]]
    X_te_tout = X_test_raw[:, [idx_tout]]

    lr_simple = LinearRegression()
    lr_simple.fit(X_tr_tout, y_train)
    y_pred_simple = lr_simple.predict(X_te_tout)

    rmse_simple = float(np.sqrt(mean_squared_error(y_test, y_pred_simple)))
    mae_simple = float(mean_absolute_error(y_test, y_pred_simple))
    r2_simple = float(r2_score(y_test, y_pred_simple))
    predictions_dict["Simple Linear (T_out)"] = y_pred_simple

    model_results.append({
        "Model Name": "Simple Linear Regression (T_out)",
        "Parameters / Hyperparameters": "Intercept & 1 Slope (T_out)",
        "Test RMSE (Wh)": round(rmse_simple, 2),
        "Test MAE (Wh)": round(mae_simple, 2),
        "Test R² Score": round(r2_simple, 4),
        "Notes": "Single-variable baseline benchmark"
    })

    # -------------------------------------------------------------
    # 2. Multiple Linear Regression (All 30 Features)
    # -------------------------------------------------------------
    print("[-] 2. Training Multiple Linear Regression (All Features)...")
    lr_multi = LinearRegression()
    lr_multi.fit(X_train_scaled, y_train)
    y_pred_multi = lr_multi.predict(X_test_scaled)

    rmse_multi = float(np.sqrt(mean_squared_error(y_test, y_pred_multi)))
    mae_multi = float(mean_absolute_error(y_test, y_pred_multi))
    r2_multi = float(r2_score(y_test, y_pred_multi))
    predictions_dict["Multiple Linear"] = y_pred_multi

    model_results.append({
        "Model Name": "Multiple Linear Regression",
        "Parameters / Hyperparameters": "OLS on 30 Standardized Predictors",
        "Test RMSE (Wh)": round(rmse_multi, 2),
        "Test MAE (Wh)": round(mae_multi, 2),
        "Test R² Score": round(r2_multi, 4),
        "Notes": "Full-rank OLS linear combination"
    })

    # Save OLS Coefficients
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": lr_multi.coef_,
        "Abs_Coefficient": np.abs(lr_multi.coef_)
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    coef_path = REPORTS_TABLES / "regression_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"    Coefficients table saved to: {coef_path}")

    # -------------------------------------------------------------
    # 3. Polynomial Regression (Degree 2 vs 3 with Validation Block Tuning)
    # -------------------------------------------------------------
    print("[-] 3. Tuning Polynomial Regression on Validation Block (Apr 16 - Apr 30)...")
    # To prevent combinatorial explosion (30^3 > 27,000 features), we apply polynomial
    # expansion to top 6 principal drivers (lights, T_out, T2, RH_1, hour_sin, hour_cos)
    poly_feature_names = ["lights", "T_out", "T2", "RH_1", "hour_sin", "hour_cos"]
    poly_indices = [feature_cols.index(col) for col in poly_feature_names]

    val_split_date = pd.Timestamp("2016-04-16")
    val_subtrain_mask = df_train["datetime"] < val_split_date
    val_block_mask = df_train["datetime"] >= val_split_date

    X_subtr_raw = df_train.loc[val_subtrain_mask, poly_feature_names].values
    y_subtr = df_train.loc[val_subtrain_mask, "Appliances"].values
    X_val_raw = df_train.loc[val_block_mask, poly_feature_names].values
    y_val = df_train.loc[val_block_mask, "Appliances"].values

    poly_scores = {}
    for deg in [2, 3]:
        poly_t = PolynomialFeatures(degree=deg, include_bias=False)
        X_subtr_poly = poly_t.fit_transform(X_subtr_raw)
        X_val_poly = poly_t.transform(X_val_raw)
        
        m_poly = LinearRegression()
        m_poly.fit(X_subtr_poly, y_subtr)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, m_poly.predict(X_val_poly))))
        poly_scores[deg] = val_rmse
        print(f"    Degree {deg} Validation RMSE: {val_rmse:.2f} Wh ({X_subtr_poly.shape[1]} features)")

    best_deg = min(poly_scores, key=poly_scores.get)
    print(f"    Selected Optimal Degree: {best_deg} (lower validation error, avoids degree 3 overfitting).")

    # Fit best degree on entire training set and evaluate on test set
    poly_transformer = PolynomialFeatures(degree=best_deg, include_bias=False)
    X_tr_poly = poly_transformer.fit_transform(df_train[poly_feature_names].values)
    X_te_poly = poly_transformer.transform(df_test[poly_feature_names].values)

    poly_model = LinearRegression()
    poly_model.fit(X_tr_poly, y_train)
    y_pred_poly = poly_model.predict(X_te_poly)

    rmse_poly = float(np.sqrt(mean_squared_error(y_test, y_pred_poly)))
    mae_poly = float(mean_absolute_error(y_test, y_pred_poly))
    r2_poly = float(r2_score(y_test, y_pred_poly))
    predictions_dict["Polynomial Regression"] = y_pred_poly

    model_results.append({
        "Model Name": f"Polynomial Regression (Degree {best_deg})",
        "Parameters / Hyperparameters": f"Degree={best_deg} on top 6 features ({X_tr_poly.shape[1]} terms)",
        "Test RMSE (Wh)": round(rmse_poly, 2),
        "Test MAE (Wh)": round(mae_poly, 2),
        "Test R² Score": round(r2_poly, 4),
        "Notes": "Non-linear interaction terms on primary drivers"
    })

    # -------------------------------------------------------------
    # 4. Ridge Regression (L2 Regularization with TimeSeriesSplit CV)
    # -------------------------------------------------------------
    print("[-] 4. Tuning Ridge Regression via TimeSeriesSplit CV...")
    alphas = np.logspace(-2, 4, 30)
    tscv = TimeSeriesSplit(n_splits=5)
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_root_mean_squared_error")
    ridge_cv.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_cv.predict(X_test_scaled)

    rmse_ridge = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    mae_ridge = float(mean_absolute_error(y_test, y_pred_ridge))
    r2_ridge = float(r2_score(y_test, y_pred_ridge))
    predictions_dict["Ridge Regression"] = y_pred_ridge

    model_results.append({
        "Model Name": "Ridge Regression (L2)",
        "Parameters / Hyperparameters": f"Tuned alpha = {ridge_cv.alpha_:.3f} (TimeSeriesSplit 5-fold)",
        "Test RMSE (Wh)": round(rmse_ridge, 2),
        "Test MAE (Wh)": round(mae_ridge, 2),
        "Test R² Score": round(r2_ridge, 4),
        "Notes": "L2 regularization shrinks collinear weights"
    })

    # -------------------------------------------------------------
    # 5. Lasso Regression (L1 Regularization with TimeSeriesSplit CV)
    # -------------------------------------------------------------
    print("[-] 5. Tuning Lasso Regression via TimeSeriesSplit CV...")
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=3000, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_cv.predict(X_test_scaled)

    rmse_lasso = float(np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
    mae_lasso = float(mean_absolute_error(y_test, y_pred_lasso))
    r2_lasso = float(r2_score(y_test, y_pred_lasso))
    predictions_dict["Lasso Regression"] = y_pred_lasso

    n_zero_coef = int(np.sum(lasso_cv.coef_ == 0))
    model_results.append({
        "Model Name": "Lasso Regression (L1)",
        "Parameters / Hyperparameters": f"Tuned alpha = {lasso_cv.alpha_:.4f} ({n_zero_coef} features zeroed)",
        "Test RMSE (Wh)": round(rmse_lasso, 2),
        "Test MAE (Wh)": round(mae_lasso, 2),
        "Test R² Score": round(r2_lasso, 4),
        "Notes": "Sparse feature selection via L1 penalty"
    })

    # -------------------------------------------------------------
    # 6. Gradient Descent from Scratch (NumPy implementation)
    # -------------------------------------------------------------
    print("[-] 6. Training Gradient Descent Linear Regression from Scratch...")
    gd_regressor = ScratchGradientDescentRegressor(learning_rate=0.03, n_iterations=1200)
    gd_regressor.fit(X_train_scaled, y_train)
    y_pred_gd = gd_regressor.predict(X_test_scaled)

    rmse_gd = float(np.sqrt(mean_squared_error(y_test, y_pred_gd)))
    mae_gd = float(mean_absolute_error(y_test, y_pred_gd))
    r2_gd = float(r2_score(y_test, y_pred_gd))
    predictions_dict["Scratch Gradient Descent"] = y_pred_gd

    model_results.append({
        "Model Name": "Scratch Gradient Descent Regressor",
        "Parameters / Hyperparameters": "lr=0.03, 1200 epochs (Vectorized NumPy)",
        "Test RMSE (Wh)": round(rmse_gd, 2),
        "Test MAE (Wh)": round(mae_gd, 2),
        "Test R² Score": round(r2_gd, 4),
        "Notes": "Pure first-principles mathematical implementation"
    })

    # Plot GD Loss History
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(gd_regressor.loss_history) + 1), gd_regressor.loss_history, color="#d62728", linewidth=2.0)
    plt.title("Scratch Gradient Descent: MSE Loss Convergence Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration (Epoch)", fontsize=10)
    plt.ylabel("Training Mean Squared Error (Wh²)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    loss_path = REPORTS_FIGURES / "10_gradient_descent_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"    Gradient descent loss curve saved to: {loss_path}")

    # Compile and save results
    comp_df = pd.DataFrame(model_results)
    comp_path = REPORTS_TABLES / "regression_model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[-] Model Comparison Table saved to: {comp_path}")
    print(comp_df[["Model Name", "Test RMSE (Wh)", "Test MAE (Wh)", "Test R² Score"]].to_string(index=False))

    return comp_df, coef_df, predictions_dict


def evaluate_logistic_regression(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
    y_train_high: np.ndarray, y_test_high: np.ndarray
) -> pd.DataFrame:
    """Trains and evaluates Logistic Regression for high_usage classification."""
    print("\n[-] 7. Training Logistic Regression for Peak Demand Classification (high_usage)...")
    
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train_high)
    
    y_pred_class = log_reg.predict(X_test_scaled)
    y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    auc = float(roc_auc_score(y_test_high, y_pred_proba))
    prec = float(precision_score(y_test_high, y_pred_class, zero_division=0))
    rec = float(recall_score(y_test_high, y_pred_class, zero_division=0))
    f1 = float(f1_score(y_test_high, y_pred_class, zero_division=0))
    cm = confusion_matrix(y_test_high, y_pred_class)

    log_metrics = pd.DataFrame([{
        "Model": "Logistic Regression (high_usage > 100 Wh)",
        "ROC-AUC Score": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1])
    }])
    log_csv = REPORTS_TABLES / "logistic_regression_metrics.csv"
    log_metrics.to_csv(log_csv, index=False)
    print(f"    Logistic metrics saved to: {log_csv}")
    print(log_metrics.to_string(index=False))

    # ROC and Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_high, y_pred_proba)
    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2.2, label=f"Logistic Regression (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", label="Random Classifier (AUC = 0.500)")
    axes[0].set_title("ROC Curve: High Usage Peak Classification", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Normal (<=100 Wh)", "High (>100 Wh)"],
                yticklabels=["Normal (<=100 Wh)", "High (>100 Wh)"])
    axes[1].set_title("Confusion Matrix on Unseen May 2016 Test Set", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("Actual Class")

    plt.tight_layout()
    fig_path = REPORTS_FIGURES / "11_logistic_regression_eval.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"    Logistic evaluation figure saved to: {fig_path}")

    return log_metrics


def plot_residuals_and_actual_vs_predicted(y_test: np.ndarray, predictions_dict: dict):
    """
    Plots Actual vs Predicted values and Residuals distribution for
    Multiple Linear Regression.
    """
    print("[-] Generating Regression Diagnostic Plots (Actual vs Predicted & Residuals)...")
    y_pred = predictions_dict["Multiple Linear"]
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Actual vs Predicted scatter
    axes[0].scatter(y_test, y_pred, alpha=0.25, color="#2b5c8f", s=18, edgecolors="none")
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=1.8, label="Ideal Perfect Fit (y = x)")
    axes[0].set_title("Multiple Linear Regression: Actual vs Predicted", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Actual Appliance Energy (Wh)")
    axes[0].set_ylabel("Predicted Appliance Energy (Wh)")
    axes[0].set_xlim(0, 800)
    axes[0].set_ylim(0, 500)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Residual Distribution
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].axvline(0, color="#d62728", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residuals Distribution (Mean Error = {residuals.mean():.2f} Wh)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual (Actual - Predicted Wh)")
    axes[1].set_ylabel("Count")
    axes[1].set_xlim(-250, 400)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = REPORTS_FIGURES / "12_regression_residuals_and_actual_vs_pred.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"    Diagnostic figures saved to: {fig_path}")


def explain_low_r2():
    """Prints plain-English explanation for viva preparation."""
    explanation = """
================================================================================
CRITICAL ACADEMIC DISCUSSION: WHY IS R² LOW-TO-MODERATE (~0.15 - 0.30)?
================================================================================
Students must understand and articulate why domestic appliance energy consumption
yields low-to-moderate R² in 10-minute time-series regression models:

1. Burst-Nature of Major Appliances:
   High-power appliances (electric kettles: 2000W, dishwashers: 1800W, washing machines: 2200W)
   turn ON suddenly for 10-40 minutes and draw intense electrical bursts. Ambient environmental
   variables (T1-T9, RH_1-RH_9) exhibit high thermal mass and inertia, changing gradually
   over hours. Consequently, slow-moving ambient features cannot explain sharp, impulsive spikes.

2. Unobserved Human Volition:
   The dataset captures environmental sensors, not human thought. Turning on a toaster or oven
   is a human choice driven by hunger or daily routine, not outdoor barometric pressure.

3. Strict Temporal Test Set (Out-of-Sample May 2016):
   Unlike random k-fold cross-validation (which artificially inflates R² via time-leakage),
   evaluating on a future test month exposes seasonal drift (May outdoor temps reach 25°C vs
   January's 2°C). This demonstrates genuine predictive reality without data leakage.
================================================================================
"""
    print(explanation)


def main():
    print("================================================================")
    print("STAGE 5: REGRESSION MODELING & EVALUATION")
    print("================================================================")
    df = pd.read_csv(PROCESSED_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])

    (
        feature_cols,
        df_train, df_test,
        X_train_raw, X_test_raw,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        y_train_high, y_test_high
    ) = prepare_features_and_splits(df)

    comp_df, coef_df, predictions_dict = evaluate_regression_models(
        feature_cols,
        df_train, df_test,
        X_train_raw, X_test_raw,
        X_train_scaled, X_test_scaled,
        y_train, y_test
    )

    evaluate_logistic_regression(
        X_train_scaled, X_test_scaled,
        y_train_high, y_test_high
    )

    plot_residuals_and_actual_vs_predicted(y_test, predictions_dict)
    explain_low_r2()
    print("[-] Regression execution completed successfully.\n")


if __name__ == "__main__":
    main()
