# Supplemental / Extra Modules (NOT in Submitted Report)

> **IMPORTANT NOTICE:**  
> The scripts in this directory contain supplemental analyses developed during project exploration that are **NOT part of the submitted report specification** (`SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`).
> 
> They are retained here for reference and viva discussion, but they are excluded from the main execution pipeline (`run_all.py`), main report verification, and primary dashboard views.

---

## Inventory of Extra Modules

1. **`05_hypothesis_tests_extra.py`**
   - **Topic:** Formal inferential statistical tests.
   - **Contents:** Mann-Whitney U tests, Welch's t-test, Kruskal-Wallis H-test, Pearson Chi-Square test of independence, effect sizes (Cohen's d, Rank-biserial r, Cramér's V).
   - **Reason for Exclusion:** The report uses exploratory data analysis (EDA) and bivariate correlation coefficients (Chapter 8 & 9) rather than classical hypothesis testing.

2. **`06_advanced_regression_extra.py`**
   - **Topic:** Non-report regression architectures.
   - **Contents:** RidgeCV (L2), LassoCV (L1), Polynomial Regression with validation-set degree tuning, Scratch Gradient Descent Regressor (NumPy), and Logistic Regression for binary `high_usage` peak classification.
   - **Reason for Exclusion:** Chapter 10 of the report focuses strictly on Simple Linear Regression (Appliances ~ lights) and Multiple Linear Regression (28 features, StandardScaler inside a Pipeline).

3. **`09_peak_demand_extra.py`**
   - **Topic:** Peak-demand business cutoff analysis.
   - **Contents:** Appliance threshold sweeps (150-300 Wh), peak interval counts, energy concentration percentages, and Time-of-Use (ToU) load shifting cost simulation.
   - **Reason for Exclusion:** The report does not include a financial cutoff or tariff simulation.

4. **`vif_analysis_extra.py`**
   - **Topic:** Variance Inflation Factor (VIF) multicollinearity screening.
   - **Contents:** VIF calculations across indoor environmental sensors.
   - **Reason for Exclusion:** Multi-collinearity is discussed in Chapter 9 via correlation matrices rather than formal VIF tables.
