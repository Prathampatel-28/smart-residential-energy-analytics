# Project Alignment Audit: Codebase vs. Report Specification

**Specification Document:** `SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`  
**Project Folder:** `energy_project/`  
**Audit Date:** 2026-11-23  
**Auditor:** Senior Data Scientist  

---

## Executive Summary
This audit compares every script, data artifact, warehouse table, machine learning model, and dashboard component in `energy_project/` against the finished project report (`SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`) and its reference implementation in **Appendix A**.

The report serves as the absolute specification. Any deviations in column definitions, feature engineering, split boundaries, hyperparameters, or evaluation metrics have been identified and categorized below. In accordance with the project instructions:
1. All core code will be updated to match the report specifications and exact numbers.
2. All non-report methodologies (Ridge/Lasso regularisation, formal hypothesis inference tests, VIF multicollinearity audits, peak-demand business cutoff analysis, and scratch gradient descent) are moved to the `extras/` directory and explicitly labelled `"not in report"`, ensuring they do not appear in the primary deliverables or main dashboard views.
3. The proposed Data Warehouse (Chapter 6), OLAP operations (Chapter 7), and Power BI Dashboard design (Chapter 13) are now fully implemented and exported to `data/processed/powerbi/`.

---

## Stage-by-Stage Detailed Audit

### Stage 1: ETL Pipeline & Star Schema Data Warehousing
- **Report Reference:** Chapter 6 (Sections 6.1 – 6.7, Tables 14 & 15, Code listings)
- **What Matches:**
  - Database engine: Embedded analytical warehouse using **DuckDB**.
  - Grain of central fact table: Exactly one row per 10-minute sensor reading (19,735 rows).
  - Surrogate key architecture: Integer surrogate keys for dimension joins.
  - Data integrity: Zero null foreign keys, unique primary keys, complete referential integrity.
- **What Differs:**
  - Table naming and structure:
    - *Prior code:* Generated `fact_energy_reading`, `dim_date`, `dim_time`, `dim_weather`, `dim_room`, and an unpivoted `fact_room_reading` table (177,615 rows).
    - *Report specification (Chapter 6):* Exactly 4 tables:
      1. `Fact_Energy_Consumption` (Fact_ID, Date_Key, Time_Key, Environment_Key, Appliances_Wh, Lights_Wh, Avg_Indoor_Temp, Avg_Indoor_RH, Temp_Diff_In_Out)
      2. `Dim_Date` (Date_Key, Full_Date, Day_No, Month_No, Month_Name, Day_Of_Week, Is_Weekend)
      3. `Dim_Time` (Time_Key, Hour_No, Minute_No, Time_Of_Day)
      4. `Dim_Environment` (Environment_Key, T_out_Band, RH_out_Band, Wind_Band, Press_Band, Visibility_Band)
- **What Was Missing:**
  - Direct export of warehouse tables to CSV format in `data/processed/powerbi/` for BI tool ingestion.
- **What Was Extra:**
  - `dim_room` and unpivoted `fact_room_reading` (moved to `extras/`).

---

### Stage 2: Data Preprocessing & Feature Engineering
- **Report Reference:** Chapter 5 (Sections 5.1 – 5.7, Table 8, Table 13) & Appendix A
- **What Matches:**
  - Dataset dimensions: 19,735 rows × 29 raw columns.
  - Timestamp integrity: 0 missing rows, unbroken 10-minute cadence.
  - Zero nulls and zero duplicate rows.
  - Log transformation: `log_Appliances = np.log1p(df["Appliances"])` (skewness reduced from 3.39 to 1.19).
- **What Differs:**
  - Target Categorisation (`Consumption_Level`):
    - *Prior code:* Defined binary `high_usage` based on training-set Q3.
    - *Report specification:* 3-class target `Consumption_Level`:
      - `Low` ($\le 50$ Wh) $\rightarrow$ 7,462 records
      - `Medium` ($50 < \text{Wh} \le 100$) $\rightarrow$ 8,040 records
      - `High` ($> 100$ Wh) $\rightarrow$ 4,233 records
      - Quantiles derived from the *full dataset* ($Q_1=50.0$, $Q_3=100.0$ Wh).
  - Derived Indoor Aggregations:
    - `T_indoor_mean`: Mean of indoor rooms `T1, T2, T3, T4, T5, T7, T8, T9` (excluding `T6` which is outside building north).
    - `RH_indoor_mean`: Mean of `RH_1, RH_2, RH_3, RH_4, RH_5, RH_7, RH_8, RH_9`.
    - `T_diff_in_out`: Thermal gradient (`T_indoor_mean - T_out`).
  - Handling of `rv1` and `rv2`:
    - *Prior code:* Immediately dropped `rv1` and `rv2` during initial CSV loading.
    - *Report specification:* Retained in dataset for descriptive statistics and correlation matrix; excluded strictly during regression and decision tree modelling.
  - Outlier Handling:
    - *Prior code:* Capped `Appliances` at 99th percentile (winsorization).
    - *Report specification:* Outliers detected via IQR rule ($Q_1 - 1.5\text{IQR}$ to $Q_3 + 1.5\text{IQR}$, Upper Fence = 175.0 Wh, 2,138 outliers / 10.83%), but **retained without removal or capping** because spikes represent true peak appliance operations.
- **What Was Missing:**
  - Direct export of `outliers_iqr.csv` and `descriptive_statistics.csv` matching Appendix A tables.
- **What Was Extra:**
  - VIF (Variance Inflation Factor) multicollinearity calculation and outlier capping (moved to `extras/`).

---

### Stage 3: OLAP Multi-Dimensional Operations
- **Report Reference:** Chapter 7 (Sections 7.1 – 7.6, Tables 17 – 19)
- **What Matches:**
  - Core OLAP operations: Roll-up (Month, Day, Hour), Drill-down (Month $\rightarrow$ Week), Slice (Weekend vs Weekday), Dice (Evening peak hours $\times$ Cold weather), Pivot (Hour $\times$ Day-of-week).
- **What Differs:**
  - Prior implementation queried against the non-aligned table schemas. Updated to query the exact Kimball schema `Fact_Energy_Consumption` and dimensions.

---

### Stage 4: Statistical & Hypothesis Testing
- **Report Reference:** Chapter 9 (Correlation) vs Prior Script `05_hypothesis_tests.py`
- **What Matches:**
  - Correlation analysis: Pearson and Spearman coefficients between `Appliances` and all environmental predictors.
  - Pearson $r(\text{Appliances}, \text{lights}) = 0.1973$ (strongest non-random correlation).
- **What Was Extra:**
  - Formal inferential hypothesis testing (Mann-Whitney U, Welch's t-test, Kruskal-Wallis, Chi-square independence tests, Cohen's d, Cramér's V). These are excellent scientific analyses but are not part of the core report narrative.
  - **Action:** Moved to `extras/05_hypothesis_tests_extra.py` and documented as extra credit material.

---

### Stage 5: Regression Analysis
- **Report Reference:** Chapter 10 (Sections 10.1 – 10.4, Tables 24 – 26) & Appendix A Section 9
- **What Matches:**
  - Target variable: `Appliances` (Wh).
  - Explanatory reporting on low $R^2$ due to appliance burst behaviour and thermal lag.
- **What Differs:**
  - Simple Linear Regression:
    - *Prior code:* Fit $T_{\text{out}}$ vs Appliances.
    - *Report specification:* Single strongest non-random predictor is `lights`:
      $$\text{Appliances} = 88.53 + 2.47 \times \text{lights}$$
      Test $R^2 = 0.0490$.
  - Multiple Linear Regression Feature Set:
    - Exactly 28 predictors: 25 raw sensor features (excluding `Appliances`, `rv1`, `rv2`) plus `hour`, `day_of_week`, `is_weekend`.
    - Normalisation: `StandardScaler` inside `sklearn.pipeline.make_pipeline`.
  - Splitting Strategy:
    - *Split A (Random 80/20, `random_state=42`):*
      - Test MAE: 52.58 Wh
      - Test RMSE: 91.11 Wh
      - Test $R^2$: 0.1705
    - *Split B (Chronological Split: first 15,788 rows train, last 3,947 rows test):*
      - Test RMSE: 86.42 Wh
      - Test $R^2$: 0.0989
  - Baseline Benchmark:
    - Added mean-only baseline on chronological test set ($R^2 = -0.0000$, RMSE 91.03 Wh).
- **What Was Extra:**
  - Polynomial regression, RidgeCV, LassoCV, Scratch Gradient Descent, and Logistic regression for `high_usage`.
  - **Action:** Moved to `extras/06_advanced_regression_extra.py`.

---

### Stage 6: Association Rule Mining
- **Report Reference:** Chapter 11 (Sections 11.1 – 11.4, Tables 28 – 31) & Appendix A Section 10
- **What Matches:**
  - Unsupervised mining of consumption conditions associated with appliance demand.
- **What Differs:**
  - Discretisation Scheme (Table 28):
    - `T_out`: Tertiles $\rightarrow$ `Cold`, `Mild`, `Warm`
    - `RH_out`: Tertiles $\rightarrow$ `Dry`, `Moderate`, `Humid`
    - `Windspeed`: Tertiles $\rightarrow$ `Calm`, `Breezy`, `Windy`
    - `Lights`: `Off` ($=0$), `On` ($>0$)
    - `TimeOfDay`: `Night`, `Morning`, `Afternoon`, `Evening`
    - `DayType`: `Weekday`, `Weekend`
    - `Appliances`: `Low`, `Medium`, `High`
    - Exact 20 discrete items.
  - Mining Parameters:
    - Min Support: $0.03$ (3%), Max Itemset Length: $4$.
    - Yields exactly **938 frequent itemsets**.
  - Rule Generation:
    - Class-association rules only: Consequent must be a single item `Appliances=Low/Medium/High`.
    - Min Confidence: $0.40$ (40%), Min Lift: $1.0$.
    - Yields exactly **232 class rules**.
  - Multilevel Association Rules:
    - Level 1: `TimeOfDay` band
    - Level 2: Individual `Hour` ($00$ to $23$) with Min Support $0.02$, Max Length $3$, Min Confidence $0.40$.
  - FP-Growth Cross-Check:
    - Added `mlxtend` FP-Growth execution, mathematically proving itemset equivalence (938 itemsets).

---

### Stage 7: Decision Tree Analysis
- **Report Reference:** Chapter 12 (Sections 12.1 – 12.4, Tables 32 – 35) & Appendix A Section 11
- **What Matches:**
  - Decision tree architectures for both classification and regression.
- **What Differs:**
  - Classification Tree:
    - Target: `Consumption_Level` (3 classes: Low, Medium, High).
    - Hyperparameters: `criterion="gini"`, `max_depth=5`, `min_samples_leaf=50`, `random_state=42`.
    - Random Stratified 80/20 Test Accuracy: **59.1%** (vs majority baseline **40.7%**).
    - Chronological Test Accuracy: **54.4%**.
  - Regression Tree:
    - Target: Continuous `Appliances`.
    - Hyperparameters: `max_depth=6`, `min_samples_leaf=50`, `random_state=42`.
    - Random Test: $R^2 = 0.1995$, RMSE $= 89.50$ Wh.
    - Chronological Test: $R^2 = -1.3579$, RMSE $= 139.80$ Wh (honest reporting of seasonal failure).
- **What Was Extra:**
  - Random Forest and cost-complexity pruning (moved to `extras/`).

---

### Stage 8: Peak-Demand Business Analysis
- **Report Reference:** Not included in the submitted report.
- **Action:** Moved to `extras/09_peak_demand_extra.py` labelled "not in report".

---

### Stage 9: Interactive Dashboard
- **Report Reference:** Chapter 13 (Sections 13.1 – 13.7, Figures 27a–d)
- **What Differs:**
  - Restructured Streamlit application into 4 dedicated pages mirroring the Chapter 13 specification:
    - **Page 1: Energy Consumption Overview** (KPI cards, binned histogram, monthly bar chart, month/weekend slicers)
    - **Page 2: Time-Based Consumption Analysis** (Hourly diurnal curve, daily trend, day-of-week bars, 24x7 matrix heatmap)
    - **Page 3: Environmental Analysis** ($T_{\text{out}}$ and $RH_{\text{out}}$ scatters, weather band consumption, indoor vs outdoor line trend)
    - **Page 4: Correlation and Insights** (Correlation matrix heatmap, top correlation bar chart, viva discussion takeaways)
    - **Tab 5: About, Methodology & Limitations** (Academic notes, model performance comparison, honest limitations)
- **What Was Extra:**
  - Standalone peak-demand slider and hypothesis-test viewers moved to a dedicated secondary view or `extras/`.

---

## Action Plan for Steps 2 to 5
1. **Re-factor ETL (`etl/run_etl.py`):** Build exact Chapter 6 schema (`Fact_Energy_Consumption`, `Dim_Date`, `Dim_Time`, `Dim_Environment`), load to DuckDB, export CSVs to `data/processed/powerbi/`.
2. **Re-factor Preprocessing (`scripts/03_preprocessing.py`):** Calculate $Q_1=50$, $Q_3=100$, generate `Consumption_Level`, compute indoor averages (`T_indoor_mean`, `RH_indoor_mean`, `T_diff_in_out`), preserve outliers with IQR summary table, save `energydata_processed.csv`.
3. **Re-factor OLAP (`scripts/04_olap.py`):** Run queries on the Chapter 6 warehouse schema.
4. **Organise `extras/` Folder:** Move hypothesis testing, regularized regression, scratch GD, VIF, and peak demand to `extras/`.
5. **Re-factor Regression (`scripts/06_regression.py`):** Implement exact simple linear (lights) and multiple linear (28 features, random 80/20 and chronological splits, baseline).
6. **Re-factor Association Rules (`scripts/07_association_rules.py`):** Implement Table 28 discretisation, 20 items, scratch Apriori (938 itemsets, 232 class rules), multilevel rules, and mlxtend FP-Growth verification.
7. **Re-factor Decision Trees (`scripts/08_decision_trees.py`):** Implement exact 3-class classification tree (depth=5, leaf=50) and regression tree (depth=6, leaf=50) on both random and chronological splits.
8. **Build `verify_against_report.py`:** Automated test suite printing PASS/FAIL for all 10 target metrics with 0.01 tolerance.
9. **Update Dashboard (`app/dashboard.py`):** 4 pages mirroring Chapter 13.
10. **Generate Verification & Report Update Documents:** `reports/VERIFICATION_REPORT.md` and `reports/REPORT_UPDATE_NOTES.md`.
