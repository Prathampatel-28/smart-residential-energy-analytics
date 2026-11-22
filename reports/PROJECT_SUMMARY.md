# Smart Residential Energy Consumption Analytics: End-to-End Project Summary & Viva Preparation Guide

**Academic Project Reference:** B.Tech / BCA Data Science & Analytics  
**Specification Document:** `SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`  
**Dataset:** Appliances Energy Prediction Dataset (19,735 readings, 10-minute sampling interval, Jan 11 – May 27, 2016)  
**Target Variable:** `Appliances` (Wh per 10-minute interval)  

---

## 1. Master Summary of Results & Metrics

| Analysis Stage | Key Metric / Measure | Observed Value | Interpretation / Viva Talking Point |
| :--- | :--- | :---: | :--- |
| **Dataset Scale** | Total Observations | 19,735 rows | 137 consecutive days at unbroken 10-minute intervals |
| **Target Distribution** | Mean Consumption | 97.69 Wh | Positively right-skewed; median (60.0 Wh) is much lower than mean |
| **Target Extreme** | Maximum Reading | 1,080.0 Wh | Peak interval demand when multiple high-power appliances run concurrently |
| **Skewness** | Raw vs log(1+x) | 3.39 $\rightarrow$ 1.19 | Logarithmic transformation reduces skewness while preserving rank ordering |
| **Outlier Detection** | IQR Upper Fence | 175.0 Wh | 2,138 records (10.83%) exceed fence; **retained** because they are genuine load spikes |
| **Target Tiers** | Low / Medium / High | 7,462 / 8,040 / 4,233 | Balanced 3-class target derived from full data quartiles ($Q_1=50$, $Q_3=100$ Wh) |
| **Correlation** | Pearson $r(\text{Appliances}, \text{lights})$ | +0.1973 | Strongest non-random correlation; lighting acts as direct human occupancy proxy |
| **Simple Regression** | OLS Equation | $\text{Wh} = 88.53 + 2.47 \times \text{lights}$ | Explains 4.90% of variance ($R^2=0.0490$) using a single lighting feature |
| **Multiple Regression** | Random 80/20 Split | MAE: 52.58 Wh, $R^2$: 0.1705 | Fits 28 standardized predictors; captures 17.05% of variance on test data |
| **Multiple Regression** | Chronological Split | RMSE: 86.42 Wh, $R^2$: 0.0989 | Outperforms mean baseline ($R^2=-0.0003$), but lower than random split due to time drift |
| **Association Rules** | Unique Discrete Items | 20 items | 938 frequent itemsets ($s \ge 0.03$), exactly 232 class-association rules ($c \ge 0.40$) |
| **FP-Growth Cross-Check** | Itemset Count | 938 itemsets | Confirms FP-Growth identically matches Apriori with faster tree execution |
| **Classification Tree** | Test Accuracy (Random) | 59.1% | Outperforms majority-class baseline (40.7%) on predicting Low/Medium/High tier |
| **Classification Tree** | Test Accuracy (Chrono) | 54.4% | Robust classification performance maintained across seasonal transition |
| **Regression Tree** | Test $R^2$ (Random) | 0.1995 (RMSE: 89.50 Wh) | Tree slightly outperforms multiple linear regression when trained on random split |
| **Regression Tree** | Test $R^2$ (Chronological) | -1.3579 (RMSE: 139.80 Wh) | Failure due to seasonal extrapolation; proves danger of unregularized non-parametric splits |
| **Monthly Means** | Jan / Feb / Mar / Apr / May | 97.03 / 100.95 / 96.95 / 98.89 / 94.20 | February records highest mean; May exhibits lowest due to mild spring temperatures |

---

## 2. Stage-by-Stage Methodological Breakdown

### Stage 1: ETL Pipeline & Kimball Star Schema Data Warehouse
- **Objective:** Design and construct a normalized, high-performance analytical warehouse from flat sensor telemetry to enable sub-second OLAP queries and BI reporting.
- **Method:**
  - Columnar embedded database: **DuckDB** (`energy_warehouse.duckdb`).
  - Implemented Kimball dimensional architecture with 1 Fact table and 3 Dimension tables:
    1. `Fact_Energy_Consumption`: Grain = 1 row per 10-minute interval (19,735 rows). Foreign keys: `Date_Key`, `Time_Key`, `Environment_Key`. Measures: `Appliances_Wh`, `Lights_Wh`, `Avg_Indoor_Temp`, `Avg_Indoor_RH`, `Temp_Diff_In_Out`.
    2. `Dim_Date`: `Date_Key` (YYYYMMDD), `Full_Date`, `Day_No`, `Month_No`, `Month_Name`, `Day_Of_Week`, `Is_Weekend`.
    3. `Dim_Time`: `Time_Key` (HHMM), `Hour_No`, `Minute_No`, `Time_Of_Day` (Night, Morning, Afternoon, Evening).
    4. `Dim_Environment`: `Environment_Key`, weather tertile bands (`T_out_Band`, `RH_out_Band`, `Wind_Band`, `Press_Band`, `Visibility_Band`).
  - Exported all 4 tables as CSVs into `data/processed/powerbi/` for native import into Microsoft Power BI Desktop.
- **Verification Result:** Zero null foreign keys, zero duplicate primary keys, 100% referential integrity across all relationships.

---

### Stage 2: Data Preprocessing, Feature Engineering & Outlier Audit
- **Objective:** Prepare raw telemetry for statistical modeling while preserving genuine smart-home usage spikes.
- **Method:**
  - Audited 29 raw columns: 0 missing values, 0 duplicate timestamps, unbroken 10-minute interval cadence across all 137 days.
  - Derived features:
    * `T_indoor_mean`: Arithmetic average of indoor sensors `T1, T2, T3, T4, T5, T7, T8, T9` (excluding `T6` which is outside building north).
    * `RH_indoor_mean`: Arithmetic average of indoor room humidities.
    * `T_diff_in_out`: Indoor thermal gradient ($T_{\text{indoor\_mean}} - T_{\text{out}}$).
    * `Consumption_Level`: 3-tier target derived from full-dataset quartiles ($Q_1 = 50.0$ Wh, $Q_3 = 100.0$ Wh): Low ($\le 50$ Wh), Medium ($50-100$ Wh), High ($> 100$ Wh).
  - Outlier Treatment Decision: Detected with IQR rule ($Q_3 + 1.5\text{IQR} = 175.0$ Wh; 2,138 records / 10.83%). **Outliers were retained without deletion or winsorization.** In smart home analytics, high energy draws reflect true physical events (concurrent operation of ovens, washing machines, dryers, kettles). Removing them would misrepresent peak demand.

---

### Stage 3: OLAP Multi-Dimensional Operations
- **Objective:** Query the warehouse cube across Date, Time, and Environment dimensions.
- **Operations:**
  - **Roll-up:** Aggregated 10-minute facts into Monthly sums/means (Jan: 97.03, Feb: 100.95, Mar: 96.95, Apr: 98.89, May: 94.20 Wh) and Daily aggregates.
  - **Drill-down:** Decomposed Monthly totals down to calendar weeks (W02 to W21), identifying peak weekly usage during cold winter weeks.
  - **Slice:** Isolated the `Is_Weekend` dimension slice, revealing that weekend daytime consumption remains elevated compared to empty weekday afternoons.
  - **Dice:** Extracted sub-cube for Evening Peak Hours (18:00 – 22:00) during Cold outdoor weather ($T_{\text{out}} < 5^\circ\text{C}$).
  - **Pivot Matrix:** Computed 24-hour $\times$ 7-day cross-tabulation heatmap, exposing the sharp residential evening peak between 18:00 and 20:00 on all weekdays.

---

### Stage 4: Regression Modeling & Baseline Benchmarking
- **Objective:** Evaluate how much variance in continuous energy draw can be predicted by ambient indoor and outdoor sensors.
- **Method & Architecture:**
  - **Simple Linear Regression:** Strongest predictor is `lights` ($r = 0.1973$). Equation: $\text{Appliances} = 88.53 + 2.47 \times \text{lights}$ ($R^2 = 0.0490$).
  - **Multiple Linear Regression:** 28 predictors (excluding synthetic noise columns `rv1` and `rv2`). `StandardScaler` fitted strictly inside a `Pipeline` on training data.
  - **Dual Splitting Strategy:**
    1. *Random 80/20 Split (`random_state=42`):* Test MAE = 52.58 Wh, Test RMSE = 91.11 Wh, Test $R^2 = 0.1705$.
    2. *Chronological Split (First 80% = 15,788 rows train, Last 20% = 3,947 rows test):* Test RMSE = 86.42 Wh, Test $R^2 = 0.0989$.
    3. *Chronological Baseline (Mean Only):* Test RMSE = 91.03 Wh, Test $R^2 = -0.0000$.

---

### Stage 5: Association Rule Mining (Apriori & FP-Growth)
- **Objective:** Discover combinatorial behavioral rules associated with high appliance energy demand.
- **Method:**
  - Binned variables into 20 discrete items (Table 28): TimeOfDay, DayType, Tertiles of $T_{\text{out}}$, $RH_{\text{out}}$, Windspeed, Lights (On/Off), and Consumption_Level (Low/Medium/High).
  - Executed level-wise Apriori ($s \ge 0.03$, max length 4), discovering **938 frequent itemsets**.
  - Generated **232 class-association rules** targeting `Appliances=...` ($c \ge 0.40$, lift $\ge 1.0$).
  - Evaluated Multilevel Rules with individual Hour ($00$ to $23$) ($s \ge 0.02$, max length 3).
  - FP-Growth Verification: Ran `mlxtend` FP-Growth, confirming it generates the exact same 938 itemsets.

---

### Stage 6: Decision Tree Analysis (Classification & Regression)
- **Objective:** Benchmark non-linear recursive partitioning against linear models.
- **Classification Tree (Consumption_Level: Low / Medium / High):**
  - Hyperparameters: CART Gini impurity, `max_depth=5`, `min_samples_leaf=50`, `random_state=42`.
  - Random Stratified 80/20 Test Accuracy: **59.1%** (significantly beating the 40.7% majority-class baseline).
  - Chronological Test Accuracy: **54.4%**.
- **Regression Tree (Continuous Appliances Wh):**
  - Hyperparameters: `max_depth=6`, `min_samples_leaf=50`, `random_state=42`.
  - Random Split Test: $R^2 = 0.1995$, RMSE = 89.50 Wh (outperforms Multiple Linear Regression's 0.1705).
  - Chronological Split Test: $R^2 = -1.3579$, RMSE = 139.80 Wh.
  - *Key Finding:* Because outdoor temperatures shifted from 0°C (winter) to 25°C (May), the regression tree encountered unobserved feature space and failed to extrapolate, demonstrating the risk of non-parametric tree models under seasonal concept drift.

---

### Stage 7: Interactive 4-Page Streamlit Dashboard
- **Objective:** Deliver an operational business intelligence interface mirroring Chapter 13.
- **Pages Implemented:**
  1. *Page 1: Energy Overview* – High-level KPIs, binned distribution histogram, monthly bar chart, month/weekend slicers.
  2. *Page 2: Time-Based Analysis* – Diurnal 24-hour curve, continuous daily trend line, day-of-week bars, 24x7 matrix heatmap.
  3. *Page 3: Environmental Analysis* – Outdoor temperature and humidity scatter plots, weather band bar charts, indoor vs outdoor thermal gradient.
  4. *Page 4: Correlation & Insights* – Heatmap matrix, top predictor rankings, model performance summary table, plain-language takeaways.
  5. *Tab 5: About & Limitations* – Kimball star schema architecture, mathematical explanation of low $R^2$, and documentation of extra modules in `extras/`.

---

## 3. Comprehensive Viva Voce Defense Guide

### Q1: Why is the $R^2$ of linear regression only ~0.10 to 0.17? Is that a poor model?
**Answer:**  
"No, in residential energy analytics, an $R^2$ between 0.10 and 0.20 on a 10-minute cadence is an authentic, expected result consistent with published literature (Candanedo et al., 2017). This occurs because:
1. High-power appliances (electric kettles, microwaves, washing machines, toasters) draw sudden, high-magnitude bursts of power (1,500–2,500 Watts) for 10–30 minutes based purely on impulsive human decisions (e.g. making tea, cooking dinner).
2. Ambient indoor and outdoor temperatures have high thermal inertia and change very slowly over hours. A slow-moving physical variable cannot predict the exact 10-minute moment an occupant switches on a kettle.
3. The dataset records environmental conditions, not occupant cognitive intentions. Therefore, an $R^2$ of ~0.17 proves the model captures the baseline diurnal cycle without falsely memorizing noise."

---

### Q2: Why did the regression tree fail with negative $R^2$ (-1.3579) on the chronological split?
**Answer:**  
"A decision tree performs recursive orthogonal splits, predicting constant piecewise values inside bounded hypercubes in the feature space. It cannot extrapolate linear slopes outside its training envelope.
- During training (January to April), outdoor temperatures ranged from $-6^\circ\text{C}$ to $15^\circ\text{C}$.
- In the May test set, outdoor temperatures rose above $25^\circ\text{C}$.
Because the tree had never seen feature values in that range, it placed May observations into boundary leaf nodes that predicted inappropriate winter-level constants. This produced massive residuals, driving MSE higher than the variance of the test set ($R^2 < 0$). This is a textbook demonstration of why chronological time-series evaluation is essential in data science."

---

### Q3: Why did you retain the 2,138 outliers detected by the IQR rule?
**Answer:**  
"The IQR rule is a statistical tool, not a physical law. For right-skewed energy data, the upper fence is $Q_3 + 1.5\text{IQR} = 175.0$ Wh. Appliances drawing 300 to 1,080 Wh represent real physical events: multiple kitchen and laundry appliances running simultaneously. In smart grid demand response, identifying and managing these specific peak load events is the primary business goal. Deleting or capping them would eliminate the most critical phenomenon the project aims to study."

---

### Q4: Why is `lights` the single best predictor of appliance consumption?
**Answer:**  
"Lighting energy consumption had the highest non-random Pearson correlation ($r = +0.1973$). While temperature sensors measure passive thermal equilibrium, lighting directly reflects active human presence. When domestic lights are turned on, residents are awake and moving between rooms, using appliances for cooking, working, and entertainment. Lighting acts as an effective proxy for household occupancy."

---

### Q5: What is the benefit of a Kimball Star Schema over a single flat CSV?
**Answer:**  
"A single flat table creates immense redundancy (repeating date names, time bands, and weather strings across 19,735 rows). In the Kimball dimensional star schema:
1. The central `Fact_Energy_Consumption` table stores only numeric surrogate keys and numerical measures, minimizing storage footprint and maximizing memory cache efficiency.
2. The surrounding `Dim_Date`, `Dim_Time`, and `Dim_Environment` dimension tables enable fast slicing, dicing, and drill-down aggregations across hierarchical attributes.
3. The schema decouples reporting tools like Power BI and Streamlit from the underlying storage, allowing sub-second analytical aggregations via DuckDB."
