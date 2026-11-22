# Smart Residential Energy Consumption Analytics

An end-to-end academic data science and data warehousing project analyzing 10-minute smart home electrical consumption telemetry.

Aligned directly with the report specification: **`SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`**.

---

## ⚡ Quick Run (How to Run in Small)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (ETL + Models + 100% Verification check in 45s)
python run_all.py

# 3. Launch the interactive dashboard (opens at http://localhost:8501)
streamlit run app/dashboard.py
```

---

## 1. Project Overview & Folder Structure

```text
energy_project/
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv          # Raw sensor telemetry (19,735 rows, never modified)
│   └── processed/
│       ├── energy_warehouse.duckdb          # Embedded analytical Kimball star-schema warehouse
│       ├── energy_cleaned.csv               # Processed dataset with engineered features
│       ├── energydata_processed.csv         # Standardized reference export
│       └── powerbi/                         # Clean dimension and fact CSVs for Power BI ingestion
│           ├── Fact_Energy_Consumption.csv
│           ├── Dim_Date.csv
│           ├── Dim_Time.csv
│           └── Dim_Environment.csv
├── etl/
│   └── run_etl.py                           # Stage 1: Dimensional star-schema ETL pipeline & DuckDB loader
├── scripts/
│   ├── 03_preprocessing.py                  # Stage 2: Feature engineering, IQR outlier detection, EDA
│   ├── 04_olap.py                           # Stage 3: DuckDB OLAP roll-up, drill-down, slice, dice, pivot
│   ├── 06_regression.py                     # Stage 4: Correlation, Simple (lights) & Multiple Linear (28 feats)
│   ├── 07_association_rules.py              # Stage 5: Apriori, Multilevel Hour rules & FP-Growth check
│   └── 08_decision_trees.py                 # Stage 6: CART Classification (3 tiers) & Regression trees
├── app/
│   └── dashboard.py                         # Stage 7: 4-Page interactive Streamlit analytics dashboard
├── extras/                                  # Supplemental modules NOT in submitted report
│   ├── 05_hypothesis_tests_extra.py         # Mann-Whitney U, Kruskal-Wallis, Chi-square tests
│   ├── 06_advanced_regression_extra.py      # Ridge, Lasso, Polynomial, Scratch Gradient Descent, Logistic
│   ├── 09_peak_demand_extra.py              # Peak demand cutoff sweeps & load shifting tariff simulation
│   └── README.md                            # Documentation explaining exclusion from report
├── reports/
│   ├── tables/                              # All generated CSV metrics, VIF, validation & rule tables
│   ├── figures/                             # High-resolution PNG exploratory and diagnostic charts
│   ├── ALIGNMENT_AUDIT.md                   # Systematic audit of codebase vs docx specification
│   ├── VERIFICATION_REPORT.md               # 28-assertion verification table (100% PASS rate)
│   ├── REPORT_UPDATE_NOTES.md               # Guide to update report for implemented warehouse & dashboard
│   └── PROJECT_SUMMARY.md                   # Complete stage-by-stage viva preparation document
├── verify_against_report.py                 # Automated assertion test suite (0.01 tolerance)
├── run_all.py                               # Master pipeline runner (executes stages in sequence)
├── requirements.txt                         # Exact Python dependencies
└── README.md                                # Project documentation and user manual
```

---

## 2. Setup & Installation Instructions

This project runs on Python 3.10+ (tested on Windows with Python 3.12).

### Step 1: Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Command Prompt:
.venv\Scripts\activate.bat
```

### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```
*Required packages:* `pandas`, `numpy`, `duckdb`, `scipy`, `statsmodels`, `scikit-learn`, `mlxtend`, `matplotlib`, `seaborn`, `plotly`, `streamlit`.

---

## 3. How to Run the Project

### Option A: Execute the Full Pipeline (All Stages)
Run the master orchestrator to build the warehouse, engineer features, generate figures, run models, and verify all metrics:
```bash
python run_all.py
```
*Pipeline execution takes approximately 45 seconds and verifies all 28 report assertions.*

### Option B: Run Automated Verification Suite
To verify that all numbers, equations, and evaluation metrics match the project report:
```bash
python verify_against_report.py
```

### Option C: Launch the Interactive Dashboard
Launch the multi-page Streamlit dashboard mirroring Chapter 13 of the report:
```bash
streamlit run app/dashboard.py
```

---

## 4. Summary of Key Analytical Findings

1. **Dataset Integrity & Outlier Reality:**
   - 19,735 records spanning 137 days with zero missing values and unbroken 10-minute cadence.
   - 2,138 readings (10.83%) exceed the IQR upper fence of 175.0 Wh. These were **retained without removal** because sudden spikes (up to 1,080 Wh) represent physical multi-appliance operations.
2. **Occupancy Proxy:**
   - Indoor lighting (`lights`) is the strongest non-random predictor of appliance energy ($r = +0.1973$). Simple linear regression: $\text{Appliances} = 88.53 + 2.47 \times \text{lights}$ ($R^2 = 0.0490$).
3. **Low-to-Moderate Predictability ($R^2 = 0.1705$):**
   - Slow-changing indoor and outdoor ambient temperatures cannot predict sudden 10-minute appliance bursts driven by human decisions.
4. **Seasonal Concept Drift Failure of Decision Tree:**
   - The regression tree performed well on a random split ($R^2 = 0.1995$, RMSE = 89.50 Wh), but collapsed on chronological evaluation ($R^2 = -1.3579$, RMSE = 139.80 Wh). Non-parametric trees cannot extrapolate when May outdoor temperatures rise well above winter training ranges.
5. **Association Rules:**
   - Level-wise Apriori discovered 938 frequent itemsets ($s \ge 0.03$) and 232 class rules. Multilevel rules with individual Hour confirmed that weekday evenings with active lighting elevate the probability of high consumption (>100 Wh) by over 2.0x.
