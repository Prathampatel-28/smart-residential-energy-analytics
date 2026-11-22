# Smart Residential Energy Consumption Analytics

An end-to-end academic data science and data warehousing system analyzing smart home electrical consumption telemetry.  
The project implements Kimball dimensional modeling, automated ETL pipelines into DuckDB, and statistical hypothesis testing.  
It benchmarks predictive regression models, association rule mining, and decision trees, fully surfaced in an interactive 4-page Streamlit dashboard.

---

## 📌 Dataset Source

The dataset used in this project is the **UCI Appliances Energy Prediction Dataset**:
- **Citation**: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house*. Energy and Buildings, 140, 81-97.
- **Span & Frequency**: 137 days (January 11, 2016 to May 27, 2016), sampled at 10-minute intervals (19,735 total readings).
- **Attributes**: 29 features comprising appliance energy consumption (`Appliances`), lighting energy (`lights`), 9 indoor temperature & humidity sensors (`T1`-`T9`, `RH_1`-`RH_9`), outdoor weather telemetry from a nearby Chievres airport station (`T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`), and random filter columns (`rv1`, `rv2`). Zero missing values across all columns.

---

## ⚡ How to Run

Follow these quick commands to set up the environment, run all stages, verify metrics, and launch the dashboard:

```bash
# 1. Create and activate a Python 3.10+ virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Command Prompt:
.venv\Scripts\activate.bat

# 2. Install all required dependencies
pip install -r requirements.txt

# 3. Execute the full end-to-end pipeline (ETL, Warehousing, Models, Figures, Reports)
python run_all.py

# 4. Run the automated assertion test suite against the report specifications
python verify_against_report.py

# 5. Launch the interactive multi-page Streamlit dashboard (opens at http://localhost:8501)
streamlit run app/dashboard.py
```

---

## 📂 Project Folder Structure

```text
energy_project/
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv          # Raw sensor telemetry (19,735 rows, never modified)
│   └── processed/
│       ├── energy_warehouse.duckdb          # Embedded analytical Kimball star-schema warehouse (local)
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
│   ├── VERIFICATION_REPORT.md               # 34-assertion verification table (100% PASS rate)
│   ├── REPORT_UPDATE_NOTES.md               # Guide to update report for implemented warehouse & dashboard
│   └── PROJECT_SUMMARY.md                   # Complete stage-by-stage viva preparation document
├── verify_against_report.py                 # Automated assertion test suite (0.01 tolerance, 34 checks)
├── run_all.py                               # Master pipeline runner (executes stages in sequence)
├── requirements.txt                         # Exact Python dependencies
└── README.md                                # Project documentation and user manual
```

---

## 📊 Key Results & Analytical Findings

1. **Energy Distribution & Outlier Realities**:
   - Total energy consumption recorded: **1,928 kWh** across 19,735 readings.
   - Mean appliance consumption is **97.69 Wh** (median 60.0 Wh, max 1,080 Wh, skewness 3.39; log1p skewness 1.19).
   - **2,138 readings (10.83%)** exceed the IQR upper fence ($175.0\text{ Wh}$). Rather than treating them as faulty sensor anomalies, they are retained as true physical spikes from simultaneous appliance usage (washing machines, dishwashers, ovens).

2. **Occupancy & Lighting Correlation**:
   - Indoor lighting (`lights`) is the single strongest instantaneous predictor ($r = +0.1973$).
   - Simple linear model: $\text{Appliances} = 88.53 + 2.47 \times \text{lights}$ ($R^2 = 0.0490$).

3. **Multiple Linear Regression (28 Features)**:
   - **Random 80/20 Train-Test Split**: $\text{MAE} = 52.58\text{ Wh}$, $\text{RMSE} = 91.11\text{ Wh}$, $R^2 = 0.1705$.
   - **Chronological Split** (first 15,788 train, last 3,947 test): $\text{RMSE} = 86.42\text{ Wh}$, $R^2 = 0.0989$.
   - Highlights the inherent unpredictability of human appliance actuation from ambient thermodynamic sensors alone.

4. **Classification & Regression Decision Trees**:
   - 3-Class Discretization (Low $\le 50$, Medium $50-100$, High $> 100\text{ Wh}$):
     - **Random Split Test Accuracy**: **59.1%** (significantly outperforming the 40.7% zero-rule majority class baseline).
     - **Chronological Split Test Accuracy**: **54.4%**.
     - `hour` is the dominant root decision feature (importance 40.7%).
   - Regression Tree:
     - Random Split: $R^2 = 0.1995$, $\text{RMSE} = 89.50\text{ Wh}$.
     - Chronological Split: $R^2 = -1.3579$, failing to extrapolate to unprecedented warmer late-spring temperatures in May.

5. **Temporal Peak Patterns & Association Rules**:
   - Distinct diurnal profile with an **evening peak at 18:00 (190.4 Wh hourly mean)**, matching post-work cooking, laundry, and lighting activities.
   - Apriori algorithm extracted **938 frequent itemsets** and **232 class rules** ($s \ge 0.03, c \ge 0.40$).
   - Multilevel rules confirm that evening hours combined with high indoor lighting yield a **lift > 2.0x** for high consumption events.

---

## ⚠️ Limitations & Real-World Engineering Insights

1. **Thermodynamic Lag vs. Instantaneous Demand**:
   - Ambient indoor temperatures and humidity change slowly over hours due to building thermal mass, whereas high-power appliances (electric kettles, microwave ovens, dryers) operate in brief 5–30 minute bursts. Ambient sensors alone cannot capture instantaneous electrical demand without sub-metering or plug-level CT clamps.
2. **Unobserved Human Occupancy & Intent**:
   - While lighting (`lights`) acts as a partial proxy for occupancy, presence sensors, PIR motion detectors, and active occupant schedules are unobserved in the telemetry.
3. **Seasonal Concept Drift & Non-Parametric Extrapolation**:
   - Tree-based models partition input feature space into rectangular orthogonal boxes and predict constant leaf means. When evaluating chronologically on May readings where outdoor temperatures exceed the winter/early-spring training distributions, decision trees cannot extrapolate trends, illustrating the critical necessity for chronological validation in time-series energy domains.
