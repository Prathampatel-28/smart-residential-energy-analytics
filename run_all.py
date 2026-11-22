"""
Master Pipeline Orchestrator: run_all.py
=========================================
Executes all project stages in strict sequential order matching the report:
1. Stage 1: etl/run_etl.py (Star Schema DuckDB creation & Power BI CSV export)
2. Stage 2: scripts/03_preprocessing.py (Features, Outliers, Validation, EDA charts)
3. Stage 3: scripts/04_olap.py (DuckDB OLAP roll-up, drill-down, slice, dice, pivot)
4. Stage 4: scripts/06_regression.py (Correlation, Simple & Multiple Linear Regression)
5. Stage 5: scripts/07_association_rules.py (Discretization, Apriori, Class rules, Multilevel, FP-Growth)
6. Stage 6: scripts/08_decision_trees.py (Classification & Regression trees)
7. Stage 7: verify_against_report.py (Automated 28-assertion verification suite)

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STAGES = [
    ("Stage 1: ETL Pipeline & Star Schema", "etl/run_etl.py"),
    ("Stage 2: Preprocessing, Features & EDA", "scripts/03_preprocessing.py"),
    ("Stage 3: OLAP Multidimensional Operations", "scripts/04_olap.py"),
    ("Stage 4: Regression & Correlation Analysis", "scripts/06_regression.py"),
    ("Stage 5: Association Rule Mining", "scripts/07_association_rules.py"),
    ("Stage 6: Decision Tree Analysis", "scripts/08_decision_trees.py"),
    ("Stage 7: Automated Report Verification Suite", "verify_against_report.py")
]


def run_stage(title: str, script_rel_path: str):
    """Executes a single python script and stops pipeline on error."""
    script_path = BASE_DIR / script_rel_path
    if not script_path.exists():
        print(f"\n[ERROR] Script not found: {script_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"RUNNING: {title}")
    print(f"Script:  {script_rel_path}")
    print("=" * 80)

    start_t = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    elapsed = time.perf_counter() - start_t

    if result.returncode != 0:
        print(f"\n[PIPELINE FAILURE] {title} failed with exit code {result.returncode}.")
        print("Stopping pipeline execution.")
        sys.exit(result.returncode)
    else:
        print(f"[-] {title} finished successfully in {elapsed:.2f} seconds.")


def main():
    print("*" * 80)
    print("SMART RESIDENTIAL ENERGY CONSUMPTION: FULL PIPELINE EXECUTION")
    print("Specification: SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx")
    print("*" * 80)

    total_start = time.perf_counter()

    for title, script in STAGES:
        run_stage(title, script)

    total_time = time.perf_counter() - total_start
    print("\n" + "*" * 80)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("You can now launch the dashboard with:")
    print("    streamlit run app/dashboard.py")
    print("*" * 80)


if __name__ == "__main__":
    main()
