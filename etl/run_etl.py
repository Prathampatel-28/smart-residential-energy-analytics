"""
Stage 1: ETL Pipeline & Star Schema Data Warehousing
=====================================================
This script implements the exact dimensional star schema specified in Chapter 6
(Sections 6.1 – 6.7, Tables 14 & 15) of SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx.

Architecture (Kimball Star Schema):
1. Fact_Energy_Consumption (Central Fact Table):
   - Grain: Exactly one row per 10-minute measurement interval (19,735 rows).
   - Foreign Keys: Date_Key, Time_Key, Environment_Key.
   - Measures: Appliances_Wh, Lights_Wh, Avg_Indoor_Temp, Avg_Indoor_RH, Temp_Diff_In_Out.
2. Dim_Date:
   - Date_Key, Full_Date, Day_No, Month_No, Month_Name, Day_Of_Week, Is_Weekend.
3. Dim_Time:
   - Time_Key, Hour_No, Minute_No, Time_Of_Day.
4. Dim_Environment:
   - Environment_Key, T_out_Band, RH_out_Band, Wind_Band, Press_Band, Visibility_Band.

Deliverables:
- Embedded analytical DuckDB database: data/processed/energy_warehouse.duckdb
- Power BI CSV export tables: data/processed/powerbi/
- Data warehouse validation audit: reports/tables/etl_validation.csv

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import os
import sys
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "energydata_complete.csv"
DB_PATH = BASE_DIR / "data" / "processed" / "energy_warehouse.duckdb"
POWERBI_DIR = BASE_DIR / "data" / "processed" / "powerbi"
OUTPUT_TABLES_DIR = BASE_DIR / "reports" / "tables"

# Ensure output directories exist
OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
POWERBI_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Loads raw 10-minute smart home dataset."""
    print("[-] Reading raw CSV dataset from:", RAW_DATA_PATH)
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"    Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def build_star_schema(df: pd.DataFrame, con: duckdb.DuckDBPyConnection):
    """
    Transforms flat dataset into the 4 Kimball star schema tables
    defined in Chapter 6 (Table 14 and Table 15).
    """
    print("[-] Transforming dataset into Chapter 6 Star Schema...")
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["Fact_ID"] = np.arange(1, len(df) + 1)

    # 1. Dimension: Dim_Date
    dates = pd.DataFrame({"date_series": pd.to_datetime(df["datetime"].dt.date.unique())})
    dim_date = pd.DataFrame({
        "Date_Key": dates["date_series"].dt.strftime("%Y%m%d").astype(int),
        "Full_Date": dates["date_series"].dt.date,
        "Day_No": dates["date_series"].dt.day.astype(int),
        "Month_No": dates["date_series"].dt.month.astype(int),
        "Month_Name": dates["date_series"].dt.strftime("%B"),
        "Day_Of_Week": dates["date_series"].dt.strftime("%A"),
        "Is_Weekend": dates["date_series"].dt.day_of_week.isin([5, 6])
    }).sort_values("Date_Key").reset_index(drop=True)

    # Map Date_Key to fact
    df["Date_Key"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    # 2. Dimension: Dim_Time
    # 144 intervals per day (24 hours x 6 ten-minute readings)
    all_hours = np.repeat(np.arange(24), 6)
    all_minutes = np.tile(np.arange(0, 60, 10), 24)

    def get_time_of_day(h):
        if 0 <= h <= 5:
            return "Night"
        elif 6 <= h <= 11:
            return "Morning"
        elif 12 <= h <= 17:
            return "Afternoon"
        else:
            return "Evening"

    dim_time = pd.DataFrame({
        "Time_Key": all_hours * 100 + all_minutes,
        "Hour_No": all_hours.astype(int),
        "Minute_No": all_minutes.astype(int),
        "Time_Of_Day": [get_time_of_day(h) for h in all_hours]
    }).drop_duplicates().sort_values("Time_Key").reset_index(drop=True)

    # Map Time_Key to fact
    df["Time_Key"] = (df["datetime"].dt.hour * 100 + df["datetime"].dt.minute).astype(int)

    # 3. Dimension: Dim_Environment (Weather tercile bands)
    t_out_q = df["T_out"].quantile([1/3, 2/3]).values
    rh_q = df["RH_out"].quantile([1/3, 2/3]).values
    ws_q = df["Windspeed"].quantile([1/3, 2/3]).values
    press_q = df["Press_mm_hg"].quantile([1/3, 2/3]).values
    vis_q = df["Visibility"].quantile([1/3, 2/3]).values

    df["T_out_Band"] = pd.cut(df["T_out"], [-np.inf, *t_out_q, np.inf], labels=["Cold", "Mild", "Warm"]).astype(str)
    df["RH_out_Band"] = pd.cut(df["RH_out"], [-np.inf, *rh_q, np.inf], labels=["Dry", "Moderate", "Humid"]).astype(str)
    df["Wind_Band"] = pd.cut(df["Windspeed"], [-np.inf, *ws_q, np.inf], labels=["Calm", "Breezy", "Windy"]).astype(str)
    df["Press_Band"] = pd.cut(df["Press_mm_hg"], [-np.inf, *press_q, np.inf], labels=["Low", "Normal", "High"]).astype(str)
    df["Visibility_Band"] = pd.cut(df["Visibility"], [-np.inf, *vis_q, np.inf], labels=["Low", "Normal", "High"]).astype(str)

    env_cols = ["T_out_Band", "RH_out_Band", "Wind_Band", "Press_Band", "Visibility_Band"]
    unique_env = df[env_cols].drop_duplicates().reset_index(drop=True)
    unique_env["Environment_Key"] = np.arange(1, len(unique_env) + 1)

    dim_environment = unique_env[["Environment_Key"] + env_cols].copy()

    # Map Environment_Key to fact
    df = df.merge(dim_environment, on=env_cols, how="left")

    # 4. Compute Derived Indoor Features for Fact Table (Chapter 5, Table 13)
    # T_indoor_mean: Mean of T1–T5 and T7–T9 (T6 is exterior outside building north)
    temp_in_cols = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
    # RH_indoor_mean: Mean of RH_1–RH_5 and RH_7–RH_9
    rh_in_cols = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]

    df["Avg_Indoor_Temp"] = df[temp_in_cols].mean(axis=1).round(3)
    df["Avg_Indoor_RH"] = df[rh_in_cols].mean(axis=1).round(3)
    df["Temp_Diff_In_Out"] = (df["Avg_Indoor_Temp"] - df["T_out"]).round(3)

    # 5. Build Fact_Energy_Consumption (Table 14)
    fact_energy_consumption = df[[
        "Fact_ID",
        "Date_Key",
        "Time_Key",
        "Environment_Key",
        "Appliances",
        "lights",
        "Avg_Indoor_Temp",
        "Avg_Indoor_RH",
        "Temp_Diff_In_Out"
    ]].copy()

    fact_energy_consumption = fact_energy_consumption.rename(columns={
        "Appliances": "Appliances_Wh",
        "lights": "Lights_Wh"
    })

    # Load tables into DuckDB warehouse
    print("[-] Loading tables into DuckDB analytical warehouse...")
    con.execute("DROP TABLE IF EXISTS Fact_Energy_Consumption;")
    con.execute("DROP TABLE IF EXISTS Dim_Environment;")
    con.execute("DROP TABLE IF EXISTS Dim_Time;")
    con.execute("DROP TABLE IF EXISTS Dim_Date;")

    con.register("df_dim_date", dim_date)
    con.execute("CREATE TABLE Dim_Date AS SELECT * FROM df_dim_date;")

    con.register("df_dim_time", dim_time)
    con.execute("CREATE TABLE Dim_Time AS SELECT * FROM df_dim_time;")

    con.register("df_dim_env", dim_environment)
    con.execute("CREATE TABLE Dim_Environment AS SELECT * FROM df_dim_env;")

    con.register("df_fact_energy", fact_energy_consumption)
    con.execute("CREATE TABLE Fact_Energy_Consumption AS SELECT * FROM df_fact_energy;")

    print("    DuckDB warehouse tables created successfully.")

    # Export tables to CSV for Power BI / external BI ingestion
    print(f"[-] Exporting CSV tables for Power BI to: {POWERBI_DIR}")
    fact_energy_consumption.to_csv(POWERBI_DIR / "Fact_Energy_Consumption.csv", index=False)
    dim_date.to_csv(POWERBI_DIR / "Dim_Date.csv", index=False)
    dim_time.to_csv(POWERBI_DIR / "Dim_Time.csv", index=False)
    dim_environment.to_csv(POWERBI_DIR / "Dim_Environment.csv", index=False)
    print("    Exported Fact_Energy_Consumption.csv, Dim_Date.csv, Dim_Time.csv, Dim_Environment.csv")

    return fact_energy_consumption, dim_date, dim_time, dim_environment


def validate_warehouse(con: duckdb.DuckDBPyConnection, expected_rows: int) -> pd.DataFrame:
    """Performs integrity verification on the Chapter 6 warehouse."""
    print("[-] Executing Data Warehouse Integrity Checks...")
    checks = []

    # 1. Fact row count reconciliation
    fact_count = con.execute("SELECT COUNT(*) FROM Fact_Energy_Consumption").fetchone()[0]
    checks.append({
        "Check Name": "Row Count Reconciliation",
        "Expected": expected_rows,
        "Observed": fact_count,
        "Status": "PASSED" if fact_count == expected_rows else "FAILED",
        "Details": "Fact_Energy_Consumption matches raw CSV row count (19,735)"
    })

    # 2. Fact_ID uniqueness
    dup_fact = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT Fact_ID, COUNT(*) FROM Fact_Energy_Consumption GROUP BY Fact_ID HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    checks.append({
        "Check Name": "Primary Key Uniqueness (Fact_ID)",
        "Expected": 0,
        "Observed": dup_fact,
        "Status": "PASSED" if dup_fact == 0 else "FAILED",
        "Details": "Zero duplicate Fact_IDs detected"
    })

    # 3. Null Foreign Keys check
    null_fks = con.execute("""
        SELECT COUNT(*) FROM Fact_Energy_Consumption
        WHERE Date_Key IS NULL OR Time_Key IS NULL OR Environment_Key IS NULL
    """).fetchone()[0]
    checks.append({
        "Check Name": "Null Foreign Key Check",
        "Expected": 0,
        "Observed": null_fks,
        "Status": "PASSED" if null_fks == 0 else "FAILED",
        "Details": "All foreign keys populated with valid integer keys"
    })

    # 4. Referential Integrity: Dim_Date
    orphan_date = con.execute("""
        SELECT COUNT(*) FROM Fact_Energy_Consumption f
        LEFT JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        WHERE d.Date_Key IS NULL
    """).fetchone()[0]
    checks.append({
        "Check Name": "Referential Integrity (Dim_Date)",
        "Expected": 0,
        "Observed": orphan_date,
        "Status": "PASSED" if orphan_date == 0 else "FAILED",
        "Details": "Every fact Date_Key resolves to Dim_Date"
    })

    # 5. Referential Integrity: Dim_Time
    orphan_time = con.execute("""
        SELECT COUNT(*) FROM Fact_Energy_Consumption f
        LEFT JOIN Dim_Time t ON f.Time_Key = t.Time_Key
        WHERE t.Time_Key IS NULL
    """).fetchone()[0]
    checks.append({
        "Check Name": "Referential Integrity (Dim_Time)",
        "Expected": 0,
        "Observed": orphan_time,
        "Status": "PASSED" if orphan_time == 0 else "FAILED",
        "Details": "Every fact Time_Key resolves to Dim_Time"
    })

    # 6. Referential Integrity: Dim_Environment
    orphan_env = con.execute("""
        SELECT COUNT(*) FROM Fact_Energy_Consumption f
        LEFT JOIN Dim_Environment e ON f.Environment_Key = e.Environment_Key
        WHERE e.Environment_Key IS NULL
    """).fetchone()[0]
    checks.append({
        "Check Name": "Referential Integrity (Dim_Environment)",
        "Expected": 0,
        "Observed": orphan_env,
        "Status": "PASSED" if orphan_env == 0 else "FAILED",
        "Details": "Every fact Environment_Key resolves to Dim_Environment"
    })

    val_df = pd.DataFrame(checks)
    val_csv_path = OUTPUT_TABLES_DIR / "etl_validation.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"[-] Validation checks saved to: {val_csv_path}")
    print(val_df.to_string(index=False))
    return val_df


def main():
    print("================================================================")
    print("STAGE 1: ETL PIPELINE & STAR SCHEMA (CHAPTER 6 SPECIFICATION)")
    print("================================================================")
    df = load_raw_data()
    con = duckdb.connect(str(DB_PATH))
    try:
        build_star_schema(df, con)
        validate_warehouse(con, len(df))
    finally:
        con.close()
    print("[-] ETL execution completed successfully.\n")


if __name__ == "__main__":
    main()
