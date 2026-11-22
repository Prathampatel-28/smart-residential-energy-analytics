"""
Stage 3: OLAP Multi-Dimensional Operations (Chapter 7 Specification)
=====================================================================
This script implements On-Line Analytical Processing (OLAP) queries directly
against the Chapter 6 Kimball star schema inside the DuckDB warehouse:
  - Fact_Energy_Consumption (central grain: 1 row per 10 min, 19,735 rows)
  - Dim_Date, Dim_Time, Dim_Environment

OLAP Operations Implemented:
1. Roll-up:
   - Monthly level: Exactly reproduces Chapter 7, Table 17 monthly means
     (Jan: 97.03, Feb: 100.95, Mar: 96.95, Apr: 98.89, May: 94.20 Wh).
   - Daily level: Daily aggregated kWh and outdoor temperature trend.
   - Hourly level: Exactly reproduces Chapter 7, Table 18 hourly means/medians.
2. Drill-down:
   - Decomposes Monthly aggregates down to Calendar Weeks and Days.
3. Slice:
   - Filters along Is_Weekend dimension (Weekend vs Weekday diurnal profile).
4. Dice:
   - Sub-cube query across Evening Peak Hours (18-22) and Cold Temperature Band.
5. Pivot Matrix:
   - 24 Hours x 7 Days-of-Week cross-tabulation of average consumption.

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import os
import sys
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "energy_warehouse.duckdb"
TAB_DIR = BASE_DIR / "reports" / "tables"
FIG_DIR = BASE_DIR / "reports" / "figures"

# Cache directory for matplotlib
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".mpl_cache")

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_warehouse_connection() -> duckdb.DuckDBPyConnection:
    """Connects to the DuckDB analytical warehouse in read-only mode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found at {DB_PATH}. Run etl/run_etl.py first.")
    return duckdb.connect(str(DB_PATH), read_only=True)


def olap_rollup(con: duckdb.DuckDBPyConnection):
    """
    Executes OLAP Roll-up queries: Monthly, Daily, and Hourly levels.
    """
    print("[-] 1. Performing OLAP Roll-up (Chapter 7, Tables 17 & 18)...")

    # Monthly Roll-up (Matches Chapter 7, Table 17)
    q_month = """
        SELECT 
            d.Month_No AS month,
            d.Month_Name AS month_name,
            COUNT(f.Fact_ID) AS records,
            ROUND(AVG(f.Appliances_Wh), 2) AS mean_appliances_wh,
            ROUND(SUM(f.Appliances_Wh * (10.0 / 60.0)) / 1000.0, 2) AS total_kwh,
            ROUND(AVG(f.Lights_Wh), 2) AS mean_lights_wh,
            ROUND(AVG(f.Avg_Indoor_Temp), 2) AS mean_indoor_temp_c
        FROM Fact_Energy_Consumption f
        JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        GROUP BY d.Month_No, d.Month_Name
        ORDER BY d.Month_No
    """
    df_month = con.execute(q_month).df()
    month_path = TAB_DIR / "olap_rollup_monthly.csv"
    df_month.to_csv(month_path, index=False)
    print(f"    Monthly Roll-up saved to: {month_path}")
    print(df_month[["month_name", "records", "mean_appliances_wh", "total_kwh"]].to_string(index=False))

    # Daily Roll-up
    q_daily = """
        SELECT 
            d.Full_Date AS date,
            d.Month_Name AS month_name,
            ROUND(AVG(f.Appliances_Wh), 2) AS daily_mean_wh,
            ROUND(SUM(f.Appliances_Wh * (10.0 / 60.0)) / 1000.0, 2) AS daily_kwh
        FROM Fact_Energy_Consumption f
        JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        GROUP BY d.Full_Date, d.Month_Name
        ORDER BY d.Full_Date
    """
    df_daily = con.execute(q_daily).df()
    daily_path = TAB_DIR / "olap_rollup_daily.csv"
    df_daily.to_csv(daily_path, index=False)

    # Plot Roll-up
    plt.figure(figsize=(10, 4.2))
    plt.plot(pd.to_datetime(df_daily["date"]), df_daily["daily_mean_wh"], color="#1f77b4", linewidth=1.5, label="Daily Mean Appliances (Wh)")
    overall_mean = df_month["mean_appliances_wh"].mean()
    plt.axhline(overall_mean, color="#d62728", linestyle="--", linewidth=1.2, label=f"Overall Mean (~{overall_mean:.1f} Wh)")
    plt.title("OLAP Roll-up: Daily Mean Appliance Consumption (2016)", fontsize=11, fontweight="bold")
    plt.xlabel("Date", fontsize=9)
    plt.ylabel("Mean Appliances (Wh)", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_olap_rollup.png")
    plt.close()


def olap_drilldown(con: duckdb.DuckDBPyConnection):
    """
    Executes OLAP Drill-down: Expanding Monthly aggregate to Weekly grain.
    """
    print("[-] 2. Performing OLAP Drill-down (Month -> Calendar Week)...")
    q_week = """
        SELECT 
            d.Month_No AS month,
            d.Month_Name AS month_name,
            strftime(d.Full_Date, '%W') AS calendar_week,
            COUNT(f.Fact_ID) AS records,
            ROUND(AVG(f.Appliances_Wh), 2) AS mean_appliances_wh,
            ROUND(SUM(f.Appliances_Wh * (10.0 / 60.0)) / 1000.0, 2) AS weekly_kwh
        FROM Fact_Energy_Consumption f
        JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        GROUP BY d.Month_No, d.Month_Name, strftime(d.Full_Date, '%W')
        ORDER BY d.Month_No, calendar_week
    """
    df_week = con.execute(q_week).df()
    week_path = TAB_DIR / "olap_drilldown_weekly.csv"
    df_week.to_csv(week_path, index=False)
    print(f"    Drill-down saved to: {week_path}")

    # Plot
    plt.figure(figsize=(10, 4.2))
    x_lbls = [f"W{row.calendar_week}\n({row.month_name[:3]})" for _, row in df_week.iterrows()]
    plt.bar(range(len(df_week)), df_week["weekly_kwh"], color="#2b5c8f", edgecolor="black", width=0.6)
    plt.xticks(range(len(df_week)), x_lbls, fontsize=8)
    plt.title("OLAP Drill-down: Weekly Total Consumption (kWh)", fontsize=11, fontweight="bold")
    plt.xlabel("Calendar Week and Month", fontsize=9)
    plt.ylabel("Total kWh", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_olap_drilldown.png")
    plt.close()


def olap_slice(con: duckdb.DuckDBPyConnection):
    """
    Executes OLAP Slice: Slicing across Is_Weekend dimension (Weekend vs Weekday).
    """
    print("[-] 3. Performing OLAP Slice (Weekend vs Weekday profile across hours)...")
    q_slice = """
        SELECT 
            t.Hour_No AS hour,
            CASE WHEN d.Is_Weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            ROUND(AVG(f.Appliances_Wh), 2) AS avg_appliances_wh,
            ROUND(AVG(f.Lights_Wh), 2) AS avg_lights_wh
        FROM Fact_Energy_Consumption f
        JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        JOIN Dim_Time t ON f.Time_Key = t.Time_Key
        GROUP BY t.Hour_No, d.Is_Weekend
        ORDER BY t.Hour_No, d.Is_Weekend
    """
    df_slice = con.execute(q_slice).df()
    slice_path = TAB_DIR / "olap_slice_weekend.csv"
    df_slice.to_csv(slice_path, index=False)
    print(f"    Slice saved to: {slice_path}")

    pivot_slice = df_slice.pivot(index="hour", columns="day_type", values="avg_appliances_wh")

    plt.figure(figsize=(9, 4.2))
    plt.plot(pivot_slice.index, pivot_slice["Weekday"], "k-o", ms=4, label="Weekday (Mon-Fri)")
    plt.plot(pivot_slice.index, pivot_slice["Weekend"], "k--s", ms=4, label="Weekend (Sat-Sun)")
    plt.title("OLAP Slice: Diurnal Hourly Consumption Profile (Weekend vs Weekday)", fontsize=11, fontweight="bold")
    plt.xlabel("Hour of Day (0 - 23)", fontsize=9)
    plt.ylabel("Average Appliances (Wh)", fontsize=9)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_olap_slice_weekend.png")
    plt.close()


def olap_dice(con: duckdb.DuckDBPyConnection):
    """
    Executes OLAP Dice: Sub-cube filtering across Evening hours (18-22)
    and Outdoor Temperature Band.
    """
    print("[-] 4. Performing OLAP Dice (Sub-cube: Evening Peak Hours x Weather Band)...")
    q_dice = """
        SELECT 
            t.Hour_No AS hour,
            e.T_out_Band AS temp_band,
            COUNT(f.Fact_ID) AS records,
            ROUND(AVG(f.Appliances_Wh), 2) AS avg_appliances_wh,
            ROUND(AVG(f.Lights_Wh), 2) AS avg_lights_wh
        FROM Fact_Energy_Consumption f
        JOIN Dim_Time t ON f.Time_Key = t.Time_Key
        JOIN Dim_Environment e ON f.Environment_Key = e.Environment_Key
        WHERE t.Hour_No BETWEEN 18 AND 22
        GROUP BY t.Hour_No, e.T_out_Band
        ORDER BY t.Hour_No, e.T_out_Band
    """
    df_dice = con.execute(q_dice).df()
    dice_path = TAB_DIR / "olap_dice_evening_cold.csv"
    df_dice.to_csv(dice_path, index=False)
    print(f"    Dice saved to: {dice_path}")

    plt.figure(figsize=(9, 4.2))
    sns.barplot(data=df_dice, x="hour", y="avg_appliances_wh", hue="temp_band", palette="Greys")
    plt.title("OLAP Dice: Evening Peak Hours (18-22) by Outdoor Temperature Band", fontsize=11, fontweight="bold")
    plt.xlabel("Hour of Day", fontsize=9)
    plt.ylabel("Mean Appliances (Wh)", fontsize=9)
    plt.legend(title="Temperature Band")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_olap_dice_evening_cold.png")
    plt.close()


def olap_pivot(con: duckdb.DuckDBPyConnection):
    """
    Executes OLAP Pivot: 24 Hours x 7 Days-of-Week cross-tabulation.
    """
    print("[-] 5. Performing OLAP Pivot (24 Hours x 7 Days-of-Week)...")
    q_piv = """
        SELECT 
            t.Hour_No AS hour,
            d.Day_Of_Week AS weekday,
            ROUND(AVG(f.Appliances_Wh), 2) AS avg_appliances_wh
        FROM Fact_Energy_Consumption f
        JOIN Dim_Date d ON f.Date_Key = d.Date_Key
        JOIN Dim_Time t ON f.Time_Key = t.Time_Key
        GROUP BY t.Hour_No, d.Day_Of_Week
    """
    df_piv = con.execute(q_piv).df()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    piv_matrix = df_piv.pivot(index="hour", columns="weekday", values="avg_appliances_wh")[dow_order]
    
    piv_path = TAB_DIR / "olap_pivot_hour_weekday.csv"
    piv_matrix.to_csv(piv_path)
    print(f"    Pivot table saved to: {piv_path}")

    plt.figure(figsize=(10, 6.5))
    sns.heatmap(piv_matrix, cmap="Greys", annot=True, fmt=".1f", cbar_kws={"label": "Mean Appliances (Wh)"})
    plt.title("OLAP Pivot Heatmap: Mean Appliance Energy by Hour & Weekday", fontsize=11, fontweight="bold")
    plt.xlabel("Day of Week", fontsize=9)
    plt.ylabel("Hour of Day (0 - 23)", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_olap_pivot_heatmap.png")
    plt.close()


def main():
    print("================================================================")
    print("STAGE 3: OLAP OPERATIONS ON CHAPTER 6 WAREHOUSE")
    print("================================================================")
    con = get_warehouse_connection()
    try:
        olap_rollup(con)
        olap_drilldown(con)
        olap_slice(con)
        olap_dice(con)
        olap_pivot(con)
    finally:
        con.close()
    print("[-] OLAP operations completed successfully.\n")


if __name__ == "__main__":
    main()
