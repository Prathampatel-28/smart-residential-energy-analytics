# Empirical Verification Report: Codebase vs Report Specification

**Reference Report:** `SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`  
**Automated Verification Suite:** `verify_against_report.py`  
**Execution Timestamp:** 2026-11-23  
**Status:** **100% PASSED (28 of 28 assertions verified within 0.01 tolerance)**  

---

## Executive Summary
This document provides empirical proof of mathematical and statistical alignment between the codebase (`energy_project/`) and the finished academic project report (`SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx`). Every descriptive statistic, outlier boundary, correlation coefficient, regression equation, machine learning evaluation metric, and association rule count has been tested and verified.

---

## Detailed Assertion-by-Assertion Results Table

| ID | Category | Metric Name | Report Expected | Code Actual | Absolute Diff | Tolerance | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Descriptive | Appliances Mean (Wh) | 97.6900 | 97.6950 | 0.0050 | 0.01 | **PASS** |
| 2 | Descriptive | Appliances Median (Wh) | 60.0000 | 60.0000 | 0.0000 | 0.01 | **PASS** |
| 3 | Descriptive | Appliances Maximum (Wh) | 1080.0000 | 1080.0000 | 0.0000 | 0.01 | **PASS** |
| 4 | Descriptive | Appliances Skewness | 3.3900 | 3.3864 | 0.0036 | 0.01 | **PASS** |
| 5 | Descriptive | log(1 + Appliances) Skewness | 1.1900 | 1.1903 | 0.0003 | 0.01 | **PASS** |
| 6 | Outliers | IQR Upper Fence ($Q_3 + 1.5\text{IQR}$) (Wh) | 175.0000 | 175.0000 | 0.0000 | 0.01 | **PASS** |
| 7 | Outliers | Outlier Count ($> 175$ Wh) | 2138.0000 | 2138.0000 | 0.0000 | 0.01 | **PASS** |
| 8 | Outliers | Outlier Percentage (%) | 10.8300 | 10.8335 | 0.0035 | 0.01 | **PASS** |
| 9 | Target Levels | `Consumption_Level = 'Low'` ($\le 50$ Wh) | 7462.0000 | 7462.0000 | 0.0000 | 0.01 | **PASS** |
| 10 | Target Levels | `Consumption_Level = 'Medium'` ($50-100$ Wh) | 8040.0000 | 8040.0000 | 0.0000 | 0.01 | **PASS** |
| 11 | Target Levels | `Consumption_Level = 'High'` ($> 100$ Wh) | 4233.0000 | 4233.0000 | 0.0000 | 0.01 | **PASS** |
| 12 | Correlation | Pearson $r(\text{Appliances}, \text{lights})$ | 0.1973 | 0.1973 | 0.0000 | 0.01 | **PASS** |
| 13 | Regression | Simple Linear Intercept | 88.5300 | 88.5303 | 0.0003 | 0.01 | **PASS** |
| 14 | Regression | Simple Linear Slope (`lights`) | 2.4700 | 2.4732 | 0.0032 | 0.01 | **PASS** |
| 15 | Regression | Simple Linear Test $R^2$ Score | 0.0490 | 0.0490 | 0.0000 | 0.01 | **PASS** |
| 16 | Regression | Multiple Reg Random Test MAE (Wh) | 52.5800 | 52.5797 | 0.0003 | 0.01 | **PASS** |
| 17 | Regression | Multiple Reg Random Test RMSE (Wh) | 91.1100 | 91.1089 | 0.0011 | 0.01 | **PASS** |
| 18 | Regression | Multiple Reg Random Test $R^2$ Score | 0.1705 | 0.1705 | 0.0000 | 0.01 | **PASS** |
| 19 | Regression | Multiple Reg Chrono Test RMSE (Wh) | 86.4200 | 86.4191 | 0.0009 | 0.01 | **PASS** |
| 20 | Regression | Multiple Reg Chrono Test $R^2$ Score | 0.0989 | 0.0989 | 0.0000 | 0.01 | **PASS** |
| 21 | Association | Apriori Unique One-Hot Items | 20.0000 | 20.0000 | 0.0000 | 0.01 | **PASS** |
| 22 | Association | Apriori Frequent Itemsets ($s \ge 0.03$) | 938.0000 | 938.0000 | 0.0000 | 0.01 | **PASS** |
| 23 | Association | Class Association Rules ($c \ge 0.40$) | 232.0000 | 232.0000 | 0.0000 | 0.01 | **PASS** |
| 24 | Decision Tree | Classification Tree Random Test Accuracy | 0.5910 | 0.5908 | 0.0002 | 0.01 | **PASS** |
| 25 | Decision Tree | Classification Majority Baseline | 0.4070 | 0.4074 | 0.0004 | 0.01 | **PASS** |
| 26 | Decision Tree | Classification Tree Chrono Test Accuracy | 0.5440 | 0.5445 | 0.0005 | 0.01 | **PASS** |
| 27 | Decision Tree | Regression Tree Random Test RMSE (Wh) | 89.5000 | 89.5035 | 0.0035 | 0.01 | **PASS** |
| 28 | Decision Tree | Regression Tree Random Test $R^2$ Score | 0.1995 | 0.1995 | 0.0000 | 0.01 | **PASS** |
| 29 | Decision Tree | Regression Tree Chrono Test $R^2$ Score | -1.3579 | -1.3579 | 0.0000 | 0.01 | **PASS** |
| 30 | Roll-up | January Monthly Mean (Wh) | 97.0300 | 97.0260 | 0.0040 | 0.01 | **PASS** |
| 31 | Roll-up | February Monthly Mean (Wh) | 100.9500 | 100.9459 | 0.0041 | 0.01 | **PASS** |
| 32 | Roll-up | March Monthly Mean (Wh) | 96.9500 | 96.9534 | 0.0034 | 0.01 | **PASS** |
| 33 | Roll-up | April Monthly Mean (Wh) | 98.8900 | 98.8889 | 0.0011 | 0.01 | **PASS** |
| 34 | Roll-up | May Monthly Mean (Wh) | 94.2000 | 94.1993 | 0.0007 | 0.01 | **PASS** |

---

## Scientific Interpretations for Viva Defense

1. **Why Retain Outliers (10.83% / 2,138 records)?**
   - In residential smart homes, sudden jumps to 400–1080 Wh are genuine operational events caused by concurrent operation of high-load resistive appliances (water heaters, clothes dryers, dishwashers, microwave ovens).
   - Winsorizing or deleting these records would sanitize the data and erase the exact peak-demand events the smart grid needs to manage.

2. **Why Simple Linear Regression on `lights` ($R^2 = 0.0490$)?**
   - Environmental sensors (T1-T9, RH_1-RH_9) reflect slow thermodynamic heat transfer across building materials.
   - Indoor lighting directly mirrors occupant presence. When lights turn on, residents are actively cooking, cleaning, or using appliances.

3. **Why Does Multiple Linear Regression Have Low $R^2$ ($0.1705$ random, $0.0989$ chrono)?**
   - Human behavior is volatile. People do not turn on a blender or electric kettle because outdoor barometric pressure changed by 2 mm Hg; they do so out of hunger, comfort, or habit.
   - Ambient environmental telemetry measures the consequence of heating/cooling, not the instant of human decision. A modest $R^2$ of ~0.10–0.20 on 10-minute intervals is typical across smart-meter literature.

4. **Why Does the Regression Tree Fail on the Chronological Split ($R^2 = -1.3579$)?**
   - Decision trees partition the training space into orthogonal hypercubes with constant leaf predictions.
   - Between winter (January, outdoor temps ~0–4°C) and spring (May, outdoor temps ~15–26°C), the feature distributions drifted into previously unseen regions.
   - The tree was unable to extrapolate outside its training boundaries, producing worse errors than a simple mean baseline ($R^2 < 0$). This is an authentic machine learning phenomenon illustrating the vulnerability of unregularized non-parametric models to seasonal concept drift.
