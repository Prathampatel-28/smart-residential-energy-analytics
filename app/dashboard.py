"""
Smart Residential Energy Consumption Analytics Dashboard
=========================================================
Interactive multi-page Streamlit application mirroring Chapter 13 of the
academic project report: SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx.

Architecture (4 Main Report Pages + Academic About/Limitations):
1. Page 1: Energy Consumption Overview (Chapter 13.2)
   - KPI metrics: Mean Wh, Max Wh, Total Electricity (SUM/1000 = ~1,928 kWh), % High Consumption.
   - Visuals: Binned frequency distribution (with non-overlapping Mean/Median legend),
     Monthly average consumption computed from filtered data matching Table 17.
   - Slicers: Month, Day type (Weekday / Weekend).
2. Page 2: Time-Based Consumption Analysis (Chapter 13.3)
   - Visuals: 24-hour diurnal profile, daily time-series trend, day-of-week bars,
     Hour x Weekday interactive heat map matrix.
   - Slicers: Month, Day Type, Hour range, Weekday selector.
3. Page 3: Environmental Analysis (Chapter 13.4)
   - Visuals: Outdoor temperature vs energy scatter, outdoor humidity scatter,
     climatic band aggregations, indoor vs outdoor thermal gradient over time.
   - Slicers: Temperature band, Humidity band, Month filter.
4. Page 4: Correlation and Insights (Chapter 13.5)
   - Visuals: Top correlation bar chart with named variables on x-axis (rotated),
     regression model comparison, classification tree results (59.1% test, 40.7% baseline,
     54.4% chrono, hour importance 40.7%), filterable association rules table, and viva takeaways.
5. Tab 5: About, Methodology & Limitations (Chapter 16)
   - Data provenance, Star schema Kimball design, real monthly mean T_out values (Jan: 4.13°C, May: 13.76°C),
     and honest scientific discussion of R².

Author: Senior Data Scientist for B.Tech/BCA Project
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Smart Home Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "energy_cleaned.csv"
POWERBI_DIR = BASE_DIR / "data" / "processed" / "powerbi"
TAB_DIR = BASE_DIR / "reports" / "tables"


@st.cache_data
def load_data():
    """Loads cleaned processed dataset."""
    if not DATA_PROCESSED.exists():
        return None
    df = pd.read_csv(DATA_PROCESSED)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_table(name: str):
    """Loads a precomputed summary table from reports/tables/."""
    path = TAB_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


df = load_data()

# Custom CSS styling for an academic dashboard
st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: 700; color: #1E293B; margin-bottom: 2px; }
    .sub-title { font-size: 14px; color: #64748B; margin-bottom: 18px; }
    .kpi-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; text-align: center; }
    .kpi-val { font-size: 24px; font-weight: 700; color: #0F172A; }
    .kpi-lbl { font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/fluency/96/smart-home-shield.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Report Page:",
    [
        "Page 1: Energy Overview",
        "Page 2: Time-Based Analysis",
        "Page 3: Environmental Analysis",
        "Page 4: Correlation & Insights",
        "About & Limitations"
    ]
)

if df is None:
    st.error("Processed data file not found! Please run `python run_all.py` first to generate analytical artifacts.")
    st.stop()


# ==============================================================================
# PAGE 1: ENERGY CONSUMPTION OVERVIEW (Chapter 13.2)
# ==============================================================================
if page == "Page 1: Energy Overview":
    st.markdown('<div class="main-title">⚡ Page 1: Energy Consumption Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-level KPIs, appliance consumption distribution, and seasonal monthly trends</div>', unsafe_allow_html=True)

    # Sidebar Slicers
    st.sidebar.markdown("### Page 1 Filters (Slicers)")
    months = ["All Months"] + list(df["month_name"].unique())
    sel_month = st.sidebar.selectbox("Filter by Month:", months, key="p1_month")

    day_options = ["All Days", "Weekday Only", "Weekend Only"]
    sel_day_type = st.sidebar.selectbox("Filter by Day Type:", day_options, key="p1_day_type")

    # Filter data dynamically
    filtered_df = df.copy()
    if sel_month != "All Months":
        filtered_df = filtered_df[filtered_df["month_name"] == sel_month]
    if sel_day_type == "Weekday Only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 0]
    elif sel_day_type == "Weekend Only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 1]

    # DAX-equivalent KPIs (Chapter 13.6)
    # Total Electricity kWh: SUM(Appliances of filtered rows) / 1000
    avg_appl = filtered_df["Appliances"].mean()
    med_appl = filtered_df["Appliances"].median()
    max_appl = filtered_df["Appliances"].max()
    tot_kwh = filtered_df["Appliances"].sum() / 1000.0
    high_pct = (filtered_df["Appliances"] > 100).mean() * 100.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_appl:.1f} <span style="font-size:15px;">Wh</span></div><div class="kpi-lbl">Avg Consumption</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{max_appl:.0f} <span style="font-size:15px;">Wh</span></div><div class="kpi-lbl">Peak Interval Demand</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{tot_kwh:,.1f} <span style="font-size:15px;">kWh</span></div><div class="kpi-lbl">Total Electricity Consumed (DAX SUM/1000)</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{high_pct:.1f}%</div><div class="kpi-lbl">High Consumption Intervals (&gt;100 Wh)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("##### Appliances Frequency Distribution (Histogram)")
        show_log = st.checkbox("View log(1 + Appliances) transformed scale", value=False)
        if show_log:
            fig_hist = px.histogram(
                filtered_df, x="log_Appliances", nbins=50,
                title="Binned Distribution: log(1 + Appliances Wh) [Skewness: 1.19]",
                color_discrete_sequence=["#475569"]
            )
            fig_hist.update_layout(xaxis_title="log(1 + Wh)", yaxis_title="Count", height=340)
        else:
            fig_hist = px.histogram(
                filtered_df, x="Appliances", nbins=60,
                title="Binned Distribution: Appliances Wh [Skewness: 3.39]",
                color_discrete_sequence=["#334155"]
            )
            # Fix overlapping Mean/Median annotations: use clean lines + legend
            fig_hist.add_vline(x=avg_appl, line_dash="dash", line_color="#EF4444", line_width=2)
            fig_hist.add_vline(x=med_appl, line_dash="dot", line_color="#10B981", line_width=2)
            
            fig_hist.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="#EF4444", dash="dash", width=2),
                name=f"Mean: {avg_appl:.1f} Wh"
            ))
            fig_hist.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="#10B981", dash="dot", width=2),
                name=f"Median: {med_appl:.1f} Wh"
            ))
            fig_hist.update_layout(
                xaxis_title="Energy per 10-min interval (Wh)", yaxis_title="Count", height=340,
                legend=dict(x=0.65, y=0.95, bgcolor="rgba(255,255,255,0.7)")
            )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("##### Monthly Mean Consumption (OLAP Roll-up)")
        # Computed from FILTERED data: with default 'All' filters, exactly matches Table 17
        mon_summary = filtered_df.groupby(["month", "month_name"])["Appliances"].agg(["mean", "count"]).reset_index()
        mon_summary = mon_summary.sort_values("month")

        fig_mon = px.bar(
            mon_summary, x="month_name", y="mean", text_auto=".2f",
            title="Average Appliance Consumption by Month (Wh) [Table 17]",
            color_discrete_sequence=["#64748B"]
        )
        fig_mon.update_layout(xaxis_title="Month (2016)", yaxis_title="Mean Consumption (Wh)", height=340)
        fig_mon.update_traces(textposition="outside")
        st.plotly_chart(fig_mon, use_container_width=True)

    # Summary table from warehouse
    st.markdown("##### Monthly Aggregation Summary (Filtered Data)")
    st.dataframe(
        mon_summary.rename(columns={"month_name": "Month", "mean": "Mean Appliances (Wh)", "count": "Record Count", "month": "Month No"}),
        use_container_width=True, hide_index=True
    )


# ==============================================================================
# PAGE 2: TIME-BASED CONSUMPTION ANALYSIS (Chapter 13.3)
# ==============================================================================
elif page == "Page 2: Time-Based Analysis":
    st.markdown('<div class="main-title">🕒 Page 2: Time-Based Consumption Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Diurnal curves, weekday variations, 24-hour heatmaps, and continuous time trends</div>', unsafe_allow_html=True)

    # Slicers
    st.sidebar.markdown("### Page 2 Filters (Slicers)")
    months_p2 = ["All Months"] + list(df["month_name"].unique())
    sel_month_p2 = st.sidebar.selectbox("Filter by Month:", months_p2, key="p2_month")

    day_opt_p2 = ["All Days", "Weekday Only", "Weekend Only"]
    sel_day_type_p2 = st.sidebar.selectbox("Filter by Day Type:", day_opt_p2, key="p2_day_type")

    hr_range = st.sidebar.slider("Select Hour Range:", 0, 23, (0, 23), key="p2_hour")
    dow_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sel_dow = st.sidebar.multiselect("Filter by Day of Week:", dow_list, default=dow_list, key="p2_dow")

    # Apply all slicers
    filtered_time = df.copy()
    if sel_month_p2 != "All Months":
        filtered_time = filtered_time[filtered_time["month_name"] == sel_month_p2]
    if sel_day_type_p2 == "Weekday Only":
        filtered_time = filtered_time[filtered_time["is_weekend"] == 0]
    elif sel_day_type_p2 == "Weekend Only":
        filtered_time = filtered_time[filtered_time["is_weekend"] == 1]
    filtered_time = filtered_time[
        (filtered_time["hour"] >= hr_range[0]) & 
        (filtered_time["hour"] <= hr_range[1]) & 
        (filtered_time["day_name"].isin(sel_dow))
    ]

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### Diurnal Hourly Profile (Mean vs Median)")
        hourly_df = filtered_time.groupby("hour")["Appliances"].agg(["mean", "median"]).reset_index()
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=hourly_df["hour"], y=hourly_df["mean"], mode="lines+markers", name="Mean Wh", line=dict(color="#1E293B", width=2.2)))
        fig_hr.add_trace(go.Scatter(x=hourly_df["hour"], y=hourly_df["median"], mode="lines+markers", name="Median Wh", line=dict(color="#94A3B8", dash="dash")))
        fig_hr.update_layout(
            title="Hourly Appliance Energy Profile [Table 18]",
            xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", dtick=2),
            yaxis_title="Energy (Wh)", height=320, legend=dict(x=0.02, y=0.95)
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    with col_t2:
        st.markdown("##### Day of Week Variation")
        dow_present = [d for d in dow_list if d in filtered_time["day_name"].unique()]
        dow_df = filtered_time.groupby("day_name")["Appliances"].mean().reindex(dow_present).reset_index()
        fig_dow = px.bar(
            dow_df, x="day_name", y="Appliances", text_auto=".1f",
            title="Mean Consumption by Day of Week",
            color_discrete_sequence=["#475569"]
        )
        fig_dow.update_layout(xaxis_title="Day of Week", yaxis_title="Mean Appliances (Wh)", height=320)
        fig_dow.update_traces(textposition="outside")
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("##### Continuous Daily Energy Trend (Jan 11 – May 27, 2016)")
    daily_df = filtered_time.set_index("date")["Appliances"].resample("D").mean().reset_index()
    fig_daily = px.line(
        daily_df, x="date", y="Appliances",
        title="Daily Mean Appliance Consumption over Time",
        color_discrete_sequence=["#0F172A"]
    )
    fig_daily.add_hline(y=df["Appliances"].mean(), line_dash="dash", line_color="#EF4444", annotation_text="Overall Mean (97.7 Wh)")
    fig_daily.update_layout(xaxis_title="Date", yaxis_title="Daily Mean Wh", height=280)
    st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown("##### 24-Hour × 7-Day Matrix Heat Map (Chapter 13.3)")
    pivot_matrix = filtered_time.pivot_table(index="day_name", columns="hour", values="Appliances", aggfunc="mean").reindex(dow_present)
    fig_heat = px.imshow(
        pivot_matrix, labels=dict(x="Hour of Day", y="Day of Week", color="Mean Wh"),
        color_continuous_scale="Greys", aspect="auto"
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# PAGE 3: ENVIRONMENTAL ANALYSIS (Chapter 13.4)
# ==============================================================================
elif page == "Page 3: Environmental Analysis":
    st.markdown('<div class="main-title">🌡️ Page 3: Environmental Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Outdoor climate, indoor thermal gradients, and weather band aggregations</div>', unsafe_allow_html=True)

    # Slicers
    st.sidebar.markdown("### Page 3 Filters (Slicers)")
    months_p3 = ["All Months"] + list(df["month_name"].unique())
    sel_month_p3 = st.sidebar.selectbox("Filter by Month:", months_p3, key="p3_month")

    t_out_q = df["T_out"].quantile([1/3, 2/3]).values
    temp_bands = ["All Bands", "Cold", "Mild", "Warm"]
    sel_temp_band = st.sidebar.selectbox("Filter Outdoor Temperature Band:", temp_bands, key="p3_temp")

    rh_q = df["RH_out"].quantile([1/3, 2/3]).values
    humidity_bands = ["All Bands", "Dry", "Moderate", "Humid"]
    sel_hum_band = st.sidebar.selectbox("Filter Outdoor Humidity Band:", humidity_bands, key="p3_hum")

    filtered_env = df.copy()
    filtered_env["T_band"] = pd.cut(filtered_env["T_out"], [-np.inf, *t_out_q, np.inf], labels=["Cold", "Mild", "Warm"]).astype(str)
    filtered_env["RH_band"] = pd.cut(filtered_env["RH_out"], [-np.inf, *rh_q, np.inf], labels=["Dry", "Moderate", "Humid"]).astype(str)

    if sel_month_p3 != "All Months":
        filtered_env = filtered_env[filtered_env["month_name"] == sel_month_p3]
    if sel_temp_band != "All Bands":
        filtered_env = filtered_env[filtered_env["T_band"] == sel_temp_band]
    if sel_hum_band != "All Bands":
        filtered_env = filtered_env[filtered_env["RH_band"] == sel_hum_band]

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("##### Outdoor Temperature vs Appliances (Scatter Plot)")
        sample_env = filtered_env.sample(min(2000, len(filtered_env)), random_state=42) if len(filtered_env) > 0 else filtered_env
        fig_sc1 = px.scatter(
            sample_env, x="T_out", y="Appliances", opacity=0.35,
            title="T_out (°C) vs Appliances (Wh) [Pearson r = +0.0991]",
            color_discrete_sequence=["#334155"]
        )
        fig_sc1.update_layout(xaxis_title="Outdoor Temperature (°C)", yaxis_title="Appliances (Wh)", height=320)
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_e2:
        st.markdown("##### Outdoor Humidity vs Appliances (Scatter Plot)")
        fig_sc2 = px.scatter(
            sample_env, x="RH_out", y="Appliances", opacity=0.35,
            title="RH_out (%) vs Appliances (Wh) [Pearson r = -0.1523]",
            color_discrete_sequence=["#475569"]
        )
        fig_sc2.update_layout(xaxis_title="Outdoor Humidity (%)", yaxis_title="Appliances (Wh)", height=320)
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("##### Indoor Thermal Gradient: Indoor Mean vs Outdoor Temperature")
    daily_temps = filtered_env.set_index("date")[["T_indoor_mean", "T_out", "T_diff_in_out"]].resample("D").mean().reset_index()
    fig_temp_line = go.Figure()
    fig_temp_line.add_trace(go.Scatter(x=daily_temps["date"], y=daily_temps["T_indoor_mean"], name="Indoor Mean Temp (°C)", line=dict(color="#0F172A", width=2)))
    fig_temp_line.add_trace(go.Scatter(x=daily_temps["date"], y=daily_temps["T_out"], name="Outdoor Temp T_out (°C)", line=dict(color="#94A3B8", dash="dash")))
    fig_temp_line.add_trace(go.Scatter(x=daily_temps["date"], y=daily_temps["T_diff_in_out"], name="Thermal Gradient (Indoor - Outdoor)", line=dict(color="#F59E0B", width=1.5)))
    fig_temp_line.update_layout(
        title="Indoor vs Outdoor Temperature & Thermal Gradient Trend",
        xaxis_title="Date", yaxis_title="Temperature (°C)", height=310, legend=dict(x=0.02, y=0.95)
    )
    st.plotly_chart(fig_temp_line, use_container_width=True)


# ==============================================================================
# PAGE 4: CORRELATION AND INSIGHTS (Chapter 13.5)
# ==============================================================================
elif page == "Page 4: Correlation & Insights":
    st.markdown('<div class="main-title">📊 Page 4: Correlation and Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical correlations, regression benchmarks, decision trees, and association rules</div>', unsafe_allow_html=True)

    # Slicers for Page 4
    st.sidebar.markdown("### Page 4 Slicers & Filters")
    top_n = st.sidebar.slider("Number of Top Correlations to Display:", 5, 20, 10, key="p4_top_n")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("##### Top Correlations with Appliances [Table 22]")
        corr_df = load_table("corr_with_appliances.csv")
        if corr_df is not None:
            # Fix column renaming so variable names appear on x-axis
            if "Unnamed: 0" in corr_df.columns:
                corr_df = corr_df.rename(columns={"Unnamed: 0": "Variable"})
            elif "index" in corr_df.columns:
                corr_df = corr_df.rename(columns={"index": "Variable"})
            elif corr_df.columns[0] != "Variable":
                corr_df = corr_df.rename(columns={corr_df.columns[0]: "Variable"})

            top_corr = corr_df.head(top_n).copy()
            fig_bar_corr = px.bar(
                top_corr, x="Variable", y="Pearson", text_auto=".3f",
                title=f"Top {top_n} Predictors Correlated with Appliances",
                color_discrete_sequence=["#334155"]
            )
            # Ensure variable names are clearly visible and rotated on the x-axis
            fig_bar_corr.update_layout(
                xaxis_title="Predictor Variable", yaxis_title="Pearson Correlation",
                xaxis=dict(tickangle=-45, type='category'), height=350
            )
            st.plotly_chart(fig_bar_corr, use_container_width=True)
        else:
            st.info("Correlation table loading...")

    with col_c2:
        st.markdown("##### Regression Model Benchmark [Tables 24 & 25]")
        reg_comp = load_table("regression_model_comparison.csv")
        if reg_comp is not None:
            st.dataframe(reg_comp, use_container_width=True, hide_index=True)
        
        st.markdown("##### Regression Tree Benchmark [Table 35]")
        tree_comp = load_table("decision_tree_metrics.csv")
        if tree_comp is not None:
            st.dataframe(tree_comp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Item 8: Classification Tree Results (Chapter 12, Table 32 & 33)
    st.markdown("##### Classification Tree Performance (Consumption_Level: Low / Medium / High)")
    st.caption("Hyperparameters: CART Gini Impurity, max_depth=5, min_samples_leaf=50, random_state=42")

    ct1, ct2, ct3, ct4 = st.columns(4)
    with ct1:
        st.markdown('<div class="kpi-card"><div class="kpi-val">59.1%</div><div class="kpi-lbl">Random Test Accuracy</div></div>', unsafe_allow_html=True)
    with ct2:
        st.markdown('<div class="kpi-card"><div class="kpi-val">40.7%</div><div class="kpi-lbl">Majority Baseline</div></div>', unsafe_allow_html=True)
    with ct3:
        st.markdown('<div class="kpi-card"><div class="kpi-val">54.4%</div><div class="kpi-lbl">Chrono Test Accuracy</div></div>', unsafe_allow_html=True)
    with ct4:
        st.markdown('<div class="kpi-card"><div class="kpi-val">40.7%</div><div class="kpi-lbl">Top Feature: hour Importance</div></div>', unsafe_allow_html=True)

    col_cr, col_imp = st.columns([1, 1])
    with col_cr:
        st.markdown("###### Detailed Classification Report [Table 33]")
        cr_table = load_table("classification_report.csv")
        if cr_table is not None:
            if "Unnamed: 0" in cr_table.columns:
                cr_table = cr_table.rename(columns={"Unnamed: 0": "Class / Metric"})
            st.dataframe(cr_table, use_container_width=True, hide_index=True)
    with col_imp:
        st.markdown("###### Top Feature Importances [Table 45]")
        imp_table = load_table("tree_feature_importance.csv")
        if imp_table is not None:
            if "Unnamed: 0" in imp_table.columns:
                imp_table = imp_table.rename(columns={"Unnamed: 0": "Feature", imp_table.columns[1]: "Gini Importance"})
            st.dataframe(imp_table.head(8), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Item 8: Filterable Association Rules Table (Chapter 11, Tables 28 – 31)
    st.markdown("##### Filterable Association Rules Explorer (Apriori & Multilevel)")
    rules_table = load_table("association_rules.csv")
    if rules_table is not None:
        consequent_options = ["All"] + sorted(list(rules_table["Consequent"].unique()))
        
        ar_col1, ar_col2, ar_col3 = st.columns(3)
        with ar_col1:
            sel_cons = st.selectbox("Filter Target Consequent:", consequent_options, key="ar_cons")
        with ar_col2:
            min_c = st.slider("Min Confidence:", 0.40, float(rules_table["Confidence"].max()), 0.40, 0.05, key="ar_conf")
        with ar_col3:
            min_l = st.slider("Min Lift:", 1.0, float(rules_table["Lift"].max()), 1.0, 0.1, key="ar_lift")

        filtered_rules = rules_table.copy()
        if sel_cons != "All":
            filtered_rules = filtered_rules[filtered_rules["Consequent"] == sel_cons]
        filtered_rules = filtered_rules[
            (filtered_rules["Confidence"] >= min_c) & 
            (filtered_rules["Lift"] >= min_l)
        ].sort_values("Lift", ascending=False)

        st.caption(f"Showing {len(filtered_rules)} of {len(rules_table)} total class rules (min support = 0.03, max length = 4)")
        st.dataframe(filtered_rules, use_container_width=True, hide_index=True)

        with st.expander("View Multilevel Rules with Individual Hour (min support = 0.02)"):
            multi_rules = load_table("association_rules_multilevel.csv")
            if multi_rules is not None:
                st.dataframe(multi_rules, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### Core Findings & Viva Discussion Takeaways (Chapter 14)")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("""
        **1. Lighting as Occupancy Proxy**  
        - `lights` is the single strongest non-random predictor ($r = 0.1973$).  
        - Simple regression: $\\text{Appliances} = 88.53 + 2.47 \\times \\text{lights}$.  
        - Turning on lights directly marks human active presence in rooms.
        """)
    with col_t2:
        st.markdown("""
        **2. Diurnal Evening Peaks**  
        - Consumption peaks sharply between 18:00 and 20:00 (reaching 190.4 Wh at 18:00).  
        - Night standby load remains at baseline 50–60 Wh.  
        - Driven by cooking, washing, entertainment, and household chores.
        """)
    with col_t3:
        st.markdown("""
        **3. Low R² & Seasonal Drift**  
        - Multiple regression random $R^2 = 0.1705$; chronological $R^2 = 0.0989$.  
        - Regression tree failed on future test ($R^2 = -1.3579$), suggesting a seasonal shift from winter to spring.  
        - High-power appliances draw sudden impulse bursts independent of slow ambient weather.
        """)


# ==============================================================================
# ABOUT & LIMITATIONS TAB
# ==============================================================================
else:
    st.markdown('<div class="main-title">ℹ️ About the Project, Methodology & Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Specification Reference: SMART_RESIDENTIAL_ENERGY_CONSUMPTION_PROJECT_REPORT.docx</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Project Background
    This project presents an empirical data analytics and data warehousing pipeline on the 
    **Appliances Energy Prediction Dataset** (Candanedo et al., 2017).
    - **Total Records:** 19,735 observations recorded at a 10-minute sampling cadence.
    - **Monitoring Period:** 11 January 2016 to 27 May 2016 (137 consecutive days).
    - **Monitored Variables:** 29 raw columns including appliance energy, lights, indoor room temperatures and humidities (T1-T9, RH_1-RH_9), and weather metrics.

    ---
    ### 2. Dimensional Star Schema (Chapter 6)
    The dimensional model organizes 10-minute sensor telemetry into Kimball star schema tables inside **DuckDB** and exported as CSVs for Power BI:
    1. **`Fact_Energy_Consumption`** (19,735 rows): Fact_ID, Date_Key, Time_Key, Environment_Key, Appliances_Wh, Lights_Wh, Avg_Indoor_Temp, Avg_Indoor_RH, Temp_Diff_In_Out.
    2. **`Dim_Date`**: Full_Date, Day_No, Month_No, Month_Name, Day_Of_Week, Is_Weekend.
    3. **`Dim_Time`**: Hour_No, Minute_No, Time_Of_Day (Night, Morning, Afternoon, Evening).
    4. **`Dim_Environment`**: T_out_Band, RH_out_Band, Wind_Band, Press_Band, Visibility_Band.

    ---
    ### 3. Machine Learning Evaluation & Honest Limitations (Chapter 16)
    - **Why is R² low-to-moderate (~0.10 - 0.20)?**  
      High-power household appliances (electric kettles, washing machines, dishwashers) operate intermittently for short durations (10 to 45 minutes) based on human decision-making. Ambient environmental sensors change very slowly with high thermal inertia and cannot directly observe human intentions.
    - **Why did the Regression Tree fail on chronological evaluation ($R^2 = -1.3579$)?**  
      Decision trees partition feature space using orthogonal cuts based on training values. Because outdoor weather transitioned from winter (January mean T_out = 4.13°C, February = 4.82°C, March = 5.44°C) to spring (April = 8.51°C, May = 13.76°C, with daily highs exceeding 26°C), the tree encountered unseen input spaces and predicted inaccurate step-function values. This demonstrates why time-based evaluation is scientifically essential.
    - **Supplemental Modules:**  
      Advanced explorations (Ridge/Lasso, scratch Gradient Descent, formal hypothesis inference tests, VIF screening, and peak demand cutoff analysis) are preserved in the `extras/` directory and labelled as supplementary research outside the main report text.
    """)
