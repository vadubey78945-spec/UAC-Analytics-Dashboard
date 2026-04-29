# app.py — UAC System Capacity & Care Load Analytics
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.metrics import compute_kpis
from src.forecasting import forecast
from src.simulation import simulate_scenario
from src.insights import generate_insights

DATA_PATH = ROOT / "data" / "HHS_Unaccompanied_Alien_Children_Program.csv"

st.set_page_config(page_title="UAC Analytics Platform", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
}

section[data-testid="stSidebar"]{background:#0f1117}
section[data-testid="stSidebar"] *{color:#e0e0e0!important;}

/* Centered & Premium KPI Cards like old project */
.kpi-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 24px 20px;
    margin-bottom: 15px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.kpi-label {
    font-size: 0.75rem;
    color: #8b92a5;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
}
.kpi-formula {
    font-size: 0.75rem;
    color: #6a7390;
    margin-top: 15px;
    font-style: italic;
}
.kpi-interp {
    font-size: 0.75rem;
    color: #a0a8c0;
    margin-top: 5px;
}

/* Cyan Section Headers */
.cyan-header {
    color: #00d2d3 !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    border-left: 4px solid #00d2d3;
    padding-left: 12px;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

.warn-box{background:#2a1f10;border-left:4px solid #e8a020;border-radius:8px;
          padding:14px 18px;margin:10px 0;font-size:1rem;color:#f0c070}
.info-box{background:#0f2030;border-left:4px solid #00d2d3;border-radius:8px;
          padding:14px 18px;margin:10px 0;font-size:1rem;color:#90c8f0}
</style>
""", unsafe_allow_html=True)

COLORS = {"cbp":"#4a90d9","hhs":"#e8734a","load":"#9b59b6",
          "pressure":"#e8a020","discharge":"#27ae60","forecast":"#e84393"}


def kpi_card(label, value, formula, interp, color="#00d2d3"):
    st.markdown(f"""<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value" style="color:{color}">{value}</div>
    <div class="kpi-formula">{formula}</div>
    <div class="kpi-interp">{interp}</div></div>""", unsafe_allow_html=True)


def section_header(title, subtitle=""):
    st.markdown(f'<div class="cyan-header">{title}</div>', unsafe_allow_html=True)
    if subtitle: st.caption(subtitle)
    st.markdown("---")


@st.cache_data(show_spinner="Loading & preprocessing dataset...")
def get_data(path):
    raw_df, debug_info = load_data(path)
    df, prep_report    = preprocess(raw_df)
    return df, debug_info, prep_report


with st.sidebar:
    st.markdown("## UAC Analytics")
    st.caption("System Capacity & Care Load Analytics")
    st.markdown("---")
    
    # Data source selection with option to upload custom CSV or use bundled dataset. 
    data_source = st.radio("Select Data Source", ["Bundled Dataset", "Upload Custom CSV"])
    
    uploaded = None
    if data_source == "Upload Custom CSV":
        uploaded = st.file_uploader("Upload custom CSV", type="csv", label_visibility="collapsed")
    else:
        st.info("Bundled Dataset Loaded")

st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#00d2d3;'>"
            "UAC System Capacity & Care Load Analytics</h1>", unsafe_allow_html=True)

# --- DATA LOADING LOGIC ---
try:
    if data_source == "Upload Custom CSV":
        if uploaded is not None:
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                shutil.copyfileobj(uploaded, tmp)
                tmp_path = Path(tmp.name)
            df, debug_info, prep_report = get_data(tmp_path)
        else:
            st.markdown("---")
            st.info("Upload a CSV from the sidebar or select the bundled dataset to begin.")
            st.stop() 
            
    else:
        df, debug_info, prep_report = get_data(DATA_PATH)
        
except (FileNotFoundError, KeyError, ValueError) as err:
    st.error(f"Data Load Failed: {err}")
    st.info("Ensure HHS_Unaccompanied_Alien_Children_Program.csv is in data/ folder.")
    st.stop()

# --- INTERACTIVE CONTROLS & DATE FILTERING ---
with st.sidebar:
    horizon = st.radio("Forecast horizon (days)", [7, 14], index=0, horizontal=True)
    st.markdown("---")
    date_placeholder = st.empty()
    st.markdown("---")
    st.caption("UAC Analytics Platform v1.0")

min_date = df.index.min().date()
max_date = df.index.max().date()

with date_placeholder.container():
    st.markdown("**🗓️ Date Range**")
    start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
    end_date   = st.date_input("End", value=max_date, min_value=start_date, max_value=max_date)

# Filter dataframe based on separate dates
if start_date <= end_date:
    df_view = df.loc[str(start_date):str(end_date)].copy()
else:
    st.sidebar.error("Error: End date cannot be before Start date")
    st.stop()

if df_view.empty:
    st.warning("No data in selected date range."); st.stop()

# --- DEVELOPER DEBUG MODE ---
if "debug" in st.query_params and st.query_params["debug"] == "true":
    st.sidebar.error("🛠️ DEVELOPER DEBUG MODE ACTIVE")
    with st.expander("🛠️ Developer Debug Panel", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw Columns**"); st.code("\n".join(debug_info["raw_columns"]))
            st.markdown("**Mapping Log**")
            st.dataframe(pd.DataFrame(debug_info["mapping_log"]), width="stretch")
        with c2:
            st.markdown("**Sample Data**")
            st.dataframe(pd.DataFrame(prep_report["head"]), width="stretch")
            st.markdown("**Null Summary**")
            st.dataframe(pd.DataFrame({"Before": prep_report["null_summary"]["before"],
                                       "After":  prep_report["null_summary"]["after"]}),
                         width="stretch")
        st.dataframe(pd.DataFrame(prep_report["constraint_violations"]), width="stretch")

# --- KPI COMPUTATION, FORECASTING, AND INSIGHTS GENERATION ---
kpis           = compute_kpis(df_view)
fc_discharge   = forecast(df_view, "HHS_Discharges",    horizon)
fc_load        = forecast(df_view, "Total_System_Load", horizon)
insights_list  = generate_insights(kpis, df_view)

st.caption(f"Data: {start_date} ➔ {end_date} | {len(df_view):,} records in view")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Overview","Trends","Bottleneck Analysis","Forecast & Simulation","Insights"])

with tab1:
    section_header("System Overview","Key performance indicators across the UAC pipeline")
    c1,c2,c3 = st.columns(3)
    with c1: kpi_card("Total Children Under Care",f"{kpis['total_under_care']:,.0f}",
                       "CBP_In_Custody + HHS_In_Care","Combined real-time headcount across both environments.")
    with c2:
        col = "#e84343" if kpis["net_intake_pressure"]>0 else "#27ae60"
        kpi_card("Net Intake Pressure",f"{kpis['net_intake_pressure']:+,.0f}",
                 "CBP_Transfers_Out - HHS_Discharges","Positive = system absorbing more than releasing.",col)
    with c3: kpi_card("Discharge Offset Ratio",f"{kpis['discharge_offset_ratio']:.2%}",
                       "HHS_Discharges / HHS_In_Care",
                       f"~{kpis['discharge_offset_ratio']:.1%} is operationally expected at {kpis['hhs_in_care_latest']:,.0f} base population.","#27ae60")
    c4,c5,c6 = st.columns(3)
    with c4: kpi_card("Care Load Volatility Index",f"{kpis['volatility_index']:,.1f}",
                       "14-day rolling std(Total_System_Load)","Higher = more unpredictable day-to-day swings.","#9b59b6")
    with c5: kpi_card("Backlog Accumulation Rate",f"{kpis['backlog_pct']:.1f}%",
                       "% days with Net_Intake_Pressure > 0",
                       f"{kpis['backlog_days']} of {kpis['total_days']} days showed positive pressure.","#e8a020")
    with c6:
        col = "#e84343" if kpis["max_consecutive_backlog"]>14 else "#e8a020"
        kpi_card("Max Consecutive Backlog",f"{kpis['max_consecutive_backlog']} days",
                 "max(Consecutive_Backlog)","Extended streaks signal chronic discharge/intake imbalance.",col)
    section_header("System Load — Last 90 Days")
    recent = df_view.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index,y=recent["Total_System_Load"],name="Daily",
                             line=dict(color=COLORS["load"],width=1,dash="dot"),opacity=0.5))
    fig.add_trace(go.Scatter(x=recent.index,y=recent["Rolling_7d_Load"],name="7d MA",
                             line=dict(color=COLORS["cbp"],width=2)))
    fig.add_trace(go.Scatter(x=recent.index,y=recent["Rolling_14d_Load"],name="14d MA",
                             line=dict(color=COLORS["hhs"],width=2)))
    fig.update_layout(template="plotly_dark",height=340,margin=dict(l=0,r=0,t=10,b=0),
                      legend=dict(orientation="h",y=1.08),yaxis_title="Children in System")
    st.plotly_chart(fig,width="stretch")

with tab2:
    section_header("Pipeline Trends","Full time-series across all five data columns")
    fig2 = go.Figure()
    for col,name,color in [("CBP_Apprehensions","CBP Apprehensions",COLORS["cbp"]),
                            ("CBP_In_Custody","CBP In Custody","#1abc9c"),
                            ("CBP_Transfers_Out","CBP Transfers Out","#f39c12"),
                            ("HHS_In_Care","HHS In Care",COLORS["hhs"]),
                            ("HHS_Discharges","HHS Discharges",COLORS["discharge"])]:
        if col in df_view.columns:
            fig2.add_trace(go.Scatter(x=df_view.index,y=df_view[col],name=name,
                                      line=dict(color=color,width=1.5)))
    fig2.update_layout(template="plotly_dark",height=380,margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(orientation="h",y=1.08),yaxis_title="Children")
    st.plotly_chart(fig2,width="stretch")

    ca,cb = st.columns([2,1])
    with ca:
        st.markdown("#### Net Intake Pressure")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df_view.index,y=df_view["Net_Intake_Pressure"],
                              marker_color=np.where(df_view["Net_Intake_Pressure"]>0,"#e84343","#27ae60")))
        fig3.add_trace(go.Scatter(x=df_view.index,y=df_view["Rolling_7d_Pressure"],
                                  name="7d MA",line=dict(color=COLORS["pressure"],width=2)))
        fig3.update_layout(template="plotly_dark",height=300,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3,width="stretch")
    with cb:
        st.markdown("#### Growth Rate Distribution")
        fig4 = px.histogram(df_view,x="Care_Load_Growth_Rate",nbins=60,template="plotly_dark",
                            color_discrete_sequence=[COLORS["load"]])
        fig4.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4,width="stretch")

    section_header("Monthly Load Heatmap")
    dvc = df_view.copy()
    dvc["Year"]  = dvc.index.year
    dvc["Month"] = dvc.index.month
    pivot = dvc.pivot_table(values="Total_System_Load",index="Year",columns="Month",aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    fig5 = px.imshow(pivot,text_auto=".0f",color_continuous_scale="Blues",template="plotly_dark")
    fig5.update_layout(height=220,margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig5,width="stretch")

with tab3:
    section_header("Bottleneck Detection","Sustained positive intake pressure periods")
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=df_view.index,y=df_view["Consecutive_Backlog"],
                              fill="tozeroy",line=dict(color="#e84343",width=1.5),
                              fillcolor="rgba(232,67,67,0.20)",name="Streak Length"))
    long = df_view[df_view["Consecutive_Backlog"]>=7]
    if not long.empty:
        fig6.add_trace(go.Scatter(x=long.index,y=long["Consecutive_Backlog"],mode="markers",
                                  marker=dict(color="#ffcc00",size=5),name=">=7 days"))
    fig6.update_layout(template="plotly_dark",height=300,margin=dict(l=0,r=0,t=10,b=0),
                       yaxis_title="Consecutive Days")
    st.plotly_chart(fig6,width="stretch")

    streaks,in_s,start = [],[],None
    for date,row in df_view.iterrows():
        if row["Consecutive_Backlog"]==1 and not in_s:
            in_s=True; start=date
        elif row["Consecutive_Backlog"]==0 and in_s:
            in_s=False
            streaks.append({"Start":start.date(),"End":date.date(),
                            "Duration":( date-start).days,
                            "Peak Pressure":df_view.loc[start:date,"Net_Intake_Pressure"].max(),
                            "Avg Load":df_view.loc[start:date,"Total_System_Load"].mean().round(0)})
    if in_s:
        end=df_view.index[-1]
        streaks.append({"Start":start.date(),"End":end.date(),"Duration":(end-start).days+1,
                        "Peak Pressure":df_view.loc[start:end,"Net_Intake_Pressure"].max(),
                        "Avg Load":df_view.loc[start:end,"Total_System_Load"].mean().round(0)})
    if streaks:
        st.dataframe(pd.DataFrame(streaks).sort_values("Duration",ascending=False).reset_index(drop=True).head(15),
                     width="stretch")

    st.markdown("#### Discharge Effectiveness Over Time")
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df_view.index,y=df_view["Discharge_Offset_Ratio"],
                              fill="tozeroy",line=dict(color=COLORS["discharge"],width=1.5),
                              fillcolor="rgba(39,174,96,0.15)"))
    fig7.update_layout(template="plotly_dark",height=260,margin=dict(l=0,r=0,t=10,b=0),
                       yaxis_tickformat=".1%")
    st.plotly_chart(fig7,width="stretch")

with tab4:
    section_header(f"Forecast & Simulation — {horizon}-Day Horizon")
    fc1,fc2 = st.columns(2)
    def plot_fc(hist,fc_df,col,label,color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist[col].tail(60).index,y=hist[col].tail(60),
                                 name="Historical",line=dict(color=color,width=2)))
        for mc in [c for c in fc_df.columns if c!="Date"]:
            fig.add_trace(go.Scatter(x=fc_df["Date"],y=fc_df[mc],mode="lines+markers",
                                     name=mc.replace("_"," "),line=dict(dash="dash",width=2),marker=dict(size=5)))
        fig.update_layout(template="plotly_dark",height=300,
                          margin=dict(l=0,r=0,t=30,b=0),title=label,
                          legend=dict(orientation="h",y=1.1,font=dict(size=10)))
        return fig
    with fc1:
        st.plotly_chart(plot_fc(df_view,fc_discharge,"HHS_Discharges","HHS Discharges Forecast",COLORS["discharge"]),width="stretch")
        st.dataframe(fc_discharge.set_index("Date").round(0),width="stretch")
    with fc2:
        st.plotly_chart(plot_fc(df_view,fc_load,"Total_System_Load","System Load Forecast",COLORS["load"]),width="stretch")
        st.dataframe(fc_load.set_index("Date").round(0),width="stretch")
    st.markdown("---")
    section_header("Scenario Simulation")
    sc1,sc2,sc3 = st.columns([1,1,2])
    with sc1: dd = st.slider("Discharge rate change",  -30,50,0,5,format="%d%%")
    with sc2: ii = st.slider("Intake volume change",   -30,50,0,5,format="%d%%")
    sim = simulate_scenario(df_view,horizon,dd,ii)
    with sc3:
        fs = go.Figure()
        fs.add_trace(go.Scatter(x=df_view.tail(30).index,y=df_view.tail(30)["Total_System_Load"],
                                name="Historical",line=dict(color=COLORS["load"],width=2)))
        fs.add_trace(go.Scatter(x=sim["Date"],y=sim["Baseline_Load"],name="Baseline",
                                line=dict(color="#aaa",dash="dash")))
        fs.add_trace(go.Scatter(x=sim["Date"],y=sim["Simulated_Load"],name="Simulated",
                                line=dict(color=COLORS["forecast"],width=2.5)))
        fs.update_layout(template="plotly_dark",height=280,margin=dict(l=0,r=0,t=10,b=0),
                         legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fs,width="stretch")
    st.dataframe(sim.set_index("Date").round(1),width="stretch")
    dl = float(sim["Simulated_Load"].iloc[-1] - sim["Baseline_Load"].iloc[-1])
    st.markdown(f'<div class="{"warn-box" if dl>0 else "info-box"}">Under this scenario the system load '
                f'will <b>{"INCREASE" if dl>0 else "DECREASE"}</b> by <b>{abs(dl):,.0f} children</b> '
                f'vs baseline at day {horizon}.</div>', unsafe_allow_html=True)

with tab5:
    section_header("Automated Insights & Recommendations")
    imap = {"critical":"warn-box","warning":"warn-box","positive":"info-box","info":"info-box"}
    iicon= {"critical":"🔴","warning":"🟡","positive":"🟢","info":"🔵","neutral":"⚪"}
    for ins in insights_list:
        sev  = ins.get("severity","info")
        icon = iicon.get(sev,"🔵")
        box  = imap.get(sev,"info-box")
        st.markdown(f'<div class="{box}"><b>{icon} {ins["title"]}</b><br>{ins["body"]}<br>'
                    f'<i>Recommendation: {ins["recommendation"]}</i></div>',
                    unsafe_allow_html=True)
    st.markdown("---")
    