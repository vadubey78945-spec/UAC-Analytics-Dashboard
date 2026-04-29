# src/metrics.py
import logging
import numpy as np
import pandas as pd
from typing import Any

log = logging.getLogger("metrics")

# This module defines a registry of key performance indicators (KPIs) relevant to the UAC system, along with their metadata such as labels, formulas, units, and severity thresholds. 
# It also provides functions to safely compute these KPIs from the raw data, classify their severity based on predefined thresholds, and retrieve metadata for display purposes. 
# The main function `compute_kpis` takes a cleaned DataFrame with all necessary columns and calculates a comprehensive set of KPIs that can be used for monitoring system performance and generating insights in the dashboard.
KPI_REGISTRY: dict[str, dict] = {
    "total_under_care":         {"label": "Total Children Under Care",       "formula": "CBP_In_Custody + HHS_In_Care (latest)",                   "unit": "children",    "direction": "higher_worse", "thresholds": {"warn": 12000,  "critical": 18000}},
    "net_intake_pressure":      {"label": "Net Intake Pressure",             "formula": "CBP_Transfers_Out - HHS_Discharges (latest)",              "unit": "children/day","direction": "higher_worse", "thresholds": {"warn": 50,     "critical": 200}},
    "discharge_offset_ratio":   {"label": "Discharge Offset Ratio",          "formula": "HHS_Discharges / HHS_In_Care (latest)",                   "unit": "ratio",       "direction": "lower_worse",  "thresholds": {"warn": 0.02,   "critical": 0.005}},
    "volatility_index":         {"label": "Care Load Volatility Index",      "formula": "14-day rolling std(Total_System_Load)",                   "unit": "std dev",     "direction": "higher_worse", "thresholds": {"warn": 500,    "critical": 1500}},
    "backlog_pct":              {"label": "Backlog Accumulation Rate",        "formula": "Count(NIP>0) / Total_Days * 100",                         "unit": "%",           "direction": "higher_worse", "thresholds": {"warn": 50,     "critical": 70}},
    "max_consecutive_backlog":  {"label": "Max Consecutive Backlog Streak",  "formula": "max(Consecutive_Backlog)",                                "unit": "days",        "direction": "higher_worse", "thresholds": {"warn": 7,      "critical": 21}},
    "peak_system_load":         {"label": "Peak System Load",                "formula": "max(Total_System_Load)",                                  "unit": "children",    "direction": "higher_worse", "thresholds": {"warn": 15000,  "critical": 22000}},
    "hhs_in_care_latest":       {"label": "HHS In Care (Latest)",            "formula": "HHS_In_Care[-1]",                                         "unit": "children",    "direction": "higher_worse", "thresholds": {"warn": 10000,  "critical": 15000}},
    "cbp_in_custody_latest":    {"label": "CBP In Custody (Latest)",         "formula": "CBP_In_Custody[-1]",                                      "unit": "children",    "direction": "higher_worse", "thresholds": {"warn": 3000,   "critical": 6000}},
    "transfer_to_intake_ratio": {"label": "Transfer-to-Intake Ratio",        "formula": "sum(CBP_Transfers_Out) / sum(CBP_Apprehensions)",         "unit": "ratio",       "direction": "lower_worse",  "thresholds": {"warn": 0.75,   "critical": 0.50}},
    "mom_load_change":          {"label": "Month-over-Month Load Change",    "formula": "(last_30d_mean - prior_30d_mean) / prior_30d_mean * 100", "unit": "%",           "direction": "higher_worse", "thresholds": {"warn": 5,      "critical": 15}},
}


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else default


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den and not np.isnan(den) and den != 0:
        return num / den
    return default


def _month_over_month(series: pd.Series) -> float:
    if len(series) < 60:
        return float("nan")
    last_30  = series.iloc[-30:].mean()
    prior_30 = series.iloc[-60:-30].mean()
    return _safe_ratio(last_30 - prior_30, prior_30, float("nan")) * 100


def classify_severity(kpi_key: str, value: float) -> str:
    meta = KPI_REGISTRY.get(kpi_key)
    if not meta or not meta.get("thresholds"):
        return "neutral"
    thresholds = meta["thresholds"]
    direction  = meta["direction"]
    warn_val   = thresholds.get("warn")
    crit_val   = thresholds.get("critical")
    if direction == "higher_worse":
        if crit_val is not None and value >= crit_val: return "critical"
        if warn_val is not None and value >= warn_val: return "warning"
        return "positive"
    if direction == "lower_worse":
        if crit_val is not None and value <= crit_val: return "critical"
        if warn_val is not None and value <= warn_val: return "warning"
        return "positive"
    return "neutral"


def get_kpi_meta(kpi_key: str) -> dict:
    return KPI_REGISTRY.get(kpi_key, {})


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    required = [
        "CBP_Apprehensions","CBP_In_Custody","CBP_Transfers_Out",
        "HHS_In_Care","HHS_Discharges","Total_System_Load",
        "Net_Intake_Pressure","Discharge_Offset_Ratio",
        "Rolling_Std_14d_Load","Backlog_Indicator","Consecutive_Backlog",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_kpis() missing columns: {missing}")

    hhs_latest      = _safe_last(df["HHS_In_Care"])
    cbp_latest      = _safe_last(df["CBP_In_Custody"])
    pressure_latest = _safe_last(df["Net_Intake_Pressure"])
    dor_latest      = _safe_last(df["Discharge_Offset_Ratio"])
    vol_latest      = _safe_last(df["Rolling_Std_14d_Load"])
    load_latest     = _safe_last(df["Total_System_Load"])

    backlog_days  = int(df["Backlog_Indicator"].sum())
    total_days    = int(len(df))
    backlog_pct   = _safe_ratio(backlog_days, total_days) * 100
    max_consec    = int(df["Consecutive_Backlog"].max())

    peak_idx  = df["Total_System_Load"].idxmax()
    peak_load = float(df["Total_System_Load"].max())

    total_app = float(df["CBP_Apprehensions"].sum())
    total_dis = float(df["HHS_Discharges"].sum())
    total_tr  = float(df["CBP_Transfers_Out"].sum())

    transfer_ratio = _safe_ratio(total_tr, total_app)
    mom_change     = _month_over_month(df["Total_System_Load"])

    if len(df) >= 60:
        pressure_trend  = ("Increasing"
                           if df["Net_Intake_Pressure"].iloc[-30:].mean()
                              > df["Net_Intake_Pressure"].iloc[-60:-30].mean()
                           else "Decreasing")
        discharge_trend = ("Improving"
                           if df["Discharge_Offset_Ratio"].iloc[-30:].mean()
                              > df["Discharge_Offset_Ratio"].iloc[-60:-30].mean()
                           else "Declining")
    else:
        pressure_trend  = "Insufficient data"
        discharge_trend = "Insufficient data"

    kpis: dict[str, Any] = {
        "total_under_care":         round(load_latest, 0),
        "net_intake_pressure":      round(pressure_latest, 1),
        "discharge_offset_ratio":   round(dor_latest, 6),
        "volatility_index":         round(vol_latest, 2),
        "backlog_pct":              round(backlog_pct, 2),
        "backlog_days":             backlog_days,
        "total_days":               total_days,
        "max_consecutive_backlog":  max_consec,
        "avg_system_load":          round(float(df["Total_System_Load"].mean()), 1),
        "peak_system_load":         round(peak_load, 0),
        "peak_load_date":           str(peak_idx.date()) if peak_idx is not None else "N/A",
        "avg_intake_pressure":      round(float(df["Net_Intake_Pressure"].mean()), 2),
        "total_apprehensions":      round(total_app, 0),
        "total_discharges":         round(total_dis, 0),
        "total_transfers":          round(total_tr, 0),
        "hhs_in_care_latest":       round(hhs_latest, 0),
        "cbp_in_custody_latest":    round(cbp_latest, 0),
        "transfer_to_intake_ratio": round(transfer_ratio, 4),
        "mom_load_change":          round(mom_change, 2) if not np.isnan(mom_change) else None,
        "pressure_trend":           pressure_trend,
        "discharge_trend":          discharge_trend,
    }

    log.info("KPI computation complete: %d metrics.", len(kpis))
    return kpis
