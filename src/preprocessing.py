"""
This module defines a comprehensive data preprocessing pipeline for the UAC analytics dashboard. 
It includes functions to clean numeric columns, parse and validate dates, handle missing values 
through interpolation, enforce logical constraints between related columns, and compute derived 
metrics such as total system load and net intake pressure.

The main `preprocess` function orchestrates these steps and generates a detailed report on the 
preprocessing outcomes, including null value summaries, constraint violations, and data quality insights. 
This ensures that the data fed into the dashboard is accurate, consistent, and enriched with 
meaningful features for analysis and forecasting.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("preprocessing")


NUMERIC_COLS = [
    "CBP_Apprehensions", "CBP_In_Custody", "CBP_Transfers_Out",
    "HHS_In_Care", "HHS_Discharges",
]

CONSTRAINTS = [
    ("CBP_Transfers_Out", "CBP_In_Custody", "Transfers <= CBP custody"),
    ("HHS_Discharges",    "HHS_In_Care",    "Discharges <= HHS care"),
]


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        original_dtype = df[col].dtype
        df[col] = (df[col].astype(str)
                   .str.replace(r"[,\$\s]", "", regex=True)
                   .str.strip()
                   .replace("nan", np.nan)
                   .replace("None", np.nan)
                   .replace("", np.nan))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n = df[col].isna().sum()
        if n:
            log.warning("Column '%s' (was %s): %d value(s) -> NaN.", col, original_dtype, n)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
   
    try:
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')
    except ValueError:
        # Fallback agar Pandas ka version purana ho
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')
        log.warning("Date parsed via pandas inference (fallback).")

    n_bad = df["Date"].isna().sum()
    if n_bad:
        log.warning("%d row(s) with unparseable dates dropped.", n_bad)
        df = df.dropna(subset=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)
    log.info("Date range: %s -> %s", df["Date"].min().date(), df["Date"].max().date())
    return df


def _validate_and_impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    null_counts = df[NUMERIC_COLS].isna().sum().to_dict()
    log.info("Null counts before imputation: %s", null_counts)

    for col, cnt in null_counts.items():
        if col == "HHS_In_Care" and cnt == len(df):
            raise ValueError(
                "CRITICAL: 'HHS_In_Care' is entirely null. Check source data."
            )

   
    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(method='linear').ffill().bfill()
    
    
    for col in NUMERIC_COLS:
        df[col] = df[col].round(0).astype(int)

    null_after = df[NUMERIC_COLS].isna().sum().to_dict()
    return df, {"before": null_counts, "after": null_after}


def _validate_constraints(df: pd.DataFrame) -> list[dict]:
    violations = []
    for val_col, cap_col, label in CONSTRAINTS:
        if val_col not in df.columns or cap_col not in df.columns:
            continue
        mask   = df[val_col] > df[cap_col]
        n_viol = mask.sum()
        if n_viol:
            log.warning("Constraint '%s' violated on %d row(s). Capping.", label, n_viol)
            df.loc[mask, val_col] = df.loc[mask, cap_col]
        violations.append({
            "constraint": label,
            "violations": int(n_viol),
            "action": "Capped" if n_viol else "No action",
        })
    return violations


def _compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Total_System_Load"]    = df["CBP_In_Custody"] + df["HHS_In_Care"]
    df["Net_Intake_Pressure"]  = df["CBP_Transfers_Out"] - df["HHS_Discharges"]
    df["Care_Load_Growth_Rate"]= (df["Total_System_Load"].pct_change() * 100).round(4)
    df["Backlog_Indicator"]    = (df["Net_Intake_Pressure"] > 0).astype(int)
    df["Discharge_Offset_Ratio"] = np.where(
        df["HHS_In_Care"] > 0, df["HHS_Discharges"] / df["HHS_In_Care"], np.nan
    )
    df["Rolling_7d_Load"]      = df["Total_System_Load"].rolling(7,  min_periods=1).mean()
    df["Rolling_14d_Load"]     = df["Total_System_Load"].rolling(14, min_periods=1).mean()
    df["Rolling_7d_Pressure"]  = df["Net_Intake_Pressure"].rolling(7, min_periods=1).mean()
    df["Rolling_Std_14d_Load"] = df["Total_System_Load"].rolling(14, min_periods=1).std()

    streak, count = [], 0
    for flag in df["Backlog_Indicator"]:
        count = count + 1 if flag else 0
        streak.append(count)
    df["Consecutive_Backlog"] = streak

    return df


def preprocess(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("-- Preprocessing started --")
    df               = _clean_numeric(raw_df)
    df               = _parse_dates(df)
    df, null_summary = _validate_and_impute(df)
    violations       = _validate_constraints(df)
    df               = _compute_derived_metrics(df)
    df               = df.set_index("Date")

    report = {
        "shape":         df.shape,
        "date_range":    (str(df.index.min().date()), str(df.index.max().date())),
        "null_summary":  null_summary,
        "constraint_violations": violations,
        "dtypes":        df.dtypes.astype(str).to_dict(),
        "head":          df.head(5).reset_index().to_dict(orient="records"),
    }
    log.info("-- Preprocessing complete: shape=%s --", report["shape"])
    return df, report