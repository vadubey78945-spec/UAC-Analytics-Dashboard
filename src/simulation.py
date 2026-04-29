# src/simulation.py
import logging
from datetime import timedelta
import numpy as np
import pandas as pd

log = logging.getLogger("simulation")

_WARMUP_WINDOW   = 14
_SHOCK_RAMP_DAYS = 3

# Simulation functions to model the impact of changes in discharge and intake rates on the future population under care. 
# The main function `simulate_scenario` takes a cleaned DataFrame and simulates a scenario based on specified percentage changes in discharge 
# and intake rates over a given horizon. The `batch_simulate` function allows running multiple scenarios across ranges of discharge and intake deltas, 
# compiling the results into a DataFrame for analysis.
def _trailing_mean(series: pd.Series, window: int) -> float:
    eff = min(window, series.dropna().shape[0])
    return float(series.dropna().iloc[-eff:].mean())


def _ramp_multiplier(day: int, delta_pct: float, ramp_days: int = _SHOCK_RAMP_DAYS) -> float:
    progress = min((day + 1) / ramp_days, 1.0)
    return 1.0 + (delta_pct / 100.0) * progress


def _derive_flow_assumptions(df: pd.DataFrame) -> dict[str, float]:
    w = _WARMUP_WINDOW
    a = {
        "cbp_in_custody":   float(df["CBP_In_Custody"].dropna().iloc[-1]),
        "hhs_in_care":      float(df["HHS_In_Care"].dropna().iloc[-1]),
        "daily_transfers":  _trailing_mean(df["CBP_Transfers_Out"], w),
        "daily_discharges": _trailing_mean(df["HHS_Discharges"],    w),
    }
    if "CBP_Apprehensions" in df.columns:
        net = df["CBP_Apprehensions"] - df["CBP_Transfers_Out"]
        a["daily_cbp_change"] = _trailing_mean(net, w)
    else:
        a["daily_cbp_change"] = 0.0
    return a


def _run_stock_flow(assumptions, horizon, discharge_factor, intake_factor):
    hhs_pop  = np.empty(horizon)
    cbp_pop  = np.empty(horizon)
    prev_hhs = assumptions["hhs_in_care"]
    prev_cbp = assumptions["cbp_in_custody"]
    d_tr     = assumptions["daily_transfers"]
    d_di     = assumptions["daily_discharges"]
    d_cbp    = assumptions["daily_cbp_change"]
    for t in range(horizon):
        tr_t      = d_tr  * intake_factor[t]
        di_t      = d_di  * discharge_factor[t]
        cbp_net_t = d_cbp * intake_factor[t]
        hhs_pop[t] = max(0.0, prev_hhs + tr_t - di_t)
        cbp_pop[t] = max(0.0, prev_cbp + cbp_net_t - tr_t)
        prev_hhs   = hhs_pop[t]
        prev_cbp   = cbp_pop[t]
    return hhs_pop, cbp_pop, hhs_pop + cbp_pop


def simulate_scenario(df: pd.DataFrame, horizon: int = 7,
                      discharge_delta: float = 0.0, intake_delta: float = 0.0) -> pd.DataFrame:
    required = ["CBP_In_Custody","HHS_In_Care","CBP_Transfers_Out","HHS_Discharges"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"simulate_scenario() missing columns: {missing}")
    if not (1 <= horizon <= 365):
        raise ValueError(f"horizon must be 1-365, got {horizon}.")

    assumptions = _derive_flow_assumptions(df)
    base_df     = np.array([_ramp_multiplier(t, 0.0) for t in range(horizon)])
    base_if     = np.array([_ramp_multiplier(t, 0.0) for t in range(horizon)])
    sim_df_fac  = np.clip([_ramp_multiplier(t, discharge_delta) for t in range(horizon)], 0, None)
    sim_if_fac  = np.clip([_ramp_multiplier(t, intake_delta)    for t in range(horizon)], 0, None)

    bh, bc, bl = _run_stock_flow(assumptions, horizon, base_df, base_if)
    sh, sc, sl = _run_stock_flow(assumptions, horizon, sim_df_fac, sim_if_fac)

    d_tr = assumptions["daily_transfers"]
    d_di = assumptions["daily_discharges"]
    ni_base = base_if * d_tr - base_df * d_di
    ni_sim  = sim_if_fac * d_tr - sim_df_fac * d_di

    future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=horizon, freq="D")
    delta_load   = sl - bl
    delta_pct    = np.where(bl != 0, delta_load / bl * 100, np.nan)

    result = pd.DataFrame({
        "Date":                 future_dates,
        "Baseline_HHS":         bh.round(1), "Baseline_CBP": bc.round(1), "Baseline_Load": bl.round(1),
        "Simulated_HHS":        sh.round(1), "Simulated_CBP": sc.round(1), "Simulated_Load": sl.round(1),
        "Delta_Load":           delta_load.round(1), "Delta_Load_Pct": delta_pct.round(2),
        "Net_Intake_Baseline":  ni_base.round(1), "Net_Intake_Simulated": ni_sim.round(1),
    })
    log.info("Simulation complete: baseline_end=%.0f, scenario_end=%.0f", bl[-1], sl[-1])
    return result


def batch_simulate(df, horizon, discharge_range, intake_range):
    rows = []
    for dd in discharge_range:
        for ii in intake_range:
            try:
                sim = simulate_scenario(df, horizon, dd, ii)
                rows.append({"discharge_delta": dd, "intake_delta": ii,
                             "final_simulated_load": round(sim["Simulated_Load"].iloc[-1], 1),
                             "final_baseline_load":  round(sim["Baseline_Load"].iloc[-1],  1),
                             "final_delta_load":     round(sim["Delta_Load"].iloc[-1],      1),
                             "final_delta_pct":      round(sim["Delta_Load_Pct"].iloc[-1],  2)})
            except Exception as exc:
                log.error("batch_simulate dd=%s ii=%s: %s", dd, ii, exc)
    return pd.DataFrame(rows)
