# src/forecasting.py
import logging
import numpy as np
import pandas as pd
from datetime import timedelta

log = logging.getLogger("forecasting")

# Attempt to import scikit-learn for linear regression forecasting. If not available, LR forecasts will be skipped.
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not found. LR forecasts will be skipped.")

_ROLLING_WINDOW  = 14
_MIN_ROWS_FOR_LR = 30
_LR_LOOKBACK     = 90

_VALID_TARGETS: dict[str, str] = {
    "HHS_Discharges":    "HHS_Discharges",
    "Total_System_Load": "Total_System_Load",
}

# Forecasting functions. The rolling mean is a simple baseline, while the linear regression uses time, moving averages, and day-of-week features to predict future values. The main forecast function generates forecasts for a specified target and horizon, while forecast_all runs forecasts for all valid targets and handles any exceptions gracefully.
def _rolling_mean_forecast(series: pd.Series, horizon: int, window: int = _ROLLING_WINDOW) -> np.ndarray:
    w        = min(window, len(series))
    baseline = series.iloc[-w:].mean()
    return np.full(horizon, baseline)

# The linear regression forecast function checks for scikit-learn availability and sufficient data length before attempting to fit a model. It uses a combination of time indices, moving averages, and seasonal features to predict future values. If any step fails, it logs the error and returns None, allowing the main forecast function to handle it gracefully.
def _linear_regression_forecast(series: pd.Series, horizon: int, lookback: int = _LR_LOOKBACK) -> np.ndarray | None:
    if not _SKLEARN_AVAILABLE or len(series) < _MIN_ROWS_FOR_LR:
        return None
    try:
        lb    = min(lookback, len(series))
        train = series.iloc[-lb:].reset_index(drop=True)
        T     = len(train)
        t_idx = np.arange(T, dtype=float)
        ma7   = train.rolling(7,  min_periods=1).mean().values
        ma14  = train.rolling(14, min_periods=1).mean().values
        dow   = np.arange(T)
        X_tr  = np.column_stack([t_idx, t_idx**2, ma7, ma14,
                                  np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)])
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_tr)
        model   = LinearRegression(fit_intercept=True)
        model.fit(X_sc, train.values)

        ft     = np.arange(T, T + horizon, dtype=float)
        fdow   = np.arange(T, T + horizon)
        X_fut  = np.column_stack([ft, ft**2,
                                   np.full(horizon, ma7[-1]),
                                   np.full(horizon, ma14[-1]),
                                   np.sin(2*np.pi*fdow/7), np.cos(2*np.pi*fdow/7)])
        preds  = model.predict(scaler.transform(X_fut))
        return np.clip(preds, 0, None)
    except Exception as exc:
        log.error("LR forecast failed: %s", exc, exc_info=True)
        return None

# Main forecasting functions. The forecast function generates forecasts for a specified target and horizon, while forecast_all runs forecasts for all valid targets and handles any exceptions gracefully.
def forecast(df: pd.DataFrame, target: str, horizon: int = 7) -> pd.DataFrame:
    if target not in _VALID_TARGETS:
        raise ValueError(f"Unknown target '{target}'. Valid: {list(_VALID_TARGETS)}")
    col = _VALID_TARGETS[target]
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in DataFrame.")
    series       = df[col].dropna()
    if series.empty:
        raise ValueError(f"Column '{col}' is entirely null.")
    future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=horizon, freq="D")
    rm   = _rolling_mean_forecast(series, horizon)
    lr   = _linear_regression_forecast(series, horizon)
    out  = pd.DataFrame({"Date": future_dates})
    out["Rolling_Mean_Forecast"] = rm.round(1)
    out["LinReg_Forecast"]       = lr.round(1) if lr is not None else np.full(horizon, np.nan)
    log.info("Forecast: target='%s' horizon=%d", target, horizon)
    return out


def forecast_all(df: pd.DataFrame, horizon: int = 7) -> dict[str, pd.DataFrame]:
    results = {}
    for target in _VALID_TARGETS:
        try:
            results[target] = forecast(df, target, horizon)
        except Exception as exc:
            log.error("forecast_all: '%s' failed: %s", target, exc)
            fd = pd.date_range(start=df.index.max() + timedelta(days=1), periods=horizon, freq="D")
            results[target] = pd.DataFrame({"Date": fd,
                                            "Rolling_Mean_Forecast": np.full(horizon, np.nan),
                                            "LinReg_Forecast":       np.full(horizon, np.nan)})
    return results
