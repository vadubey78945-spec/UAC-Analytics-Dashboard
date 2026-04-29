# src/__init__.py
# Makes src/ a proper Python package.
# Exposes top-level imports for convenience.

from src.data_loader   import load_data
from src.preprocessing import preprocess
from src.metrics       import compute_kpis, classify_severity, get_kpi_meta
from src.forecasting   import forecast, forecast_all
from src.simulation    import simulate_scenario, batch_simulate
from src.insights      import generate_insights

__all__ = [
    "load_data",
    "preprocess",
    "compute_kpis",
    "classify_severity",
    "get_kpi_meta",
    "forecast",
    "forecast_all",
    "simulate_scenario",
    "batch_simulate",
    "generate_insights",
]
