# src/data_loader.py
import pandas as pd
import logging
from pathlib import Path
from difflib import get_close_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("data_loader")

# Define the expected columns and their canonical names
COLUMN_MAP: dict[str, str] = {
    "Children apprehended and placed in CBP custody*": "CBP_Apprehensions",
    "Children in CBP custody":                         "CBP_In_Custody",
    "Children transferred out of CBP custody":         "CBP_Transfers_Out",
    "Children in HHS Care":                            "HHS_In_Care",
    "Children discharged from HHS Care":               "HHS_Discharges",
}

REQUIRED_INTERNAL = list(COLUMN_MAP.values())
DATE_COLUMN_RAW   = "Date"

# This function loads the dataset, performs column mapping (with fuzzy matching), and returns a cleaned DataFrame along with debug information.
def load_data(filepath: str | Path) -> tuple[pd.DataFrame, dict]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Place the CSV inside the data/ folder and update DATA_PATH in app.py."
        )

    raw_df   = pd.read_csv(filepath, dtype=str)
    raw_cols = raw_df.columns.tolist()

    log.info("=" * 60)
    log.info("RAW COLUMNS DETECTED (%d total):", len(raw_cols))
    for i, c in enumerate(raw_cols):
        log.info("  [%02d] '%s'", i, c)
    log.info("=" * 60)

    raw_df.columns = raw_df.columns.str.strip()
    stripped_cols  = raw_df.columns.tolist()

    rename_map:  dict[str, str] = {}
    mapping_log: list[dict]     = []

    for raw_key, canonical in COLUMN_MAP.items():
        if raw_key in stripped_cols:
            rename_map[raw_key] = canonical
            mapping_log.append({"raw": raw_key, "canonical": canonical,
                                 "match_type": "EXACT", "status": "OK"})
            log.info("EXACT MATCH: '%s' -> '%s'", raw_key, canonical)
        else:
            candidates = get_close_matches(raw_key, stripped_cols, n=1, cutoff=0.75)
            if candidates:
                best = candidates[0]
                rename_map[best] = canonical
                mapping_log.append({"raw": best, "canonical": canonical,
                                     "match_type": "FUZZY", "status": "FUZZY - verify"})
                log.warning("FUZZY MATCH: '%s' matched to '%s' -> '%s'",
                            raw_key, best, canonical)
            else:
                mapping_log.append({"raw": raw_key, "canonical": canonical,
                                     "match_type": "MISSING", "status": "NOT FOUND"})
                raise KeyError(
                    f"Required column '{raw_key}' (-> '{canonical}') missing.\n"
                    f"Available: {stripped_cols}"
                )

    df = raw_df.rename(columns=rename_map)

    if DATE_COLUMN_RAW not in df.columns:
        candidates = get_close_matches(DATE_COLUMN_RAW, df.columns.tolist(), n=1, cutoff=0.8)
        if candidates:
            df = df.rename(columns={candidates[0]: DATE_COLUMN_RAW})
        else:
            raise KeyError(f"'Date' column not found. Available: {df.columns.tolist()}")

    keep_cols = [DATE_COLUMN_RAW] + REQUIRED_INTERNAL
    df = df[keep_cols].copy()

    hhs_null  = df["HHS_In_Care"].isna().sum()
    hhs_empty = (df["HHS_In_Care"].astype(str).str.strip() == "").sum()
    if hhs_null + hhs_empty == len(df):
        raise ValueError("CRITICAL: 'HHS_In_Care' is entirely null. Cannot compute KPIs.")

    debug_info = {
        "raw_columns":      raw_cols,
        "stripped_columns": stripped_cols,
        "mapping_log":      mapping_log,
        "df_shape":         df.shape,
        "sample_head":      df.head(5).to_dict(orient="records"),
    }

    log.info("Column mapping complete. Shape: %s", df.shape)
    return df, debug_info
