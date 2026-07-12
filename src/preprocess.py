"""
src/preprocess.py
=================
Feature engineering and preprocessing for the EV Charger Reliability model.
"""

import ast
import json
import logging
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABELED_PATH = PROCESSED_DIR / "labeled_poi.csv"
FEATURES_OUTPUT = PROCESSED_DIR / "features.csv"
TARGET_OUTPUT = PROCESSED_DIR / "target.csv"
ENCODER_MAP_OUTPUT = PROCESSED_DIR / "label_encoder_map.json"

# Columns representing target variables or post-event information (prevent leakage)
LEAKAGE_COLUMNS = {
    "StatusType.ID", "StatusTypeID", "StatusType_ID",
    "StatusType.Title", "StatusType.IsOperational", "StatusType",
    "DateLastStatusUpdate", "DateLastVerified", "DateCreated",
    "DateLastConfirmed", "GeneralComments", "UserComments",
    "label",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_connections(raw) -> list[dict]:
    """Parse a station's Connections cell into a list of dicts using ast.literal_eval."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _mode_or_first(vals: list[int]) -> float:
    """Return the statistical mode of a list of ints. First occurrence tie-breaker."""
    if not vals:
        return np.nan
    counts = Counter(vals)
    max_count = max(counts.values())
    for v in vals:
        if counts[v] == max_count:
            return float(v)
    return float(vals[0])


def flatten_connections(raw) -> dict:
    """
    Flatten per-station nested Connections list into station-level scalar features.
    Extracts max power, max voltage, max amps, and connector types modes.
    """
    defaults: dict = {
        "max_power_kw": np.nan,
        "primary_connection_type_id": np.nan,
        "n_distinct_connection_types": 0,
        "max_amps": np.nan,
        "max_voltage": np.nan,
        "primary_current_type_id": np.nan,
        "n_connections": 0,
        "has_fast_charger": 0,
        "has_dc_fast": 0,
    }

    connections = _parse_connections(raw)
    if not connections:
        return defaults

    power_kws: list[float] = []
    amps_vals: list[float] = []
    voltage_vals: list[float] = []
    conn_type_ids: list[int] = []
    curr_type_ids: list[int] = []

    for conn in connections:
        if not isinstance(conn, dict):
            continue

        pkw = conn.get("PowerKW")
        if pkw is not None:
            try:
                power_kws.append(float(pkw))
            except (ValueError, TypeError):
                pass

        amps = conn.get("Amps")
        if amps is not None:
            try:
                amps_vals.append(float(amps))
            except (ValueError, TypeError):
                pass

        voltage = conn.get("Voltage")
        if voltage is not None:
            try:
                voltage_vals.append(float(voltage))
            except (ValueError, TypeError):
                pass

        ct_id = conn.get("ConnectionTypeID")
        if ct_id is not None:
            try:
                conn_type_ids.append(int(ct_id))
            except (ValueError, TypeError):
                pass

        cur_id = conn.get("CurrentTypeID")
        if cur_id is not None:
            try:
                curr_type_ids.append(int(cur_id))
            except (ValueError, TypeError):
                pass

    return {
        "max_power_kw": max(power_kws) if power_kws else np.nan,
        "primary_connection_type_id": _mode_or_first(conn_type_ids),
        "n_distinct_connection_types": len(set(conn_type_ids)),
        "max_amps": max(amps_vals) if amps_vals else np.nan,
        "max_voltage": max(voltage_vals) if voltage_vals else np.nan,
        "primary_current_type_id": _mode_or_first(curr_type_ids),
        "n_connections": len(connections),
        "has_fast_charger": int(max(power_kws, default=0) >= 50),
        "has_dc_fast": int(
            any(
                conn.get("CurrentTypeID") == 30
                and (conn.get("PowerKW") or 0) >= 50
                for conn in connections
                if isinstance(conn, dict)
            )
        ),
    }


def extract_features(df: pd.DataFrame, include_country: bool = True) -> pd.DataFrame:
    """Extract infrastructure features from the labeled DataFrame."""
    features = pd.DataFrame(index=df.index)

    features["operator_id"] = (
        df.get("OperatorInfo.ID", pd.Series(dtype="float64"))
        .fillna(df.get("OperatorID", pd.Series(dtype="float64")))
        .fillna(-1)
        .astype(int)
    )
    features["operator_title"] = (
        df.get("OperatorInfo.Title", pd.Series(dtype="object"))
        .fillna("Unknown")
        .str.strip()
        .str.upper()
    )

    features["usage_type_id"] = (
        df.get("UsageType.ID", pd.Series(dtype="float64"))
        .fillna(df.get("UsageTypeID", pd.Series(dtype="float64")))
        .fillna(-1)
        .astype(int)
    )
    features["usage_type_title"] = (
        df.get("UsageType.Title", pd.Series(dtype="object"))
        .fillna("Unknown")
        .str.strip()
        .str.upper()
    )

    features["country_id"] = (
        df.get("AddressInfo.CountryID", pd.Series(dtype="float64"))
        .fillna(-1)
        .astype(int)
    )
    if include_country:
        features["country_iso"] = (
            df.get("AddressInfo.Country.ISOCode", pd.Series(dtype="object"))
            .fillna("XX")
            .str.strip()
            .str.upper()
        )
    features["state_or_province"] = (
        df.get("AddressInfo.StateOrProvince", pd.Series(dtype="object"))
        .fillna("Unknown")
        .str.strip()
        .str.upper()
    )
    features["town"] = (
        df.get("AddressInfo.Town", pd.Series(dtype="object"))
        .fillna("Unknown")
        .str.strip()
        .str.upper()
    )

    features["number_of_points"] = (
        df.get("NumberOfPoints", pd.Series(dtype="float64"))
        .fillna(0)
        .astype(int)
    )

    features["data_provider_id"] = (
        df.get("DataProviderID", pd.Series(dtype="float64"))
        .fillna(-1)
        .astype(int)
    )

    conn_col = next((c for c in df.columns if c == "Connections"), None)
    if conn_col is not None:
        conn_feats = df[conn_col].apply(flatten_connections).apply(pd.Series)
        features = pd.concat([features, conn_feats], axis=1)
    else:
        for col in [
            "max_power_kw", "primary_connection_type_id",
            "n_distinct_connection_types", "max_amps", "max_voltage",
            "primary_current_type_id", "n_connections",
            "has_fast_charger", "has_dc_fast",
        ]:
            features[col] = np.nan

    return features


def encode_categoricals(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode low-cardinality string columns; frequency-encode town."""
    encoder_map: dict = {}
    categorical_cols = [
        "operator_title", "usage_type_title",
        "state_or_province", "country_iso",
    ]
    high_card_cols = ["town"]

    for col in categorical_cols:
        if col not in features.columns:
            continue
        le = LabelEncoder()
        features[f"{col}_enc"] = le.fit_transform(features[col].astype(str))
        encoder_map[col] = {
            str(cls): int(idx) for idx, cls in enumerate(le.classes_)
        }
        features = features.drop(columns=[col])

    for col in high_card_cols:
        if col not in features.columns:
            continue
        freq = features[col].value_counts()
        features[f"{col}_freq"] = features[col].map(freq).fillna(0).astype(int)
        features = features.drop(columns=[col])

    return features, encoder_map


def impute_numerics(features: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with column medians."""
    bool_cols = ["has_fast_charger", "has_dc_fast"]
    for col in bool_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    num_cols = features.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val)

    return features


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Raise ValueError if any leakage column appears in the feature matrix."""
    leaked = set(features.columns) & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED in columns: {leaked}")


def preprocess(
    labeled_path: Path = LABELED_PATH,
    include_country: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess labeled POIs and save features and targets."""
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")

    df = pd.read_csv(labeled_path, low_memory=False)

    y = (df["label"] == "Operational").astype(int)
    y.name = "is_operational"

    X = extract_features(df, include_country=include_country)
    assert_no_leakage(X)
    X, encoder_map = encode_categoricals(X)
    X = impute_numerics(X)

    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)

    suffix = "with_country" if include_country else "without_country"
    X.to_csv(PROCESSED_DIR / f"features_{suffix}.csv", index=False)
    y.to_csv(PROCESSED_DIR / "target.csv", index=False)
    with open(ENCODER_MAP_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(encoder_map, fh, indent=2)

    return X, y


if __name__ == "__main__":
    X_geo, y = preprocess(include_country=True)
    print(f"Features with country shape: {X_geo.shape}")
    X_nogeo, _ = preprocess(include_country=False)
    print(f"Features without country shape: {X_nogeo.shape}")
