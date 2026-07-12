"""
src/label_target.py
===================
Derive the binary target label (Operational / Non-Operational) from raw OCM POI
data, using the live StatusTypes reference as the source of truth.
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_POI_PATH = RAW_DIR / "raw_poi_all.json"
STATUS_TYPES_PATH = RAW_DIR / "reference_status_types.json"
LABELED_OUTPUT_PATH = PROCESSED_DIR / "labeled_poi.csv"

# Deliberate status exclusions based on data clarity or pre-operational state
EXCLUDED_STATUS_IDS = {0, 150, 200, 210}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_status_type_map(status_types_path: Path) -> dict[int, dict]:
    """Load the StatusTypes reference map mapping ID to status metadata."""
    if not status_types_path.exists():
        raise FileNotFoundError(f"StatusTypes reference file not found: {status_types_path}")
    with open(status_types_path, encoding="utf-8") as fh:
        status_types: list[dict] = json.load(fh)
    status_map = {int(st["ID"]): st for st in status_types if "ID" in st}
    logger.info("Loaded StatusTypes lookup: %d entries", len(status_map))
    return status_map


def load_raw_poi(raw_poi_path: Path) -> pd.DataFrame:
    """Load raw POI JSON into a DataFrame."""
    if not raw_poi_path.exists():
        raise FileNotFoundError(f"Raw POI file not found: {raw_poi_path}")
    with open(raw_poi_path, encoding="utf-8") as fh:
        records = json.load(fh)
    df = pd.json_normalize(records, sep=".")
    logger.info("Loaded raw POI data: %d records", len(df))
    return df


def extract_status_type_id(df: pd.DataFrame) -> pd.Series:
    """Extract StatusTypeID checking both nested and flat field names."""
    candidates = ["StatusType.ID", "StatusTypeID", "StatusType_ID"]
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def assign_label(status_id, status_map: dict[int, dict]) -> str | None:
    """Map StatusTypeID to Operational, Non-Operational, or None (exclude)."""
    try:
        if pd.isna(status_id):
            return None
    except (TypeError, ValueError):
        pass

    try:
        sid = int(status_id)
    except (ValueError, TypeError):
        return None

    if sid in EXCLUDED_STATUS_IDS:
        return None

    if sid not in status_map:
        return None

    is_op = status_map[sid].get("IsOperational")
    if is_op is True:
        return "Operational"
    elif is_op is False:
        return "Non-Operational"
    return None


def build_labeled_dataset(
    df: pd.DataFrame, status_map: dict[int, dict]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply exclusion filters and labeling logic to raw POIs."""
    funnel: dict[str, int] = {}
    funnel["1_total_raw"] = len(df)

    id_col = next((c for c in ["ID", "id", "Id"] if c in df.columns), None)
    if id_col:
        df = df.drop_duplicates(subset=[id_col])
    funnel["2_after_dedup"] = len(df)

    df = df.copy()
    df["_status_id"] = extract_status_type_id(df)

    missing_mask = df["_status_id"].isna()
    funnel["3_after_removing_missing_status"] = len(df) - missing_mask.sum()
    df = df[~missing_mask].copy()

    excluded_mask = df["_status_id"].apply(
        lambda sid: False if pd.isna(sid) else int(sid) in EXCLUDED_STATUS_IDS
    )
    funnel["4_after_removing_excluded_ids"] = len(df) - excluded_mask.sum()
    df = df[~excluded_mask].copy()

    df["label"] = df["_status_id"].apply(
        lambda sid: assign_label(sid, status_map)
    )
    df = df[df["label"].notna()].copy()

    label_counts = df["label"].value_counts()
    funnel["5_final_operational"] = int(label_counts.get("Operational", 0))
    funnel["6_final_non_operational"] = int(label_counts.get("Non-Operational", 0))
    funnel["7_final_total"] = len(df)

    df = df.drop(columns=["_status_id"])
    return df, funnel


def label(raw_poi_path: Path = RAW_POI_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """Execute labeling pipeline and save outputs."""
    status_map = load_status_type_map(STATUS_TYPES_PATH)
    df_raw = load_raw_poi(raw_poi_path)
    df_labeled, funnel = build_labeled_dataset(df_raw, status_map)

    df_labeled.to_csv(LABELED_OUTPUT_PATH, index=False)
    funnel_path = PROCESSED_DIR / "labeling_funnel.json"
    funnel_serializable = {k: int(v) for k, v in funnel.items()}
    with open(funnel_path, "w", encoding="utf-8") as fh:
        json.dump(funnel_serializable, fh, indent=2)

    return df_labeled, funnel


if __name__ == "__main__":
    labeled_df, funnel_counts = label()
    for stage, count in funnel_counts.items():
        print(f"{stage}: {count:,}")
    sys.exit(0)
