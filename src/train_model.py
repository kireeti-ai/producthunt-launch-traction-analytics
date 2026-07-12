"""
src/train_model.py
==================
Train the machine learning models (Random Forest and XGBoost, both with and
without country features) on the processed dataset and save the serialized models.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_and_save() -> None:
    # 1. Load data
    logger.info("Loading processed datasets...")
    X_with = pd.read_csv(PROCESSED_DIR / "features_with_country.csv")
    X_wo = pd.read_csv(PROCESSED_DIR / "features_without_country.csv")
    X_wo = X_wo.drop(columns=["country_id"], errors="ignore")
    y = pd.read_csv(PROCESSED_DIR / "target.csv").squeeze()

    # Split to prevent evaluation leakage
    from sklearn.model_selection import train_test_split
    X_tr_w, _, y_tr, _ = train_test_split(
        X_with, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr_wo, _, _, _ = train_test_split(
        X_wo, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Imbalance configurations
    n_op = int((y_tr == 1).sum())
    n_nonop = int((y_tr == 0).sum())
    scale_pos_weight = n_op / max(n_nonop, 1)

    logger.info(
        "Dataset loaded and split. Training set size: %d | Imbalance ratio: %.1f : 1",
        len(X_tr_w),
        scale_pos_weight,
    )

    # 3. Train models
    # -- RF WITH Country --
    logger.info("Training Random Forest (WITH Country)...")
    rf_with = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_with.fit(X_tr_w, y_tr)

    # -- RF WITHOUT Country --
    logger.info("Training Random Forest (WITHOUT Country)...")
    rf_wo = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_wo.fit(X_tr_wo, y_tr)

    # -- XGB WITH Country --
    logger.info("Training XGBoost (WITH Country)...")
    xgb_with = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    xgb_with.fit(X_tr_w, 1 - y_tr)

    # -- XGB WITHOUT Country --
    logger.info("Training XGBoost (WITHOUT Country)...")
    xgb_wo = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    xgb_wo.fit(X_tr_wo, 1 - y_tr)

    # 4. Serialize models
    logger.info("Serializing split-trained models to disk...")
    with open(MODEL_DIR / "rf_with_country.pkl", "wb") as f:
        pickle.dump(rf_with, f)
    with open(MODEL_DIR / "rf_without_country.pkl", "wb") as f:
        pickle.dump(rf_wo, f)
    with open(MODEL_DIR / "xgb_with_country.pkl", "wb") as f:
        pickle.dump(xgb_with, f)
    with open(MODEL_DIR / "xgb_without_country.pkl", "wb") as f:
        pickle.dump(xgb_wo, f)

    logger.info("All models trained and saved to %s", MODEL_DIR)


if __name__ == "__main__":
    train_and_save()
