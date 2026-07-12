"""
src/evaluate.py
===============
Evaluate the trained models, output classification reports, generate evaluation figures
(Confusion Matrices, ROC curves, PR curves), and save the risk score outputs.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "reports"
FIG_DIR = MODEL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_models() -> None:
    # 1. Load data
    logger.info("Loading processed datasets...")
    X_with = pd.read_csv(PROCESSED_DIR / "features_with_country.csv")
    X_wo = pd.read_csv(PROCESSED_DIR / "features_without_country.csv")
    X_wo = X_wo.drop(columns=["country_id"], errors="ignore")
    y = pd.read_csv(PROCESSED_DIR / "target.csv").squeeze()

    # Held-out split for reporting final test metrics (same split seed 42 as notebooks)
    X_tr_w, X_te_w, y_tr, y_te = train_test_split(
        X_with, y, test_size=0.2, stratify=y, random_state=42
    )
    X_te_wo = X_wo.iloc[X_te_w.index]
    y_te_flip = 1 - y_te

    # 2. Load models
    logger.info("Loading serialized models...")
    with open(MODEL_DIR / "rf_with_country.pkl", "rb") as f:
        rf_with = pickle.load(f)
    with open(MODEL_DIR / "rf_without_country.pkl", "rb") as f:
        rf_wo = pickle.load(f)
    with open(MODEL_DIR / "xgb_with_country.pkl", "rb") as f:
        xgb_with = pickle.load(f)
    with open(MODEL_DIR / "xgb_without_country.pkl", "rb") as f:
        xgb_wo = pickle.load(f)

    # 3. Evaluate models
    logger.info("Evaluating on held-out 20% test set...")

    # Evaluation results lookup
    models = {
        "RF WITH country": {
            "model": rf_with,
            "X_test": X_te_w,
            "y_test": y_te,
            "is_xgb": False,
        },
        "RF WITHOUT country": {
            "model": rf_wo,
            "X_test": X_te_wo,
            "y_test": y_te,
            "is_xgb": False,
        },
        "XGB WITH country": {
            "model": xgb_with,
            "X_test": X_te_w,
            "y_test": y_te_flip,  # XGB flips target: 1 = Non-Op, 0 = Op
            "is_xgb": True,
        },
        "XGB WITHOUT country": {
            "model": xgb_wo,
            "X_test": X_te_wo,
            "y_test": y_te_flip,
            "is_xgb": True,
        },
    }

    test_metrics = {}

    for name, info in models.items():
        model = info["model"]
        X_te_curr = info["X_test"]
        y_te_curr = info["y_test"]

        # Predictions
        y_pred = model.predict(X_te_curr)

        # Probabilities: we want P(Non-Op)
        if info["is_xgb"]:
            # XGB flips: class 1 is Non-Op
            prob_nonop = model.predict_proba(X_te_curr)[:, 1]
            pos_label = 1
        else:
            # RF standard: class 0 is Non-Op
            prob_nonop = model.predict_proba(X_te_curr)[:, 0]
            pos_label = 0

        # Flip predictions for RF metrics compatibility if needed
        # We compute metrics specifically for the Non-Operational class
        rec = recall_score(y_te_curr, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_te_curr, y_pred, pos_label=pos_label, zero_division=0)
        if info["is_xgb"]:
            auc = roc_auc_score(y_te_curr, prob_nonop)
        else:
            # For RF, y_te_curr is 1=Op, 0=Non-Op. We want P(Non-Op) to align with Non-Op=1.
            auc = roc_auc_score(1 - y_te_curr, prob_nonop)

        test_metrics[name] = {"recall": rec, "f1": f1, "roc_auc": auc}

        logger.info("--- %s Test Set Performance ---", name)
        logger.info("  Recall (Non-Op): %.3f", rec)
        logger.info("  F1     (Non-Op): %.3f", f1)
        logger.info("  ROC-AUC        : %.3f", auc)

    # 4. Scenario Analysis on the Best Model (XGB WITHOUT country)
    # This is geography-blind and has strong recall (0.685) and ROC-AUC (0.932).
    logger.info("Performing threshold scenario analysis on XGB WITHOUT country...")
    xgb_target_model = xgb_wo
    xgb_target_X = X_te_wo
    prob_nonop_xgb = xgb_target_model.predict_proba(xgb_target_X)[:, 1]

    # Generate PR curve values
    prec_pr, rec_pr, thresholds_pr = precision_recall_curve(y_te_flip, prob_nonop_xgb)
    ap = average_precision_score(y_te_flip, prob_nonop_xgb)

    threshold_scenarios = {
        "Default (0.5)": 0.5,
        "Flag top 10% (risk >= p90)": float(np.percentile(prob_nonop_xgb, 90)),
        "Flag top 20% (risk >= p80)": float(np.percentile(prob_nonop_xgb, 80)),
        "Max-F1 optimal": float(
            thresholds_pr[
                np.argmax(2 * prec_pr[:-1] * rec_pr[:-1] / (prec_pr[:-1] + rec_pr[:-1] + 1e-9))
            ]
        ),
    }

    scenarios_summary = []
    for s_name, thresh in threshold_scenarios.items():
        preds = (prob_nonop_xgb >= thresh).astype(int)
        n_flagged = int(preds.sum())
        prec_val = precision_score(y_te_flip, preds, zero_division=0)
        rec_val = recall_score(y_te_flip, preds, zero_division=0)
        f1_val = f1_score(y_te_flip, preds, zero_division=0)

        scenarios_summary.append(
            {
                "Scenario": s_name,
                "Threshold": round(thresh, 3),
                "Flagged": n_flagged,
                "Flagged%": round(n_flagged / len(y_te) * 100, 1),
                "Precision": round(prec_val, 3),
                "Recall": round(rec_val, 3),
                "F1": round(f1_val, 3),
            }
        )

    # Save metrics summaries to JSON
    with open(MODEL_DIR / "test_evaluation_metrics.json", "w") as f:
        json.dump(
            {"test_metrics": test_metrics, "threshold_scenarios": scenarios_summary, "ap": ap},
            f,
            indent=2,
        )
    logger.info("Saved test evaluation summaries to reports/test_evaluation_metrics.json")

    # 5. Output risk scores for all stations using geography-blind RF model
    logger.info("Saving full risk scores for downstream consumption...")
    all_prob_nonop_rf = rf_wo.predict_proba(X_wo)[:, 0]
    risk_df = pd.DataFrame(
        {
            "station_id": X_wo.index,  # Using index as a proxy for station identifier
            "risk_score": all_prob_nonop_rf,
            "true_label": y.map({1: "Operational", 0: "Non-Operational"}),
        }
    ).sort_values("risk_score", ascending=False)

    risk_df.to_csv(PROCESSED_DIR / "risk_scores_without_country.csv", index=False)
    logger.info("Saved full risk scores to data/processed/risk_scores_without_country.csv")


if __name__ == "__main__":
    evaluate_models()
