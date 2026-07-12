# EV Charger Reliability Analytics

Predicting the operational status of public EV charging stations using infrastructure characteristics, to support maintenance prioritization and network reliability planning.

## Problem

Public EV charging networks operate thousands of stations across regions, operators, and hardware configurations. Manually monitoring every station for failures is expensive and slow. This project asks:

**Can infrastructure characteristics (operator, connector type, power rating, usage type, location) predict whether a charging station is likely to be operational or non-operational?**

## Data

Collected via the [Open Charge Map API](https://openchargemap.org/site/develop/api) — a global public registry of EV charging locations.

Target variable is derived from the API's own `StatusType.IsOperational` flag (not a manually guessed status mapping), with the following handling:

| Status | Included? | Label |
|---|---|---|
| Currently Available, Currently In Use, Temporarily Unavailable, Operational, Partly Operational | ✅ | Operational |
| Not Operational | ✅ | Non-operational |
| Unknown, Planned For Future Date, Removed (Decommissioned/Duplicate) | ❌ Excluded | — |

Rationale for each exclusion/grouping decision is documented in `notebooks/01_eda_and_labeling.ipynb`.

## Approach

1. **EDA** — class distribution, missing values, feature quality, correlation checks
2. **Feature selection** — infrastructure-only predictors (operator, connection type, power, current type, usage type, region); status-derived and post-event fields explicitly excluded to prevent leakage
3. **Modeling** — binary classifier (Random Forest / XGBoost), evaluated on recall/F1 for the non-operational class (accuracy is not the primary metric due to class imbalance)
4. **Explainability** — feature importance / SHAP to identify which infrastructure characteristics are associated with higher operational risk
5. **Business output** — model confidence exposed as a per-station **operational risk score** for maintenance prioritization (not presented as a calibrated probability unless calibration is explicitly validated)

## Repo structure
data/               raw and processed datasets (gitignored if large)
notebooks/          EDA, labeling, modeling, evaluation
src/                extraction, preprocessing, training scripts
reports/            final case study writeup, figures
.env.example        template for API key config
## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OCM API key
python src/extract_data.py
```

## Status

🚧 In progress — B.Tech final year Business Analytics case study.

## Author

Kireeti — B.Tech CSE, Amrita Vishwa Vidyapeetham
