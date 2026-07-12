# EV Charger Reliability Analytics

Predicting public EV charging station reliability (Operational vs. Non-Operational) from infrastructure characteristics using Open Charge Map API data. Supports maintenance prioritization and regional grid vulnerability analysis.

## Repository Structure

```
ev-charger-reliability-analytics/
├── data/
│   ├── raw/                      # Raw API responses (gitignored)
│   └── processed/                # Cleaned/labeled/features datasets (gitignored)
├── notebooks/
│   ├── 01_eda_and_labeling.ipynb # Target labeling & funnel checks
│   ├── 02_preprocessing_and_features.ipynb # Leakage audit & connection flattening
│   ├── 03_modeling_and_evaluation.ipynb # Dual RF, XGB, stratified 5-fold CV
│   └── 04_explainability_and_recommendations.ipynb # SHAP explainability & business recommendations
├── src/
│   ├── extract_data.py           # Ingestion from Open Charge Map API
│   ├── label_target.py           # Implements the target labeling logic
│   ├── preprocess.py             # Feature engineering, leakage exclusions
│   ├── train_model.py            # Model training on split data
│   └── evaluate.py               # Model evaluation & scenario threshold analysis
├── reports/
│   ├── figures/                  # Visualization outputs
│   ├── final_case_study.md       # Final markdown case study report
│   ├── rf_with_country.pkl       # Serialized models
│   ├── rf_without_country.pkl
│   ├── xgb_with_country.pkl
│   └── xgb_without_country.pkl
├── build.js                      # Compiles docx report with embedded figures
├── EV_Charger_Reliability_Case_Study.docx # Compiled case study report
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Ingestion & Environment Setup
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
npm install docx
cp .env.example .env
# Open .env and add your OCM_API_KEY
```

### 2. Run the Pipeline
Run the source scripts sequentially to ingest data, engineer features, train models, and run evaluations:
```bash
python src/extract_data.py
python src/label_target.py
python src/preprocess.py
python src/train_model.py
python src/evaluate.py
```

### 3. Compile Word Document Report
Compile the final `.docx` report containing all embedded figures:
```bash
node build.js
```
*Output: `EV_Charger_Reliability_Case_Study.docx`*

---

## Methodology & Scrutiny Findings

### ⚠️ Headline Finding: South Africa Grid Confounder (29.1% Non-Op Rate)
During EDA, we discovered that South Africa has a **29.1% non-operational rate** (143/492 stations) compared to Denmark (5.4%), Portugal (2.3%), and a global median of **~0.7%**.

- **Factual Grid Context:** Fact-checking confirms that national **load-shedding** (Eskom's rotational power cuts) was not active in South Africa during the months immediately preceding the July 2026 data pull, having been avoided for over 420 consecutive days (since May 16, 2025). However, localized **load reduction** (rotational cuts to protect overloaded transformers) remained active in several provinces (such as Gauteng and KwaZulu-Natal) through mid-2026.
- **Data Audit:** 115 of the 143 non-operational South African stations (80.4%) had no operator ID (`operator_id == -1`).
- **Modeling Confounder:** When geography features were included, the model learned `operator_id == -1` as a shortcut for failure, proxying South African geography.
- **Confounder Mitigation:** When running SHAP, we analyzed the importances twice: once with the full feature set (with `-1`), and once with the `-1` missing data bucket excluded. Excluding the `-1` bucket revealed that physical infrastructure features (station size, power capability, and density) are the true physical drivers of reliability.

### 🔌 Power & Connector Verification (Excluding South Africa)
To check if the "low power / AC connector -> higher risk" pattern holds independent of South Africa, we cross-tabulated the non-operational rates in the rest of the dataset:

- **By Power Bracket (Excluding South Africa):**
  - **<7kW:** 1.89% non-op rate (13 / 689 stations)
  - **7-22kW:** 1.06% non-op rate (37 / 3,495 stations)
  - **22-50kW:** 0.49% non-op rate (9 / 1,832 stations)
  - **50kW+:** 0.37% non-op rate (14 / 3,822 stations)
- **By Connector Category (Excluding South Africa):**
  - **AC Chargers:** 1.16% non-op rate (46 / 3,969 stations)
  - **DC Chargers:** 0.46% non-op rate (27 / 5,869 stations)

*Conclusion:* The pattern is genuine and geography-independent. Slower AC infrastructure shows a 2.5x higher failure rate than DC chargers globally.

### Business Operating Scenarios (XGB WITHOUT Country)
Because default probability cutoffs (0.5) are designed for balanced classes, they make poor decision boundaries under 46.8:1 class imbalance:

| Operating Scenario | Threshold | Flags | Precision | Recall | F1 | Business Context |
|---|---|---|---|---|---|---|
| **Max-F1 Optimal** | 0.906 | 2.5% | **44.2%** | 53.5% | **0.484** | High efficiency: ~1 in 2 inspections finds failure. |
| **Top 10% Risk** | 0.146 | 10.1% | 19.1% | **93.0%** | 0.317 | Safety first: catches 93% of failures, inspects top 10% risk. |
| **Top 20% Risk** | 0.010 | 20.0% | 10.4% | **100.0%** | 0.188 | Conservative search: sweeps 20% of network to catch all failures. |
