# Final Case Study: EV Charger Reliability Analytics

This report presents the final predictive modeling and explainability results for the EV Charger Reliability Analytics case study, designed to support maintenance prioritization and network reliability planning.

---

## 1. Executive Summary

- **Business Objective:** Public EV charging networks operate thousands of stations across diverse regions, operators, and hardware configurations. Manually monitoring every station is expensive and inefficient. The goal of this project is to build an infrastructure-only binary classifier to predict whether a charging station is likely to be **Operational** or **Non-Operational** and expose model confidence as a per-station risk score for maintenance prioritization.
- **Headline Results:** 
  - The best-performing model is the geography-blind **XGBoost Classifier** (trained *without* country features to prevent geographic memorization), achieving a cross-validated **ROC-AUC of 0.932 ± 0.021** on the Non-Operational class.
  - Under a realistic business operating scenario, flagging the **top 10% highest-risk stations** for preventative inspections successfully catches **93.0% of all non-operational stations** in the held-out test set, providing a robust, data-driven alternative to manual monitoring.

---

## 2. Key Findings

We present the four verified findings from our exploratory data analysis, modeling, and SHAP explainability work, ordered by strength of evidence:

### I. Slower AC Charger Infrastructure Exhibits Significantly Higher Failure Risk (Genuine Signal)
We verified that charger power capacity and connector category (AC vs. DC) are genuine, geography-independent predictors of reliability. The monotonic trend of decreasing risk as power capacity increases holds perfectly outside South Africa:

- **By Power Bracket (Excluding South Africa):**
  - **<7kW:** 1.89% non-op rate (13 / 689 stations)
  - **7-22kW:** 1.06% non-op rate (37 / 3,495 stations)
  - **22-50kW:** 0.49% non-op rate (9 / 1,832 stations)
  - **50kW+:** 0.37% non-op rate (14 / 3,822 stations)
- **By Connector Category (Excluding South Africa):**
  - **AC Chargers:** 1.16% non-op rate (46 / 3,969 stations)
  - **DC Chargers:** 0.46% non-op rate (27 / 5,869 stations)

> **Sample Size Caveat:** While the downward monotonic trend is directionally convincing and physically logical (operators prioritize high-revenue fast DC chargers for immediate repairs), the overall number of non-operational stations outside South Africa is small (~73 total). Consequently, individual bracket percentages carry meaningful statistical uncertainty due to low absolute sample counts.

### II. South Africa Shows a Large, Statistically Real Regional Gap (Grid-Correlated)
South Africa is a massive outlier, exhibiting a **29.1% non-operational rate** (143 / 492 stations) compared to Denmark (5.4%), Portugal (2.3%), and a global median of **~0.7%**. 
- **Factual Context:** Fact-checking confirms that national **load-shedding** (Eskom's rotational power cuts) was not active in South Africa during the months immediately preceding the July 2026 data pull, having been avoided for over 420 consecutive days (since May 16, 2025). However, localized **load reduction** (rotational cuts to protect overloaded transformers) remained active in several provinces (such as Gauteng and KwaZulu-Natal) through mid-2026.
- **Hypothesis:** A plausible explanation for the high failure rate is a combination of (a) stale/unmaintained status records remaining from the severe 2022–2024 load-shedding crisis that were never updated, and (b) localized hardware degradation or connectivity drops from ongoing load reduction. This dataset contains no direct evidence of failure cause and cannot confirm causation — only the geographic correlation is empirically supported.

### III. Operator Identity is Confounded by Reporting Gaps (Metadata Artifact)
- **Scrutiny Verdict:** **Inconclusive.** In the full model, missing operator data (`operator_id == -1`) was identified as the top predictor of failure.
- **The Confounder:** 115 of the 143 non-operational South African stations (80.4%) had no operator ID (`operator_id == -1`). Because the South African stations suffer from localized grid/maintenance issues, the model exploited the missing operator ID as a proxy for South African geography. For stations with valid operator IDs outside of South Africa, failure rates are consistently low (~1.1%). Thus, operator identity signal is confounded by the reporting gap and cannot be cleanly evaluated with the current data.

### IV. Risk Ranking is Excellent, but Requires Cost-Tradeoff Thresholds
While the model's threshold-free ranking quality is very high (ROC-AUC of ~0.93), the default 0.5 classification threshold is inappropriate for a heavily imbalanced dataset (46.8 : 1 class ratio). The optimal operating point depends on a business cost tradeoff:

| Business Scenario | Decision Cutoff (Prob) | Stations Flagged | % of Network Flagged | Precision (Inspection Success) | Recall (% Failures Caught) | F1 Score |
|---|---|---|---|---|---|---|
| **Max-F1 Optimal** | 0.906 | 52 | 2.5% | **44.2%** | 53.5% | **0.484** |
| **Top 10% Risk Sweep** | 0.146 | 209 | 10.1% | 19.1% | **0.930** | 0.317 |
| **Top 20% Risk Sweep** | 0.010 | 414 | 20.0% | 10.4% | **1.000** | 0.188 |

---

## 3. Business Recommendations

Based on our verified findings, we recommend the following strategic actions:

1. **Focus Maintenance Sweeps on AC/Low-Power Plugs:** Allocate routine, preventative maintenance cycles toward slower AC charging stations (<22kW). Since these destination chargers have low revenue margins, operators rarely monitor them in real-time, resulting in a 2.5x higher failure rate compared to DC chargers.
2. **Implement a Top-10% Risk Inspection Cycle:** Adopt the **Top 10% Risk Scenario** (threshold 0.146) as the operational threshold for quarterly scheduling. This catches **93.0% of all failures** while restricting the inspection field to just 10% of the active network, maximizing resource efficiency.
3. **Isolate South African Risk as a Regional Infrastructure Case:** Treat South African charger failures as a localized infrastructure risk rather than an operator-performance issue. Invest in hardware mitigations (voltage surge protectors to buffer local load reduction spikes) rather than penalizing operator service level agreements (SLAs).
4. **Mandate Operator Registration Standards:** Resolve the metadata reporting gap by requiring operators to populate valid IDs. Filling this missing data gap will unconfound the operator characteristics feature, allowing future models to identify underperforming maintenance operators.

---

## 4. Limitations & Future Work

### Limitations
- **Class Imbalance:** With only 216 Non-Operational stations out of 10,330, training suffers from thin positive-class representation.
- **Snapshot Status Data:** The dataset represents a static snapshot. It contains no time-series tracking and no failure-cause metadata (e.g., vandalism, communications drop, or component wear).
- **Geographic Bias:** OCM sampling caps records at 500 per country, which may under-represent high-density regional networks.
- **Operator Confounding:** The high proportion of missing operator metadata in South Africa prevents clean isolation of operator quality.

### Future Work
- **Time-Series Monitoring:** Transition from a snapshot to longitudinal status tracking to model mean-time-to-repair (MTTR) and failure rates over time.
- **Failure Cause Metadata:** Capture explicit reasons for downtime (grid outage vs. hardware breakdown) to train separate risk models.
- **Probability Calibration:** Validate and calibrate model outputs using Platt Scaling or Isotonic Regression if risk scores are to be treated as probabilities of failure.
- **Expand Country Coverage:** Increase the country list and remove the OCM pagination cap to capture denser network dynamics in high-adoption markets.
