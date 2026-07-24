# Product Hunt Launch Traction Analysis: Project Summary

This project implements a data-driven approach to investigate the pre-launch, controllable factors that influence launch-day success on Product Hunt. The analysis is built upon a raw web-scraped dataset spanning several weeks of leaderboard history.

---

## 1. Data Collection & Scraping
We collected the dataset by scraping publicly accessible Product Hunt leaderboards and individual product launch pages using browser automation (Playwright):
- **Core Scraping**: Retrieved top daily listings, including `rank`, `name`, `tagline`, product URL, `upvotes` (traction), `comments`, and `categories`.
- **Detail Scraping**: Visited individual product pages to gather extended features: `description`, `launch_timestamp`, `media_count` (images and videos), `maker_names` (and co-maker links), and co-maker team sizes.
- **Aggregation**: Unified raw JSON files into a structured flat database ([producthunt_final.csv](producthunt_final.csv)).

---

## 2. Preprocessing & Feature Engineering
Before running models, the data was cleaned, audited, and transformed:
- **Scraping Limitations Audit**: We identified that certain metrics (such as Hunter profiles) are hidden behind dynamic redirection shields and login requirements, leaving them empty in basic scraping logs. We documented this as a data collection constraint.
- **Timing Transformation**: 
  - Parsed launch timestamps into calendar weekdays to analyze weekly volume distributions.
  - Formatted `launch_hour` into cyclic sine and cosine coordinates (`launch_hour_sin` and `launch_hour_cos`) to capture the boundary proximity between hour 23 and hour 0.
- **Categorical Processing**: Cleaned string lists under `categories` and grouped lower frequency entries under an `"Other"` category, followed by one-hot encoding.
- **Traction Target Sourcing**:
  - **Regression Target**: Log-transformed upvotes (`log(upvotes + 1)`) to normalize highly skewed popularity numbers.
  - **Classification Target (`high_traction`)**: Dynamically computed a binary indicator flagging products that placed in the **top 20% by upvotes within their specific launch date**.

---

## 3. Exploratory Data Analysis (EDA)
Inside the analytics workspace, we visualized the variables to understand baseline patterns:
* **Timing Effects**: Highlighted that launching around **00:01 PST** (Midnight PST) maximizes average upvotes by allowing the post to be visible for the full 24-hour leaderboard cycle. Tuesday and Wednesday are the busiest days, whereas weekends (Saturday/Sunday) represent a lower-barrier opportunity to rank in the top 20% due to reduced competition.
* **Presentation Effects**: Verified positive associations between vote volume and larger team sizes (`team_size`) or media asset counts (`media_count`).

---

## 4. Machine Learning Modeling & Evaluation
We split the dataset into **80% Training (457 rows)** and **20% Testing (115 rows)** and implemented both prediction tasks:

### A. Regression Task (Predicting Upvote Volume)
* **Baseline**: Ridge Regression (linear).
* **Advanced**: XGBoost Regressor (non-linear).
* **Performance**: The non-linear XGBoost Regressor achieved an $R^2$ of **0.7532** on the log scale, indicating that timing, team size, media assets, and category explain three-quarters of launch upvote variance.

### B. Classification Task (Predicting Top 20% Traction)
* **Baseline**: Logistic Regression.
* **Advanced**: XGBoost Classifier.
* **Performance**: The XGBoost Classifier achieved **96.52% accuracy**, an F1-Score of **92.31%**, and an ROC-AUC of **0.9948** on the test partition.

---

## 5. Actionable Launch Recommendations
- **Avoid Midnight Misses**: Configure launches exactly at 00:01 PST to capture the maximum daily traffic.
- **Co-Maker Mobilization**: Assemble larger launch teams (higher `team_size`) to expand co-promotion circles on launch day.
- **Polished Presentation**: Include at least 5 screenshots and a product video walkthrough to increase conversion.
