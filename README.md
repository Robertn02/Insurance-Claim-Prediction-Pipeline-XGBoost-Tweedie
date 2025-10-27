# Insurance-Claim-Prediction-Pipeline-XGBoost-Tweedie

This project develops a clean, reproducible machine learning pipeline to predict insurance claim outcomes.  
It combines advanced **feature engineering**, **winsorization**, and **gradient boosting (XGBoost with a Tweedie objective)**  
to model both claim **severity (LC, HALC)** and **frequency (CS)** in a unified, production-ready workflow.

---

## Overview

Accurately predicting claim cost and likelihood is critical for insurers to manage risk, price policies, and allocate reserves.  
This project applies a **data-driven approach** to forecast three key targets:

| Target | Description |
|---------|--------------|
| **LC** | Loss Cost (`X.15 / X.16`) — average claim cost per policy |
| **HALC** | Historically Adjusted Loss Cost (`LC × X.18`) — adjusted for claim inflation |
| **CS** | Claim Status (`1 if X.16 > 0 else 0`) — binary claim occurrence |

By integrating time-based and ratio-based engineered features with scalable model pipelines, the notebook ensures robust, interpretable, and reproducible results suitable for analytics and actuarial teams.

---

## Key Features

- **Unified Preprocessing:** One consistent data-prep pipeline for train, test, and submission.
- **Feature Engineering:** Date differentials (e.g., policy age), value/power ratios, premium ratios, and derived vehicle features.
- **Target Winsorization:** Reduces skew at the 99th percentile for LC and HALC stability.
- **Modeling:**
  - `XGBRegressor` with Tweedie objective for LC & HALC.
  - `RandomForestClassifier` for CS classification.
- **Evaluation:** RMSE for regression, ROC-AUC for classification.
- **Feature Importance:** Matplotlib visualizations for top predictive factors.
- **Deployment-Ready Submission:** Automated test preprocessing and CSV export.

---

## Tech Stack

- **Language:** Python (3.x)
- **Libraries:**  
  `Pandas`, `NumPy`, `Scikit-Learn`, `XGBoost`, `Matplotlib`
- **Environment:** Google Colab / Jupyter Notebook


