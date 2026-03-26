# Credit Risk Classification

A machine learning pipeline for predicting credit default risk on an imbalanced dataset, comparing Logistic Regression, Random Forest, and K-Nearest Neighbors classifiers with optimized classification thresholds.

## Overview

Standard classification models trained on imbalanced data tend to over-predict the majority class — in credit risk, that means approving borrowers who will default. This project addresses that problem directly by optimizing classification thresholds using the **Youden Index** (maximizing sensitivity + specificity − 1), producing models that make a more informed trade-off between false approvals and false denials.

**Dataset:** [South German Credit Dataset](https://archive.ics.uci.edu/dataset/573/south+german+credit) — 1,000 loan applicants, 20 features, 70/30 class split (good/bad credit).

---

## Key Results

| Model | Accuracy | Sensitivity | Specificity | MSE |
|---|---|---|---|---|
| Logistic Regression | **74.4%** | **0.793** | 0.642 | **0.256** |
| Random Forest | 64.8% | 0.550 | **0.852** | 0.352 |
| KNN (k=14) | 64.8% | 0.556 | 0.840 | 0.352 |

Logistic Regression achieved the best overall balance — highest accuracy, lowest MSE, and strong sensitivity after threshold tuning. Random Forest and KNN trade sensitivity for specificity, which may be preferable in contexts where false approvals are more costly than false denials.

---

## Methodology

### Feature Engineering
- Split `personal_status` into separate `sex` and `marital_status` variables to isolate the independent effects of each on credit outcomes.
- Scaled all continuous features prior to KNN fitting to prevent distance-metric distortion.

### Exploratory Clustering
Applied K-Means (k=3) on continuous features to identify natural groupings in the applicant population before modeling:

| Cluster | Duration | Credit Amount | Age | Interpretation |
|---|---|---|---|---|
| 1 | 16.2 mo | $2,419 | 46.9 yrs | Established borrowers, lower risk |
| 2 | 39.6 mo | $8,023 | 35.3 yrs | High-value loans, moderate age |
| 3 | 17.5 mo | $2,225 | 29.7 yrs | Young borrowers, smaller loans |

### Model Training
- **75/25 train/test split** with `set.seed(1)` for reproducibility
- **Logistic Regression:** Forward stepwise selection via AIC (reduced AIC from 752.3 → 738.1)
- **Random Forest:** 500 trees, all features, variable importance via Mean Decrease Accuracy and Gini
- **KNN:** 25-fold cross-validation over k=1..25; optimal k=14

### Threshold Optimization
All three models were tuned using the **Youden Index** rather than a fixed 0.5 threshold, accounting for class imbalance and the asymmetric costs of misclassification in lending decisions.

```r
maximize_youden <- function(predictions) {
  thresholds <- seq(0, 1, by = 0.01)
  # ... sweep thresholds, compute sensitivity + specificity - 1
  best_row <- results[which.max(results$youden_index), ]
  return(best_row)
}
```

Optimized thresholds:
- Logistic Regression: **0.63** (default: 0.50)
- Random Forest: **0.75** (default: 0.50)
- KNN: **0.72** (default: 0.50)

### Variable Importance (Random Forest)
Top predictors by Mean Decrease Accuracy: `checking_status`, `duration`, `credit_amount`, `age`, `employment`, `purpose` — consistent with domain knowledge that account standing and loan size are primary drivers of default risk.

---

## Files

```
├── credit_risk_analysis.R     # Full analysis script
├── credit_customers.csv       # Source dataset
└── README.md
```

---

## Requirements

```r
install.packages(c("tidyverse", "caret", "randomForest", "knitr", "cluster"))
```

---

## How to Run

```r
# Clone the repo and set your working directory, then:
source("credit_risk_analysis.R")
```

All random seeds are fixed (`set.seed(1)`) for full reproducibility.

---

## Takeaways

- Threshold tuning meaningfully changes model behavior — the Random Forest's specificity improved from 48% to 85% by raising the threshold from 0.50 to 0.75, at the cost of sensitivity.
- Logistic Regression outperformed both ensemble and distance-based methods on this dataset, likely because the decision boundary is approximately linear in the most predictive features.
- The Youden Index is a practical threshold selection criterion when false positives and false negatives carry different costs — a common situation in financial and clinical applications.
