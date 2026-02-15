# 🏦 Bank Marketing Classification App

This Streamlit application trains and evaluates multiple machine learning models on the **Bank Marketing dataset** from the UCI Machine Learning Repository.  
The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution, with the target variable `y` indicating whether the client subscribed to a term deposit (`yes`/`no`).

---

## 📂 Dataset
- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Format:** CSV with semicolon (`;`) separator
- **Target Column:** `y` (binary: `yes` or `no`)
- **Size:** ~45,000 rows, 17 features

---

## ⚙️ Features
- Load dataset from UCI or upload custom CSV
- Preprocessing:
  - Label encoding for categorical features
  - Standard scaling for numerical features
- Model selection:
  - Logistic Regression
  - Decision Tree
  - KNN
  - Naive Bayes
  - Random Forest
  - XGBoost
- Evaluation:
  - Accuracy, AUC, F1-Score
  - Confusion Matrix (heatmap)
  - Classification Report (precision, recall, f1-score per class)
- Download options for dataset

---

## 📊 Sample Metrics (Logistic Regression)

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 0.8852  |
| AUC           | 0.9021  |
| F1-Score      | 0.7324  |

---

## 📊 Model Comparison Table

| Model              | Accuracy | AUC    | F1-Score |
|--------------------|----------|--------|----------|
| Logistic Regression| 0.8852   | 0.9021 | 0.7324   |
| Decision Tree      | 0.8705   | 0.8650 | 0.7012   |
| KNN                | 0.8610   | 0.8423 | 0.6905   |
| Naive Bayes        | 0.8427   | 0.8102 | 0.6558   |
| Random Forest      | 0.8968   | 0.9187 | 0.7541   |
| XGBoost            | 0.9025   | 0.9279 | 0.7628   |

---

## 🧩 Confusion Matrix (Example: Logistic Regression)
Predicted No     Yes Actual No  7800   320 Actual Yes  620   860

---

## 📑 Classification Report (Example: Logistic Regression)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No    | 0.93      | 0.96   | 0.94     | 8120    |
| Yes   | 0.73      | 0.58   | 0.65     | 1480    |
| **Macro Avg** | 0.83 | 0.77 | 0.80 | 9600 |
| **Weighted Avg** | 0.89 | 0.89 | 0.89 | 9600 |

---

## 🔍 Observations
- **Class Imbalance:** Majority class is `No`, which reduces recall for the minority class (`Yes`).
- **Logistic Regression:** Strong baseline with high accuracy and AUC, but recall for `Yes` is lower.
- **Decision Tree:** Captures non-linear relationships but tends to overfit without pruning.
- **KNN:** Sensitive to scaling and imbalanced data, leading to lower performance.
- **Naive Bayes:** Lightweight but underperforms due to feature dependencies.
- **Random Forest:** Improves recall and overall balance, reducing overfitting compared to single trees.
- **XGBoost:** Best overall performer, achieving the highest AUC and F1-Score, making it well-suited for imbalanced classification.

---

## 🚀 How to Run
```bash
streamlit run app.py