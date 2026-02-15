# 🏦 Bank Marketing Classification App

This Streamlit application allows users to train and evaluate multiple machine learning models on the **Bank Marketing dataset** from the UCI Machine Learning Repository.  
The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution, with the target variable `y` indicating whether the client subscribed to a term deposit (`yes`/`no`).

---

## 📂 Dataset
- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Format:** CSV with semicolon (`;`) separator
- **Target Column:** `y` (binary: `yes` or `no`)

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
- Download options for test CSV

---

## 📊 Sample Metrics Comparison

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
