# Bank Marketing Dataset - ML Assignment 2

## Problem Statement
The goal of this project is to build and evaluate multiple classification models on the **Bank Marketing Dataset** (UCI repository).  
The task is to predict whether a client subscribes to a term deposit (`y` = yes/no) based on socio-economic and marketing campaign features.

This project demonstrates an end-to-end ML workflow: dataset preprocessing, model training, evaluation, and deployment using **Streamlit Community Cloud**.

---

## Dataset Description
- **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
- **Instances**: 45,211  
- **Features**: 16 (mix of categorical and numerical)  
- **Target Variable**: `y` (binary: yes/no)  
- **Objective**: Predict subscription to a term deposit.  

---

## Models Implemented
Six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## Evaluation Metrics
For each model, the following metrics were calculated:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Comparison Table

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression  |          |     |           |        |    |     |
| Decision Tree        |          |     |           |        |    |     |
| kNN                  |          |     |           |        |    |     |
| Naive Bayes          |          |     |           |        |    |     |
| Random Forest        |          |     |           |        |    |     |
| XGBoost              |          |     |           |        |    |     |

*(Fill in values after running your app.)*

---

## Observations

| ML Model Name        | Observation about model performance |
|----------------------|-------------------------------------|
| Logistic Regression  |                                     |
| Decision Tree        |                                     |
| kNN                  |                                     |
| Naive Bayes          |                                     |
| Random Forest        |                                     |
| XGBoost              |                                     |

*(Add insights on which models performed best, trade-offs, and dataset-specific behavior.)*

---

## Streamlit App Deployment
- **App Link**: [Your Streamlit Cloud App URL]  
- **Features**:
  - Dataset upload option (CSV)  
  - Model selection dropdown  
  - Display of evaluation metrics  
  - Confusion matrix & classification report  
  - Download predictions as CSV  

---

## Repository Structure