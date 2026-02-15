# train_models.py
# Trains all ML models on Bank Marketing dataset and saves them for Streamlit app

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Load Dataset (Bank Marketing from UCI)
# --------------------------------------------------
data = pd.read_csv("data/bank.csv", sep=";")

# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------
# Drop rows with missing values if any
data.dropna(inplace=True)

# Encode categorical columns
encoder = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = encoder.fit_transform(data[col])

# --------------------------------------------------
# 3. Feature / Target Split
# --------------------------------------------------
X = data.drop("y", axis=1)
y = data["y"].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0)

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler (important for Streamlit)
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# --------------------------------------------------
# 6. Model Definitions
# --------------------------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "dt": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(n_estimators=120, random_state=42),
    "xgb": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

# --------------------------------------------------
# 7. Evaluation Function
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = "NA"

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# --------------------------------------------------
# 8. Train, Evaluate & Save Models
# --------------------------------------------------
print("\nMODEL PERFORMANCE SUMMARY\n")

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    results[name] = metrics

    joblib.dump(model, f"model/{name}.pkl")

    print(f"Model: {name.upper()}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("-" * 40)

print("\nAll models trained and saved successfully.")