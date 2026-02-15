from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0
    except Exception:
        auc = 0.0
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(auc),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    return y_pred, metrics
