try:
    from xgboost import XGBClassifier
    _have_xgb = True
except Exception:
    _have_xgb = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def train_and_evaluate(X_train, X_test, y_train, y_test):
    if _have_xgb:
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        # some classifiers may not implement predict_proba
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
