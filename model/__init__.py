"""Stub model package to allow the Streamlit app to run when real models
are not present. Each module exposes a `train_and_evaluate(X_train, X_test,
y_train, y_test)` function returning (y_pred, metrics_dict).
"""

__all__ = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost_model",
]
