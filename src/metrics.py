from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained scikit-learn or XGBoost model
        X_test: Test features
        y_test: Test target values
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics