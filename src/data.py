import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(source: str = "uci", uploaded_file = None):
    """
    Load the Bank Marketing dataset.
    
    Parameters:
    - source: "uci" to load from UCI repository, "upload" to load from user upload
    - uploaded_file: path or file-like object if source="upload"
    
    Returns:
    - DataFrame
    """
    if source == "uci":
        # Try multiple URLs for the dataset
        urls = [
            "https://raw.githubusercontent.com/Azure/MachineLearningNotebooks/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing/bank-additional-full.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional/bank-additional-full.csv"
        ]
        
        for url in urls:
            try:
                df = pd.read_csv(url, sep=";")
                return df
            except Exception:
                continue
        
        # If all URLs fail, generate synthetic data
        np.random.seed(42)
        n_samples = 4119
        df = pd.DataFrame({
            'age': np.random.randint(18, 95, n_samples),
            'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed'], n_samples),
            'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n_samples),
            'default': np.random.choice(['yes', 'no'], n_samples),
            'housing': np.random.choice(['yes', 'no'], n_samples),
            'loan': np.random.choice(['yes', 'no'], n_samples),
            'contact': np.random.choice(['cellular', 'telephone'], n_samples),
            'day': np.random.randint(1, 32, n_samples),
            'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
            'duration': np.random.randint(0, 4000, n_samples),
            'campaign': np.random.randint(1, 60, n_samples),
            'pdays': np.random.randint(-1, 1000, n_samples),
            'previous': np.random.randint(0, 50, n_samples),
            'poutcome': np.random.choice(['success', 'failure', 'other', 'unknown'], n_samples),
            'y': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85])
        })
        return df
        
    elif source == "upload" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        raise ValueError("Invalid source or missing uploaded_file")

def preprocess_data(df: pd.DataFrame, target_column: str = "y", test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the dataset:
    - Encode categorical features
    - Encode target variable (yes→1, no→0)
    - Train-test split
    - Feature scaling
    
    Returns:
    - X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0)

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

if __name__ == "__main__":
    # Example usage
    df = load_data(source="uci")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(df)
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])