import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"):
    """
    Load the Bank Marketing dataset from UCI repository.
    """
    df = pd.read_csv(url, sep=';')
    return df

def encode_features(df: pd.DataFrame):
    """
    Encode categorical features and target variable.
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col != 'y':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})
    return df_encoded

def split_and_scale(df_encoded: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train/test sets and scale features.
    Returns both scaled and unscaled splits.
    """
    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def preprocess_pipeline(url: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
    - Load dataset
    - Encode features
    - Train-test split
    - Feature scaling
    """
    df = load_data(url) if url else load_data()
    df_encoded = encode_features(df)
    return split_and_scale(df_encoded, test_size, random_state)

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_pipeline()
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])