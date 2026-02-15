import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, target_column="income"):
    """
    Preprocess Adult Income dataset:
    - Strip spaces
    - Drop missing values
    - Encode categorical features
    - Encode target column (income >50K → 1, else 0)
    - Scale numeric features
    Returns: X_scaled, y
    """
    # Clean strings and drop missing
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna()

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_column)
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Encode target
    y = df[target_column].apply(lambda x: 1 if str(x).strip() in [">50K", "1"] else 0)
    X = df.drop(target_column, axis=1)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y