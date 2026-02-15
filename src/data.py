import pandas as pd
import logging
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Default dataset URL
BANK_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv"
BANK_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def load_data(source: str = "url", uploaded_file=None) -> pd.DataFrame:
    """
    Load the Bank Marketing dataset from UCI URL or from an uploaded CSV file.

    Parameters
    ----------
    source : str
        "url" to load from UCI repository, "upload" to load from user-uploaded file.
    uploaded_file : UploadedFile
        Streamlit UploadedFile object if source="upload".

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    try:
        if source == "url":
            try:
                df = pd.read_csv(BANK_DATA_URL, sep=";")
                st.info("Loading Bank Marketing dataset from UCI repository...")
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                return df
            except Exception as e:
                # CSV from UCI URL failed; silently attempt local file and ZIP fallback
                # try local copy first
                try:
                    import os
                    if os.path.exists('data/bank.csv'):
                        df_local = pd.read_csv('data/bank.csv', sep=';')
                        st.success(f"✅ Loaded local dataset: data/bank.csv (shape: {df_local.shape})")
                        return df_local
                except Exception:
                    pass

                # try ZIP archive from UCI
                try:
                    import urllib.request, io, zipfile
                    with urllib.request.urlopen(BANK_ZIP_URL, timeout=20) as resp:
                        data = resp.read()
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        for name in ('bank-full.csv', 'bank.csv'):
                                    if name in z.namelist():
                                        with z.open(name) as f:
                                            df_zip = pd.read_csv(f, sep=';')
                                            return df_zip
                except Exception as e2:
                    logging.exception("Failed to load dataset from UCI (CSV and ZIP attempts failed).")
                    return None

        elif source == "upload" and uploaded_file is not None:
            # try semicolon first, fall back to default delimiter
            try:
                df = pd.read_csv(uploaded_file, sep=";")
            except Exception:
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                df = pd.read_csv(uploaded_file)
            return df

        else:
            return None
    except Exception as e:
        logging.exception("Failed to load dataset")
        return None


def show_sample(df: pd.DataFrame, rows: int = 10):
    """
    Display a sample of the dataset in Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to display.
    rows : int
        Number of rows to show.
    """
    if df is not None:
        st.dataframe(df.head(rows))


def preprocess_data(df: pd.DataFrame, target_column: str = "y"):
    """
    Preprocess the Bank Marketing dataset:
    - Encode categorical features
    - Scale numerical features
    - Convert target column to binary (yes=1, no=0)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    target_column : str
        Name of target column (default: "y")

    Returns
    -------
    X_scaled : np.ndarray
        Preprocessed feature matrix
    y : pd.Series
        Binary target vector
    categorical_cols : list
        List of categorical columns encoded
    """
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found in dataset.")
        return None, None, None

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0)

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return X_scaled, y, list(categorical_cols)