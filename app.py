import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.metrics import confusion_matrix, classification_report

# Import preprocessing functions
from data import load_data, preprocess_data

# Import model training functions
from model.logistic_regression import train_and_evaluate as lr
from model.decision_tree import train_and_evaluate as dt
from model.knn import train_and_evaluate as knn
from model.naive_bayes import train_and_evaluate as nb
from model.random_forest import train_and_evaluate as rf
from model.xgboost_model import train_and_evaluate as xgb

st.set_page_config(page_title="Bank Marketing Classification")
st.title("🏦 Bank Marketing Classification")

# Dataset source selection
data_option = st.radio(
    "Select dataset source:",
    ("Load from UCI URL", "Upload CSV file")
)

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

# Target column input
target_column = st.text_input(
    "Target Column Name",
    value="y",
    help="Enter the name of your target/label column (default: y)"
)

df = None
if data_option == "Load from UCI URL":
    st.info("Loading Bank Marketing dataset from UCI repository...")
    df = load_data(source="uci")
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head(10), width='stretch')

elif data_option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload Bank Marketing CSV (Test Data Only)", type=["csv"])
    if uploaded_file:
        df = load_data(source="upload", uploaded_file=uploaded_file)
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        st.dataframe(df.head(10), width='stretch')

if df is not None:
    try:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(df, target_column=target_column)

        if st.button("Run Model"):
            model_map = {
                "Logistic Regression": lr,
                "Decision Tree": dt,
                "KNN": knn,
                "Naive Bayes": nb,
                "Random Forest": rf,
                "XGBoost": xgb
            }

            # Run selected model
            y_pred, metrics = model_map[model_name](
                X_train_scaled, X_test_scaled, y_train, y_test
            )

            st.subheader("📊 Evaluation Metrics")
            metrics_df = pd.DataFrame(metrics, index=["Score"]).T
            st.table(metrics_df)

            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No", "Yes"],
                        yticklabels=["No", "Yes"],
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("📑 Classification Report")
            report = classification_report(y_test, y_pred,
                                           target_names=["No", "Yes"],
                                           output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)

            # Download buttons
            st.download_button(
                label="⬇️ Download Evaluation Metrics (CSV)",
                data=metrics_df.to_csv().encode("utf-8"),
                file_name="evaluation_metrics.csv",
                mime="text/csv"
            )

            st.download_button(
                label="⬇️ Download Classification Report (CSV)",
                data=report_df.to_csv().encode("utf-8"),
                file_name="classification_report.csv",
                mime="text/csv"
            )
    except ValueError as e:
        st.error(str(e))