import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    from model.logistic_regression import train_and_evaluate as lr
    from model.decision_tree import train_and_evaluate as dt
    from model.knn import train_and_evaluate as knn
    from model.naive_bayes import train_and_evaluate as nb
    from model.random_forest import train_and_evaluate as rf
    from model.xgboost_model import train_and_evaluate as xgb
    models_available = True
    missing_model_import_error = None
except Exception as e:
    lr = dt = knn = nb = rf = xgb = None
    models_available = False
    import traceback
    missing_model_import_error = traceback.format_exc()

# Streamlit page setup
st.set_page_config(page_title="Bank Marketing Classification")
st.title("🏦 Bank Marketing Classification")

from data import load_data, show_sample, preprocess_data

# --------------------------------------------------
# Dataset Loading Section
# --------------------------------------------------
data_option = st.radio("Select dataset source:", ("Load from UCI URL", "Upload CSV file"))
df = None

if data_option == "Load from UCI URL":
    df = load_data(source="url")
    if df is not None:
        show_sample(df)
        st.download_button(
            label="📥 Download Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bank_marketing.csv",
            mime="text/csv"
        )

elif data_option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload Bank Marketing CSV", type=["csv"])
    if uploaded_file:
        df = load_data(source="upload", uploaded_file=uploaded_file)
        if df is not None:
            show_sample(df)
            # Download button placed right after title for uploaded file
            st.download_button(
                label="📥 Download Uploaded CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="bank_marketing_uploaded.csv",
                mime="text/csv"
            )

# --------------------------------------------------
# Model Selection & Target Column
# --------------------------------------------------
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

target_column = st.text_input(
    "Target Column Name",
    value="y",
    help="Enter the name of your target/label column (default: y)"
)

# --------------------------------------------------
# Preprocessing & Training
# --------------------------------------------------
if df is not None:
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found in dataset.")
    else:
        X_scaled, y, categorical_cols = preprocess_data(df, target_column=target_column)
        if X_scaled is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            if st.button("Run Model"):
                if not models_available:
                    st.error("Model package not found. Ensure `train_models.py` exists with required functions.")
                    if missing_model_import_error:
                        st.text_area("Import error details", missing_model_import_error, height=200)
                    st.stop()

                model_map = {
                    "Logistic Regression": lr,
                    "Decision Tree": dt,
                    "KNN": knn,
                    "Naive Bayes": nb,
                    "Random Forest": rf,
                    "XGBoost": xgb
                }

                y_pred, metrics = model_map[model_name](X_train, X_test, y_train, y_test)

                st.subheader("📊 Evaluation Metrics")
                metrics_df = pd.DataFrame(metrics, index=["Score"]).T.round(4)
                st.table(metrics_df)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('Accuracy', 0.0):.4f}")
                with col2:
                    st.metric("AUC", f"{metrics.get('AUC', 0.0):.4f}")
                with col3:
                    st.metric("F1-Score", f"{metrics.get('F1-Score', 0.0):.4f}")
                with col4:
                    st.metric("MCC", f"{metrics.get('MCC', 0.0):.4f}")

                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"],
                    ax=ax,
                    cbar_kws={"label": "Count"},
                    annot_kws={"size": 14, "weight": "bold"}
                )
                ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
                ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
                ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
                st.pyplot(fig)

                st.subheader("📑 Classification Report")
                report = classification_report(y_test, y_pred,
                                               target_names=["No", "Yes"],
                                               output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(4)
                st.table(report_df)
