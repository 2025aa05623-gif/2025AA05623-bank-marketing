import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from model.logistic_regression import train_and_evaluate as lr
from model.decision_tree import train_and_evaluate as dt
from model.knn import train_and_evaluate as knn
from model.naive_bayes import train_and_evaluate as nb
from model.random_forest import train_and_evaluate as rf
from model.xgboost_model import train_and_evaluate as xgb

# Streamlit page setup
st.set_page_config(page_title="Bank Marketing Classification")
st.title("🏦 Bank Marketing Classification")

# Dataset URL and columns
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv"
df = pd.read_csv(url, sep=";")

# Download button for test CSV
st.download_button(
    label="📥 Download Test CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="bank_marketing_test.csv",
    mime="text/csv"
)

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

if data_option == "Load from UCI URL":
    st.info("Loading Bank Marketing dataset from UCI repository...")
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head(10), width='stretch')

elif data_option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload Bank Marketing CSV (Test Data Only)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        st.dataframe(df.head(10), width='stretch')

        # Download button for uploaded test CSV
        st.download_button(
            label="📥 Download Test CSV",
            data=uploaded_file.getvalue(),
            file_name="bank_marketing_test.csv",
            mime="text/csv"
        )

if df is not None:
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found in dataset.")
    else:
        # Features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1"] else 0)

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if st.button("Run Model"):
            model_map = {
                "Logistic Regression": lr,
                "Decision Tree": dt,
                "KNN": knn,
                "Naive Bayes": nb,
                "Random Forest": rf,
                "XGBoost": xgb
            }

            y_pred, metrics = model_map[model_name](
                X_train, X_test, y_train, y_test
            )

            st.subheader("📊 Evaluation Metrics")
            metrics_df = pd.DataFrame(metrics, index=["Score"]).T
            metrics_df_display = metrics_df.round(4)
            st.table(metrics_df_display)

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC", f"{metrics['AUC']:.4f}")
            with col3:
                st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

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
            report_df = pd.DataFrame(report).transpose()
            report_df_display = report_df.round(4)
            st.table(report_df_display)