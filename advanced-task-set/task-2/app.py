from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==================== PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ====================
st.set_page_config(
    page_title="Telco Churn Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD FILE PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "churn_pipeline.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

# ==================== VALIDATION ====================
if not PIPELINE_PATH.exists():
    st.error(f"Missing file: {PIPELINE_PATH}")
    st.stop()

if not METADATA_PATH.exists():
    st.error(f"Missing file: {METADATA_PATH}")
    st.stop()

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata

pipeline, metadata = load_model()

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
.main {
    padding: 0rem 0rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("# 🔮 Customer Churn Prediction System")
st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown(f"""
    **Model Performance:**
    - Accuracy: {metadata['accuracy']:.4f}
    - Precision: {metadata['precision']:.4f}
    - Recall: {metadata['recall']:.4f}
    - F1-Score: {metadata['f1']:.4f}
    - ROC-AUC: {metadata['roc_auc']:.4f}
    """)

    for key, value in metadata['best_params'].items():
        st.write(f"- {key}: {value}")

    st.markdown("---")
    st.info("Telco Churn Prediction App using ML Pipeline")

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Prediction",
    "📤 Batch Prediction",
    "📈 Model Details"
])

# ==================== TAB 1 ====================
with tab1:
    st.subheader("Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 45)
        tenure = st.slider("Tenure", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    input_data = pd.DataFrame([{
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Gender": gender,
        "Contract": contract,
        "PaymentMethod": payment
    }])

    if st.button("Predict"):
        try:
            pred = pipeline.predict(input_data)[0]
            prob = pipeline.predict_proba(input_data)[0]

            st.metric("Prediction", "Churn" if pred == 1 else "Stay")
            st.metric("Churn Probability", f"{prob[1]:.2%}")

        except Exception as e:
            st.error(str(e))

# ==================== TAB 2 ====================
with tab2:
    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        if st.button("Predict Batch"):
            try:
                preds = pipeline.predict(df)
                probs = pipeline.predict_proba(df)

                df["Prediction"] = preds
                df["Probability"] = probs[:, 1]

                st.dataframe(df)

                st.download_button(
                    "Download Results",
                    df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(str(e))

# ==================== TAB 3 ====================
with tab3:
    st.subheader("Model Info")

    st.json(metadata)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn Pipeline")
