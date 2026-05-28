from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==================== PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ====================
st.set_page_config(
    page_title="Telco Churn Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "churn_pipeline.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

# ==================== TRAINING SCHEMA ====================
FEATURE_COLUMNS = [
    "SeniorCitizen",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

GENDER_OPTIONS = ["Male", "Female"]
YES_NO_OPTIONS = ["Yes", "No"]
MULTIPLE_LINES_OPTIONS = ["Yes", "No", "No phone service"]
INTERNET_OPTIONS = ["Fiber optic", "DSL", "No"]
SERVICE_YN_OPTIONS = ["Yes", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAPERLESS_OPTIONS = ["Yes", "No"]
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# ==================== HELPERS ====================
def ensure_files_exist() -> None:
    if not PIPELINE_PATH.exists():
        st.error(f"❌ Missing file: {PIPELINE_PATH.name}")
        st.stop()

    if not METADATA_PATH.exists():
        st.error(f"❌ Missing file: {METADATA_PATH.name}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


def normalize_telco_row(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure dependent fields stay valid so the encoder sees only legal Telco combinations."""
    df = df.copy()

    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        df.loc[df["PhoneService"] == "No", "MultipleLines"] = "No phone service"

    internet_dependent_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    if "InternetService" in df.columns:
        mask = df["InternetService"] == "No"
        for col in internet_dependent_cols:
            if col in df.columns:
                df.loc[mask, col] = "No internet service"

    return df


def build_input_df(
    senior_citizen,
    tenure,
    monthly_charges,
    total_charges,
    gender,
    partner,
    dependents,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
):
    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }
    df = pd.DataFrame([input_data])
    return normalize_telco_row(df)[FEATURE_COLUMNS]


def clean_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Strip text values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Rename common variants to the exact training names
    rename_map = {
        "seniorcitizen": "SeniorCitizen",
        "tenure": "Tenure",
        "monthlycharges": "MonthlyCharges",
        "totalcharges": "TotalCharges",
        "gender": "Gender",
        "partner": "Partner",
        "dependents": "Dependents",
        "phoneservice": "PhoneService",
        "multiplelines": "MultipleLines",
        "internetservice": "InternetService",
        "onlinesecurity": "OnlineSecurity",
        "onlinebackup": "OnlineBackup",
        "deviceprotection": "DeviceProtection",
        "techsupport": "TechSupport",
        "streamingtv": "StreamingTV",
        "streamingmovies": "StreamingMovies",
        "contract": "Contract",
        "paperlessbilling": "PaperlessBilling",
        "paymentmethod": "PaymentMethod",
    }

    normalized = {}
    for col in df.columns:
        key = col.lower().replace(" ", "").replace("_", "")
        if key in rename_map:
            normalized[col] = rename_map[key]

    if normalized:
        df = df.rename(columns=normalized)

    return normalize_telco_row(df)


def predict_customer(input_df: pd.DataFrame):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]
    return int(prediction), probability


# ==================== VALIDATION + LOAD ====================
ensure_files_exist()
pipeline, metadata = load_model()

# ==================== STYLING ====================
st.markdown(
    """
    <style>
        .main { padding: 0rem 0rem; }
        .hero {
            padding: 1.2rem 1.2rem 0.2rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            border: 1px solid #e8eef7;
            margin-bottom: 1rem;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e9eef5;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        }
        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== HEADER ====================
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom: 0.2rem;">🔮 Telco Churn Prediction System</h1>
        <p style="margin-top: 0; color: #475569;">
            Production-ready ML app using Scikit-learn Pipeline API + Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 Model Snapshot")
    st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
    st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
    st.metric("F1-Score", f"{metadata['f1']:.4f}")

    st.markdown("### Best Parameters")
    for key, value in metadata["best_params"].items():
        st.write(f"- {key}: {value}")

    st.markdown("### Input Tips")
    st.info(
        "Use the exact Telco values shown in the dropdowns. "
        "Dependent service fields are auto-corrected when InternetService or PhoneService is set to No."
    )

    st.markdown("### Training Schema")
    st.write(f"- Numerical features: {len(metadata['numerical_cols'])}")
    st.write(f"- Categorical features: {len(metadata['categorical_cols'])}")
    st.write(f"- Total features: {len(metadata['feature_names'])}")

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📤 Batch Prediction", "📈 Model Details"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.markdown("## Single Customer Prediction")
    st.markdown(
        "Enter customer details below. The model will predict churn probability "
        "using the saved pipeline."
    )

    with st.form("single_customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Numerical Features")
            senior_citizen = st.selectbox("Senior Citizen", YES_NO_OPTIONS, index=1)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, value=65.0, step=0.5
            )
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=1000.0, step=10.0
            )
            st.caption("Age is not used by the trained Telco model, so it is intentionally omitted.")

        with col2:
            st.markdown("### 🏷️ Categorical Features")
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            partner = st.selectbox("Has Partner", YES_NO_OPTIONS)
            dependents = st.selectbox("Has Dependents", YES_NO_OPTIONS)
            phone_service = st.selectbox("Phone Service", YES_NO_OPTIONS)

            if phone_service == "No":
                multiple_lines = "No phone service"
                st.info("Multiple Lines is locked to 'No phone service' because Phone Service is 'No'.")
            else:
                multiple_lines = st.selectbox("Multiple Lines", MULTIPLE_LINES_OPTIONS)

            internet_service = st.selectbox("Internet Service", INTERNET_OPTIONS)

            if internet_service == "No":
                st.info("Internet-related features are locked to 'No internet service' because Internet Service is 'No'.")
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
            else:
                online_security = st.selectbox("Online Security", SERVICE_YN_OPTIONS)
                online_backup = st.selectbox("Online Backup", SERVICE_YN_OPTIONS)
                device_protection = st.selectbox("Device Protection", SERVICE_YN_OPTIONS)
                tech_support = st.selectbox("Tech Support", SERVICE_YN_OPTIONS)
                streaming_tv = st.selectbox("Streaming TV", SERVICE_YN_OPTIONS)
                streaming_movies = st.selectbox("Streaming Movies", SERVICE_YN_OPTIONS)

            contract = st.selectbox("Contract", CONTRACT_OPTIONS)
            paperless_billing = st.selectbox("Paperless Billing", PAPERLESS_OPTIONS)
            payment_method = st.selectbox("Payment Method", PAYMENT_OPTIONS)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        try:
            input_df = build_input_df(
                senior_citizen=senior_citizen,
                tenure=tenure,
                monthly_charges=monthly_charges,
                total_charges=total_charges,
                gender=gender,
                partner=partner,
                dependents=dependents,
                phone_service=phone_service,
                multiple_lines=multiple_lines,
                internet_service=internet_service,
                online_security=online_security,
                online_backup=online_backup,
                device_protection=device_protection,
                tech_support=tech_support,
                streaming_tv=streaming_tv,
                streaming_movies=streaming_movies,
                contract=contract,
                paperless_billing=paperless_billing,
                payment_method=payment_method,
            )

            prediction, probability = predict_customer(input_df)

            st.markdown("---")
            st.markdown("## Prediction Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Prediction",
                    "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay",
                )
            with c2:
                st.metric("Churn Probability", f"{probability[1]:.2%}")
            with c3:
                st.metric("Retention Probability", f"{probability[0]:.2%}")

            st.progress(float(probability[1]))

            result_df = pd.DataFrame(
                {
                    "Class": ["Will Stay", "Will Churn"],
                    "Probability": [probability[0], probability[1]],
                }
            )
            st.bar_chart(result_df.set_index("Class"))

            st.markdown("### Recommended Action")
            if prediction == 1:
                st.warning(
                    """
                    **High churn risk detected.**
                    - Offer a retention discount
                    - Contact the customer proactively
                    - Review service quality and billing experience
                    - Consider a loyalty incentive
                    """
                )
            else:
                st.success(
                    """
                    **Low churn risk.**
                    - Maintain service quality
                    - Continue engagement
                    - Consider upsell opportunities
                    - Track satisfaction over time
                    """
                )

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# ==================== TAB 2: BATCH PREDICTION ====================
with tab2:
    st.markdown("## Batch Prediction")
    st.markdown(
        "Upload a CSV file with the same customer feature columns used in training. "
        "The app will standardize column names and fix dependent-service combinations."
    )

    template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="telco_churn_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            batch_data = clean_uploaded_df(batch_data)

            missing = [c for c in FEATURE_COLUMNS if c not in batch_data.columns]
            if missing:
                st.error(
                    "❌ Your CSV is missing required columns:\n\n"
                    + "\n".join([f"- {c}" for c in missing])
                )
                st.stop()

            batch_input = batch_data[FEATURE_COLUMNS].copy()

            st.markdown(f"### Uploaded Data ({len(batch_input)} rows)")
            st.dataframe(batch_input.head(10), use_container_width=True)

            if st.button("🔮 Predict Churn for All Customers", use_container_width=True):
                try:
                    predictions = pipeline.predict(batch_input)
                    probabilities = pipeline.predict_proba(batch_input)

                    results_df = batch_input.copy()
                    results_df["Prediction"] = predictions
                    results_df["Churn_Probability"] = probabilities[:, 1]
                    results_df["Churn_Status"] = results_df["Prediction"].apply(
                        lambda x: "Will Churn" if x == 1 else "Will Stay"
                    )

                    st.markdown("### Prediction Summary")
                    c1, c2, c3 = st.columns(3)
                    total_customers = len(results_df)
                    churn_count = int((results_df["Prediction"] == 1).sum())
                    churn_rate = churn_count / total_customers if total_customers else 0

                    with c1:
                        st.metric("Total Customers", total_customers)
                    with c2:
                        st.metric("Predicted Churn", churn_count)
                    with c3:
                        st.metric("Churn Rate", f"{churn_rate:.2%}")

                    st.dataframe(
                        results_df[["Churn_Status", "Churn_Probability"]],
                        use_container_width=True,
                    )

                    st.download_button(
                        label="📥 Download Predictions",
                        data=results_df.to_csv(index=False),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"❌ Error during batch prediction: {e}")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ==================== TAB 3: MODEL DETAILS ====================
with tab3:
    st.markdown("## Model Details")

    left, right = st.columns(2)

    with left:
        st.markdown("### Performance Metrics")
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                "Score": [
                    metadata["accuracy"],
                    metadata["precision"],
                    metadata["recall"],
                    metadata["f1"],
                    metadata["roc_auc"],
                ],
            }
        )
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown("### Pipeline Summary")
        st.write("1. Preprocessing: StandardScaler + OneHotEncoder")
        st.write("2. Model: Random Forest Classifier")
        st.write("3. Optimization: GridSearchCV with 5-fold CV")

    with right:
        st.markdown("### Feature Information")
        st.write(f"**Total Features:** {len(metadata['feature_names'])}")
        st.write(f"**Numerical Features:** {len(metadata['numerical_cols'])}")
        st.write(f"**Categorical Features:** {len(metadata['categorical_cols'])}")

        st.markdown("### Training Configuration")
        st.write("- Train/Test Split: 80/20")
        st.write("- Cross-Validation: 5-Fold")
        st.write("- Scoring Metric: ROC-AUC")
        st.write("- Stratified split for class balance")

    st.markdown("---")
    st.markdown("### Best Hyperparameters")
    params_df = pd.DataFrame(
        list(metadata["best_params"].items()),
        columns=["Parameter", "Value"],
    )
    st.dataframe(params_df, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn Pipeline API")
