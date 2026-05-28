from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# MUST be the first Streamlit command in the file
st.set_page_config(
    page_title="Telco Churn Prediction System",
    page_icon="🔮",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "churn_pipeline.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

if not PIPELINE_PATH.exists():
    st.error(f"Missing file: {PIPELINE_PATH}")
    st.stop()

if not METADATA_PATH.exists():
    st.error(f"Missing file: {METADATA_PATH}")
    st.stop()

pipeline = joblib.load(PIPELINE_PATH)
metadata = joblib.load(METADATA_PATH)

# now all your st.markdown, st.title, sidebar, etc. can come after this

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained pipeline and metadata"""
    try:
        pipeline = joblib.load('churn_pipeline.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return pipeline, metadata
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure 'churn_pipeline.pkl' and 'model_metadata.pkl' are in the same directory as app.py")
        st.stop()

pipeline, metadata = load_model()

# ==================== PAGE LAYOUT ====================
st.markdown("# 🔮 Customer Churn Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown(f"""
    **Model Performance:**
    - Accuracy: {metadata['accuracy']:.4f}
    - Precision: {metadata['precision']:.4f}
    - Recall: {metadata['recall']:.4f}
    - F1-Score: {metadata['f1']:.4f}
    - ROC-AUC: {metadata['roc_auc']:.4f}
    
    **Best Parameters:**
    """)
    for key, value in metadata['best_params'].items():
        st.write(f"- {key}: {value}")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app uses a **Random Forest Pipeline** trained on the 
    **Telco Customer Churn Dataset** to predict customer churn 
    with high accuracy.
    """)

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📤 Batch Prediction", "📈 Model Details"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.markdown("## Single Customer Prediction")
    st.markdown("Enter customer details below to predict churn probability:")
    
    col1, col2 = st.columns(2)
    
    # Numerical inputs
    with col1:
        st.markdown("### 📊 Numerical Features")
        age = st.slider("Age", min_value=18, max_value=80, value=45)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=10.0)
    
    # Categorical inputs
    with col2:
        st.markdown("### 🏷️ Categorical Features")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", options=["Yes", "No"])
        partner = st.selectbox("Has Partner", options=["Yes", "No"])
        dependents = st.selectbox("Has Dependents", options=["Yes", "No"])
        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", options=["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                      options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Create input dataframe
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Gender': gender,
        'Partner': partner,
        'Dependents': dependents
    }
    
    # Make prediction
    if st.button("🎯 Predict Churn", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_label = "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay"
                st.metric(label="Prediction", value=churn_label, delta=None)
            
            with col2:
                st.metric(label="Churn Probability", value=f"{probability[1]:.2%}")
            
            with col3:
                st.metric(label="Retention Probability", value=f"{probability[0]:.2%}")
            
            # Probability bar chart
            st.markdown("### Confidence Distribution")
            prob_df = pd.DataFrame({
                'Class': ['Will Stay', 'Will Churn'],
                'Probability': [probability[0], probability[1]]
            })
            
            st.bar_chart(prob_df.set_index('Class'))
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            if prediction == 1:
                st.warning("""
                **⚠️ High Churn Risk Detected!**
                
                Recommended Actions:
                - 🎁 Offer special loyalty discounts
                - 📞 Contact customer for satisfaction survey
                - 🚀 Upgrade services or add premium features
                - 💬 Provide dedicated customer support
                """)
            else:
                st.success("""
                **✅ Low Churn Risk - Customer Satisfied**
                
                Recommended Actions:
                - 🎯 Maintain service quality
                - 📊 Monitor for any service issues
                - 🎁 Consider upselling opportunities
                - 🤝 Build long-term relationship
                """)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# ==================== TAB 2: BATCH PREDICTION ====================
with tab2:
    st.markdown("## 📤 Batch Prediction")
    st.markdown("Upload a CSV file to make predictions for multiple customers")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            batch_data = pd.read_csv(uploaded_file)
            
            st.markdown(f"### 📋 Uploaded Data ({len(batch_data)} rows)")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("🔮 Predict Churn for All Customers", use_container_width=True):
                try:
                    # Make predictions
                    predictions = pipeline.predict(batch_data)
                    probabilities = pipeline.predict_proba(batch_data)
                    
                    # Create results dataframe
                    results_df = batch_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Churn_Probability'] = probabilities[:, 1]
                    results_df['Churn_Status'] = results_df['Prediction'].apply(
                        lambda x: 'Will Churn' if x == 1 else 'Will Stay'
                    )
                    
                    # Display results
                    st.markdown("### 🎯 Predictions")
                    st.dataframe(results_df[['Churn_Status', 'Churn_Probability']], use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    total_customers = len(results_df)
                    churn_count = (results_df['Prediction'] == 1).sum()
                    churn_rate = churn_count / total_customers
                    
                    with col1:
                        st.metric("Total Customers", total_customers)
                    with col2:
                        st.metric("Predicted Churn", churn_count, delta_color="inverse")
                    with col3:
                        st.metric("Churn Rate", f"{churn_rate:.2%}")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# ==================== TAB 3: MODEL DETAILS ====================
with tab3:
    st.markdown("## 📈 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Metrics")
        metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [
                metadata['accuracy'],
                metadata['precision'],
                metadata['recall'],
                metadata['f1'],
                metadata['roc_auc']
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("### Pipeline Steps")
        st.write("1. **Preprocessing**: StandardScaler + OneHotEncoder")
        st.write("2. **Model**: Random Forest Classifier")
        st.write("3. **Hyperparameters**: GridSearchCV optimized")
    
    with col2:
        st.markdown("### Feature Information")
        st.write(f"**Total Features**: {len(metadata['feature_names'])}")
        st.write(f"**Numerical Features**: {len(metadata['numerical_cols'])}")
        st.write(f"**Categorical Features**: {len(metadata['categorical_cols'])}")
        
        st.markdown("### Training Configuration")
        st.write("- **Train-Test Split**: 80-20")
        st.write("- **Cross-Validation**: 5-Fold")
        st.write("- **Scoring Metric**: ROC-AUC")
        st.write("- **Class Imbalance**: Stratified Split")
    
    # Detailed metrics
    st.markdown("---")
    st.markdown("### Best Hyperparameters")
    params_df = pd.DataFrame(
        list(metadata['best_params'].items()),
        columns=['Parameter', 'Value']
    )
    st.dataframe(params_df, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <b>Customer Churn Prediction System</b></p>
    <p>Built with Scikit-learn Pipeline API | Deployed on Streamlit</p>
</div>
""", unsafe_allow_html=True)
