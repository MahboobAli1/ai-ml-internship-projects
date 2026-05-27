import streamlit as st
import pandas as pd
import joblib

import os
import joblib

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.joblib")

model = joblib.load(model_path)

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet
}])

if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.success("Churn" if result == 1 else "No Churn")