"""Streamlit app: enter a customer profile, get a churn risk score and explanation."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import explain_prediction, predict_churn

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("Customer Churn Predictor")
st.write(
    "Enter a customer's account details to predict their churn risk and see which "
    "factors are driving that prediction. Model: tuned Random Forest, ROC-AUC 0.846 "
    "on held-out test data — see the [GitHub repo](https://github.com/nazni-sr/customer-churn-prediction) "
    "for the full analysis."
)

with st.form("customer_form"):
    st.subheader("Account")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    st.subheader("Services")
    col3, col4 = st.columns(2)
    with col3:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col4:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("Demographics")
    col5, col6 = st.columns(2)
    with col5:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col6:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn Risk", type="primary")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
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
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    result = predict_churn(customer)
    probability = result["churn_probability"]

    st.divider()
    st.subheader("Prediction")

    risk_label = "High Risk" if probability >= 0.5 else "Low Risk"
    risk_color = "red" if probability >= 0.5 else "green"
    st.metric("Churn Probability", f"{probability:.1%}")
    st.markdown(f"**Risk level:** :{risk_color}[{risk_label}]")
    st.progress(min(probability, 1.0))

    st.subheader("Top Factors Driving This Prediction")
    contributions = explain_prediction(customer)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in contributions["shap_value"]]
    ax.barh(contributions["feature"], contributions["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on churn probability (SHAP value)")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward churn; blue bars push away from it. "
        "Bar length shows the size of the effect for this specific customer."
    )
