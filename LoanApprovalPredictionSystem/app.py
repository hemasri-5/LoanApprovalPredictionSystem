
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "loan_approval_model.joblib"

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.caption("Educational Machine Learning Demonstration")

# Check model availability
if not Path(MODEL_FILE).exists():
    st.error("Trained model not found. Please run train_model.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

st.subheader("Enter Applicant Information")

with st.form("loan_form"):

    applicant_income = st.number_input(
        "Applicant Income", min_value=0, value=50000, step=1000
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income", min_value=0, value=0, step=1000
    )

    loan_amount = st.number_input(
        "Loan Amount", min_value=1, value=150, step=10
    )

    loan_term = st.selectbox(
        "Loan Term (Months)", [120, 180, 240, 300, 360, 420], index=4
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x: "Positive / Available (1)" if x == 1
        else "Negative / Not Available (0)"
    )

    education = st.selectbox(
        "Education", ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed", ["No", "Yes"]
    )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:

    sample = pd.DataFrame([{
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Education": education,
        "Self_Employed": self_employed
    }])

    # Model prediction
    predicted_class = int(model.predict(sample)[0])

    # Probability calculation
    probabilities = model.predict_proba(sample)[0]
    st.write("Model Classes:", model.classes_)
    st.write("Raw Probabilities:", probabilities)

    for class_value, probability in zip(model.classes_, probabilities):
        st.write(
            f"Class {class_value}: {probability:.10f}"
        )

    approval_index = list(model.classes_).index(1)
    rejection_index = list(model.classes_).index(0)

    approval_probability = float(probabilities[approval_index])
    rejection_probability = float(probabilities[rejection_index])

    st.divider()
    st.subheader("Prediction Results")

    if predicted_class == 1:
        st.success("Predicted Class: Loan Approved (Y)")
    else:
        st.warning("Predicted Class: Loan Rejected (N)")

    # Display both probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approval Probability",
            f"{approval_probability * 100:.4f}%"
        )

    with col2:
        st.metric(
            "Rejection Probability",
            f"{rejection_probability * 100:.4f}%"
        )

    st.write("Approval Probability Distribution")
    st.progress(approval_probability)

    # Debug information
    with st.expander("Model Details"):
        st.write("Model Classes:", model.classes_)
        st.write("Raw probabilities:", probabilities)
        st.write("Input data:", sample)

    st.info(
        "Educational demonstration only. This model uses synthetic data "
        "and must not be used to make real lending decisions."
    )
