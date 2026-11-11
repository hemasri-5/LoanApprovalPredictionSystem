
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------
# 1. Application Configuration
# --------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.caption("Educational Machine Learning Demonstration")


# --------------------------------
# 2. Load Trained Model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "loan_approval_model.joblib"

if not MODEL_FILE.exists():
    st.error("Trained model not found.")
    st.write("Expected model location:", str(MODEL_FILE))
    st.info(
        "Please ensure loan_approval_model.joblib "
        "is present in the same folder as app.py."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# --------------------------------
# 3. Applicant Information Form
# --------------------------------

st.subheader("Enter Applicant Information")
st.write("DEBUG: Loan form section reached")

with st.form("loan_form"):

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=50000,
        step=1000
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0,
        value=0,
        step=1000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=150,
        step=10
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [120, 180, 240, 300, 360, 420],
        index=4
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x:
        "Positive / Available (1)" if x == 1
        else "Negative / Not Available (0)"
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button(
        "Predict Loan Status",
        use_container_width=True
    )


# --------------------------------
# 4. Prediction Process
# --------------------------------

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

    try:
        # Predict loan status
        predicted_class = int(model.predict(sample)[0])

        # Calculate prediction probabilities
        probabilities = model.predict_proba(sample)[0]

        # Identify class positions
        approval_index = list(model.classes_).index(1)
        rejection_index = list(model.classes_).index(0)

        approval_probability = float(
            probabilities[approval_index]
        )

        rejection_probability = float(
            probabilities[rejection_index]
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()


    # --------------------------------
    # 5. Display Prediction Results
    # --------------------------------

    st.divider()
    st.subheader("Prediction Results")

    if predicted_class == 1:
        st.success("Predicted Class: Loan Approved (Y)")
    else:
        st.warning("Predicted Class: Loan Rejected (N)")

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

    # Probability distribution
    st.write("Approval Probability Distribution")

    st.progress(approval_probability)

    # --------------------------------
    # 6. Model Details
    # --------------------------------

    with st.expander("Model Details"):

        st.write("Model Classes:", model.classes_)

        st.write("Raw Probabilities:", probabilities)

        st.write("Input Data:")

        st.dataframe(
            sample,
            use_container_width=True
        )

        st.write("Class-wise Probabilities:")

        for class_value, probability in zip(
            model.classes_, probabilities
        ):
            st.write(
                f"Class {class_value}: {probability:.10f}"
            )


# --------------------------------
# 7. Educational Disclaimer
# --------------------------------

st.divider()

st.info(
    "Educational demonstration only. This model uses synthetic "
    "data and must not be used to make real lending decisions. "
    "Its predictions and probabilities are not real credit assessments."
)
