import numpy as np
import pandas as pd

def create_dataset(rows=1200, output_file="loan_prediction_dataset.xlsx", random_state=42):
    """
    Creates a SYNTHETIC teaching dataset.
    Its labels are generated from an illustrative rule and are not real bank decisions.
    """
    rng = np.random.default_rng(random_state)

    applicant_income = np.clip(rng.lognormal(mean=8.55, sigma=0.55, size=rows), 1500, 150000).round(0)
    coapplicant_income = np.where(
        rng.random(rows) < 0.42,
        0,
        np.clip(rng.lognormal(mean=7.7, sigma=0.75, size=rows), 0, 100000)
    ).round(0)

    loan_amount = np.clip(
        0.018 * (applicant_income + coapplicant_income) + rng.normal(90, 65, rows),
        20, 700
    ).round(0)

    loan_term = rng.choice([120, 180, 240, 300, 360, 420], size=rows,
                           p=[0.04, 0.06, 0.10, 0.10, 0.64, 0.06])
    credit_history = rng.choice([0, 1], size=rows, p=[0.22, 0.78])
    education = rng.choice(["Graduate", "Not Graduate"], size=rows, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=rows, p=[0.16, 0.84])

    # Illustrative label-generation rule for a classroom demonstration only.
    total_income = applicant_income + coapplicant_income
    income_to_loan = total_income / np.maximum(loan_amount, 1)
    score = (
        -1.0
        + 2.5 * credit_history
        + 0.55 * (income_to_loan >= 25)
        + 0.30 * (education == "Graduate")
        - 0.25 * (self_employed == "Yes")
        + 0.20 * (loan_term <= 360)
        + rng.normal(0, 0.9, rows)
    )
    probability = 1 / (1 + np.exp(-score))
    loan_status = np.where(rng.random(rows) < probability, "Y", "N")

    df = pd.DataFrame({
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Education": education,
        "Self_Employed": self_employed,
        "Loan_Status": loan_status
    })

    # Add a small number of missing values to demonstrate preprocessing.
    for column in ["CoapplicantIncome", "LoanAmount", "Credit_History", "Self_Employed"]:
        missing_rows = rng.choice(df.index, size=max(1, int(rows * 0.02)), replace=False)
        df.loc[missing_rows, column] = np.nan

    df.to_excel(output_file, index=False)
    print(f"Created {output_file} with {len(df)} rows.")
    print(df.head())

if __name__ == "__main__":
    create_dataset()
