# Loan Approval Prediction System

A classroom machine-learning project using Logistic Regression and a synthetic Excel dataset.

## Important limitation
This project is an educational demonstration only. The included dataset is synthetic and its labels are created using an illustrative formula. The output is not a reliable assessment of creditworthiness and must not be used for real lending decisions.

## 1. Create and activate a virtual environment (optional)

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Generate the Excel dataset

```bash
python generate_dataset.py
```

This creates `loan_prediction_dataset.xlsx`.

## 4. Train and evaluate the model

```bash
python train_model.py
```

This prints accuracy and a classification report, saves `loan_approval_model.joblib`, `model_metrics.json`, and `confusion_matrix.png`.

## 5. Run the Streamlit interface

```bash
streamlit run app.py
```

## Dataset columns

- `ApplicantIncome`: applicant income
- `CoapplicantIncome`: co-applicant income
- `LoanAmount`: requested loan amount
- `Loan_Amount_Term`: loan duration in months
- `Credit_History`: example numeric indicator (0 or 1)
- `Education`: Graduate / Not Graduate
- `Self_Employed`: Yes / No
- `Loan_Status`: Y / N target label

## Workflow

1. Generate/load data.
2. Separate input features (`X`) and target (`y`).
3. Split data into training and test sets.
4. Impute missing values, scale numeric features, and one-hot encode categorical features.
5. Train Logistic Regression.
6. Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.
7. Save the complete preprocessing + model pipeline.
8. Load the model in the Streamlit demo.
