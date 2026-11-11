
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# File names
DATA_FILE = "loan_prediction_dataset.xlsx"
MODEL_FILE = "loan_approval_model.joblib"
METRICS_FILE = "model_metrics.json"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"

# Input features
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

CATEGORICAL_FEATURES = [
    "Education",
    "Self_Employed"
]

TARGET = "Loan_Status"


def main():

    # 1. Check whether dataset exists
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run generate_dataset.py first."
        )

    # 2. Load dataset
    df = pd.read_excel(DATA_FILE)

    required_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    )

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    df = df[required_columns].copy()

    # 3. Convert target labels into numbers
    df[TARGET] = df[TARGET].map({
        "Y": 1,
        "N": 0
    })

    # Remove rows with missing target values
    df = df.dropna(subset=[TARGET])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    # 4. Check dataset and target classes
    print("\nDataset shape:", df.shape)

    print("\nLoan status distribution:")
    print(y.value_counts().rename({
        0: "Rejected (N)",
        1: "Approved (Y)"
    }))

    if y.nunique() != 2:
        raise ValueError(
            "Dataset must contain both approved (Y) "
            "and rejected (N) loan records."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class must have at least two records."
        )

    # 5. Preprocessing for numerical features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 6. Preprocessing for categorical features
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 7. Combine preprocessing
    preprocessing = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    # 8. Create machine learning model
    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 9. Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # 10. Train the model
    model.fit(X_train, y_train)

    # 11. Make predictions
    predictions = model.predict(X_test)

    # 12. Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("\nTest accuracy: {:.4f}".format(accuracy))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Rejected (N)", "Approved (Y)"],
        zero_division=0
    ))

    # 13. Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Rejected", "Approved"]
    )

    display.plot(values_format="d")
    plt.title("Loan Status - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()

    # 14. Check prediction probabilities
    probabilities = model.predict_proba(X_test)

    approval_index = list(model.classes_).index(1)
    rejection_index = list(model.classes_).index(0)

    print("\nModel classes:", model.classes_)
    print(
        "First test record approval probability: {:.6f}".format(
            probabilities[0][approval_index]
        )
    )
    print(
        "First test record rejection probability: {:.6f}".format(
            probabilities[0][rejection_index]
        )
    )

    # 15. Save trained model
    joblib.dump(model, MODEL_FILE)

    # 16. Save evaluation metrics
    metrics = {
        "test_accuracy": float(accuracy),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "classes": [int(value) for value in model.classes_],
        "class_distribution": {
            "Rejected (N)": int((y == 0).sum()),
            "Approved (Y)": int((y == 1).sum())
        }
    }

    Path(METRICS_FILE).write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print("\nTraining completed successfully!")
    print("Saved model:", MODEL_FILE)
    print("Saved metrics:", METRICS_FILE)
    print("Saved confusion matrix:", CONFUSION_MATRIX_FILE)


if __name__ == "__main__":
    main()
