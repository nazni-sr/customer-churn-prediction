"""Single-customer inference: raw form input -> churn prediction + explanation."""

import json
from pathlib import Path

import pandas as pd
import shap

from src.data_cleaning import clean_data
from src.explainability import load_pipeline, transform_features
from src.feature_engineering import add_engineered_features

FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")

# Categorical base columns that get one-hot encoded during training, i.e. the
# columns that produce "<base>_<category>" feature names like
# "InternetService_Fiber optic". Sorted longest-first so a base column whose
# own name contains an underscore (tenure_group) is matched before a shorter
# prefix could match it incorrectly.
CATEGORICAL_BASE_COLUMNS = sorted(
    [
        "gender",
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
        "tenure_group",
    ],
    key=len,
    reverse=True,
)


def load_feature_columns(path: Path = FEATURE_COLUMNS_PATH) -> list[str]:
    with open(path) as f:
        return json.load(f)


def prepare_input(raw_customer: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Turn a single raw customer record into the exact feature shape the model expects.

    Deliberately does NOT use pd.get_dummies() here: on a single-row
    DataFrame, get_dummies only creates a column for whichever category is
    actually present in that row — e.g. a customer with
    InternetService="Fiber optic" would silently get 0 for
    "InternetService_Fiber optic", because get_dummies never saw any other
    category to compare against. That's a wrong prediction with no error
    raised. Instead, each expected one-hot column is built explicitly by
    comparing the raw value against the category encoded in its name.
    """
    df = pd.DataFrame([raw_customer])
    df["customerID"] = "inference-row"
    df["Churn"] = "No"  # placeholder; clean_data() requires the column but it's discarded below

    df = clean_data(df)
    df = add_engineered_features(df)
    df = df.drop(columns=["Churn"])
    row = df.iloc[0]

    values = {}
    for col in feature_columns:
        if col in row.index:
            values[col] = row[col]
            continue
        base_col = next((b for b in CATEGORICAL_BASE_COLUMNS if col.startswith(b + "_")), None)
        if base_col is None:
            raise ValueError(f"Could not resolve feature column: {col}")
        category = col[len(base_col) + 1 :]
        values[col] = int(str(row[base_col]) == category)

    return pd.DataFrame([values], columns=feature_columns)


def predict_churn(raw_customer: dict) -> dict:
    """Predict churn probability for a single customer record."""
    pipeline = load_pipeline()
    feature_columns = load_feature_columns()
    X = prepare_input(raw_customer, feature_columns)

    probability = pipeline.predict_proba(X)[0, 1]
    prediction = pipeline.predict(X)[0]

    return {
        "churn_probability": float(probability),
        "prediction": "Churn" if prediction == 1 else "No Churn",
    }


def explain_prediction(raw_customer: dict, top_n: int = 8):
    """Explain a single customer's prediction with SHAP, returning the top
    contributing factors (feature, value, SHAP contribution) sorted by
    absolute impact.
    """
    pipeline = load_pipeline()
    feature_columns = load_feature_columns()
    X = prepare_input(raw_customer, feature_columns)

    X_transformed = transform_features(pipeline, X)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    values = shap_values.values[0]
    if values.ndim == 2:
        values = values[:, 1]  # positive (churn) class

    contributions = pd.DataFrame(
        {
            "feature": X_transformed.columns,
            "value": X_transformed.iloc[0].values,
            "shap_value": values,
        }
    )
    contributions["abs_shap"] = contributions["shap_value"].abs()
    return contributions.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
