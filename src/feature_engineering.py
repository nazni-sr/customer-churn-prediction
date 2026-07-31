"""Feature engineering for the Telco Customer Churn dataset."""

import numpy as np
import pandas as pd

ADDITIONAL_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the cleaned Telco churn data.

    Expects `df` to already be cleaned (see `data_cleaning.clean_data`):
    `TotalCharges` numeric, `customerID` dropped, `Churn` mapped to 0/1.
    """
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )

    # Customers with tenure == 0 have TotalCharges == 0 (see data_cleaning),
    # so dividing would understate their spend; use MonthlyCharges instead.
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    df["num_additional_services"] = (df[ADDITIONAL_SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["has_multiple_services"] = (df["num_additional_services"] >= 3).astype(int)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns (drop_first to avoid multicollinearity)."""
    df = df.copy()
    categorical_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def build_feature_set(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: engineer features, then encode."""
    df = add_engineered_features(df_clean)
    return encode_categorical_features(df)


if __name__ == "__main__":
    from src.data_cleaning import load_clean_data

    clean = load_clean_data()
    featured = build_feature_set(clean)
    print(f"Cleaned shape: {clean.shape}")
    print(f"Feature-engineered + encoded shape: {featured.shape}")
