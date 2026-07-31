"""Data loading and cleaning utilities for the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn data.

    Handles the known data quality issues in this dataset:
    - `TotalCharges` is read as text because new customers (tenure == 0)
      have a blank string instead of a numeric value. Those customers have
      paid nothing yet, so blanks are filled with 0 rather than dropped,
      to avoid losing valid records.
    - `customerID` is a unique identifier, not a predictive feature.
    - `Churn` is recoded from Yes/No to 1/0 for modeling.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def load_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load raw data and return it cleaned."""
    return clean_data(load_raw_data(path))


if __name__ == "__main__":
    data = load_clean_data()
    print(f"Cleaned data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum().sum()} total")
    print(f"Churn rate: {data['Churn'].mean():.3f}")
