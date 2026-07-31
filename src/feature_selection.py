"""Feature selection utilities for the Telco Customer Churn dataset."""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def compute_mutual_information(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """Rank features by mutual information with the target."""
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


def find_highly_correlated_pairs(X: pd.DataFrame, threshold: float = 0.9) -> list[tuple[str, str, float]]:
    """Find feature pairs with absolute correlation above `threshold`.

    Redundant features (near-duplicate information) can hurt model
    interpretability and inflate variance in linear models, even if they
    don't hurt tree-based model accuracy much.
    """
    corr = X.corr().abs()
    cols = corr.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if value > threshold:
                pairs.append((cols[i], cols[j], round(float(value), 3)))
    return sorted(pairs, key=lambda p: p[2], reverse=True)
