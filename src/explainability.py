"""SHAP-based explainability for the churn prediction model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def load_pipeline(path: Path = Path("models/best_model.joblib")) -> Pipeline:
    return joblib.load(path)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's preprocessing step and return a labeled DataFrame.

    SHAP needs feature names attached to produce readable plots, but
    ColumnTransformer.transform() returns a plain array, so we reattach
    names via get_feature_names_out().
    """
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    # ColumnTransformer prefixes names with the transformer name (e.g.
    # "scale__tenure", "remainder__Contract_Two year"); strip that for
    # readable SHAP plots and reports.
    feature_names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def compute_shap_values(pipeline: Pipeline, X_transformed: pd.DataFrame):
    """Compute SHAP values for the fitted tree-based model inside the pipeline.

    TreeExplainer is used (rather than the model-agnostic KernelExplainer)
    because it's exact and fast for tree ensembles like Random Forest.
    """
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    return explainer(X_transformed)


def top_features_by_shap(shap_values, n: int = 10) -> pd.Series:
    """Rank features by mean absolute SHAP value (global importance)."""
    values = shap_values.values
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) — take the positive (churn) class
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=shap_values.feature_names).sort_values(ascending=False).head(n)
