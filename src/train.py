"""Training, tuning, and evaluation utilities for churn prediction models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from src.model_pipeline import PARAM_GRIDS, build_pipeline, get_candidate_models

PROCESSED_DATA_PATH = Path("data/processed/telco_features.csv")
RANDOM_STATE = 42


def load_features(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split — stratification matters here because the
    target is imbalanced (~26.5% churn); a plain random split risks producing
    a test set with a meaningfully different churn rate than the population.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def cross_validate_candidates(X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """Compare baseline (untuned) models via stratified k-fold cross-validation."""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, estimator in get_candidate_models().items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="roc_auc")
        results.append({"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()})
    return pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def tune_model(name: str, X_train, y_train, cv: int = 5):
    """Hyperparameter tuning via grid search, optimizing for ROC-AUC."""
    estimator = get_candidate_models()[name]
    pipeline = build_pipeline(estimator)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, PARAM_GRIDS[name], cv=cv_splitter, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_on_test(model, X_test, y_test) -> dict:
    """Compute the full evaluation suite required for the project: accuracy,
    precision, recall, F1, ROC-AUC, and the confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def save_model(model, path: Path | str = Path("models/best_model.joblib")) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
