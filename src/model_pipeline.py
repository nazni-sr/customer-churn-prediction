"""Model pipeline construction: preprocessing and candidate models."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Only continuous features need scaling; one-hot/binary columns are already
# on a 0/1 scale and are left untouched via ColumnTransformer's "remainder".
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "num_additional_services"]


def build_preprocessor(numeric_features: list[str] = NUMERIC_FEATURES) -> ColumnTransformer:
    """Scale continuous features; pass binary/one-hot features through unchanged."""
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)],
        remainder="passthrough",
    )


def get_candidate_models(random_state: int = 42) -> dict:
    """Baseline models to compare, each with class balancing for the ~26.5% churn rate."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        ),
    }


def build_pipeline(estimator, numeric_features: list[str] = NUMERIC_FEATURES) -> Pipeline:
    """Wrap an estimator with the shared preprocessing step."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features)),
            ("model", estimator),
        ]
    )


PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 5, 10],
        "model__max_iter": [100, 200],
    },
}
