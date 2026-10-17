"""Pipeline construction module for feature preprocessing and model estimation."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES: list[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

CATEGORICAL_FEATURES: list[str] = [
    "Month",
    "VisitorType",
    "Weekend",
]


def build_numerical_transformer() -> Pipeline:
    """Construct preprocessing pipeline for numerical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_transformer() -> Pipeline:
    """Construct preprocessing pipeline for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    """Combine numerical and categorical transformers."""
    return ColumnTransformer(
        transformers=[
            ("num", build_numerical_transformer(), NUMERICAL_FEATURES),
            ("cat", build_categorical_transformer(), CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline(
    seed: int = 42,
    n_estimators: int = 150,
    max_depth: int = 8,
    min_samples_split: int = 5,
) -> Pipeline:
    """Build full end-to-end Pipeline combining preprocessor and classifier.

    Args:
        seed: Random state seed.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        min_samples_split: Minimum samples required to split a node.

    Returns:
        Configured scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
