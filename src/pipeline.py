"""
Model pipelines for the two baselines.

WHAT IS BEING ESTIMATED
-----------------------
Both models estimate the class posterior p(x) = P(default = 1 | x). Both are
DISCRIMINATIVE: they model the posterior directly rather than modelling the
feature distribution per class and inverting it (which is what generative
models such as Naive Bayes do). The decision rule built on top of p(x) lives in
src/costs.py.

THE NaN ASYMMETRY, AND HOW IT IS RESOLVED
-----------------------------------------
LightGBM handles NaN natively, learning a default split direction for missing
values. Logistic regression cannot accept NaN at all. A careless fix -- impute
for LR, native NaN for LightGBM -- would make the AUC comparison measure
preprocessing rather than algorithms.

Resolution: IMPUTE AND INDICATE. Both models receive the nine
`<col>_is_missing` flags from src/features.py. Once the flag is present, median
imputation is a placeholder rather than an assertion about the borrower: LR fits
one coefficient on the value and one on the flag, which lets it express "use
this number, unless it was missing, in which case shift the log-odds". That is
close to what LightGBM extracts from a native NaN split.

What is compared, stated plainly: two PIPELINES, each given the preprocessing
its algorithm requires, both told exactly which values were missing.

NO CLASS WEIGHTING, NO RESAMPLING
---------------------------------
`class_weight='balanced'` and SMOTE shift predicted probabilities away from the
true base rate: a balanced model outputs ~0.5 for an average loan rather than
~0.18. The expected-cost threshold and the calibration check both need p(x) to
mean what it says. Asymmetric cost is a property of the DECISION, not of the
training data; handling it by reweighting AND by moving the threshold would
double-count the same asymmetry.

COLLINEARITY CAVEAT
-------------------
`grade`, `sub_grade` and `int_rate` are near-deterministic functions of one
another. All three are kept because an investor observes all three, but the
INDIVIDUAL LOGISTIC COEFFICIENTS ON THEM ARE NOT INTERPRETABLE -- L2
regularisation splits their shared signal arbitrarily. Predictions are
unaffected.

LEAKAGE INSIDE THE PIPELINE
---------------------------
Imputation medians, scaling parameters and one-hot vocabularies are fitted
INSIDE the sklearn Pipeline, i.e. on training rows only. Fitting them on the
full dataset before splitting would leak the test distribution into training.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LGBM_PARAMS, SEED, TEST_SIZE
from src.features import engineer_features
from src.leakage_audit import get_feature_columns

# Raw columns replaced by parsed equivalents in src/features.py.
# Keeping both would feed the same information twice.
SUPERSEDED_BY_DERIVED = ["term", "emp_length", "earliest_cr_line"]


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Audited KEEP columns + engineered columns - superseded raw columns.

    Returns (X, numeric_columns, categorical_columns).
    """
    out, engineered = engineer_features(df)
    features = [c for c in get_feature_columns(df) if c not in SUPERSEDED_BY_DERIVED]
    features += engineered

    X = out[features]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in features if c not in numeric]
    return X, numeric, categorical


def split_indices(y: pd.Series, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split, returned as POSITIONAL INDICES.

    Indices rather than frames, because the cost model needs columns that are
    NOT features (amount funded, realised payments). Splitting by index keeps
    every array -- features, target, costs -- aligned to the same rows.

    Stratified rather than time-ordered: within a single vintage there is no
    meaningful time axis to hold out along. Stratification keeps the default
    rate identical in both halves, which matters because the cost-optimal
    threshold is sensitive to the base rate.
    """
    positions = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        positions, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    return np.sort(train_idx), np.sort(test_idx)


def build_lr_pipeline(numeric: list[str], categorical: list[str], seed: int = SEED) -> Pipeline:
    """Logistic regression: impute, scale, one-hot -- all fitted on training rows."""
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Unseen test categories encode as all-zeros instead of raising.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)]
    )
    return Pipeline(
        [
            ("prep", preprocess),
            # L2 (the default) is needed: grade / sub_grade / int_rate are collinear.
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def build_lgbm_model(seed: int = SEED) -> lgb.LGBMClassifier:
    """LightGBM with native NaN and native categorical handling.

    Hyperparameters are modest and UNTUNED on purpose. Tuning needs its own
    validation split to avoid selecting on the test set, and the project's
    contribution is the decision layer, not a leaderboard score.
    """
    return lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)


def to_lgbm_frame(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast categoricals to pandas `category` dtype for LightGBM.

    Call this on the FULL frame, BEFORE splitting, so train and test share one
    category vocabulary. Safe: it uses only the set of category labels, never
    the target, so no outcome information crosses the split.
    """
    out = X.copy()
    for c in categorical:
        out[c] = out[c].astype("category")
    return out
