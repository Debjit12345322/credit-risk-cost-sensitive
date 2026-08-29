"""
Model pipelines for the two baselines.

THE PROBLEM THIS MODULE SOLVES
------------------------------
LightGBM handles NaN natively: at each split it learns a default direction for
missing values. Logistic regression cannot accept NaN at all. So the two models
cannot literally receive identical matrices, and a careless fix -- impute for LR,
native NaN for LightGBM -- means the AUC comparison is measuring preprocessing
rather than algorithms.

THE RESOLUTION: IMPUTE AND INDICATE
-----------------------------------
Both models receive the nine `<col>_is_missing` indicators from
`src.features`. Once the indicator is present, median imputation is no longer
an assertion about the borrower -- the filled value is a placeholder, and the
indicator lets the model correct for it. Logistic regression fits one
coefficient on the value and another on the flag, which lets it express
"use this number, unless it was missing, in which case shift the log-odds by
beta_indicator". That is close to what LightGBM extracts from a native NaN
split, so the remaining difference is the idiomatic treatment each algorithm
needs rather than an information gap.

What is being compared, stated plainly: two PIPELINES, each given the
preprocessing its algorithm requires, both told exactly which values were
missing. Not two algorithms on identical matrices -- that comparison is not
available, because handling missingness IS part of what a GBM does.

WHY THERE IS NO CLASS WEIGHTING OR RESAMPLING
---------------------------------------------
The default rate is 18.45%, and the reflex is `class_weight='balanced'` or
SMOTE. Both are declined deliberately.

Class weighting distorts predicted probabilities away from the true base rate.
A model trained with balanced weights will output something near 0.5 for an
average loan rather than near 0.18. Requirement 5 chooses a decision threshold
by minimising expected cost, and requirement 6 checks whether the predicted
probabilities are calibrated. Both depend on the probabilities meaning what
they say.

Asymmetric costs are a property of the DECISION, not of the training data.
Handling them by reweighting the loss AND by tuning the threshold double-counts
the same asymmetry. The cleaner separation: train on the natural class balance
so probabilities stay calibrated, then move the threshold to reflect the cost
matrix. This also makes the threshold choice auditable -- it is one number with
an explicit cost justification, not an interaction between a sampling scheme
and a cutoff.

COLLINEARITY CAVEAT
-------------------
`grade`, `sub_grade`, and `int_rate` are near-deterministic functions of one
another: LendingClub assigns the sub-grade, which sets the rate. All three are
kept because a lender genuinely observes all three at decision time, but the
consequence is that INDIVIDUAL LOGISTIC REGRESSION COEFFICIENTS ON THESE THREE
ARE NOT INTERPRETABLE. L2 regularisation splits the shared signal among them
arbitrarily. Predictions and AUC are unaffected; coefficient sign and magnitude
for this group are not to be read as effects.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import engineer_features
from src.leakage_audit import get_feature_columns

SEED = 42
TEST_SIZE = 0.25

# Raw columns replaced by parsed/derived equivalents in src.features.
# Keeping both would double-count the same information.
SUPERSEDED_BY_DERIVED = ["term", "emp_length", "earliest_cr_line"]


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply feature engineering and return (X, numeric_cols, categorical_cols).

    The audited KEEP list plus engineered columns, minus the raw columns their
    derived versions replace.
    """
    out, new_cols = engineer_features(df)
    features = [c for c in get_feature_columns(df) if c not in SUPERSEDED_BY_DERIVED]
    features = features + new_cols

    X = out[features]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in features if c not in numeric]
    return X, numeric, categorical


def make_split(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Stratified train/test split.

    Stratified rather than time-ordered: this is a single origination vintage,
    so there is no meaningful time axis to hold out along. Stratification keeps
    the 18.45% default rate identical in both folds, which matters because the
    cost-optimal threshold is sensitive to the base rate.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )


def build_lr_pipeline(numeric: list[str], categorical: list[str], seed: int = SEED) -> Pipeline:
    """Logistic regression with imputation, scaling, and one-hot encoding.

    Every preprocessing step lives INSIDE the Pipeline so that medians, scaling
    parameters, and category vocabularies are fitted on training data only.
    Imputing before splitting would leak test-set distribution into training --
    a quieter version of the same error the leakage audit exists to catch.

    `handle_unknown='ignore'` guards against a category appearing in test but
    not train; it encodes as all-zeros rather than raising.
    """
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", categorical_steps, categorical),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    # L2 by default. Needed here because grade/sub_grade/int_rate
                    # are near-collinear (see module docstring).
                ),
            ),
        ]
    )


def build_lgbm_model(seed: int = SEED) -> lgb.LGBMClassifier:
    """LightGBM using native NaN and native categorical handling.

    No imputation: the model learns a default split direction for missing
    values, which is strictly more expressive than filling with a median.
    Categoricals are passed as pandas `category` dtype and handled natively
    rather than one-hot encoded, which is why LightGBM was chosen over XGBoost
    for this project.

    Hyperparameters are deliberately modest and untuned. This is a BASELINE;
    the project's contribution is the decision layer, not a leaderboard score.
    Tuning would also need a separate validation fold to avoid selecting on the
    test set.
    """
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )


def to_lgbm_frame(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast categorical columns to pandas `category` dtype for LightGBM.

    Categories are taken from the full frame before splitting so that train and
    test share a vocabulary. This is safe: it uses only the SET of category
    labels, never the target, so no outcome information crosses the split.
    """
    out = X.copy()
    for c in categorical:
        out[c] = out[c].astype("category")
    return out
