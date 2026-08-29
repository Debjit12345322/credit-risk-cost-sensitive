"""
Segment-level error analysis.

TWO QUESTIONS, NOT ONE
----------------------
1. Where does the model discriminate well or badly? (segment AUC, calibration)
2. Where does the DECISION lose money, and would a segment-specific threshold
   recover any of it?

The second question is the one with an action attached, and it comes with a
methodological trap described below.

FINDING 1: MOST OF THE AUC IS THE GRADE
---------------------------------------
    grade    n        default rate   AUC
    A        9,077     5.1%          0.6364
    B       14,958    12.0%          0.6310
    C       15,605    19.5%          0.6307
    D        9,850    27.4%          0.6375
    E        4,468    34.9%          0.6309
    F        1,451    37.8%          0.6455
    G          367    47.7%          0.5568   (small sample)

Overall AUC is 0.7198, but WITHIN any grade it drops to roughly 0.63. Most of
the model's ranking power comes from re-deriving LendingClub's own grade
assignment; conditional on grade, the remaining 70 features add modestly.

This is the quantified version of the circularity caveat noted in
`src/leakage_audit.py` when `grade`/`sub_grade`/`int_rate` were kept. It is not
a defect -- an investor genuinely observes the grade -- but it means the honest
description of this model is "a funding decision layered on top of LendingClub's
underwriting", not "a model that beats LendingClub's underwriting".

FINDING 2: CALIBRATION SURVIVES SEGMENTATION
--------------------------------------------
Mean predicted probability tracks the realised default rate inside every
segment: grade A 0.0538 vs 0.0514 actual, grade G 0.4752 vs 0.4768, income
quartile 1 0.2163 vs 0.2168. Aggregate calibration can hide offsetting errors
across subgroups; this does not.

FINDING 3: DOLLAR COST RISES WITH INCOME DESPITE FALLING RISK
-------------------------------------------------------------
    income quartile   default rate   cost per application
    Q1 (low)          21.7%          $715
    Q2                20.1%          $1,000
    Q3                17.0%          $1,061
    Q4 (high)         14.7%          $1,150

Higher earners default less but borrow more, and both error costs scale with
amount funded. The dollar metric therefore mixes risk with exposure. Worth
stating explicitly so the table is not misread as "high earners are riskier".

FINDING 4: SEGMENT THRESHOLDS MOSTLY DO NOT SURVIVE HONEST EVALUATION
---------------------------------------------------------------------
Fitting a separate threshold per segment ON THE TEST SET suggests large gains:

    segment    apparent gain (test-fitted)
    term       +$7.24 / application
    grade      +$5.14 / application
    purpose    +$6.41 / application

Fitting the same thresholds on OUT-OF-FOLD training predictions and evaluating
on the held-out test set gives a very different answer:

    segment    honest gain
    term       +$6.60 / application
    grade      +$0.14 / application
    purpose    -$2.02 / application   (actively harmful)

Only the term split survives, and for a reason that is economic rather than
statistical: 60-month loans return 29.6% when repaid against 15.9% for 36-month
loans, with near-identical downside, so their break-even genuinely differs.
Grade and purpose thresholds differed only in where sampling noise put the
bottom of the cost curve.

WHY OUT-OF-FOLD PREDICTIONS ARE REQUIRED
----------------------------------------
A threshold must be chosen against the probability distribution the model
produces on UNSEEN data. In-sample training predictions are overconfident, so a
threshold fitted to them is tuned to a distribution the model will never emit
again. Measured here: fitting on in-sample predictions gave 0.380 and cost
$1,006.79 per application on test, WORSE than the naive 0.5 threshold
($981.06). Fitting on out-of-fold predictions gave 0.470 and $979.15 -- within
$0.32 of the unattainable test-fitted oracle ($978.83 at 0.460).

This is the same discipline as the leakage audit, applied one level up: there it
kept outcome information out of features, here it keeps it out of a
hyperparameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.costs import optimal_threshold, sweep_thresholds, total_cost_at_threshold


def out_of_fold_predictions(
    model_factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """Cross-validated predictions for every training row.

    `model_factory` is a zero-argument callable returning an unfitted model, so
    each fold trains a fresh instance rather than continuing a fitted one.
    """
    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(X, y):
        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def segment_report(
    labels: pd.Series,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Per-segment performance and realised cost at a single global threshold."""
    decline = p_pred >= threshold
    realised = np.where(
        (~decline) & (y_true == 1),
        cost_fn,
        np.where(decline & (y_true == 0), cost_fp, 0.0),
    )

    frame = pd.DataFrame(
        {
            "segment": labels.to_numpy(),
            "y": y_true,
            "p": p_pred,
            "cost": realised,
            "decline": decline,
        }
    )

    rows = []
    for name, group in frame.groupby("segment", observed=True):
        rows.append(
            {
                "segment": name,
                "n": len(group),
                "default_rate": group["y"].mean(),
                "mean_predicted": group["p"].mean(),
                # AUC is undefined in a segment with only one outcome class.
                "auc": (
                    roc_auc_score(group["y"], group["p"])
                    if group["y"].nunique() > 1
                    else np.nan
                ),
                "decline_rate": group["decline"].mean(),
                "cost_per_application": group["cost"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def evaluate_segment_thresholds(
    train_labels: pd.Series,
    test_labels: pd.Series,
    y_train: np.ndarray,
    oof_train: np.ndarray,
    cost_fn_train: np.ndarray,
    cost_fp_train: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    cost_fn_test: np.ndarray,
    cost_fp_test: np.ndarray,
    global_threshold: float,
    min_segment_size: int = 500,
) -> tuple[float, dict]:
    """Fit per-segment thresholds on OOF training data, score on test.

    Returns (test_cost_per_application, {segment: threshold}).

    Segments smaller than `min_segment_size` in training fall back to the global
    threshold rather than fitting a threshold to a few hundred loans.
    """
    thresholds: dict = {}
    total = 0.0

    for value in pd.unique(train_labels):
        train_mask = (train_labels == value).to_numpy()
        test_mask = (test_labels == value).to_numpy()
        if test_mask.sum() == 0:
            continue

        if train_mask.sum() < min_segment_size:
            chosen = global_threshold
            thresholds[value] = "global (segment too small)"
        else:
            sweep = sweep_thresholds(
                y_train[train_mask],
                oof_train[train_mask],
                cost_fn_train[train_mask],
                cost_fp_train[train_mask],
            )
            chosen, _ = optimal_threshold(sweep)
            thresholds[value] = round(chosen, 3)

        total += total_cost_at_threshold(
            y_test[test_mask],
            p_test[test_mask],
            chosen,
            cost_fn_test[test_mask],
            cost_fp_test[test_mask],
        )

    return total / len(y_test), thresholds