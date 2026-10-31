"""
Segment-level error analysis, and honest threshold fitting.

TWO QUESTIONS
-------------
1. Where does the model discriminate or calibrate badly? (per-segment AUC,
   mean predicted vs observed default rate)
2. Where does the DECISION lose money, and would a segment-specific threshold
   recover any of it?

The second question has an action attached, and a trap.

THE TRAP: CHOOSING THRESHOLDS ON THE DATA THAT SCORES THEM
----------------------------------------------------------
Fitting a separate threshold per segment on the test set, then reporting the
test-set gain, selects on the evaluation data: every segment's threshold lands
where sampling noise happened to put the bottom of its cost curve. The apparent
gain is optimistic, and more so the more segments there are.

Honest procedure, implemented here:
  1. Produce OUT-OF-FOLD predictions for every training row (each row is scored
     by a model that never saw it).
  2. Fit thresholds on those OOF predictions.
  3. Apply the fitted thresholds, unchanged, to the held-out test set.

Why OOF and not plain in-sample training predictions: a model is overconfident
on rows it was trained on, so in-sample predictions have a different
distribution from what the model emits on new data. A threshold tuned to that
distribution transfers badly. reports/results.md quantifies all three versions
(in-sample, OOF, test-fitted oracle).

Segments smaller than MIN_SEGMENT_SIZE in training fall back to the global
threshold instead of fitting a cutoff to a few hundred loans.

WHY A SEGMENT THRESHOLD SHOULD SURVIVE ONLY FOR AN ECONOMIC REASON
-----------------------------------------------------------------
Under a calibrated model, the Bayes threshold G/(L+G) depends only on costs.
So a segment deserves its own threshold when its COSTS differ, not merely when
its default rate differs (a calibrated p already carries the default rate).
Term qualifies: 60-month loans earn far more when repaid. Grade and purpose do
not have their own cost rates here, so any apparent gain from them should be
expected to vanish out-of-fold -- and the results check whether it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import MIN_SEGMENT_SIZE, N_CV_FOLDS
from src.costs import optimal_threshold, sweep_thresholds, total_cost_of_decisions


def out_of_fold_predictions(model_factory, X: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """P(default) for every training row from a model that never saw that row.

    `model_factory` is a zero-argument callable returning a FRESH unfitted
    model, so no fold continues training a model fitted on another fold.
    """
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, val_idx in folds.split(X, y):
        model = model_factory()
        model.fit(X.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def segment_report(
    labels: pd.Series | np.ndarray,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    decline: np.ndarray,
) -> pd.DataFrame:
    """Per-segment discrimination, calibration and realised cost."""
    y_true = np.asarray(y_true)
    decline = np.asarray(decline, dtype=bool)
    regret = np.where(
        (~decline) & (y_true == 1), cost_fn, np.where(decline & (y_true == 0), cost_fp, 0.0)
    )
    frame = pd.DataFrame(
        {"segment": np.asarray(labels), "y": y_true, "p": p_pred,
         "decline": decline, "regret": regret}
    )
    rows = []
    for name, g in frame.groupby("segment", observed=True):
        rows.append(
            {
                "segment": name,
                "n": len(g),
                "default_rate": g["y"].mean(),
                "mean_predicted": g["p"].mean(),
                # AUC is undefined when a segment contains only one class.
                "auc": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() > 1 else np.nan,
                "decline_rate": g["decline"].mean(),
                "cost_per_application": g["regret"].mean(),
            }
        )
    return pd.DataFrame(rows)


def fit_segment_thresholds(
    labels: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    fallback: float,
    min_size: int = MIN_SEGMENT_SIZE,
) -> dict:
    """One threshold per segment, fitted on the data passed in."""
    labels = np.asarray(labels)
    y = np.asarray(y)
    thresholds = {}
    for value in pd.unique(labels):
        mask = labels == value
        if mask.sum() < min_size:
            thresholds[value] = fallback
            continue
        sweep = sweep_thresholds(y[mask], p[mask], cost_fn[mask], cost_fp[mask])
        thresholds[value], _ = optimal_threshold(sweep)
    return thresholds


def apply_segment_thresholds(labels: np.ndarray, p: np.ndarray, thresholds: dict, fallback: float) -> np.ndarray:
    """Decline mask using each loan's segment threshold (fallback if unseen)."""
    cutoffs = np.array([thresholds.get(v, fallback) for v in np.asarray(labels)])
    return p >= cutoffs


def segment_threshold_cost(
    labels_fit: np.ndarray, y_fit: np.ndarray, p_fit: np.ndarray,
    fn_fit: np.ndarray, fp_fit: np.ndarray,
    labels_eval: np.ndarray, y_eval: np.ndarray, p_eval: np.ndarray,
    fn_eval: np.ndarray, fp_eval: np.ndarray,
    fallback: float,
) -> tuple[float, dict, float]:
    """Fit per-segment thresholds on one set, score on another.

    Returns (cost per application on the evaluation set, thresholds, decline rate).
    Passing the SAME arrays as fit and eval reproduces the optimistic
    test-fitted number; passing OOF-train as fit and test as eval is honest.
    """
    thresholds = fit_segment_thresholds(labels_fit, y_fit, p_fit, fn_fit, fp_fit, fallback)
    decline = apply_segment_thresholds(labels_eval, p_eval, thresholds, fallback)
    cost = total_cost_of_decisions(y_eval, decline, fn_eval, fp_eval) / len(y_eval)
    return cost, thresholds, float(decline.mean())
