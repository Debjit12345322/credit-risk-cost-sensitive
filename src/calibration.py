"""
Calibration: are the predicted probabilities actually probabilities?

WHY IT MATTERS HERE
-------------------
AUC measures ranking. Calibration measures whether p = 0.23 really means "23 of
100 such loans default". The two are independent: multiply every prediction by
0.5 and AUC is unchanged while every expected-cost calculation breaks.

This project needs calibration because the decision rule compares p against
G/(L+G), a threshold built from DOLLARS. If p is not on the probability scale,
that comparison is meaningless.

WHAT IS CHECKED
---------------
  * Aggregate reliability: predicted vs observed default rate in quantile bins.
  * Within-segment calibration (src/segments.py): aggregate calibration can
    hide offsetting errors across subgroups, so it is checked per grade and per
    income band as well.
  * Theory vs practice: if p is calibrated, the Bayes threshold G/(L+G) should
    sit close to the threshold found by sweeping realised costs. Small gaps are
    expected because costs are weighted by loan amount and amount correlates
    with risk -- larger loans pull the cost-weighted optimum upward.

No Platt scaling or isotonic regression is applied unless the diagnostics call
for it; fitting a correction to already-calibrated output adds variance for no
gain. Results are in reports/results.md and reports/figures/calibration.png.

Quantile bins rather than equal-width bins: predictions concentrate below 0.4,
so equal-width bins would estimate the upper range from a handful of loans.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_table(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table."""
    bins = pd.qcut(p_pred, q=n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins + 1, "y": y_true, "p": p_pred})
    table = (
        frame.groupby("bin")
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def calibration_summary(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Scalar diagnostics: Brier score, mean prediction vs base rate, bin gaps."""
    table = calibration_table(y_true, p_pred)
    return {
        "brier": float(brier_score_loss(y_true, p_pred)),
        "mean_predicted": float(np.mean(p_pred)),
        "base_rate": float(np.mean(y_true)),
        "mean_abs_bin_gap": float(table["gap"].abs().mean()),
        "max_abs_bin_gap": float(table["gap"].abs().max()),
        "top_bin_gap": float(table["gap"].iloc[-1]),
    }


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, n_bins: int = 10
) -> None:
    """Reliability diagram plus prediction histogram for each model."""
    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), height_ratios=[3, 1], sharex=True
    )
    ax_rel.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

    for name, (y_true, p_pred) in curves.items():
        observed, predicted = calibration_curve(
            y_true, p_pred, n_bins=n_bins, strategy="quantile"
        )
        ax_rel.plot(predicted, observed, "o-", lw=1.5, ms=5, label=name)
        ax_hist.hist(p_pred, bins=50, alpha=0.5, label=name)

    upper = max(max(np.max(p) for _, p in curves.values()), 0.5)
    ax_rel.set(xlim=(0, upper), ylim=(0, upper), ylabel="observed default rate",
               title="Calibration (quantile bins, held-out test set)")
    ax_rel.legend(loc="upper left")
    ax_rel.grid(alpha=0.3)
    ax_hist.set(xlabel="predicted probability of default", ylabel="count (log)", yscale="log")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
