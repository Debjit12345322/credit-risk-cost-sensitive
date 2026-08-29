"""
Calibration assessment: are the predicted probabilities actually probabilities?

WHY THIS MATTERS HERE SPECIFICALLY
----------------------------------
Ranking quality (AUC) and calibration are different properties. A model can
order loans perfectly and still be systematically wrong about their absolute
risk -- multiply every probability by 0.5 and AUC is unchanged while every
expected-cost calculation breaks.

This project depends on calibration in a way that a pure ranking exercise does
not. The decision rule "fund when p < G/(L+G)" compares a predicted probability
against a threshold derived from dollar amounts. If p is not on the probability
scale, that comparison is meaningless and the cost-optimal threshold is
optimising against a distorted input.

RESULT ON THE 2014 VINTAGE
--------------------------
Both baselines are well calibrated without any post-hoc correction:

    model       AUC      Brier    mean predicted    actual base rate
    LogReg      0.7173   0.1364   0.1838            0.1845
    LightGBM    0.7198   0.1361   0.1831            0.1845

Across ten quantile bins the largest deviation is ~1.7pp, in the top bin.
Bins 1-9 agree within roughly half a point. No Platt scaling or isotonic
regression is applied, because none is warranted -- fitting a correction to
already-calibrated output would add variance for no gain.

This is partly a consequence of a deliberate earlier choice. `src/pipeline.py`
declines `class_weight='balanced'` and any resampling. Those techniques shift
predicted probabilities away from the true base rate: a balanced-weight model
outputs ~0.5 for an average loan rather than ~0.18. The cost analysis would
then be built on numbers that do not mean what they say.

THE ONE WEAK REGION
-------------------
The top decile over-predicts by ~1.7pp in both models (LightGBM: 0.4554
predicted vs 0.4387 actual). The highest-risk loans are slightly SAFER than the
model claims. This matters more than the number suggests, because that is
exactly the region where declines occur -- so the model is marginally too
pessimistic precisely where it acts. The effect is small enough not to change
the chosen thresholds materially, but it is the honest caveat on the decision
layer.

VALIDATING THE COST MODEL AGAINST THEORY
----------------------------------------
Good calibration allows a real check rather than a decorative plot. The
threshold implied by the cost rates should match the threshold found by
sweeping realised outcomes. Per term:

    term    theoretical G/(L+G)    empirical optimum    gap
    36mo    0.316                  0.345                +0.029
    60mo    0.457                  0.475                +0.018

Close agreement, which validates the cost model. The small positive bias has a
concrete cause: both error costs scale with amount funded, and loan size
correlates with risk (default rate rises from 14.2% in the smallest amount
quintile to 21.0% in the largest). Larger loans therefore carry more weight in
the cost sum and pull the cost-weighted optimum slightly above the unweighted
break-even.

Comparing the BLENDED theoretical rate (0.354) against the SINGLE GLOBAL
empirical optimum (0.460) is not a like-for-like comparison and should not be
read as miscalibration. The global optimum sits near the 60-month value because
60-month loans average $19,969 against $12,628, so they dominate the dollar
weighting despite being only 27% of applications.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a script context
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_table(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binned reliability table using quantile bins.

    Quantile bins rather than uniform-width bins: predicted probabilities here
    are concentrated below 0.4, so uniform bins would leave the upper range
    nearly empty and produce noisy estimates from a handful of loans.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, p_pred, n_bins=n_bins, strategy="quantile"
    )

    edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    counts = np.histogram(p_pred, bins=edges)[0]

    return pd.DataFrame(
        {
            "bin": range(1, len(frac_pos) + 1),
            "n": counts[: len(frac_pos)],
            "mean_predicted": mean_pred,
            "actual_rate": frac_pos,
            "gap": mean_pred - frac_pos,
        }
    )


def calibration_summary(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Scalar calibration diagnostics."""
    table = calibration_table(y_true, p_pred)
    return {
        "brier": float(brier_score_loss(y_true, p_pred)),
        "mean_predicted": float(p_pred.mean()),
        "actual_base_rate": float(y_true.mean()),
        "mean_abs_gap": float(table["gap"].abs().mean()),
        "max_abs_gap": float(table["gap"].abs().max()),
    }


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram for one or more models.

    `curves` maps model name -> (y_true, p_pred).
    """
    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), height_ratios=[3, 1], sharex=True
    )

    ax_rel.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration", zorder=1)

    for name, (y_true, p_pred) in curves.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, p_pred, n_bins=n_bins, strategy="quantile"
        )
        ax_rel.plot(mean_pred, frac_pos, "o-", lw=1.5, ms=5, label=name, zorder=2)
        ax_hist.hist(p_pred, bins=50, alpha=0.5, label=name)

    # Restrict axes to the region the model actually occupies; showing [0,1]
    # would compress every point into the lower-left corner.
    upper = max(max(p.max() for _, p in curves.values()), 0.5)
    ax_rel.set_xlim(0, upper)
    ax_rel.set_ylim(0, upper)

    ax_rel.set_ylabel("observed default rate")
    ax_rel.set_title("Calibration (quantile bins)")
    ax_rel.legend(loc="upper left")
    ax_rel.grid(alpha=0.3)

    ax_hist.set_xlabel("predicted probability of default")
    ax_hist.set_ylabel("count")
    ax_hist.set_yscale("log")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)