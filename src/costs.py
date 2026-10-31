"""
Asymmetric cost model and the cost-minimising decision threshold.

THE DECISION
------------
A platform investor sees a listed loan -- already graded and priced by
LendingClub -- and chooses FUND or DECLINE. This framing is what makes keeping
`grade`, `sub_grade` and `int_rate` coherent: the investor genuinely observes
LendingClub's pricing at decision time. From the originator's seat those
features would be circular.

WHERE THE COSTS COME FROM: MEASURED, NOT ASSUMED
------------------------------------------------
The dataset records realised investor cash flows, so both error costs are
measured:

    net return = total_pymnt_inv - funded_amnt_inv     (as a fraction of funded)

    loss rate   L = mean net loss over DEFAULTED loans
    profit rate G = mean net gain over REPAID loans

`total_pymnt_inv` is a post-origination column and is correctly excluded from
the FEATURES. Using it to estimate a BUSINESS PARAMETER is a different act: the
cost matrix is an input to the decision, not to the model. The estimate still
comes from TRAINING rows only -- setting costs from test outcomes and then
scoring a threshold on those same outcomes would be circular.

Two things the measurement shows (exact figures in reports/results.md):
  * L is about 35%, well below the loss on outstanding principal at
    charge-off, because a defaulting borrower usually pays for a year or more
    first and recoveries add a little back.
  * G is about 19% on average and much higher for 60-month loans, because these
    are double-digit-APR loans held for years.

RATES ARE SPLIT BY TERM
-----------------------
60-month loans have near-identical downside to 36-month loans but nearly double
the upside when repaid, so their break-even threshold genuinely differs. A
blended rate would over-decline the loans with the most to gain.

THE DECISION RULE (BAYES-OPTIMAL UNDER THIS LOSS)
-------------------------------------------------
For a loan with posterior p = P(default | x) and funded amount A:

    E[cost | fund]    = p * L * A  -  (1 - p) * G * A
    E[cost | decline] = 0

Decline when funding has positive expected cost:

    p > G / (L + G)

The amount A cancels for a single loan. The EMPIRICAL optimum found by sweeping
thresholds can still differ slightly, because across many loans the costs are
weighted by amount and loan size correlates with risk. src/calibration.py
reports both and explains the gap.

HOW COST IS SCORED
------------------
Regret against a perfect decision. Funded-and-defaulted costs L*A; declined-
but-would-have-repaid costs G*A (foregone profit); the two correct decisions
cost zero. So "$X per application" means "this much worse than an oracle", not
"money lost".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import THRESHOLD_GRID_STEP


@dataclass(frozen=True)
class CostRates:
    """Loss and profit as fractions of the amount funded (both positive)."""

    loss_rate: float
    profit_rate: float
    n_defaults: int = 0
    n_repaid: int = 0
    source: str = "measured"

    @property
    def breakeven_threshold(self) -> float:
        """Posterior above which funding has negative expected value: G/(L+G)."""
        return self.profit_rate / (self.loss_rate + self.profit_rate)

    @property
    def cost_ratio(self) -> float:
        return self.loss_rate / self.profit_rate

    @classmethod
    def assumed(cls, loss_rate: float, profit_rate: float) -> "CostRates":
        """Illustrative rates for sensitivity analysis -- not measured, not cited."""
        return cls(loss_rate, profit_rate, source="assumed")


def estimate_cost_rates(
    df_train: pd.DataFrame,
    funded_col: str = "funded_amnt_inv",
    received_col: str = "total_pymnt_inv",
    target_col: str = "target",
) -> CostRates:
    """Measure L and G from realised cash flows on TRAINING rows.

    Means of per-loan returns, so each decision is weighted equally. Summing
    dollars first would instead answer a portfolio question.
    """
    funded = df_train[funded_col]
    net = (df_train[received_col] - funded) / funded
    is_default = (df_train[target_col] == 1).to_numpy()
    return CostRates(
        loss_rate=float(-net[is_default].mean()),
        profit_rate=float(net[~is_default].mean()),
        n_defaults=int(is_default.sum()),
        n_repaid=int((~is_default).sum()),
    )


def estimate_cost_rates_by_term(
    df_train: pd.DataFrame, term_col: str = "term_months", **kwargs
) -> dict[int, CostRates]:
    """{36: CostRates, 60: CostRates}, each measured on that term's training rows."""
    return {
        int(term): estimate_cost_rates(group, **kwargs)
        for term, group in df_train.groupby(term_col)
    }


def per_loan_costs(
    funded: pd.Series | np.ndarray,
    terms: pd.Series | np.ndarray,
    rates_by_term: dict[int, CostRates],
) -> tuple[np.ndarray, np.ndarray]:
    """Dollar cost of each error type for every loan.

    Returns (cost_fn, cost_fp):
      cost_fn = L * A   funded a loan that defaulted
      cost_fp = G * A   declined a loan that would have repaid
    """
    funded = np.asarray(funded, dtype=float)
    terms = np.asarray(terms)
    loss = np.full(len(funded), np.nan)
    profit = np.full(len(funded), np.nan)
    for term, rates in rates_by_term.items():
        mask = terms == term
        loss[mask] = rates.loss_rate
        profit[mask] = rates.profit_rate
    if np.isnan(loss).any():
        raise ValueError("Some loans have a term with no cost rates.")
    return loss * funded, profit * funded


def total_cost_of_decisions(
    y_true: np.ndarray, decline: np.ndarray, cost_fn: np.ndarray, cost_fp: np.ndarray
) -> float:
    """Total regret of a set of fund/decline decisions."""
    y_true = np.asarray(y_true)
    decline = np.asarray(decline, dtype=bool)
    funded_bad = (~decline) & (y_true == 1)
    declined_good = decline & (y_true == 0)
    return float(cost_fn[funded_bad].sum() + cost_fp[declined_good].sum())


def total_cost_at_threshold(
    y_true: np.ndarray,
    p_default: np.ndarray,
    threshold: float,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
) -> float:
    """Total regret when declining every loan with p_default >= threshold."""
    return total_cost_of_decisions(y_true, p_default >= threshold, cost_fn, cost_fp)


def threshold_grid(step: float = THRESHOLD_GRID_STEP) -> np.ndarray:
    return np.round(np.arange(step, 1.0, step), 6)


def sweep_thresholds(
    y_true: np.ndarray,
    p_default: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cost per application, decline rate and approved default rate per threshold."""
    y_true = np.asarray(y_true)
    grid = threshold_grid() if grid is None else grid
    n = len(y_true)
    rows = []
    for t in grid:
        decline = p_default >= t
        approved = ~decline
        rows.append(
            {
                "threshold": float(t),
                "cost_per_application": total_cost_of_decisions(
                    y_true, decline, cost_fn, cost_fp
                )
                / n,
                "decline_rate": float(decline.mean()),
                "approved_default_rate": float(y_true[approved].mean())
                if approved.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(sweep: pd.DataFrame) -> tuple[float, float]:
    """(threshold, cost per application) at the minimum of a sweep."""
    best = sweep.loc[sweep["cost_per_application"].idxmin()]
    return float(best["threshold"]), float(best["cost_per_application"])
