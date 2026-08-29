"""
Asymmetric cost model and cost-minimising decision threshold.

THE DECISION BEING MODELLED
---------------------------
A platform investor sees a listed loan -- already graded and priced by
LendingClub -- and chooses to FUND it or DECLINE it. This framing is what makes
keeping `grade`, `sub_grade` and `int_rate` as features coherent: the investor
genuinely observes LendingClub's pricing at decision time. (From the
originator's seat those features would be circular, since the originator
produces them.)

WHERE THE COST NUMBERS COME FROM
--------------------------------
Not from industry rules of thumb. The dataset contains realised investor cash
flows, so the two costs are measured rather than assumed:

    net_return = total_pymnt_inv - funded_amnt_inv

Measured over the 2014 vintage:

    outcome        n          mean funded   mean received   net % of funded
    defaulted      41,162     $15,568       $10,167         -34.8%
    fully paid     181,941    $14,354       $17,250         +19.1%

Two things this corrects against the usual assumptions:

  * Loss given default is ~35%, not the 55-75% often quoted for unsecured
    consumer credit. That figure describes loss on principal outstanding at
    charge-off. But a defaulting borrower typically pays for a year or more
    first -- principal AND interest -- and recoveries add a little back.
    Averaged over all defaults the investor recovers about 65 cents per dollar.

  * Profit on a repaid loan is ~19%, not a thin margin. These are 13-15% APR
    loans held for years, so foregoing a good loan costs real money.

Both corrections compress the asymmetry to roughly 1.8:1, not the 6:1 or worse
that a naive principal-loss view implies.

WHY THE RATES ARE SPLIT BY TERM
-------------------------------
    term        repaid net %   defaulted net %
    36 months        +15.9%           -34.5%
    60 months        +29.6%           -35.1%

Same downside, nearly double the upside on 60-month loans. A single blended
rate would systematically over-decline the loans with the most to gain. The
consequence is that the cost-optimal threshold DIFFERS BY TERM -- the threshold
is an economic quantity, not a property of the model.

Caveat: 60-month figures come from resolved loans only, and that stratum is
16.3% right-censored (see src/data_loading.py). The CONDITIONAL rates used here
are fairly robust to this, since censoring determines which loans resolve
rather than what happens once they do.

WHY RATES ARE ESTIMATED ON THE TRAINING SPLIT ONLY
--------------------------------------------------
`total_pymnt_inv` is a post-origination column and is correctly excluded from
the feature set. Using it to estimate a BUSINESS PARAMETER is a different act
from using it as a model input -- the cost matrix is an input to the decision,
not to the model.

But the estimate must still come from training rows only. Setting the cost
matrix from test-set outcomes and then evaluating the resulting threshold on
that same test set would be circular: the threshold would be tuned on the
answers it is later scored against. Same reasoning that kept the missingness
indicators from being selected on full-data lifts.

THE THRESHOLD ARITHMETIC
------------------------
For a loan with default probability p, funding amount A, loss rate L and profit
rate G:

    E[cost of funding]   = p * L * A  -  (1 - p) * G * A
    E[cost of declining] = 0

Fund when the first is negative, i.e. when

    p < G / (L + G)

With blended rates that is 0.191 / (0.348 + 0.191) ~= 0.354. Note this is the
break-even for a SINGLE loan under known rates; the empirical optimum found by
sweeping thresholds against realised outcomes can differ, because predicted
probabilities are imperfect and their errors are not symmetric across the
range. Both numbers are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostRates:
    """Loss and profit rates as fractions of the amount funded."""

    loss_rate: float  # positive number; fraction lost when a funded loan defaults
    profit_rate: float  # positive number; fraction gained when a funded loan repays
    n_defaults: int
    n_repaid: int

    @property
    def breakeven_threshold(self) -> float:
        """Default probability above which funding has negative expected value."""
        return self.profit_rate / (self.loss_rate + self.profit_rate)

    @property
    def cost_ratio(self) -> float:
        return self.loss_rate / self.profit_rate


def estimate_cost_rates(
    df_train: pd.DataFrame,
    funded_col: str = "funded_amnt_inv",
    received_col: str = "total_pymnt_inv",
    target_col: str = "target",
) -> CostRates:
    """Measure loss and profit rates from realised cash flows on TRAINING rows.

    Rates are means of per-loan net return as a fraction of the amount funded,
    which weights each decision equally. (Aggregating dollars first would weight
    by loan size and answer a different question: the return of the portfolio
    rather than the expected cost of one decision.)
    """
    funded = df_train[funded_col]
    net_pct = (df_train[received_col] - funded) / funded

    is_default = df_train[target_col] == 1
    loss_rate = -net_pct[is_default].mean()  # negate: stored as a positive cost
    profit_rate = net_pct[~is_default].mean()

    return CostRates(
        loss_rate=float(loss_rate),
        profit_rate=float(profit_rate),
        n_defaults=int(is_default.sum()),
        n_repaid=int((~is_default).sum()),
    )


def estimate_cost_rates_by_term(
    df_train: pd.DataFrame, term_col: str = "term_months", **kwargs
) -> dict[int, CostRates]:
    """Per-term cost rates. Returns {36: CostRates, 60: CostRates}."""
    return {
        int(term): estimate_cost_rates(group, **kwargs)
        for term, group in df_train.groupby(term_col)
    }


def per_loan_costs(
    funded: pd.Series,
    terms: pd.Series,
    rates_by_term: dict[int, CostRates],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-loan cost of each error type, in dollars.

    Returns (cost_false_negative, cost_false_positive):
      * FN = funded a loan that defaulted -> realised loss
      * FP = declined a loan that would have repaid -> foregone profit

    A constant matrix would price a $2,000 36-month loan the same as a $35,000
    60-month one. These are per-loan because the decision is per-loan.
    """
    loss = np.zeros(len(funded), dtype=float)
    profit = np.zeros(len(funded), dtype=float)

    for term, rates in rates_by_term.items():
        mask = (terms == term).to_numpy()
        loss[mask] = rates.loss_rate
        profit[mask] = rates.profit_rate

    amounts = funded.to_numpy(dtype=float)
    return loss * amounts, profit * amounts


def total_cost_at_threshold(
    y_true: np.ndarray,
    p_default: np.ndarray,
    threshold: float,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
) -> float:
    """Total realised cost of applying a decision threshold.

    Decision rule: DECLINE when p_default >= threshold, otherwise FUND.
      * funded and defaulted  -> pay cost_fn
      * declined but repaid   -> pay cost_fp (opportunity cost)
      * declined and defaulted -> 0 (loss avoided)
      * funded and repaid      -> 0 (baseline; profit is the reference point)

    Note the frame: costs are measured RELATIVE to a perfect decision. Correctly
    funding a good loan scores zero rather than a profit, so the reported number
    is regret against an oracle, not portfolio P&L.
    """
    decline = p_default >= threshold

    false_negative = (~decline) & (y_true == 1)
    false_positive = decline & (y_true == 0)

    return float(cost_fn[false_negative].sum() + cost_fp[false_positive].sum())


def sweep_thresholds(
    y_true: np.ndarray,
    p_default: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Total and per-application cost across a grid of thresholds."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 197)

    n = len(y_true)
    rows = []
    for t in grid:
        decline = p_default >= t
        total = total_cost_at_threshold(y_true, p_default, t, cost_fn, cost_fp)
        rows.append(
            {
                "threshold": t,
                "total_cost": total,
                "cost_per_application": total / n,
                "decline_rate": float(decline.mean()),
                "approved_default_rate": float(
                    y_true[~decline].mean() if (~decline).any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(sweep: pd.DataFrame) -> tuple[float, float]:
    """Threshold minimising cost per application. Returns (threshold, cost)."""
    best = sweep.loc[sweep["cost_per_application"].idxmin()]
    return float(best["threshold"]), float(best["cost_per_application"])