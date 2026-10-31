"""Tests for feature parsing, the cost model, the decision rule and OOF fitting."""

import numpy as np
import pandas as pd
import pytest

from src.costs import (CostRates, optimal_threshold, per_loan_costs, sweep_thresholds,
                       total_cost_at_threshold)
from src.features import (credit_history_months, engineer_features, parse_emp_length,
                          parse_term)
from src.segments import apply_segment_thresholds, out_of_fold_predictions


# ------------------------------------------------------------ features

def test_emp_length_less_than_one_year_is_zero_not_one():
    # Regression test for a real bug: a bare digit regex reads "< 1 year" as 1,
    # silently merging two distinct groups of borrowers.
    out = parse_emp_length(pd.Series(["< 1 year", "1 year", "10+ years", "4 years", None]))
    assert out.iloc[:4].tolist() == [0, 1, 10, 4]
    assert pd.isna(out.iloc[4])


def test_term_parsing():
    assert parse_term(pd.Series([" 36 months", " 60 months"])).tolist() == [36, 60]


def test_credit_history_months():
    out = credit_history_months(pd.Series(["Aug-2001", "Jan-2020"]),
                                pd.Series(["Dec-2014", "Dec-2014"]))
    assert out.iloc[0] == 13 * 12 + 4
    assert pd.isna(out.iloc[1])  # first credit line after origination is impossible


def _minimal_frame(emp_length):
    from src.features import MISSINGNESS_INDICATOR_COLS
    df = pd.DataFrame({c: [1.0] for c in MISSINGNESS_INDICATOR_COLS})
    df["emp_length"] = [emp_length]
    df["term"] = [" 36 months"]
    df["earliest_cr_line"] = ["Aug-2001"]
    df["issue_d"] = ["Dec-2014"]
    return df


def test_missingness_indicator_reflects_raw_data():
    out, new_cols = engineer_features(_minimal_frame(None))
    assert out["emp_length_is_missing"].iloc[0] == 1
    assert len(new_cols) == 12


# ------------------------------------------------------------ costs

def test_bayes_threshold_formula():
    r = CostRates.assumed(loss_rate=0.35, profit_rate=0.20)
    assert r.breakeven_threshold == pytest.approx(0.20 / 0.55)


def test_regret_accounting_by_hand():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    fn = np.array([100.0, 100.0, 100.0, 100.0])
    fp = np.array([10.0, 10.0, 10.0, 10.0])
    # threshold 0.5: declines rows 0,2. Row 1 = funded default (100), row 2 = declined good (10).
    assert total_cost_at_threshold(y, p, 0.5, fn, fp) == pytest.approx(110.0)


def test_per_loan_costs_scale_with_amount_and_reject_unknown_term():
    rates = {36: CostRates.assumed(0.3, 0.1)}
    fn, fp = per_loan_costs(np.array([1000.0, 2000.0]), np.array([36, 36]), rates)
    assert fn.tolist() == pytest.approx([300.0, 600.0])
    assert fp.tolist() == pytest.approx([100.0, 200.0])
    with pytest.raises(ValueError):
        per_loan_costs(np.array([1000.0]), np.array([60]), rates)


def test_swept_optimum_recovers_bayes_threshold_on_calibrated_data():
    # When p is the TRUE probability and amounts are equal, the threshold that
    # minimises realised cost must sit at G/(L+G). This checks the decision
    # theory, not just the arithmetic.
    rng = np.random.default_rng(0)
    n = 400_000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    L, G = 0.35, 0.19
    fn, fp = np.full(n, L), np.full(n, G)
    t, _ = optimal_threshold(sweep_thresholds(y, p, fn, fp))
    assert t == pytest.approx(G / (L + G), abs=0.02)


# ------------------------------------------------------------ segments / OOF

def test_unseen_segment_uses_fallback_threshold():
    decline = apply_segment_thresholds(np.array(["A", "Z"]), np.array([0.4, 0.4]),
                                       {"A": 0.3}, fallback=0.5)
    assert decline.tolist() == [True, False]


class _RecordingModel:
    """Remembers which rows it was trained on and predicts their positions."""

    trained_on: list = []

    def fit(self, X, y):
        self.seen = set(X["row"])
        _RecordingModel.trained_on.append(self.seen)
        return self

    def predict_proba(self, X):
        overlap = [r in self.seen for r in X["row"]]
        assert not any(overlap), "OOF leak: model predicted a row it was trained on"
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])


def test_out_of_fold_never_scores_a_row_with_a_model_that_saw_it():
    X = pd.DataFrame({"row": np.arange(100)})
    y = np.array([0, 1] * 50)
    oof = out_of_fold_predictions(_RecordingModel, X, y, seed=0)
    assert len(oof) == 100 and np.all(oof == 0.5)
