"""Tests for target definition, vintage parsing and the leakage audit.

All synthetic: no dataset download needed, runs in seconds.
"""

import pandas as pd
import pytest

from src.data_loading import censoring_table, label_target, parse_issue_year
from src.leakage_audit import (COLUMN_AUDIT, KEEP, LEAK_BUCKETS, audit_columns,
                               get_feature_columns)


# ------------------------------------------------------------ target

def test_terminal_statuses_map_to_labels():
    s = pd.Series(["Fully Paid", "Charged Off", "Default"])
    assert label_target(s).tolist() == [0, 1, 1]


@pytest.mark.parametrize("status", ["Current", "Late (31-120 days)", "Late (16-30 days)",
                                    "In Grace Period"])
def test_unresolved_statuses_are_NA_not_zero(status):
    # The bug this guards against: silently labelling an unfinished loan as "repaid".
    assert label_target(pd.Series([status])).isna().all()


def test_unknown_status_is_NA():
    assert label_target(pd.Series(["Something New"])).isna().all()


def test_status_whitespace_is_tolerated():
    assert label_target(pd.Series(["  Fully Paid "])).tolist() == [0]


def test_issue_year_parsing():
    assert parse_issue_year(pd.Series(["Dec-2014", "Jan-2015", "garbage"])).tolist()[:2] == [2014, 2015]
    assert pd.isna(parse_issue_year(pd.Series(["garbage"])).iloc[0])


def test_censoring_table_arithmetic():
    census = [
        {"year": 2014, "term": "36 months", "status": "Fully Paid", "n": 80},
        {"year": 2014, "term": "36 months", "status": "Charged Off", "n": 20},
        {"year": 2014, "term": "60 months", "status": "Current", "n": 100},
    ]
    row = censoring_table(census).iloc[0]
    assert row["loans"] == 200
    assert row["pct_unresolved"] == pytest.approx(50.0)
    assert row["resolved_default_share"] == pytest.approx(20.0)


# ------------------------------------------------------------ leakage audit

def test_audit_raises_on_unclassified_column():
    # The guarantee: an unvetted column can never silently reach the model.
    df = pd.DataFrame({"loan_amnt": [1], "brand_new_column": [2]})
    with pytest.raises(KeyError, match="brand_new_column"):
        audit_columns(df)


@pytest.mark.parametrize("col", ["total_pymnt", "recoveries", "last_fico_range_low",
                                 "last_pymnt_d", "debt_settlement_flag", "out_prncp"])
def test_known_post_origination_fields_are_leaks(col):
    assert COLUMN_AUDIT[col][0] in LEAK_BUCKETS


@pytest.mark.parametrize("col", ["fico_range_low", "dti", "annual_inc", "grade", "int_rate"])
def test_known_origination_fields_are_kept(col):
    assert COLUMN_AUDIT[col][0] == KEEP


def test_target_and_protected_proxies_never_become_features():
    cols = ["target", "loan_status", "addr_state", "zip_code", "loan_amnt"]
    feats = get_feature_columns(pd.DataFrame({c: [0] for c in cols}))
    assert feats == ["loan_amnt"]


def test_every_classification_has_a_rationale():
    assert all(len(rationale) > 10 for _, rationale in COLUMN_AUDIT.values())
