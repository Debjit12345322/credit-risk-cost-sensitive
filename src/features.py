"""
Feature engineering: missingness indicators and type conversions.

MISSINGNESS AS SIGNAL
---------------------
Nine columns in the audited feature set contain missing values. Blind median
imputation would assert something false about the borrower -- filling
`mths_since_last_delinq` with the median tells the model "this borrower's last
delinquency was ~31 months ago" about someone who has NEVER been delinquent.
That does not fill a gap; it invents a fact and erases genuine information.

Measured default rate by missingness (n=223,103, base rate 18.45%):

    column                          n_missing   lift(pp)   p-value
    emp_length                         11,554     +5.56    7.9e-51
    mths_since_recent_inq              20,185     -5.18    5.7e-73
    bc_util                             2,482     +1.76    2.7e-02
    revol_util                            115     +1.55    7.6e-01   (n.s.)
    mths_since_last_major_derog       159,591     -1.33    2.4e-13
    mths_since_last_record            183,144     -1.31    1.1e-09
    mths_since_recent_bc_dlq          163,991     -0.99    1.1e-07
    mths_since_last_delinq            109,646     -0.88    8.8e-08
    mths_since_recent_revol_delinq    142,779     -0.56    1.2e-03

Note the SIGN FLIP. Missing `mths_since_last_delinq` means "never delinquent"
-> slightly LOWER default risk. Missing `emp_length` means the borrower did not
supply employment information -> substantially HIGHER default risk. Same
mechanism, opposite meaning. A single imputation strategy cannot express both.

WHY ALL NINE GET AN INDICATOR, INCLUDING THE WEAK ONES
------------------------------------------------------
The table above was computed on the full dataset, which includes rows that will
later become the test set. Using it to decide which features exist would let
outcome information influence model design -- a mild version of the same
mistake the leakage audit exists to prevent. Selecting the "significant" seven
would mean answering "how did you choose?" with "I looked at the answers."

The cost of keeping all nine is negligible: 9 extra columns on 71, which
LightGBM simply will not split on if uninformative and which L2-regularised
logistic regression shrinks toward zero. Correct selection would compute these
lifts inside training folds only; that is more machinery than a 9-feature
decision warrants here.

Note also that significance at n=223,103 is cheap. `mths_since_recent_revol_delinq`
reaches p=1.2e-03 on a 0.56pp effect. The p-values mostly measure sample size;
the lift column is what carries practical meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns whose missingness is preserved as an explicit binary feature.
MISSINGNESS_INDICATOR_COLS = [
    "emp_length",
    "mths_since_recent_inq",
    "bc_util",
    "revol_util",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_last_delinq",
    "mths_since_recent_revol_delinq",
]


def parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, NaN -> NaN.

    The bucketing is LendingClub's, not ours: the field is already censored at
    both ends, so 10 means "10 or more" and 0 means "less than one year".
    Missing stays missing -- `emp_length_is_missing` carries that signal.

    NOTE: '< 1 year' must be handled BEFORE digit extraction. A bare
    `str.extract(r'(\\d+)')` pulls the 1 out of '< 1 year' and collapses it into
    the same value as '1 year', silently merging two distinct groups (11,676
    borrowers with under a year of employment and 14,808 with exactly one).
    """
    if series.isna().all():
        return pd.Series(np.nan, index=series.index, dtype="float64")

    text = series.astype("string").str.strip()
    extracted = text.str.extract(r"(\d+)")[0]
    years = pd.to_numeric(extracted, errors="coerce")
    # '< 1 year' is the only value where the extracted digit misrepresents the bucket.
    return years.mask(text.str.startswith("<").fillna(False), 0.0)


def parse_term(series: pd.Series) -> pd.Series:
    """' 36 months' -> 36. Kept numeric so it can serve as an analysis segment."""
    extracted = series.astype("string").str.extract(r"(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def credit_history_months(
    earliest_cr_line: pd.Series, issue_d: pd.Series
) -> pd.Series:
    """Months between first credit line and loan origination.

    The raw date is useless to a model -- 'Aug-2001' carries no ordering the
    model can exploit, and using the year alone would smuggle in a time trend.
    Length of credit history at the moment of application is what an underwriter
    actually reads, and it is well defined at decision time.
    """
    first = pd.to_datetime(earliest_cr_line, format="%b-%Y", errors="coerce")
    issued = pd.to_datetime(issue_d, format="%b-%Y", errors="coerce")
    months = (issued.dt.year - first.dt.year) * 12 + (issued.dt.month - first.dt.month)
    # A first credit line after origination is impossible; treat as bad data.
    return months.where(months >= 0)


def add_missingness_indicators(
    df: pd.DataFrame, cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Append `<col>_is_missing` int8 columns. Returns (df, indicator_names).

    Indicators are added BEFORE any imputation, so they record the raw state of
    the data rather than the state after filling.
    """
    cols = MISSINGNESS_INDICATOR_COLS if cols is None else cols
    present = [c for c in cols if c in df.columns]

    missing_from_frame = set(cols) - set(present)
    if missing_from_frame:
        raise KeyError(
            f"Expected columns absent from dataframe: {sorted(missing_from_frame)}"
        )

    indicators = {f"{c}_is_missing": df[c].isna().astype("int8") for c in present}
    out = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)
    return out, list(indicators.keys())


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply all transformations. Returns (dataframe, new_column_names).

    Order matters: missingness indicators are computed first, so that parsing
    `emp_length` cannot alter what is recorded as missing.
    """
    out, indicator_cols = add_missingness_indicators(df)

    out["emp_length_years"] = parse_emp_length(out["emp_length"])
    out["term_months"] = parse_term(out["term"])
    out["credit_history_months"] = credit_history_months(
        out["earliest_cr_line"], out["issue_d"]
    )

    derived = ["emp_length_years", "term_months", "credit_history_months"]
    return out, indicator_cols + derived
