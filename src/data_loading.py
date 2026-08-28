"""
Load the LendingClub accepted-loans file, subset to a single origination
vintage, and define the binary default target.

Design decisions (defensible in review):

1. VINTAGE = 2014.

   MATURED != RESOLVED. A loan MATURES when its scheduled term elapses. It
   RESOLVES when it reaches a terminal state, which usually happens EARLIER:
   the borrower either prepays (common on LendingClub -- borrowers refinance
   out) or defaults and is charged off. Neither waits for the end date.
   Measured on this vintage: 97.2% of resolved 60-month loans resolved before
   month 60, at a median of ~25 months.

   The snapshot runs to Mar-2019 (despite the "2018Q4" filename). 2014
   60-month loans mature Jan-Dec 2019, so essentially NONE have matured --
   yet 83.7% have already resolved, via the mechanisms above.

   Recent vintages are still heavily right-censored: 59% of 2017 loans are
   "Current" at snapshot. Dropping unresolved loans is correct (they have no
   outcome) but it is NOT outcome-neutral. Defaults resolve fast; healthy
   loans that neither prepay nor default stay "Current" to the end. So the
   loans dropped are disproportionately GOOD ones, and the observed default
   rate is biased UPWARD.

   2014 chosen as the most recent vintage where nearly all loans have
   resolved: 36-month 100% resolved (162,570/162,570), 60-month 16.3%
   censored. Both terms kept so that `term` remains an error-analysis segment.

   QUANTIFIED CENSORING BIAS (60-month stratum):
       resolved            60,533   observed default rate 31.1%
       still Current       11,919
       if all Current pay           implied default rate  26.0%
   True 60-month rate lies in [26.0%, 31.1%] and is likely near the lower
   bound, since the censored loans are on-schedule payers. Report the
   60-month segment with this caveat attached. The 36-month stratum (13.7%)
   is unaffected -- it is fully resolved.

2. TARGET.
   Only terminal outcomes are labelled:
       1 = Charged Off, Default
       0 = Fully Paid
   Everything else (Current, Late, In Grace Period) is DROPPED, not imputed and
   not folded into the positive class. "Late (31-120 days)" is an interim state:
   the loan may still cure. Labelling it as default would assert an outcome we
   have not observed, and would make the positive class mean two different
   things at once.

3. MEMORY.
   The gzipped source is ~375MB and expands well beyond available RAM, so the
   file is streamed in chunks and filtered per chunk rather than loaded whole.
"""

from __future__ import annotations

import pandas as pd

# --- target definition -----------------------------------------------------

DEFAULT_STATUSES = {"Charged Off", "Default"}
PAID_STATUSES = {"Fully Paid"}

# Explicitly enumerated so that an unexpected new status value causes a visible
# drop we can inspect, rather than silently landing in a catch-all bucket.
UNRESOLVED_STATUSES = {
    "Current",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "In Grace Period",
}

TERMINAL_STATUSES = DEFAULT_STATUSES | PAID_STATUSES


def label_target(loan_status: pd.Series) -> pd.Series:
    """Map loan_status -> {1: default, 0: fully paid, NA: unresolved/unknown}.

    NA is deliberate: the caller drops these rows explicitly so the count of
    dropped loans is observable, rather than being hidden inside this function.
    """
    status = loan_status.str.strip()
    target = pd.Series(pd.NA, index=status.index, dtype="Int8")
    target[status.isin(PAID_STATUSES)] = 0
    target[status.isin(DEFAULT_STATUSES)] = 1
    return target


def load_vintage(
    path: str,
    year: int = 2014,
    chunksize: int = 200_000,
    usecols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Stream the source file, keep one origination year, attach the target.

    Returns (dataframe, audit_counts). The audit dict records how many rows were
    dropped at each stage so the numbers can be reported rather than assumed.
    """
    kept_chunks = []
    audit = {
        "rows_scanned": 0,
        "rows_in_vintage": 0,
        "dropped_unresolved": 0,
        "dropped_unknown_status": 0,
        "rows_final": 0,
        "status_counts_in_vintage": {},
    }

    reader = pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )

    for chunk in reader:
        audit["rows_scanned"] += len(chunk)

        issue_year = pd.to_datetime(
            chunk["issue_d"], format="%b-%Y", errors="coerce"
        ).dt.year
        chunk = chunk[issue_year == year]
        if chunk.empty:
            continue
        audit["rows_in_vintage"] += len(chunk)

        for status, n in chunk["loan_status"].value_counts().items():
            audit["status_counts_in_vintage"][status] = (
                audit["status_counts_in_vintage"].get(status, 0) + int(n)
            )

        target = label_target(chunk["loan_status"])
        status = chunk["loan_status"].str.strip()

        audit["dropped_unresolved"] += int(status.isin(UNRESOLVED_STATUSES).sum())
        audit["dropped_unknown_status"] += int(
            (~status.isin(TERMINAL_STATUSES | UNRESOLVED_STATUSES)).sum()
        )

        chunk = chunk.assign(target=target)
        kept_chunks.append(chunk[chunk["target"].notna()])

    df = pd.concat(kept_chunks, ignore_index=True)
    df["target"] = df["target"].astype("int8")
    audit["rows_final"] = len(df)
    audit["default_rate"] = float(df["target"].mean())

    return df, audit
