"""
Load the LendingClub accepted-loans file, keep one origination vintage, and
define the binary default target.

1. TARGET: TERMINAL STATES ONLY
-------------------------------
    1  = Charged Off, Default
    0  = Fully Paid
    -- = Current, Late (16-30), Late (31-120), In Grace Period  -> dropped

A supervised model needs an observed outcome. A "Current" loan has none yet: the
borrower is paying on schedule and the story has not ended. A "Late" loan is an
interim state that may still cure, so labelling it 1 would assert an outcome
nobody has observed and make the positive class mean two different things
(finished-and-failed versus currently struggling).

2. MATURED != RESOLVED
----------------------
A loan MATURES when its scheduled term elapses. It RESOLVES when it reaches a
terminal state, which usually happens much earlier: borrowers prepay (common on
LendingClub, where people refinance out) or stop paying and are charged off
within months. Resolution, not maturity, is what makes a loan usable.

3. WHY DROPPING UNRESOLVED LOANS IS NOT NEUTRAL (RIGHT-CENSORING)
-----------------------------------------------------------------
The snapshot cuts every loan's story off at the same calendar date. Defaults
resolve fast; healthy loans that neither prepay nor default stay "Current"
until their final payment. So the loans removed for lack of an outcome are
disproportionately GOOD loans, and the default rate among the survivors is
biased UPWARD. The more censored a vintage, the larger the bias -- which is why
the charged-off share of resolved loans climbs with recency even though
lending standards did not collapse.

4. VINTAGE = 2014
-----------------
`load_vintage` records a full year x term x status census in the same pass it
uses to extract the vintage; `scripts/build_dataset.py` writes that census to
reports/vintage_censoring.md. 2014 is the most recent vintage with low
censoring: 36-month loans are fully resolved and 60-month loans are partially
censored, with the bias bounded rather than ignored (see reports/results.md).
2015 offers more rows at roughly double the censoring; 223k loans are already
far more than this analysis needs, so the extra rows buy little.

5. MEMORY
---------
The gzipped source expands well beyond a few GB in memory, so it is streamed in
chunks and filtered per chunk. Only the chosen vintage ever accumulates.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.config import CHUNK_SIZE

DEFAULT_STATUSES = frozenset({"Charged Off", "Default"})
PAID_STATUSES = frozenset({"Fully Paid"})

# Enumerated explicitly rather than handled by a catch-all `else`: a status that
# matches neither list is counted separately, so schema surprises surface.
UNRESOLVED_STATUSES = frozenset(
    {"Current", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"}
)
TERMINAL_STATUSES = DEFAULT_STATUSES | PAID_STATUSES


def label_target(loan_status: pd.Series) -> pd.Series:
    """Map status to 1 (default), 0 (fully paid) or NA (no observed outcome).

    NA is deliberate. The function refuses to guess; the CALLER drops NA rows,
    where the number dropped can be counted and reported. Silently mapping
    "Current" to 0 here would mislabel thousands of loans with no trace.
    """
    status = loan_status.astype("string").str.strip()
    target = pd.Series(pd.NA, index=loan_status.index, dtype="Int8")
    target[status.isin(PAID_STATUSES)] = 0
    target[status.isin(DEFAULT_STATUSES)] = 1
    return target


def parse_issue_year(issue_d: pd.Series) -> pd.Series:
    """'Dec-2014' -> 2014. Unparseable values become NA."""
    return pd.to_datetime(issue_d, format="%b-%Y", errors="coerce").dt.year


def load_vintage(
    path,
    year: int,
    chunksize: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Stream the source, keep one origination year, attach the target.

    Returns (dataframe, audit). The audit records:
      * row counts at every filtering stage, so drops are reported, not assumed
      * status counts by term for the chosen vintage (feeds the censoring bound)
      * a year x term x status census of the WHOLE file (justifies the vintage)
    """
    census: Counter = Counter()
    vintage_by_term: Counter = Counter()
    kept = []
    audit = {
        "source_rows": 0,
        "vintage_year": year,
        "vintage_rows": 0,
        "dropped_unresolved": 0,
        "dropped_unknown_status": 0,
    }

    reader = pd.read_csv(path, compression="gzip", low_memory=False, chunksize=chunksize)

    for chunk in reader:
        audit["source_rows"] += len(chunk)

        issue_year = parse_issue_year(chunk["issue_d"])
        term = chunk["term"].astype("string").str.strip()
        status = chunk["loan_status"].astype("string").str.strip()

        # Census of the full file, before any filtering.
        counts = pd.DataFrame({"year": issue_year, "term": term, "status": status})
        for key, n in counts.value_counts(dropna=False).items():
            census[key] += int(n)

        in_vintage = (issue_year == year).to_numpy()
        if not in_vintage.any():
            continue
        chunk = chunk.loc[in_vintage]
        status = status[in_vintage]
        audit["vintage_rows"] += len(chunk)

        for key, n in pd.DataFrame(
            {"term": term[in_vintage], "status": status}
        ).value_counts().items():
            vintage_by_term[key] += int(n)

        audit["dropped_unresolved"] += int(status.isin(UNRESOLVED_STATUSES).sum())
        audit["dropped_unknown_status"] += int(
            (~status.isin(TERMINAL_STATUSES | UNRESOLVED_STATUSES)).sum()
        )

        target = label_target(chunk["loan_status"]).rename("target")
        has_outcome = target.notna().to_numpy()
        # concat rather than assign: avoids pandas' fragmented-frame warning
        # on a 151-column slice.
        kept.append(pd.concat([chunk.loc[has_outcome], target[has_outcome]], axis=1))

    df = pd.concat(kept, ignore_index=True)
    df["target"] = df["target"].astype("int8")

    audit["final_rows"] = len(df)
    audit["default_rate"] = float(df["target"].mean())
    audit["vintage_status_by_term"] = {
        f"{t}|{s}": n for (t, s), n in sorted(vintage_by_term.items())
    }
    audit["census"] = [
        {"year": None if pd.isna(y) else int(y), "term": t, "status": s, "n": n}
        for (y, t, s), n in sorted(census.items(), key=lambda kv: str(kv[0]))
    ]
    return df, audit


def censoring_table(census: list[dict]) -> pd.DataFrame:
    """Per-year share of loans with no observed outcome at snapshot.

    `resolved_default_share` is charged-off share among RESOLVED loans only.
    Its rise with recency is the censoring bias made visible.
    """
    frame = pd.DataFrame(census).dropna(subset=["year"])
    frame["year"] = frame["year"].astype(int)
    frame["is_unresolved"] = frame["status"].isin(UNRESOLVED_STATUSES)
    frame["is_default"] = frame["status"].isin(DEFAULT_STATUSES)
    frame["is_paid"] = frame["status"].isin(PAID_STATUSES)

    rows = []
    for year, g in frame.groupby("year"):
        total = g["n"].sum()
        unresolved = g.loc[g["is_unresolved"], "n"].sum()
        defaults = g.loc[g["is_default"], "n"].sum()
        paid = g.loc[g["is_paid"], "n"].sum()
        rows.append(
            {
                "year": year,
                "loans": int(total),
                "unresolved": int(unresolved),
                "pct_unresolved": 100 * unresolved / total,
                "resolved_default_share": 100 * defaults / (defaults + paid)
                if defaults + paid
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
