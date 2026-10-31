"""Tiny helpers for writing generated markdown reports.

Reports are GENERATED from code, never hand-edited, so they cannot drift from
the analysis that produced them.
"""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    """Render a dataframe as a GitHub-flavoured markdown table.

    `formats` maps column -> format spec (e.g. {"auc": ".4f", "n": ",d"}).
    """
    formats = formats or {}

    def fmt(col, value):
        if pd.isna(value):
            return "—"
        spec = formats.get(col)
        if spec:
            try:
                return format(value, spec)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    # to_dict("records") keeps each column's own type; iterrows() would
    # upcast integer columns to float in mixed-type frames.
    body = [
        "| " + " | ".join(fmt(c, row[c]) for c in df.columns) + " |"
        for row in df.to_dict("records")
    ]
    return "\n".join([header, rule, *body])
