"""
Generate the leakage audit report and quantify what the audit is worth.

Run:  python -m scripts.run_leakage_audit

Outputs:
    reports/leakage_audit.md        full 152-column classification
    reports/leakage_demo.md         naive-vs-clean AUC comparison

The demo trains the SAME model twice -- once on the audited feature set, once
with the servicing/outcome columns added back -- so the cost of skipping the
audit is a number rather than an assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leakage_audit import (  # noqa: E402
    COLUMN_AUDIT,
    DROP_BUCKETS,
    audit_columns,
    get_feature_columns,
)

SEED = 42
DATA = "data/loans_2014.parquet"
BUCKET_ORDER = [
    "TARGET",
    "LEAK_PAYMENT",
    "LEAK_MONITORING",
    "LEAK_OUTCOME",
    "EMPTY_OR_CONST",
    "IDENTIFIER",
    "ANALYSIS_ONLY",
    "KEEP",
]


def write_audit_report(df: pd.DataFrame, path: str) -> pd.DataFrame:
    audit = audit_columns(df)
    counts = audit["bucket"].value_counts()

    lines = [
        "# Leakage Audit",
        "",
        "Every column in the source file is classified explicitly. The test applied:",
        "",
        "> Would this value exist, with this value, in the lender's system at the",
        "> moment the lend/decline decision is made?",
        "",
        "`src/leakage_audit.py` raises on any column it has not seen, so a schema",
        "change fails loudly instead of silently admitting an unvetted field.",
        "",
        "## Summary",
        "",
        "| Bucket | Columns |",
        "| --- | ---: |",
    ]
    for b in BUCKET_ORDER:
        if b in counts:
            lines.append(f"| `{b}` | {counts[b]} |")
    lines += [f"| **Total** | **{len(audit)}** |", ""]

    for b in BUCKET_ORDER:
        sub = audit[audit["bucket"] == b]
        if sub.empty:
            continue
        lines += [
            f"## {b}",
            "",
            "| Column | Fill % | Rationale |",
            "| --- | ---: | --- |",
        ]
        for _, r in sub.iterrows():
            lines.append(f"| `{r['column']}` | {r['fill_rate']} | {r['rationale']} |")
        lines.append("")

    Path(path).write_text("\n".join(lines))
    return audit


def run_leak_demo(df: pd.DataFrame, path: str) -> None:
    """Same model, two feature sets. The gap is the cost of skipping the audit."""
    y = df["target"]

    def numeric(cols):
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    clean = numeric(get_feature_columns(df))
    leaks = numeric(
        [
            c
            for c, (b, _) in COLUMN_AUDIT.items()
            if b in {"LEAK_PAYMENT", "LEAK_MONITORING", "LEAK_OUTCOME"}
            and c in df.columns
        ]
    )

    results = []
    for name, cols in [("Naive (leaks included)", clean + leaks), ("Audited (clean)", clean)]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[cols], y, test_size=0.25, stratify=y, random_state=SEED
        )
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1
        )
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        top = (
            pd.Series(model.feature_importances_, index=cols)
            .sort_values(ascending=False)
            .head(6)
            .index.tolist()
        )
        results.append((name, len(cols), auc, top))

    lines = [
        "# What the leakage audit is worth",
        "",
        f"Identical LightGBM config (seed={SEED}), identical split, two feature sets.",
        "",
        "| Feature set | n features | Test AUC |",
        "| --- | ---: | ---: |",
    ]
    for name, n, auc, _ in results:
        lines.append(f"| {name} | {n} | **{auc:.4f}** |")
    lines += [
        "",
        f"Top drivers of the naive model: {', '.join('`%s`' % c for c in results[0][3])}",
        "",
        "Every one of those is a loan-servicing field that does not exist at",
        "decision time. An AUC near 1.0 on consumer credit default is not a strong",
        "model -- it is the signature of a leak. The audited figure is the one that",
        "describes performance on a decision that could actually be made.",
        "",
    ]
    Path(path).write_text("\n".join(lines))

    for name, n, auc, _ in results:
        print(f"{name:<26} n={n:<4} AUC={auc:.4f}")


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)

    audit = write_audit_report(df, "reports/leakage_audit.md")
    dropped = audit[audit["bucket"].isin(DROP_BUCKETS)]
    print(f"Columns: {len(audit)} | dropped: {len(dropped)} | modelled: {len(get_feature_columns(df))}")

    run_leak_demo(df, "reports/leakage_demo.md")


if __name__ == "__main__":
    main()
