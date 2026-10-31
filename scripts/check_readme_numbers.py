"""
Check that every number quoted in README.md exists in a generated report.

    python scripts/check_readme_numbers.py

The README is written by hand; the reports are written by code. This catches
the README drifting away from what the code actually produces -- for example
after re-running on a different machine, where floating-point details can move
the last decimal of a LightGBM result.

A number passes if it appears in reports/ as a whole number (not inside a
longer one), or if it is a percentage
written with a % sign whose fraction appears there (README "18.45%" <->
report "0.1845"). Plain numbers never match as fractions, otherwise "5.06"
would pass as "0.051". This is a
sanity check, not a proof: it verifies a number exists in the reports, not that
it is quoted in the right place. The marked version-history block is skipped:
it quotes v1 results, which are sourced from the v1-single-split git tag.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Numbers the README states that are NOT verbatim analysis outputs, each with
# its source. Rounded summaries are declared here rather than passing by luck.
ALLOWED = {
    "374": "raw file size in MB",
    "3.12": "minimum Python version",
    "0.72": "overall AUC rounded (0.7206)",
    "0.63": "within-grade AUC rounded (0.6319-0.6452, grade G excluded)",
    "0.4": "where the cost curve flattens (figures/cost_curve.png)",
    "1.3": "largest non-top calibration bin gap rounded (1.26 points)",
    "0.15": "bound on |mean predicted - base rate| (0.07 and 0.13 points)",
}


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    # The version-history block quotes v1's historical numbers, which live in
    # the v1-single-split tag rather than in the current reports.
    readme = re.sub(r"<!-- v1-start -->.*?<!-- v1-end -->", "", readme, flags=re.S)
    reports = "\n".join(p.read_text() for p in (ROOT / "reports").glob("*.md"))
    reports += (ROOT / "reports" / "results.json").read_text()

    def present(text: str) -> bool:
        # Whole-number match: "1.75" must not count as found inside "971.75".
        return re.search(rf"(?<![\d.,]){re.escape(text)}(?!\d)", reports) is not None

    def found(token: str, is_percent: bool) -> bool:
        if token in ALLOWED or present(token):
            return True
        if not is_percent:
            return False
        # Only a number written with % may match its fraction ("18.45%" <-> "0.1845").
        value = float(token.replace(",", "")) / 100
        return any(present(f"{value:.{d}f}") for d in (3, 4))

    # Comma-grouped numbers first ("9,077", "2,381.91"), then decimals, then
    # plain integers of 3+ digits.
    pattern = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,}"
    tokens = sorted({(m.group(), readme[m.end():m.end() + 1] == "%")
                     for m in re.finditer(pattern, readme)})
    missing = sorted({t for t, pct in tokens if not found(t, pct)})
    if missing:
        print(f"README quotes {len(missing)} number(s) not found in reports/: {missing}")
        print("Re-run `make analysis` and update the README from reports/results.md.")
        return 1
    print(f"OK: all {len(tokens)} numbers in README.md are present in reports/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())