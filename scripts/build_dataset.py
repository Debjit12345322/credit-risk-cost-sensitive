"""Build the 2014 vintage dataset and record the target-definition audit."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import load_vintage

df, audit = load_vintage("data/accepted_full.csv.gz", year=2014)

Path("reports").mkdir(exist_ok=True)
with open("reports/target_audit.json", "w") as f:
    json.dump(audit, f, indent=2, default=str)

df.to_parquet("data/loans_2014.parquet", index=False)

print(f"{len(df):,} loans | default rate {df.target.mean():.4f}")
print()
print("Status counts in vintage:")
for status, n in sorted(audit["status_counts_in_vintage"].items(), key=lambda x: -x[1]):
    print(f"  {status:<25} {n:>7,}")
