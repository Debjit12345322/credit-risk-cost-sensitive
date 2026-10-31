"""
Single source of truth for paths, seeds and fixed constants.

Every module imports from here rather than hardcoding values, so changing the
vintage, the seed or a file location happens in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Kaggle filename, unchanged, so `kaggle datasets download` drops it in place.
RAW_FILE = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
KAGGLE_DATASET = "wordsforthewise/lending-club"

VINTAGE_YEAR = 2014
DATASET_FILE = DATA_DIR / f"loans_{VINTAGE_YEAR}.parquet"
TARGET_AUDIT_FILE = REPORTS_DIR / "target_audit.json"

# ---------------------------------------------------------------- randomness
# Primary seed: every table in the README comes from this seed.
SEED = 42
# Additional seeds for the robustness check, which reports mean and spread
# across different train/test splits instead of trusting a single split.
ROBUSTNESS_SEEDS = [42, 7, 123, 2024, 31337]

# ---------------------------------------------------------------- evaluation
TEST_SIZE = 0.25
N_CV_FOLDS = 5

# Threshold grid for expected-cost sweeps. Step 0.005 is finer than any
# difference worth acting on; the grid stays well inside (0, 1).
THRESHOLD_GRID_STEP = 0.005

# Segments smaller than this in TRAINING fall back to the global threshold
# rather than fitting a threshold to a few hundred loans.
MIN_SEGMENT_SIZE = 500

# ---------------------------------------------------------------- streaming
CHUNK_SIZE = 200_000

# ---------------------------------------------------------------- model
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)
