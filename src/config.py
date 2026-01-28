"""
----
Global experiment configuration.
----
Centralize all experiment-wide settings to ensure:
    1) Reproducibility.
    2) Consistent paths.
    3) Clean separation between code and parameters.
"""

from pathlib import Path

# Resolve project root dynamically based on this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reproducibility.
RANDOM_STATE = 42

# Data paths.
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Subfolders inside results.
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create folders if non-existent.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "DDG_DATA.csv"
PROCESSED_PATH = DATA_DIR / "fireprotdb_processed_sample.csv"

# Sampling configuration.
SAMPLE_SIZE = 200_000
USE_SAMPLE = False
SAVE_PROCESSED = False

# Class imbalance handling.
DOWNSAMPLE = False