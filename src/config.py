"""Configuration for Predictive Maintenance RUL project."""
from pathlib import Path
from typing import Dict, List

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure dirs exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
SEED = 42

# RUL cap (piecewise linear)
RUL_CAP = 125

# Window sizes
WINDOW_SIZES_CLASSICAL = [5, 10, 30]
WINDOW_LEN_DEEP = {
    "FD001": 30,
    "FD002": 50,
    "FD003": 30,
    "FD004": 50,
}

# Subsets
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Expected engine counts (train / test)
EXPECTED_COUNTS = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (249, 248),
}

# Column names
COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors known to be constant / near-constant on FD001/FD003
# (will be identified dynamically as well)
CONSTANT_SENSORS_CANDIDATES = [
    "s1", "s5", "s6", "s10", "s16", "s18", "s19"
]

# Operating condition clustering for multi-condition subsets
N_CLUSTERS = 6
OP_SETTING_COLS = ["setting_1", "setting_2", "setting_3"]

# Training hyperparams
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0

# Decision thresholds (cycles)
THRESH_OK = 60          # RUL > this -> OK
THRESH_INSPECT = 30     # RUL > this -> Schedule Inspection; else Urgent

# Cost model assumptions (documented in decision.py)
COST_UNPLANNED_FAILURE = 100.0
COST_EARLY_MAINTENANCE = 10.0
COST_INSPECTION = 2.0
