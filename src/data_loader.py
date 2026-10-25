"""Load and parse NASA C-MAPSS turbofan data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COLUMNS,
    DATA_RAW,
    EXPECTED_COUNTS,
    SEED,
    SUBSETS,
)


def _read_txt(path: Path) -> pd.DataFrame:
    """Read space-separated C-MAPSS file into DataFrame."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    # Drop trailing empty columns if any
    df = df.dropna(axis=1, how="all")
    return df


def load_subset(subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load train, test, and true RUL for one subset (e.g. 'FD001').

    Returns
    -------
    train_df, test_df, rul_true
        rul_true has shape (n_test_engines,)
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset}. Choose from {SUBSETS}")

    train_path = DATA_RAW / f"train_{subset}.txt"
    test_path = DATA_RAW / f"test_{subset}.txt"
    rul_path = DATA_RAW / f"RUL_{subset}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Place C-MAPSS files in {DATA_RAW}"
            )

    train = _read_txt(train_path)
    test = _read_txt(test_path)
    rul = pd.read_csv(rul_path, header=None).values.ravel().astype(np.float64)

    return train, test, rul


def verify_counts(subset: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Assert engine counts match official documentation."""
    n_train = train["unit"].nunique()
    n_test = test["unit"].nunique()
    expected = EXPECTED_COUNTS[subset]
    print(f"{subset}: train engines={n_train}, test engines={n_test}")
    if (n_train, n_test) != expected:
        print(
            f"  WARNING: expected {expected}, got ({n_train}, {n_test}). "
            "Continuing anyway."
        )
    else:
        print(f"  ✓ counts match expected {expected}")


def load_all() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Load every subset and verify counts."""
    data = {}
    for subset in SUBSETS:
        train, test, rul = load_subset(subset)
        verify_counts(subset, train, test)
        data[subset] = (train, test, rul)
    return data


if __name__ == "__main__":
    np.random.seed(SEED)
    data = load_all()
    for subset, (train, test, rul) in data.items():
        print(
            f"{subset} shapes: train={train.shape}, test={test.shape}, "
            f"RUL={rul.shape}"
        )
