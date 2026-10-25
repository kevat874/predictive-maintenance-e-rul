"""Anomaly detection / health index via Isolation Forest on early-life data."""
from __future__ import annotations

from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import MODELS_DIR, SEED


def train_isolation_forest(
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    early_cycles: int = 30,
    contamination: float = 0.05,
) -> IsolationForest:
    """Train on early-life (healthy) cycles of training engines."""
    early = (
        train_df.groupby("unit", group_keys=False)
        .apply(lambda g: g.nsmallest(early_cycles, "cycle"))
    )
    X = early[sensor_cols].values
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def health_index(
    model: IsolationForest, df: pd.DataFrame, sensor_cols: List[str]
) -> np.ndarray:
    """
    Higher score = more anomalous (worse health).
    IsolationForest decision_function: higher = more normal, so we negate.
    """
    scores = -model.decision_function(df[sensor_cols].values)
    # Normalize to [0, 1] roughly
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores.astype(np.float32)


def add_health_index(
    df: pd.DataFrame,
    model: IsolationForest,
    sensor_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out["health_index"] = health_index(model, out, sensor_cols)
    return out


def save_anomaly_model(model: IsolationForest, subset: str) -> None:
    joblib.dump(model, MODELS_DIR / f"anomaly_{subset}.joblib")


def load_anomaly_model(subset: str) -> IsolationForest:
    return joblib.load(MODELS_DIR / f"anomaly_{subset}.joblib")
