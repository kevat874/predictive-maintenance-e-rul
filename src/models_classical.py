"""Classical ML models: Linear Regression, Random Forest, XGBoost."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.config import SEED


def train_linear(
    X: np.ndarray, y: np.ndarray, **kwargs
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 12,
    **kwargs,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 30,
    **kwargs,
) -> XGBRegressor:
    """Train XGBoost with optional GroupKFold early-stopping validation."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=SEED,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    if groups is not None and early_stopping_rounds:
        gkf = GroupKFold(n_splits=5)
        # Use first fold for early stopping
        train_idx, val_idx = next(gkf.split(X, y, groups))
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
    else:
        model.fit(X, y, verbose=False)

    return model


CLASSICAL_TRAINERS = {
    "linear": train_linear,
    "rf": train_rf,
    "xgb": train_xgb,
}


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X).astype(np.float32)
