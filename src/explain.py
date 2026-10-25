"""Explainability: SHAP for tree models, permutation importance for deep."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import FIGURES_DIR, SEED


def shap_summary_xgb(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 15,
    save_path: Optional[Path] = None,
) -> shap.Explanation:
    """SHAP summary plot for XGBoost."""
    # Use a sample for speed
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    Xs = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    plt.figure()
    shap.summary_plot(
        shap_values,
        Xs,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return shap.Explanation(
        values=shap_values,
        data=Xs,
        feature_names=feature_names,
    )


def shap_for_engine(
    model,
    X_engine: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> dict:
    """Top SHAP features for a single engine (last window / last row)."""
    explainer = shap.TreeExplainer(model)
    # X_engine shape (1, n_features) or (n_features,)
    x = X_engine.reshape(1, -1)
    sv = explainer.shap_values(x)[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return {
        feature_names[i]: float(sv[i]) for i in order
    }


def permutation_importance_deep(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
) -> dict:
    """Simple permutation importance for deep models (sensor-level)."""
    from src.models_deep import predict_deep

    base_pred = predict_deep(model, X)
    base_rmse = float(np.sqrt(np.mean((base_pred - y) ** 2)))

    n_feat = X.shape[-1]
    importances = {}
    rng = np.random.RandomState(SEED)

    for f in range(n_feat):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # Permute feature f across samples
            Xp[:, :, f] = rng.permutation(Xp[:, :, f].ravel()).reshape(Xp.shape[0], Xp.shape[1])
            pred = predict_deep(model, Xp)
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            deltas.append(rmse - base_rmse)
        name = feature_names[f] if f < len(feature_names) else f"f{f}"
        importances[name] = float(np.mean(deltas))

    return dict(sorted(importances.items(), key=lambda kv: -kv[1]))
