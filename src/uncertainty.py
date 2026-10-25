"""Uncertainty estimation via MC Dropout for deep models."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models_deep import DEVICE


def enable_dropout(model: nn.Module) -> None:
    """Keep dropout layers active during eval for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray,
    n_samples: int = 30,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns mean, lower (10th), upper (90th) percentiles → ~80% interval.
    """
    model.eval()
    enable_dropout(model)

    all_preds = []
    for _ in range(n_samples):
        preds = []
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
        all_preds.append(np.concatenate(preds))

    stack = np.stack(all_preds, axis=0)  # (n_samples, n)
    mean = stack.mean(axis=0).astype(np.float32)
    lower = np.percentile(stack, 10, axis=0).astype(np.float32)
    upper = np.percentile(stack, 90, axis=0).astype(np.float32)
    return mean, lower, upper


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of true values inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
