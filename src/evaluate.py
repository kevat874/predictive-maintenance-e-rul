"""Evaluation metrics: RMSE, NASA score, plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RUL_CAP, REPORTS_DIR, SEED


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetric scoring function.
    d = pred - true
    score = sum( exp(-d/13) - 1  if d < 0 else exp(d/10) - 1 )
    """
    d = y_pred - y_true
    return float(
        np.sum(
            np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
        )
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cap: bool = True,
) -> Dict[str, float]:
    yt = np.clip(y_true, None, RUL_CAP) if cap else y_true
    yp = np.clip(y_pred, 0, None)  # RUL cannot be negative
    return {
        "rmse": rmse(yt, yp),
        "nasa_score": nasa_score(yt, yp),
        "mae": float(np.mean(np.abs(yt - yp))),
    }


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_error_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=30, edgecolor="k", alpha=0.75)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Prediction error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_rul_trajectories(
    train_df: pd.DataFrame,
    unit_ids: List[int],
    sensor: str = "s2",
    title: str = "RUL trajectories",
    save_path: Optional[Path] = None,
) -> None:
    fig, axes = plt.subplots(1, len(unit_ids), figsize=(4 * len(unit_ids), 3.5))
    if len(unit_ids) == 1:
        axes = [axes]
    for ax, uid in zip(axes, unit_ids):
        g = train_df[train_df["unit"] == uid].sort_values("cycle")
        ax.plot(g["cycle"], g["RUL"], label="RUL")
        ax2 = ax.twinx()
        ax2.plot(g["cycle"], g[sensor], color="orange", alpha=0.6, label=sensor)
        ax.set_title(f"Engine {uid}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def save_results_table(rows: List[dict], path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = path or (REPORTS_DIR / "results.csv")
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["subset", "model"], keep="last"
        )
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    return df
