"""EDA script (run as notebook equivalent). Saves figures to reports/figures/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, SEED
from src.data_loader import load_subset
from src.preprocessing import add_rul_labels, identify_constant_sensors

np.random.seed(SEED)
sns.set_theme(style="whitegrid")


def main():
    train, test, rul = load_subset("FD001")
    train = add_rul_labels(train)

    # 1. Cycle-length distribution
    max_cycles = train.groupby("unit")["cycle"].max()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(max_cycles, bins=20, edgecolor="k", alpha=0.8)
    ax.set_xlabel("Engine lifetime (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("FD001 – Cycle-length distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_cycle_lengths_FD001.png", dpi=120)
    plt.close()

    # 2. Sensor trends for a few engines
    units = [1, 2, 3]
    sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, s in zip(axes.ravel(), sensors):
        for u in units:
            g = train[train.unit == u]
            ax.plot(g.cycle, g[s], label=f"eng {u}", alpha=0.8)
        ax.set_title(s)
        ax.legend(fontsize=7)
    fig.suptitle("Sensor trends vs cycle (FD001)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_sensor_trends_FD001.png", dpi=120)
    plt.close()

    # 3. Constant sensors
    const = identify_constant_sensors(train)
    print("Constant / near-constant sensors:", const)

    # 4. Correlation with RUL
    sensor_cols = [c for c in train.columns if c.startswith("s") and c not in const]
    corrs = train[sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    corrs.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Sensor correlation with RUL (FD001)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_corr_rul_FD001.png", dpi=120)
    plt.close()

    # 5. Operating condition clusters for FD002
    train2, _, _ = load_subset("FD002")
    from sklearn.cluster import KMeans
    from src.config import OP_SETTING_COLS, N_CLUSTERS

    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(train2[OP_SETTING_COLS])
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        train2["setting_1"],
        train2["setting_2"],
        c=labels,
        cmap="tab10",
        s=2,
        alpha=0.3,
    )
    ax.set_xlabel("setting_1")
    ax.set_ylabel("setting_2")
    ax.set_title("FD002 operating-condition clusters (k=6)")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_op_clusters_FD002.png", dpi=120)
    plt.close()

    print("EDA figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
