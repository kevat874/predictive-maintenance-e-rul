"""Preprocessing: RUL labels, constant-sensor drop, normalization, clustering."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    N_CLUSTERS,
    OP_SETTING_COLS,
    RUL_CAP,
    SEED,
)


def add_rul_labels(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add RUL column: max_cycle - cycle, then piecewise-linear cap."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def identify_constant_sensors(
    df: pd.DataFrame, threshold: float = 1e-6
) -> List[str]:
    """Return sensor columns with near-zero variance."""
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    stds = df[sensor_cols].std()
    return stds[stds < threshold].index.tolist()


def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def fit_cluster_model(
    train_df: pd.DataFrame, n_clusters: int = N_CLUSTERS
) -> KMeans:
    """Fit KMeans on operating settings of TRAIN only (FD002/FD004)."""
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(train_df[OP_SETTING_COLS].values)
    return km


def assign_clusters(df: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    out = df.copy()
    out["op_cluster"] = km.predict(out[OP_SETTING_COLS].values).astype(np.int32)
    return out


def fit_scalers_single_condition(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    method: str = "minmax",
) -> object:
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float64))
    return scaler


def fit_scalers_per_cluster(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    method: str = "minmax",
) -> Dict[int, object]:
    scalers = {}
    for c in sorted(train_df["op_cluster"].unique()):
        mask = train_df["op_cluster"] == c
        if method == "minmax":
            sc = MinMaxScaler()
        else:
            sc = StandardScaler()
        sc.fit(train_df.loc[mask, feature_cols].values.astype(np.float64))
        scalers[int(c)] = sc
    return scalers


def transform_single(
    df: pd.DataFrame, scaler: object, feature_cols: List[str]
) -> pd.DataFrame:
    out = df.copy()
    vals = scaler.transform(out[feature_cols].values.astype(np.float64))
    out[feature_cols] = vals.astype(np.float64)
    return out


def transform_per_cluster(
    df: pd.DataFrame,
    scalers: Dict[int, object],
    feature_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    # Ensure float dtype for sensor columns
    for c in feature_cols:
        out[c] = out[c].astype(np.float64)
    for cluster_id, sc in scalers.items():
        mask = out["op_cluster"] == cluster_id
        if mask.any():
            vals = sc.transform(out.loc[mask, feature_cols].values.astype(np.float64))
            out.loc[mask, feature_cols] = vals
    return out


def preprocess_subset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    subset: str,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    multi_condition = subset in ("FD002", "FD004")

    train = add_rul_labels(train)

    const = identify_constant_sensors(train)
    # Also drop constant settings if any
    setting_std = train[[c for c in train.columns if c.startswith("setting_")]].std()
    const_settings = setting_std[setting_std < 1e-6].index.tolist()
    const = list(set(const + const_settings))
    print(f"  Dropping constant columns: {const}")
    train = drop_columns(train, const)
    test = drop_columns(test, const)

    feature_cols = [c for c in train.columns if c.startswith("s")]

    artifacts: dict = {"constant_sensors": const, "feature_cols": feature_cols}

    if multi_condition:
        km = fit_cluster_model(train)
        train = assign_clusters(train, km)
        test = assign_clusters(test, km)
        scalers = fit_scalers_per_cluster(train, feature_cols)
        train = transform_per_cluster(train, scalers, feature_cols)
        test = transform_per_cluster(test, scalers, feature_cols)
        artifacts["kmeans"] = km
        artifacts["scalers"] = scalers
        if save:
            joblib.dump(km, MODELS_DIR / f"kmeans_{subset}.joblib")
            joblib.dump(scalers, MODELS_DIR / f"scalers_{subset}.joblib")
    else:
        scaler = fit_scalers_single_condition(train, feature_cols)
        train = transform_single(train, scaler, feature_cols)
        test = transform_single(test, scaler, feature_cols)
        artifacts["scaler"] = scaler
        if save:
            joblib.dump(scaler, MODELS_DIR / f"scaler_{subset}.joblib")

    if save:
        joblib.dump(artifacts, MODELS_DIR / f"prep_artifacts_{subset}.joblib")
        train.to_csv(DATA_PROCESSED / f"train_{subset}.csv", index=False)
        test.to_csv(DATA_PROCESSED / f"test_{subset}.csv", index=False)

    return train, test, artifacts


if __name__ == "__main__":
    from src.data_loader import load_subset

    for sub in ["FD001", "FD002"]:
        print(f"\n=== Preprocessing {sub} ===")
        tr, te, rul = load_subset(sub)
        tr_p, te_p, art = preprocess_subset(tr, te, sub)
        print(f"  train shape={tr_p.shape}, test shape={te_p.shape}")
        print(f"  feature_cols={art['feature_cols']}")
