"""Classical rolling features and deep sliding-window construction."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import WINDOW_LEN_DEEP, WINDOW_SIZES_CLASSICAL


def _fast_rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized approximate slope via (last - first) / window."""
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out
    for i in range(1, n):
        start = max(0, i - window + 1)
        seg = values[start : i + 1]
        out[i] = (seg[-1] - seg[0]) / max(len(seg) - 1, 1)
    return out


def add_classical_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Per-engine rolling mean/std/slope, baseline difference, cycle.
    Applied independently per unit so no leakage across engines.
    Uses a reduced set of windows for speed: [5, 20].
    """
    if windows is None:
        windows = [5, 20]  # faster than 5/10/30

    # Limit sensors used for rolling features if many
    use_sensors = sensor_cols[:14] if len(sensor_cols) > 14 else sensor_cols

    parts = []
    for unit, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle").reset_index(drop=True)
        baseline = g[use_sensors].iloc[:5].mean()

        new_cols = {}
        for s in use_sensors:
            vals = g[s].values.astype(np.float32)
            new_cols[f"{s}_baseline_diff"] = vals - float(baseline[s])
            for w in windows:
                s_ser = pd.Series(vals)
                new_cols[f"{s}_rmean_{w}"] = s_ser.rolling(w, min_periods=1).mean().values
                new_cols[f"{s}_rstd_{w}"] = (
                    s_ser.rolling(w, min_periods=1).std().fillna(0).values
                )
                new_cols[f"{s}_rslope_{w}"] = _fast_rolling_slope(vals, w)

        feat_df = pd.DataFrame(new_cols)
        parts.append(pd.concat([g, feat_df], axis=1))

    out = pd.concat(parts, ignore_index=True)
    return out


def get_classical_feature_matrix(
    df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str = "RUL",
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Build X, y, units from a dataframe that already has classical features."""
    meta = {"unit", "cycle", "RUL", "op_cluster"} | set(
        c for c in df.columns if c.startswith("setting_")
    )
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].values.astype(np.float32)
    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[target_col].values.astype(np.float32) if target_col in df.columns else None
    units = df["unit"].values
    return X, y, units


def create_sliding_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_len: int,
    target_col: Optional[str] = "RUL",
    last_only: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Build (n_samples, window_len, n_features) tensors.
    """
    Xs, ys, units_list = [], [], []

    for unit, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        sensors = g[sensor_cols].values.astype(np.float32)
        n = len(sensors)

        if last_only:
            if n >= window_len:
                window = sensors[-window_len:]
            else:
                pad = np.repeat(sensors[:1], window_len - n, axis=0)
                window = np.vstack([pad, sensors])
            Xs.append(window)
            units_list.append(unit)
            if target_col and target_col in g.columns:
                ys.append(g[target_col].iloc[-1])
        else:
            if n < window_len:
                continue
            for i in range(window_len - 1, n):
                Xs.append(sensors[i - window_len + 1 : i + 1])
                units_list.append(unit)
                if target_col and target_col in g.columns:
                    ys.append(g[target_col].iloc[i])

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.float32) if ys else None
    units = np.array(units_list)
    return X, y, units


def get_window_len(subset: str) -> int:
    return WINDOW_LEN_DEEP.get(subset, 30)
