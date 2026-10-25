"""Unit tests for RUL labeling, windowing, NASA score, no test leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RUL_CAP, SEED, EXPECTED_COUNTS
from src.evaluate import nasa_score, rmse
from src.features import create_sliding_windows
from src.preprocessing import add_rul_labels


def test_rul_labeling():
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    out = add_rul_labels(df, cap=125)
    assert list(out.loc[out.unit == 1, "RUL"]) == [2, 1, 0]
    assert list(out.loc[out.unit == 2, "RUL"]) == [1, 0]


def test_rul_cap():
    df = pd.DataFrame(
        {
            "unit": [1] * 200,
            "cycle": list(range(1, 201)),
            "s1": np.random.randn(200),
        }
    )
    out = add_rul_labels(df, cap=RUL_CAP)
    assert out["RUL"].max() == RUL_CAP
    assert out["RUL"].min() == 0


def test_windowing_shape():
    n_cycles = 40
    n_feat = 5
    df = pd.DataFrame(
        {
            "unit": [1] * n_cycles,
            "cycle": list(range(1, n_cycles + 1)),
            **{f"s{i}": np.random.randn(n_cycles) for i in range(n_feat)},
            "RUL": list(range(n_cycles - 1, -1, -1)),
        }
    )
    sensor_cols = [f"s{i}" for i in range(n_feat)]
    wlen = 30
    X, y, units = create_sliding_windows(
        df, sensor_cols, wlen, target_col="RUL", last_only=False
    )
    assert X.shape == (n_cycles - wlen + 1, wlen, n_feat)
    assert y.shape == (n_cycles - wlen + 1,)
    assert len(units) == X.shape[0]

    X2, y2, u2 = create_sliding_windows(
        df, sensor_cols, wlen, target_col="RUL", last_only=True
    )
    assert X2.shape == (1, wlen, n_feat)
    assert y2[0] == 0


def test_windowing_pad_short():
    df = pd.DataFrame(
        {
            "unit": [1] * 10,
            "cycle": list(range(1, 11)),
            "s0": np.arange(10.0),
            "RUL": list(range(9, -1, -1)),
        }
    )
    X, y, _ = create_sliding_windows(
        df, ["s0"], window_len=30, target_col="RUL", last_only=True
    )
    assert X.shape == (1, 30, 1)
    assert np.allclose(X[0, :20, 0], 0.0)
    assert np.allclose(X[0, 20:, 0], np.arange(10.0))


def test_nasa_score():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert nasa_score(y_true, y_pred) == pytest.approx(0.0)

    y_pred_late = np.array([15.0, 20.0, 30.0])
    y_pred_early = np.array([5.0, 20.0, 30.0])
    s_late = nasa_score(y_true, y_pred_late)
    s_early = nasa_score(y_true, y_pred_early)
    assert s_late > 0 and s_early > 0
    assert s_late > s_early


def test_rmse():
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_expected_counts():
    assert EXPECTED_COUNTS["FD001"] == (100, 100)
    assert EXPECTED_COUNTS["FD002"] == (260, 259)
