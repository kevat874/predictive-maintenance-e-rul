"""
Main training entry point.

Usage:
  python -m src.train --subset FD001 --model all
  python -m src.train --subset FD001 --model xgb
  python -m src.train --subset all --model xgb,lstm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly import (
    add_health_index,
    save_anomaly_model,
    train_isolation_forest,
)
from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RUL_CAP,
    SEED,
    SUBSETS,
)
from src.data_loader import load_subset
from src.evaluate import (
    evaluate_predictions,
    plot_error_hist,
    plot_pred_vs_true,
    plot_rul_trajectories,
    save_results_table,
)
from src.features import (
    add_classical_features,
    create_sliding_windows,
    get_classical_feature_matrix,
    get_window_len,
)
from src.models_classical import CLASSICAL_TRAINERS, predict as predict_classical
from src.models_deep import predict_deep, train_deep
from src.preprocessing import preprocess_subset
from src.uncertainty import interval_coverage, mc_dropout_predict

np.random.seed(SEED)
torch.manual_seed(SEED)


def _last_cycle_mask(df: pd.DataFrame) -> pd.Series:
    return df.groupby("unit")["cycle"].transform("max") == df["cycle"]


def train_classical_pipeline(
    subset: str,
    model_names: List[str],
    results: List[dict],
) -> Dict:
    """Train classical models, evaluate on last cycle of test engines."""
    print(f"\n{'='*60}\nClassical models on {subset}\n{'='*60}")
    train_raw, test_raw, rul_true = load_subset(subset)
    train, test, artifacts = preprocess_subset(train_raw, test_raw, subset)
    sensor_cols = artifacts["feature_cols"]

    # Classical features
    train_f = add_classical_features(train, sensor_cols)
    test_f = add_classical_features(test, sensor_cols)

    X_tr, y_tr, units_tr = get_classical_feature_matrix(train_f, sensor_cols)
    # Test: only last cycle per engine (order by unit matches RUL file)
    test_last = test_f.loc[_last_cycle_mask(test_f)].sort_values("unit")
    X_te, _, units_te = get_classical_feature_matrix(
        test_last, sensor_cols, target_col=None
    )

    # Align RUL
    assert len(X_te) == len(rul_true), (
        f"Test engines {len(X_te)} vs RUL {len(rul_true)}"
    )

    # Feature names for SHAP
    meta = {"unit", "cycle", "RUL", "op_cluster"} | {
        c for c in train_f.columns if c.startswith("setting_")
    }
    feat_names = [c for c in train_f.columns if c not in meta]

    trained = {}
    for name in model_names:
        if name not in CLASSICAL_TRAINERS:
            continue
        print(f"  Training {name} ...")
        trainer = CLASSICAL_TRAINERS[name]
        model = trainer(X_tr, y_tr, groups=units_tr)
        y_pred = predict_classical(model, X_te)

        metrics_cap = evaluate_predictions(rul_true, y_pred, cap=True)
        metrics_uncap = evaluate_predictions(rul_true, y_pred, cap=False)
        print(
            f"    RMSE(capped)={metrics_cap['rmse']:.2f}  "
            f"NASA={metrics_cap['nasa_score']:.1f}  "
            f"RMSE(uncapped)={metrics_uncap['rmse']:.2f}"
        )

        results.append(
            {
                "subset": subset,
                "model": name,
                "rmse_capped": metrics_cap["rmse"],
                "nasa_score": metrics_cap["nasa_score"],
                "mae_capped": metrics_cap["mae"],
                "rmse_uncapped": metrics_uncap["rmse"],
            }
        )

        # Save
        joblib.dump(model, MODELS_DIR / f"{name}_{subset}.joblib")
        trained[name] = {
            "model": model,
            "y_pred": y_pred,
            "feat_names": feat_names,
            "X_te": X_te,
            "units_te": units_te,
        }

        # Plots for first classical model of the subset
        if name == model_names[0] or name == "xgb":
            plot_pred_vs_true(
                np.clip(rul_true, None, RUL_CAP),
                y_pred,
                f"{name.upper()} – {subset}",
                FIGURES_DIR / f"pred_vs_true_{name}_{subset}.png",
            )
            plot_error_hist(
                np.clip(rul_true, None, RUL_CAP),
                y_pred,
                f"Error hist {name} {subset}",
                FIGURES_DIR / f"error_hist_{name}_{subset}.png",
            )

    # Failure-risk classifier (fails within 30 cycles) using XGB if available
    if "xgb" in trained:
        from xgboost import XGBClassifier

        y_fail = (y_tr <= 30).astype(int)
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf.fit(X_tr, y_fail)
        joblib.dump(clf, MODELS_DIR / f"fail_clf_{subset}.joblib")
        # Evaluate on test last cycle
        y_fail_true = (np.clip(rul_true, None, RUL_CAP) <= 30).astype(int)
        proba = clf.predict_proba(X_te)[:, 1]
        pred_fail = (proba >= 0.5).astype(int)
        print(
            f"  Fail-within-30 classifier: "
            f"P={precision_score(y_fail_true, pred_fail, zero_division=0):.3f} "
            f"R={recall_score(y_fail_true, pred_fail, zero_division=0):.3f} "
            f"F1={f1_score(y_fail_true, pred_fail, zero_division=0):.3f} "
            f"AP={average_precision_score(y_fail_true, proba):.3f}"
        )
        trained["fail_clf"] = clf

    # Anomaly / health index
    if_model = train_isolation_forest(train, sensor_cols)
    save_anomaly_model(if_model, subset)
    trained["anomaly"] = if_model

    # Sample trajectory plot
    sample_units = sorted(train["unit"].unique())[:3]
    plot_rul_trajectories(
        train,
        sample_units,
        sensor=sensor_cols[0] if sensor_cols else "s2",
        title=f"RUL trajectories {subset}",
        save_path=FIGURES_DIR / f"trajectories_{subset}.png",
    )

    trained["artifacts"] = artifacts
    trained["rul_true"] = rul_true
    trained["sensor_cols"] = sensor_cols
    return trained


def train_deep_pipeline(
    subset: str,
    model_names: List[str],
    results: List[dict],
) -> Dict:
    """Train deep models on sliding windows."""
    print(f"\n{'='*60}\nDeep models on {subset}\n{'='*60}")
    train_raw, test_raw, rul_true = load_subset(subset)
    train, test, artifacts = preprocess_subset(train_raw, test_raw, subset)
    sensor_cols = artifacts["feature_cols"]
    wlen = get_window_len(subset)

    X_tr, y_tr, units_tr = create_sliding_windows(
        train, sensor_cols, wlen, target_col="RUL", last_only=False
    )
    X_te, _, units_te = create_sliding_windows(
        test, sensor_cols, wlen, target_col=None, last_only=True
    )
    # Ensure unit order matches RUL file (sorted unit ids)
    order = np.argsort(units_te)
    X_te = X_te[order]
    units_te = units_te[order]
    assert len(X_te) == len(rul_true)

    print(f"  Window len={wlen}, X_tr={X_tr.shape}, X_te={X_te.shape}")

    trained = {}
    for name in model_names:
        if name not in ("lstm", "cnn_lstm"):
            continue
        print(f"  Training {name} ...")
        model = train_deep(X_tr, y_tr, units_tr, model_type=name)
        y_pred = predict_deep(model, X_te)

        metrics_cap = evaluate_predictions(rul_true, y_pred, cap=True)
        metrics_uncap = evaluate_predictions(rul_true, y_pred, cap=False)
        print(
            f"    RMSE(capped)={metrics_cap['rmse']:.2f}  "
            f"NASA={metrics_cap['nasa_score']:.1f}"
        )
        results.append(
            {
                "subset": subset,
                "model": name,
                "rmse_capped": metrics_cap["rmse"],
                "nasa_score": metrics_cap["nasa_score"],
                "mae_capped": metrics_cap["mae"],
                "rmse_uncapped": metrics_uncap["rmse"],
            }
        )

        torch.save(model.state_dict(), MODELS_DIR / f"{name}_{subset}.pt")
        # Also save architecture meta
        meta = {
            "n_features": X_tr.shape[-1],
            "window_len": wlen,
            "model_type": name,
        }
        with open(MODELS_DIR / f"{name}_{subset}_meta.json", "w") as f:
            json.dump(meta, f)

        trained[name] = {
            "model": model,
            "y_pred": y_pred,
            "X_te": X_te,
            "units_te": units_te,
        }

        plot_pred_vs_true(
            np.clip(rul_true, None, RUL_CAP),
            y_pred,
            f"{name.upper()} – {subset}",
            FIGURES_DIR / f"pred_vs_true_{name}_{subset}.png",
        )

        # MC Dropout uncertainty (only for lstm)
        if name == "lstm":
            mean, lo, hi = mc_dropout_predict(model, X_te, n_samples=20)
            cov = interval_coverage(np.clip(rul_true, None, RUL_CAP), lo, hi)
            print(f"    MC-Dropout 80% interval coverage: {cov:.3f}")
            trained["uncertainty"] = {"mean": mean, "lower": lo, "upper": hi, "coverage": cov}

    trained["artifacts"] = artifacts
    trained["rul_true"] = rul_true
    trained["sensor_cols"] = sensor_cols
    return trained


def parse_args():
    p = argparse.ArgumentParser(description="Train RUL models on C-MAPSS")
    p.add_argument(
        "--subset",
        default="FD001",
        help="FD001|FD002|FD003|FD004|all",
    )
    p.add_argument(
        "--model",
        default="all",
        help="Comma-separated: linear,rf,xgb,lstm,cnn_lstm,all",
    )
    return p.parse_args()


def main():
    args = parse_args()
    subsets = SUBSETS if args.subset.lower() == "all" else [args.subset.upper()]
    if args.model.lower() == "all":
        models = ["linear", "rf", "xgb", "lstm", "cnn_lstm"]
    else:
        models = [m.strip().lower() for m in args.model.split(",")]

    classical = [m for m in models if m in ("linear", "rf", "xgb")]
    deep = [m for m in models if m in ("lstm", "cnn_lstm")]

    results: List[dict] = []

    for subset in subsets:
        if classical:
            train_classical_pipeline(subset, classical, results)
        if deep:
            train_deep_pipeline(subset, deep, results)

    if results:
        save_results_table(results)
        print(f"\nResults saved to {REPORTS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
