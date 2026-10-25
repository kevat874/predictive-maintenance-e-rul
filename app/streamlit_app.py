"""
Streamlit dashboard for Predictive Maintenance RUL.
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, DATA_PROCESSED, RUL_CAP, SUBSETS
from src.data_loader import load_subset
from src.preprocessing import preprocess_subset
from src.features import (
    add_classical_features,
    create_sliding_windows,
    get_classical_feature_matrix,
    get_window_len,
)
from src.models_deep import LSTMModel, CNNLSTM, predict_deep, DEVICE
from src.anomaly import health_index
from src.decision import recommend_action
from src.explain import shap_for_engine
from src.uncertainty import mc_dropout_predict

st.set_page_config(page_title="Predictive Maintenance RUL", layout="wide")
st.title("🔧 Predictive Maintenance – Remaining Useful Life")
st.caption("NASA C-MAPSS turbofan engines (SIMULATED data)")


@st.cache_resource
def load_models(subset: str):
    artifacts = {}
    # Classical
    for name in ("xgb", "rf", "linear"):
        p = MODELS_DIR / f"{name}_{subset}.joblib"
        if p.exists():
            artifacts[name] = joblib.load(p)
    # Fail classifier
    p = MODELS_DIR / f"fail_clf_{subset}.joblib"
    if p.exists():
        artifacts["fail_clf"] = joblib.load(p)
    # Anomaly
    p = MODELS_DIR / f"anomaly_{subset}.joblib"
    if p.exists():
        artifacts["anomaly"] = joblib.load(p)
    # Prep
    p = MODELS_DIR / f"prep_artifacts_{subset}.joblib"
    if p.exists():
        artifacts["prep"] = joblib.load(p)
    # Deep
    for name in ("lstm", "cnn_lstm"):
        meta_p = MODELS_DIR / f"{name}_{subset}_meta.json"
        wt_p = MODELS_DIR / f"{name}_{subset}.pt"
        if meta_p.exists() and wt_p.exists():
            with open(meta_p) as f:
                meta = json.load(f)
            if name == "lstm":
                m = LSTMModel(meta["n_features"]).to(DEVICE)
            else:
                m = CNNLSTM(meta["n_features"]).to(DEVICE)
            m.load_state_dict(torch.load(wt_p, map_location=DEVICE))
            m.eval()
            artifacts[name] = m
            artifacts[f"{name}_meta"] = meta
    return artifacts


@st.cache_data
def load_processed(subset: str):
    train_raw, test_raw, rul = load_subset(subset)
    train, test, art = preprocess_subset(train_raw, test_raw, subset, save=False)
    return train, test, rul, art


def main():
    subset = st.sidebar.selectbox("Subset", SUBSETS, index=0)
    models = load_models(subset)

    if not models:
        st.warning(
            f"No trained models found for {subset}. "
            "Run: `python -m src.train --subset FD001 --model all`"
        )
        return

    train, test, rul_true, art = load_processed(subset)
    sensor_cols = art["feature_cols"]
    engines = sorted(test["unit"].unique())
    engine = st.sidebar.selectbox("Test engine", engines, index=0)
    model_choice = st.sidebar.selectbox(
        "Primary model",
        [m for m in ("xgb", "lstm", "rf", "cnn_lstm", "linear") if m in models],
        index=0,
    )

    eng_df = test[test["unit"] == engine].sort_values("cycle")
    true_rul = float(rul_true[list(engines).index(engine)]) if engine in engines else None

    st.subheader(f"Engine {engine} ({subset})")
    col1, col2, col3 = st.columns(3)

    # Sensor trends
    with st.expander("Sensor trends", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 3))
        for s in sensor_cols[:6]:
            ax.plot(eng_df["cycle"], eng_df[s], label=s, alpha=0.8)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Normalized sensor")
        ax.legend(ncol=3, fontsize=8)
        ax.set_title(f"Sensors – engine {engine}")
        st.pyplot(fig)
        plt.close()

    # Predictions
    y_pred = None
    lo = hi = None
    shap_dict = None

    if model_choice in ("xgb", "rf", "linear") and model_choice in models:
        test_f = add_classical_features(test, sensor_cols)
        eng_f = test_f[test_f["unit"] == engine].sort_values("cycle")
        last = eng_f.iloc[[-1]]
        X, _, _ = get_classical_feature_matrix(last, sensor_cols, target_col=None)
        model = models[model_choice]
        y_pred = float(model.predict(X)[0])

        if model_choice == "xgb":
            meta = {"unit", "cycle", "RUL", "op_cluster"} | {
                c for c in eng_f.columns if c.startswith("setting_")
            }
            feat_names = [c for c in eng_f.columns if c not in meta]
            try:
                shap_dict = shap_for_engine(model, X, feat_names, top_k=8)
            except Exception:
                shap_dict = None

    elif model_choice in ("lstm", "cnn_lstm") and model_choice in models:
        wlen = get_window_len(subset)
        X_te, _, _ = create_sliding_windows(
            eng_df, sensor_cols, wlen, target_col=None, last_only=True
        )
        model = models[model_choice]
        y_pred = float(predict_deep(model, X_te)[0])
        if model_choice == "lstm":
            mean, lo_arr, hi_arr = mc_dropout_predict(model, X_te, n_samples=15)
            lo, hi = float(lo_arr[0]), float(hi_arr[0])
            y_pred = float(mean[0])

    # Health index
    hi_val = 0.0
    if "anomaly" in models:
        scores = health_index(models["anomaly"], eng_df, sensor_cols)
        hi_val = float(scores[-1])
        with st.expander("Health index curve", expanded=True):
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.plot(eng_df["cycle"], scores, color="crimson")
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Health index (↑ worse)")
            ax.set_title("Anomaly / health index")
            st.pyplot(fig)
            plt.close()

    # Risk
    fails_prob = 0.0
    if "fail_clf" in models and model_choice in ("xgb", "rf", "linear"):
        test_f = add_classical_features(test, sensor_cols)
        last = test_f[test_f["unit"] == engine].sort_values("cycle").iloc[[-1]]
        X, _, _ = get_classical_feature_matrix(last, sensor_cols, target_col=None)
        fails_prob = float(models["fail_clf"].predict_proba(X)[0, 1])

    risk = "High" if fails_prob > 0.5 or (y_pred is not None and y_pred < 30) else (
        "Medium" if fails_prob > 0.3 or (y_pred is not None and y_pred < 60) else "Low"
    )

    with col1:
        st.metric("Predicted RUL", f"{y_pred:.1f}" if y_pred is not None else "N/A")
        if lo is not None:
            st.caption(f"80% interval: [{lo:.1f}, {hi:.1f}]")
    with col2:
        st.metric("True RUL (capped)", f"{min(true_rul, RUL_CAP):.0f}" if true_rul else "N/A")
        st.metric("Fail-within-30 prob", f"{fails_prob:.2f}")
    with col3:
        st.metric("Health index", f"{hi_val:.3f}")
        st.metric("Risk class", risk)

    if y_pred is not None:
        rec = recommend_action(y_pred, fails_prob, hi_val)
        st.info(f"**Recommended action: {rec['action']}** — {rec['reason']}")
        st.caption(rec["cost_assumptions"])

    if shap_dict:
        st.subheader("Top SHAP features (this engine)")
        shap_df = pd.DataFrame(
            {"feature": list(shap_dict.keys()), "shap": list(shap_dict.values())}
        )
        st.bar_chart(shap_df.set_index("feature"))


if __name__ == "__main__":
    main()
