import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.ensemble import ensemble_proba
from src.features import prepare, to_features
from src.split import load_splits

st.set_page_config(page_title="Fraud Detection", layout="wide")

DATA = ROOT / "data" / "creditcard.csv"
MODEL = ROOT / "models" / "ensemble.joblib"
NAMES = ["Logistic regression", "Random forest", "XGBoost"]

if not MODEL.exists() or not DATA.exists():
    st.error("Needs models/ensemble.joblib and data/creditcard.csv (run the notebook first).")
    st.stop()


@st.cache_resource
def load_artifacts():
    art = joblib.load(MODEL)
    return art, shap.TreeExplainer(art["models"][2])  # XGBoost component


@st.cache_data
def load_data():
    _, val, test = load_splits(DATA)
    return val.reset_index(drop=True), test.reset_index(drop=True)


@st.cache_data
def validation_scores():
    art, _ = load_artifacts()
    val, _ = load_data()
    X, y = prepare(val)
    return ensemble_proba(art["models"], X[art["features"]]), y.values


art, explainer = load_artifacts()
val_df, test_df = load_data()
default_t = float(art["threshold"])

st.title("Credit card fraud detection")
st.sidebar.header("About")
st.sidebar.write(
    "Soft-voting ensemble of logistic regression, random forest and XGBoost, "
    f"trained on the first 70% of a 48-hour transaction log. Saved threshold: {default_t}."
)
st.sidebar.caption(
    "Features V1-V28 are anonymised, so explanations show what the model relies on, "
    "not what causes fraud."
)

tab1, tab2, tab3 = st.tabs(["Score a transaction", "Upload a CSV", "Threshold explorer"])

# ---------- Tab 1: single transaction ----------
with tab1:
    st.caption("Examples come from the held-out test period (hours 42-48).")
    kind = st.radio("Show a", ["Fraud", "Normal"], horizontal=True)
    subset = test_df[test_df["Class"] == (1 if kind == "Fraud" else 0)].head(100)
    pick = st.selectbox("Transaction", subset.index.tolist(),
                        format_func=lambda i: f"row {i}  |  amount {test_df.loc[i, 'Amount']:.2f}")

    row = test_df.loc[[pick]]
    X_row = to_features(row)[art["features"]]
    per_model = [m.predict_proba(X_row)[0, 1] for m in art["models"]]
    p = float(np.mean(per_model))

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{p:.3f}")
    c2.metric("Decision", "FLAG" if p >= default_t else "OK")
    c3.metric("Actual label in dataset", "Fraud" if row["Class"].iloc[0] == 1 else "Normal")

    st.subheader("Each model's score")
    st.dataframe(pd.DataFrame({"model": NAMES, "fraud probability": [f"{x:.4f}" for x in per_model]}),
                 hide_index=True)

    st.subheader("Why? (XGBoost component only)")
    sv = explainer(X_row)
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Bars are log-odds contributions. Red pushes toward fraud, blue toward normal.")

# ---------- Tab 2: upload ----------
with tab2:
    st.write("Upload a CSV with columns Amount and V1-V28 (Time and Class are optional).")
    up = st.file_uploader("CSV file", type="csv")
    thr = st.slider("Decision threshold", 0.05, 0.95, default_t, 0.05, key="upload_thr")
    if up is not None:
        raw = pd.read_csv(up)
        X = to_features(raw)
        missing = [c for c in art["features"] if c not in X.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            scores = ensemble_proba(art["models"], X[art["features"]])
            out = raw.copy()
            out["fraud_probability"] = np.round(scores, 4)
            out["flagged"] = scores >= thr
            st.write(f"{len(out)} transactions, {int(out['flagged'].sum())} flagged at {thr:.2f}")
            if "Class" in raw.columns:
                tn, fp, fn, tp = confusion_matrix(raw["Class"], out["flagged"], labels=[0, 1]).ravel()
                st.write(f"Caught {tp}, missed {fn}, false alarms {fp}")
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(200))
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv")

# ---------- Tab 3: threshold explorer (validation only) ----------
with tab3:
    st.caption("Uses the validation split only. The test set is never used to choose a threshold.")
    p_val, y_val = validation_scores()
    t = st.slider("Threshold", 0.05, 0.95, default_t, 0.05, key="explore_thr")
    tn, fp, fn, tp = confusion_matrix(y_val, p_val >= t, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Caught", int(tp))
    m2.metric("Missed", int(fn))
    m3.metric("False alarms", int(fp))
    m4.metric("Precision / recall", f"{precision:.2f} / {recall:.2f}")

    pr, rc, _ = precision_recall_curve(y_val, p_val)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rc, pr)
    ax.scatter([recall], [precision], color="red", zorder=3, label=f"threshold {t:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
