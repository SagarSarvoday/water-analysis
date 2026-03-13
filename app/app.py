"""
app.py  —  AI Water Quality Prediction System
=========================================================
Sections:
  1. Water parameter inputs
  2. Predict button
  3. Prediction Results  →  7 tabs:
       📈 Parameter Analysis
       🎯 Risk Gauge
       💡 Feature Selection
       1️⃣ Logistic Regression      (own prediction + metrics)
       2️⃣ Random Forest            (own prediction + metrics)
       3️⃣ Gradient Boosting        (own prediction + metrics)
       4️⃣ Improved LR              (own prediction + metrics)
  4. Model Evaluation  →  5 tabs:
       1️⃣ LR metrics
       2️⃣ RF metrics
       3️⃣ GB metrics
       4️⃣ Improved LR metrics
       📊 Full Comparison
  5. AI Report  (Groq)
  6. AI Assistant  (Groq)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
st.cache_resource.clear()
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from utils.risk_scorer      import analyze_parameters, calculate_risk_score
from utils.report_generator import generate_report, generate_assistant_response

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "water_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PLOT_DIR    = os.path.join(MODEL_DIR, "plots")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Water Quality System", page_icon="💧", layout="wide")

st.markdown("""
<style>
    .main-title    { font-size:2.2rem; font-weight:700; color:#1a6fa8; text-align:center; }
    .subtitle      { text-align:center; color:#666; font-size:1rem; margin-bottom:1.5rem; }
    .sec-header    { font-size:1.15rem; font-weight:700; color:#1a6fa8;
                     border-bottom:2px solid #1a6fa8; padding-bottom:4px; margin-top:1.2rem; }
    .result-safe   { background:#e8f5e9; border-left:6px solid #2e7d32; padding:.9rem 1.4rem;
                     border-radius:8px; font-size:1.05rem; font-weight:bold; color:#2e7d32; margin-bottom:.5rem; }
    .result-unsafe { background:#ffebee; border-left:6px solid #c62828; padding:.9rem 1.4rem;
                     border-radius:8px; font-size:1.05rem; font-weight:bold; color:#c62828; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💧 AI Water Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Logistic Regression · Random Forest · Gradient Boosting · Improved LR · AI Reports</div>', unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────

def load_all_models():
    scaler      = joblib.load(SCALER_PATH)
    final_model = joblib.load(MODEL_PATH)
    names_files = {
        "Logistic Regression":          "logistic_regression.pkl",
        "Random Forest":                "random_forest.pkl",
        "Gradient Boosting":            "gradient_boosting.pkl",
        "Improved Logistic Regression": "improved_logistic_regression.pkl",
    }
    individual = {}
    for name, fname in names_files.items():
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            individual[name] = joblib.load(p)
    rfe_selector = None
    rfe_path = os.path.join(MODEL_DIR, "rfe_selector.pkl")
    if os.path.exists(rfe_path):
        rfe_selector = joblib.load(rfe_path)
    return final_model, scaler, individual, rfe_selector
try:
   model, scaler, all_models, rfe_selector = load_all_models()
except Exception as e:
    st.error(f"⚠️ Models not found. Run `python model/train_model.py` first.\n\nError: {e}")
    st.stop()

# ── Load metrics ───────────────────────────────────────────────────────────────
def load_metrics():
    p = os.path.join(PLOT_DIR, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}

all_metrics = load_metrics()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Get free key at console.groq.com")
    st.markdown("---")
    st.markdown("### 📊 Sample Inputs")
    if st.button("🔴 Load Unsafe Sample"):
        st.session_state["sample"] = "unsafe"
    if st.button("🟢 Load Safe Sample"):
        st.session_state["sample"] = "safe"
    st.markdown("---")
    st.info("This system compares 4 ML models for water potability prediction with AI-powered reporting.")

# ── Defaults ───────────────────────────────────────────────────────────────────
UNSAFE = dict(ph=5.2, hardness=450, solids=50000, chloramines=10.0,
              sulfate=450, conductivity=750, organic_carbon=18.0,
              turbidity=7.5, trihalomethanes=110.0)
SAFE = dict(ph=7.2, hardness=180, solids=500,  chloramines=2.0,
            sulfate=200, conductivity=310, organic_carbon=3.0,
            turbidity=1.5, trihalomethanes=40.0)
DEF = dict(ph=7.0, hardness=200, solids=400, chloramines=2.0,
           sulfate=200, conductivity=300, organic_carbon=3.0,
           turbidity=2.0, trihalomethanes=50.0)
dv = (UNSAFE if st.session_state.get("sample")=="unsafe"
      else SAFE if st.session_state.get("sample")=="safe" else DEF)

# ── Input section ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">🧪 Enter Water Parameters</div>', unsafe_allow_html=True)
st.markdown("")
c1, c2, c3 = st.columns(3)
with c1:
    ph            = st.number_input("pH",                0.0, 14.0,    float(dv["ph"]),             0.1)
    hardness      = st.number_input("Hardness (mg/L)",   0.0, 600.0,   float(dv["hardness"]),       1.0)
    solids        = st.number_input("Solids (ppm)",      0.0, 60000.0, float(dv["solids"]),         100.0)
with c2:
    chloramines   = st.number_input("Chloramines (ppm)", 0.0, 15.0,    float(dv["chloramines"]),    0.1)
    sulfate       = st.number_input("Sulfate (mg/L)",    0.0, 700.0,   float(dv["sulfate"]),        1.0)
    conductivity  = st.number_input("Conductivity",      0.0, 800.0,   float(dv["conductivity"]),   1.0)
with c3:
    organic_carbon  = st.number_input("Organic Carbon",  0.0, 30.0,   float(dv["organic_carbon"]),  0.1)
    turbidity       = st.number_input("Turbidity (NTU)", 0.0, 10.0,   float(dv["turbidity"]),       0.1)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, 130.0,  float(dv["trihalomethanes"]), 0.1)

inputs = dict(ph=ph, hardness=hardness, solids=solids, chloramines=chloramines,
              sulfate=sulfate, conductivity=conductivity, organic_carbon=organic_carbon,
              turbidity=turbidity, trihalomethanes=trihalomethanes)

st.markdown("")
pcol, _ = st.columns([1, 3])
with pcol:
    predict_btn = st.button("🔍 Predict Water Quality", use_container_width=True)

# ── Run prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    arr    = np.array([[ph, hardness, solids, chloramines, sulfate,
                         conductivity, organic_carbon, turbidity, trihalomethanes]])
    scaled = scaler.transform(arr)

    # Final model
    final_proba = model.predict_proba(scaled)[0]
    pred   = 1 if float(final_proba[1]) >= 0.45 else 0
    proba  = final_proba

    # Individual model predictions
    # Individual model predictions
    ind_preds = {}
    for name, m in all_models.items():
        try:
            # Use RFE-transformed input for Improved Logistic Regression
            if "Improved" in name and rfe_selector is not None:
                input_for_model = rfe_selector.transform(scaled)
            else:
                input_for_model = scaled

            prob     = m.predict_proba(input_for_model)[0]
            safe_p   = float(prob[1]) if len(prob) > 1 else float(prob[0])
            unsafe_p = float(prob[0])
            threshold = 0.35 if "Logistic" in name else 0.45
            p = 1 if safe_p >= threshold else 0
            ind_preds[name] = {
                "prediction":  p,
                "probability": safe_p if p == 1 else unsafe_p,
                "safe_prob":   safe_p,
                "unsafe_prob": unsafe_p,
            }
        except Exception as ex:
            ind_preds[name] = {"prediction": -1, "probability": 0.0,
                               "safe_prob": 0.0, "unsafe_prob": 0.0}

    pa   = analyze_parameters(inputs)
    risk = calculate_risk_score(float(proba[0]), pa)

    st.session_state.update({
        "pred": int(pred), "proba": proba.tolist(),
        "probability": float(proba[pred]),
        "param_analysis": pa, "risk": risk,
        "inputs": inputs, "ind_preds": ind_preds,
        "chat_history": [], "ai_report": None,
    })

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if "pred" in st.session_state:
    pred     = st.session_state["pred"]
    proba    = st.session_state["proba"]
    prob     = st.session_state["probability"]
    pa       = st.session_state["param_analysis"]
    risk     = st.session_state["risk"]
    inputs   = st.session_state["inputs"]
    ind_preds = st.session_state["ind_preds"]

    st.markdown("---")
    st.markdown('<div class="sec-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── 7 tabs inside Prediction Results ──────────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📈 Parameter Analysis",
        "🎯 Risk Gauge",
        "💡 Feature Selection",
        "1️⃣ Logistic Regression",
        "2️⃣ Random Forest",
        "3️⃣ Gradient Boosting",
        "4️⃣ Improved LR",
    ])

    # ── Tab 1: Parameter Analysis ──────────────────────────────────────────────
    with t1:

        st.markdown("#### WHO Parameter Safety Analysis")
        st.markdown("")
        perc_rows = []
        for p in pa:
            safe_max = p["safe_high"] if p["safe_high"] > 0 else 1
            perc_rows.append({"Parameter": p["label"],
                               "% of Safe Limit": round((p["value"]/safe_max)*100, 1),
                               "Status": p["status"]})
        fig = px.bar(pd.DataFrame(perc_rows), x="Parameter", y="% of Safe Limit", color="Status",
                     color_discrete_map={"Safe":"#2e7d32","Unsafe":"#c62828"},
                     title="Each Parameter as % of Safe Limit (100% = at the limit)",
                     text="% of Safe Limit")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Safe Limit")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=460)
        st.plotly_chart(fig, use_container_width=True)
        tbl = pd.DataFrame([{"Parameter": p["label"], "Your Value": p["value"],
                              "Safe Range": f"{p['safe_low']} – {p['safe_high']}", "Status": p["status"]}
                             for p in pa])
        def hl(v):
            return "background-color:#ffcdd2;color:#c62828" if v=="Unsafe" else "background-color:#c8e6c9;color:#2e7d32"
        st.dataframe(tbl.style.applymap(hl, subset=["Status"]), use_container_width=True)

    # ── Tab 2: Risk Gauge ──────────────────────────────────────────────────────
    with t2:
        if pred == 1:
            st.markdown('<div class="result-safe">✅ SAFE FOR DRINKING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-unsafe">⛔ NOT SAFE FOR DRINKING</div>', unsafe_allow_html=True)
        st.markdown("")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model Confidence",  f"{round(prob*100,1)}%")
        mc2.metric("Risk Score",        f"{risk['score']} / 100")
        mc3.metric("Risk Level",        f"{risk['badge']} {risk['level']}")
        mc4.metric("Unsafe Parameters", f"{risk['unsafe_params']} / {risk['total_params']}")
        st.markdown("")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk["score"],
            title={"text": f"Risk Score<br><span style='font-size:.8em'>{risk['badge']} {risk['level']}</span>"},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":"#1a6fa8"},
                   "steps":[{"range":[0,30],"color":"#c8e6c9"},
                             {"range":[30,60],"color":"#fff9c4"},
                             {"range":[60,100],"color":"#ffcdd2"}],
                   "threshold":{"line":{"color":"black","width":3},"thickness":.75,"value":risk["score"]}}))
        gauge.update_layout(height=400)
        st.plotly_chart(gauge, use_container_width=True)

    # ── Tab 3: Feature Selection ───────────────────────────────────────────────
    with t3:
        col_a, col_b = st.columns(2)
        with col_a:
            fi = os.path.join(PLOT_DIR, "feature_importance_selector.png")
            if os.path.exists(fi):
                st.image(fi, caption="Random Forest — Feature Importance")
        with col_b:
            rp = os.path.join(PLOT_DIR, "rfe_feature_ranking.png")
            if os.path.exists(rp):
                st.image(rp, caption="RFE — Feature Ranking (Green=Selected)")

   # ── Helper: render each model prediction tab ───────────────────────────────
    def model_pred_tab(model_name):
        _preds = st.session_state.get("ind_preds", {})
        _risk  = st.session_state.get("risk", {})
        p = _preds.get(model_name)
        if p is None or p.get("prediction", -1) == -1:
            st.info("👆 Click **Predict Water Quality** to see this model's prediction.")
            return

        if p["prediction"] == 1:
            st.markdown(f'<div class="result-safe">✅ {model_name}: SAFE FOR DRINKING</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-unsafe">⛔ {model_name}: NOT SAFE FOR DRINKING</div>', unsafe_allow_html=True)
        st.markdown("")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model Confidence",  f"{round(p['probability']*100, 1)}%")
        mc2.metric("Risk Score",        f"{_risk.get('score', 0)} / 100")
        mc3.metric("Risk Level",        f"{_risk.get('badge', '')} {_risk.get('level', '')}")
        mc4.metric("Unsafe Parameters", f"{_risk.get('unsafe_params', 0)} / {_risk.get('total_params', 9)}")
        st.markdown("")

        fig = go.Figure(go.Bar(
            x=["Safe for Drinking", "Not Safe for Drinking"],
            y=[round(p["safe_prob"]*100,1), round(p["unsafe_prob"]*100,1)],
            marker_color=["#2e7d32","#c62828"],
            text=[f"{round(p['safe_prob']*100,1)}%", f"{round(p['unsafe_prob']*100,1)}%"],
            textposition="outside",
        ))
        fig.update_layout(title=f"{model_name} — Confidence Breakdown",
                          yaxis_range=[0,130], height=320,
                          yaxis_title="Confidence %", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if model_name in all_metrics:
            m = all_metrics[model_name]
            st.markdown("#### Training Evaluation Metrics")
            e1,e2,e3,e4,e5 = st.columns(5)
            e1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
            e2.metric("Precision", f"{m['precision']*100:.1f}%")
            e3.metric("Recall",    f"{m['recall']*100:.1f}%")
            e4.metric("F1 Score",  f"{m['f1']*100:.1f}%")
            e5.metric("AUC",       f"{m['auc']*100:.1f}%")
            mfig = px.bar(
                x=["Accuracy","Precision","Recall","F1","AUC"],
                y=[m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc"]],
                color_discrete_sequence=["#1a6fa8"],
                title=f"{model_name} — Training Metrics",
                text=[f"{v*100:.1f}%" for v in [m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc"]]],
            )
            mfig.update_traces(textposition="outside")
            mfig.update_layout(yaxis_range=[0,1.25], height=320, showlegend=False)
            st.plotly_chart(mfig, use_container_width=True)

    # ── Tabs 4-7: Individual model predictions ─────────────────────────────────
    with t4:
        model_pred_tab("Logistic Regression")
        st.markdown("---")
        st.info("**How it works:** Finds a linear boundary between Safe/Unsafe using sigmoid function.")
        st.error("**Limitation:** Cannot capture non-linear relationships in water quality data.")

    with t5:
        model_pred_tab("Random Forest")
        st.markdown("---")
        st.info("**How it works:** 200 decision trees vote — majority decides the prediction.")
        st.success("**Strength:** Best accuracy — handles non-linear patterns through ensemble voting.")

    with t6:
        model_pred_tab("Gradient Boosting")
        st.markdown("---")
        st.info("**How it works:** Each tree corrects errors of the previous tree sequentially.")
        st.success("**Strength:** Very strong on tabular data due to sequential error correction.")

    with t7:
        model_pred_tab("Improved Logistic Regression")
        st.markdown("---")
        st.info("**How it works:** LR with SMOTE + RobustScaler + GridSearchCV + RFE applied.")
        i1,i2,i3,i4 = st.columns(4)
        i1.info("**SMOTE**\n\nSynthetic samples to fix class imbalance")
        i2.info("**RobustScaler**\n\nMedian/IQR scaling handles outliers better")
        i3.info("**GridSearchCV**\n\n5-fold CV to find best C, penalty, solver")
        i4.info("**RFE**\n\nKept best 6 out of 9 features")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION SECTION (always visible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-header">📐 Model Evaluation Metrics</div>', unsafe_allow_html=True)
st.caption("Training/test performance of each model on the held-out test set.")
st.markdown("")

ev1, ev2, ev3, ev4, ev5 = st.tabs([
    "1️⃣ Logistic Regression",
    "2️⃣ Random Forest",
    "3️⃣ Gradient Boosting",
    "4️⃣ Improved LR",
    "📊 Full Comparison",
])

COLORS = {"Logistic Regression":"#1565c0","Random Forest":"#2e7d32",
          "Gradient Boosting":"#e65100","Improved Logistic Regression":"#6a1b9a"}

def eval_tab(model_name, why):
    st.markdown(f"### {model_name} — Evaluation Results")
    if model_name not in all_metrics:
        st.warning("No metrics. Run `python model/train_model.py` first.")
        return
    m = all_metrics[model_name]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['precision']*100:.1f}%")
    c3.metric("Recall",    f"{m['recall']*100:.1f}%")
    c4.metric("F1 Score",  f"{m['f1']*100:.1f}%")
    c5.metric("AUC",       f"{m['auc']*100:.1f}%")
    color = COLORS.get(model_name,"#1a6fa8")
    fig = px.bar(
        x=["Accuracy","Precision","Recall","F1 Score","AUC"],
        y=[m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc"]],
        color_discrete_sequence=[color],
        title=f"{model_name} — All Metrics",
        text=[f"{v*100:.1f}%" for v in [m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0,1.25], height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Metric Explanations")
    e1,e2,e3,e4,e5 = st.columns(5)
    e1.info(f"**Accuracy**\n\n{m['accuracy']*100:.1f}%\n\nOverall correct predictions")
    e2.info(f"**Precision**\n\n{m['precision']*100:.1f}%\n\nOf predicted Safe, actually Safe")
    e3.info(f"**Recall**\n\n{m['recall']*100:.1f}%\n\nOf actual Safe, model caught")
    e4.info(f"**F1**\n\n{m['f1']*100:.1f}%\n\nHarmonic mean of P & R")
    e5.info(f"**AUC**\n\n{m['auc']*100:.1f}%\n\nROC area, 1.0 = perfect")
    st.markdown("---")
    st.markdown("#### Why These Results?")
    st.warning(why)

with ev1:
    eval_tab("Logistic Regression",
             "LR is a linear model — draws a straight boundary between Safe/Unsafe. "
             "Water quality patterns are non-linear, which limits its accuracy ceiling.")

with ev2:
    eval_tab("Random Forest",
             "200 trees vote together, capturing non-linear relationships. "
             "Ensemble voting reduces variance and improves robustness.")
    fi = os.path.join(PLOT_DIR, "feature_importance_selector.png")
    if os.path.exists(fi):
        st.image(fi, caption="Feature Importance")

with ev3:
    eval_tab("Gradient Boosting",
             "Builds trees sequentially where each tree fixes the previous one's mistakes. "
             "Very strong on structured tabular data like water quality measurements.")

with ev4:
    eval_tab("Improved Logistic Regression",
             "SMOTE fixes class imbalance, RobustScaler handles outliers, "
             "GridSearchCV tunes hyperparameters, RFE selects best 6 features. "
             "All improvements push LR further, but linear boundary is still its ceiling.")
    st.markdown("#### Baseline vs Improved Comparison")
    if "Logistic Regression" in all_metrics and "Improved Logistic Regression" in all_metrics:
        base = all_metrics["Logistic Regression"]
        imp  = all_metrics["Improved Logistic Regression"]
        rows = []
        for k, lbl in [("accuracy","Accuracy"),("precision","Precision"),
                        ("recall","Recall"),("f1","F1"),("auc","AUC")]:
            gain = (imp[k]-base[k])*100
            rows.append({"Metric": lbl,
                         "Baseline LR": f"{base[k]*100:.1f}%",
                         "Improved LR": f"{imp[k]*100:.1f}%",
                         "Gain": f"+{gain:.1f}%" if gain>=0 else f"{gain:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        lrp = os.path.join(PLOT_DIR, "lr_improvement.png")
        if os.path.exists(lrp):
            st.image(lrp, caption="Baseline vs Improved LR")
    with col_b:
        rfep = os.path.join(PLOT_DIR, "rfe_feature_ranking.png")
        if os.path.exists(rfep):
            st.image(rfep, caption="RFE Feature Ranking")

with ev5:
    st.markdown("### All 4 Models — Complete Comparison")
    if all_metrics:
        rows = []
        for name, m in all_metrics.items():
            rows.append({"Model": name,
                         "Accuracy":  f"{m['accuracy']*100:.1f}%",
                         "Precision": f"{m['precision']*100:.1f}%",
                         "Recall":    f"{m['recall']*100:.1f}%",
                         "F1 Score":  f"{m['f1']*100:.1f}%",
                         "AUC":       f"{m['auc']*100:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("")
        mnames = list(all_metrics.keys())
        mkeys  = ["accuracy","precision","recall","f1","auc"]
        mlbls  = ["Accuracy","Precision","Recall","F1","AUC"]
        clrs   = ["#1565c0","#2e7d32","#e65100","#6a1b9a"]
        fig_all = go.Figure()
        for i, name in enumerate(mnames):
            fig_all.add_trace(go.Bar(
                name=name, x=mlbls,
                y=[all_metrics[name][k] for k in mkeys],
                marker_color=clrs[i % len(clrs)],
                text=[f"{all_metrics[name][k]*100:.1f}%" for k in mkeys],
                textposition="outside",
            ))
        fig_all.update_layout(barmode="group", title="All 4 Models — Full Comparison",
                              yaxis_range=[0,1.3], height=520,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_all, use_container_width=True)
        st.markdown("""
#### Why Different Accuracies?
| Model | Reason |
|---|---|
| **Logistic Regression** | Linear model — limited by non-linear water quality patterns |
| **Random Forest** | 200 trees ensemble — captures complex non-linear relationships |
| **Gradient Boosting** | Sequential error correction — strongest on tabular data |
| **Improved LR** | SMOTE + tuning helps but linear boundary remains the ceiling |
        """)
        st.markdown("---")
        pc1, pc2 = st.columns(2)
        with pc1:
            cm = os.path.join(PLOT_DIR, "confusion_matrix.png")
            if os.path.exists(cm):
                st.image(cm, caption="Confusion Matrix — Final Model (RF)")
        with pc2:
            rc = os.path.join(PLOT_DIR, "roc_curve.png")
            if os.path.exists(rc):
                st.image(rc, caption="ROC Curve — Final Model (RF)")
    else:
        st.warning("No metrics found. Please retrain first.")

# ══════════════════════════════════════════════════════════════════════════════
# AI REPORT + ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
if "pred" in st.session_state:
    pred   = st.session_state["pred"]
    prob   = st.session_state["probability"]
    pa     = st.session_state["param_analysis"]
    risk   = st.session_state["risk"]
    inputs = st.session_state["inputs"]

    st.markdown("---")
    st.markdown('<div class="sec-header">📝 AI-Generated Water Quality Report</div>', unsafe_allow_html=True)
    st.markdown("")
    if not api_key:
        st.warning("Enter your Groq API Key in the sidebar to generate the AI report.")
    else:
        if st.button("🤖 Generate AI Report"):
            with st.spinner("Generating report..."):
                try:
                    st.session_state["ai_report"] = generate_report(
                        pred, prob, inputs, pa, risk, api_key)
                except Exception as e:
                    st.error(f"API Error: {e}")
        if st.session_state.get("ai_report"):
            st.text_area("", value=st.session_state["ai_report"], height=400,
                         label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sec-header">🤖 AI Water Quality Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask anything about this water sample.")
    st.markdown("")
    if not api_key:
        st.warning("Enter your Groq API Key in the sidebar to use the assistant.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        suggested = ["Why is this water unsafe?", "Which parameter is biggest risk?",
                     "What treatment is needed?", "Is the pH acceptable?", "What are health risks?"]
        st.markdown("**Quick questions:**")
        sq_cols = st.columns(len(suggested))
        for i, q in enumerate(suggested):
            if sq_cols[i].button(q, key=f"sq_{i}"):
                st.session_state["pending_q"] = q
        user_input = st.chat_input("Ask the assistant...")
        if "pending_q" in st.session_state:
            user_input = st.session_state.pop("pending_q")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    resp = generate_assistant_response(
                        question=user_input, prediction=pred, probability=prob,
                        inputs=inputs, param_analysis=pa, risk=risk,
                        chat_history=st.session_state["chat_history"],
                        api_key=api_key)
                    st.session_state["chat_history"].append({"role":"user","content":user_input})
                    st.session_state["chat_history"].append({"role":"assistant","content":resp})
                except Exception as e:
                    st.error(f"Assistant error: {e}")
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()
