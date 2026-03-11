"""
app.py
------
Main Streamlit web application for the
AI-Based Water Quality Prediction System.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

from utils.risk_scorer     import analyze_parameters, calculate_risk_score
from utils.report_generator import generate_report, generate_assistant_response


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "..", "model", "water_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Water Quality System",
    page_icon="💧",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a6fa8;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-safe {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .result-unsafe {
        background: #ffebee;
        border-left: 6px solid #c62828;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #c62828;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a6fa8;
        border-bottom: 2px solid #1a6fa8;
        padding-bottom: 4px;
        margin-top: 1.5rem;
    }
    .chat-user {
        background: #e3f2fd;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 4px 0;
        text-align: right;
    }
    .chat-ai {
        background: #f1f8e9;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">💧 AI Water Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Logistic Regression + Explainable AI + Claude Report Generation</div>', unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model not found. Please run `python model/train_model.py` first.\n\nError: {e}")
    model_loaded = False
    st.stop()

# ─── Sidebar: API Key ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-...")
    st.caption("Required for AI Report and Assistant features.")

    st.markdown("---")
    st.markdown("### 📋 About This System")
    st.info(
        "This system uses Logistic Regression to predict "
        "water potability from 9 chemical and physical parameters. "
        "It generates an AI-powered analysis report and supports "
        "an interactive water quality assistant."
    )
    st.markdown("---")
    st.markdown("### 📊 Sample Inputs")
    if st.button("Load Unsafe Sample"):
        st.session_state["sample"] = "unsafe"
    if st.button("Load Safe Sample"):
        st.session_state["sample"] = "safe"

# ─── Sample values ────────────────────────────────────────────────────────────
UNSAFE_SAMPLE = dict(ph=5.8, hardness=330, solids=22000, chloramines=9.0,
                     sulfate=420, conductivity=560, organic_carbon=18.0,
                     turbidity=7.5, trihalomethanes=90.0)

SAFE_SAMPLE   = dict(ph=7.2, hardness=180, solids=8000,  chloramines=3.0,
                     sulfate=200, conductivity=310, organic_carbon=8.0,
                     turbidity=1.5, trihalomethanes=40.0)

defaults = UNSAFE_SAMPLE if st.session_state.get("sample") == "unsafe" \
           else SAFE_SAMPLE if st.session_state.get("sample") == "safe" \
           else dict(ph=7.0, hardness=200, solids=15000, chloramines=7.0,
                     sulfate=300, conductivity=400, organic_carbon=10.0,
                     turbidity=4.0, trihalomethanes=70.0)

# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧪 Enter Water Parameters</div>', unsafe_allow_html=True)
st.markdown("")

c1, c2, c3 = st.columns(3)

with c1:
    ph            = st.number_input("pH",               min_value=0.0,  max_value=14.0, value=float(defaults["ph"]),            step=0.1)
    hardness      = st.number_input("Hardness (mg/L)",  min_value=0.0,  max_value=600.0, value=float(defaults["hardness"]),     step=1.0)
    solids        = st.number_input("Solids (ppm)",     min_value=0.0,  max_value=60000.0, value=float(defaults["solids"]),     step=100.0)

with c2:
    chloramines   = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=float(defaults["chloramines"]),   step=0.1)
    sulfate       = st.number_input("Sulfate (mg/L)",    min_value=0.0, max_value=700.0, value=float(defaults["sulfate"]),      step=1.0)
    conductivity  = st.number_input("Conductivity",      min_value=0.0, max_value=800.0, value=float(defaults["conductivity"]), step=1.0)

with c3:
    organic_carbon  = st.number_input("Organic Carbon",     min_value=0.0, max_value=30.0,  value=float(defaults["organic_carbon"]),  step=0.1)
    turbidity       = st.number_input("Turbidity (NTU)",     min_value=0.0, max_value=10.0,  value=float(defaults["turbidity"]),       step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes",     min_value=0.0, max_value=130.0, value=float(defaults["trihalomethanes"]), step=0.1)

inputs = dict(
    ph=ph, hardness=hardness, solids=solids,
    chloramines=chloramines, sulfate=sulfate, conductivity=conductivity,
    organic_carbon=organic_carbon, turbidity=turbidity,
    trihalomethanes=trihalomethanes,
)

# ─── Predict Button ───────────────────────────────────────────────────────────
st.markdown("")
predict_col, _ = st.columns([1, 3])
with predict_col:
    predict_btn = st.button("🔍 Predict Water Quality", use_container_width=True)

# ─── Run Prediction ───────────────────────────────────────────────────────────
if predict_btn:
    sample = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, turbidity, trihalomethanes]])
    sample_scaled = scaler.transform(sample)

    prediction   = model.predict(sample_scaled)[0]
    proba        = model.predict_proba(sample_scaled)[0]
    probability  = proba[prediction]

    param_analysis = analyze_parameters(inputs)
    risk           = calculate_risk_score(proba[0], param_analysis)

    # Store in session
    st.session_state["prediction"]     = prediction
    st.session_state["probability"]    = probability
    st.session_state["proba"]          = proba
    st.session_state["param_analysis"] = param_analysis
    st.session_state["risk"]           = risk
    st.session_state["inputs"]         = inputs
    st.session_state["chat_history"]   = []
    st.session_state["ai_report"]      = None


# ─── Results Section ──────────────────────────────────────────────────────────
if "prediction" in st.session_state:
    prediction     = st.session_state["prediction"]
    probability    = st.session_state["probability"]
    proba          = st.session_state["proba"]
    param_analysis = st.session_state["param_analysis"]
    risk           = st.session_state["risk"]
    inputs         = st.session_state["inputs"]

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Prediction Banner ──────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown('<div class="result-safe">✅ SAFE FOR DRINKING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-unsafe">⛔ NOT SAFE FOR DRINKING</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics Row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Confidence",  f"{round(probability*100, 1)}%")
    m2.metric("Risk Score",        f"{risk['score']} / 100")
    m3.metric("Risk Level",        f"{risk['badge']} {risk['level']}")
    m4.metric("Unsafe Parameters", f"{risk['unsafe_params']} / {risk['total_params']}")

    # ── Tabs: Visualizations ──────────────────────────────────────────────────
    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["📈 Parameter Analysis", "🎯 Risk Gauge", "💡 Feature Importance"])

    with tab1:
        rows = []
        for p in param_analysis:
            rows.append({
                "Parameter": p["label"],
                "Your Value": p["value"],
                "Safe Max": p["safe_high"],
                "Status": p["status"],
            })
        df_p = pd.DataFrame(rows)

                # Calculate percentage of safe limit exceeded
        perc_rows = []
        for p in param_analysis:
            safe_max = p["safe_high"] if p["safe_high"] > 0 else 1
            pct = round((p["value"] / safe_max) * 100, 1)
            perc_rows.append({
                "Parameter": p["label"],
                "% of Safe Limit": pct,
                "Status": p["status"],
            })
        perc_df = pd.DataFrame(perc_rows)

        fig = px.bar(
            perc_df,
            x="Parameter",
            y="% of Safe Limit",
            color="Status",
            color_discrete_map={"Safe": "#2e7d32", "Unsafe": "#c62828"},
            title="Each Parameter as % of Safe Limit (100% = exactly at safe limit)",
            text="% of Safe Limit",
        )
        fig.add_hline(y=100, line_dash="dash", line_color="orange",
                      annotation_text="Safe Limit (100%)")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=-30, height=450, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Status table
        display_df = df_p[["Parameter", "Your Value", "Status"]]
        def color_status(val):
            if val == "Unsafe":
                return "background-color: #ffcdd2; color: #c62828"
            return "background-color: #c8e6c9; color: #2e7d32"
        st.dataframe(display_df.style.applymap(color_status, subset=["Status"]),
                     use_container_width=True)

    with tab2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk["score"],
            title={"text": f"Water Quality Risk Score<br><span style='font-size:0.8em'>{risk['badge']} {risk['level']}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#1a6fa8"},
                "steps": [
                    {"range": [0, 30],   "color": "#c8e6c9"},
                    {"range": [30, 60],  "color": "#fff9c4"},
                    {"range": [60, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": risk["score"],
                },
            },
        ))
        fig_gauge.update_layout(height=380)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab3:
        coefs      = model.coef_[0]
        feat_names = ["pH", "Hardness", "Solids", "Chloramines", "Sulfate",
                      "Conductivity", "Organic Carbon", "Turbidity", "Trihalomethanes"]
        feat_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
        feat_df = feat_df.reindex(feat_df["Coefficient"].abs().sort_values(ascending=True).index)

        fig_feat = px.bar(feat_df, x="Coefficient", y="Feature",
                          orientation="h", color="Coefficient",
                          color_continuous_scale="RdBu_r",
                          title="Logistic Regression Feature Coefficients")
        fig_feat.update_layout(height=400)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.caption("Positive coefficients increase the probability of unsafe water. Negative coefficients reduce it.")
# ── Evaluation Metrics Section ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)

plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "plots")

col1, col2 = st.columns(2)
with col1:
    cm_path = os.path.join(plot_dir, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix")

with col2:
    roc_path = os.path.join(plot_dir, "roc_curve.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curve")

col3, col4 = st.columns(2)
with col3:
    fi_path = os.path.join(plot_dir, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Feature Importance")

with col4:
    mc_path = os.path.join(plot_dir, "model_comparison.png")
    if os.path.exists(mc_path):
        st.image(mc_path, caption="Model Comparison")
    # ─── AI Report Section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📝 AI-Generated Water Quality Report</div>', unsafe_allow_html=True)
    st.markdown("")

    if not api_key:
        st.warning("⚠️ Enter your Anthropic API Key in the sidebar to generate the AI report.")
    else:
        gen_report_btn = st.button("🤖 Generate AI Report")
        if gen_report_btn or st.session_state.get("ai_report"):
            if gen_report_btn or not st.session_state.get("ai_report"):
                with st.spinner("Claude is analyzing the water sample..."):
                    try:
                        report_text = generate_report(
                            prediction, probability,
                            inputs, param_analysis, risk, api_key
                        )
                        st.session_state["ai_report"] = report_text
                    except Exception as e:
                        st.error(f"API Error: {e}")
                        report_text = None
            else:
                report_text = st.session_state["ai_report"]

            if report_text:
                st.text_area("", value=report_text, height=380, label_visibility="collapsed")

               

    # ─── AI Water Quality Assistant ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Water Quality Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask me anything about this water sample — health risks, treatment, parameters, etc.")

    if not api_key:
        st.warning("⚠️ Enter your Anthropic API Key in the sidebar to use the assistant.")
    else:
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Suggested questions
        suggested = [
            "Why is this water unsafe?",
            "Which parameter is the biggest risk?",
            "What treatment is needed?",
            "Is the pH level acceptable?",
            "What are the health risks?",
        ]
        st.markdown("**Quick questions:**")
        cols = st.columns(len(suggested))
        for i, q in enumerate(suggested):
            if cols[i].button(q, key=f"sq_{i}"):
                st.session_state["pending_question"] = q

        # Chat input
        user_input = st.chat_input("Ask the AI assistant...")

        # Handle pending question from quick buttons
        if "pending_question" in st.session_state:
            user_input = st.session_state.pop("pending_question")

        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = generate_assistant_response(
                        question=user_input,
                        prediction=prediction,
                        probability=probability,
                        inputs=inputs,
                        param_analysis=param_analysis,
                        risk=risk,
                        chat_history=st.session_state["chat_history"],
                        api_key=api_key,
                    )
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": user_input}
                    )
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Assistant error: {e}")

        # Render chat history
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()
