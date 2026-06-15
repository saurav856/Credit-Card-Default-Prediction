import streamlit as st
import requests
import psycopg2
import pandas as pd
from datetime import datetime
import os

# --- Page config ---
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .title-block p {
        color: #888;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    .card {
        background: #1a1d26;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
    }

    .result-high {
        background: linear-gradient(135deg, #3d1a1a, #1a0f0f);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #0f3d1a, #0a1f0f);
        border: 1px solid #00cc66;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 0.4rem;
    }
    .result-risk-high {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ff4444;
    }
    .result-risk-low {
        font-size: 2.8rem;
        font-weight: 700;
        color: #00cc66;
    }
    .result-prob {
        font-size: 1.1rem;
        color: #ccc;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: #1a1d26;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 0.3rem;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.85;
    }

    .stNumberInput input, .stSelectbox select {
        background-color: #1a1d26 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #13151f;
    }

    .log-table {
        background: #1a1d26;
        border-radius: 10px;
        padding: 1rem;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- API config ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432),
        dbname=os.getenv("PG_DB", "mlops_logs"),
        user=os.getenv("PG_USER", "mlops_user"),
        password=os.getenv("PG_PASSWORD", "mlops_pass")
    )

def fetch_prediction_history(limit=10):
    try:
        conn = get_pg_conn()
        df = pd.read_sql(
            f"SELECT timestamp, age, limit_bal, prediction, probability, risk FROM prediction_log ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        return None

# --- Header ---
st.markdown("""
<div class="title-block">
    <h1>💳 Credit Risk Predictor</h1>
    <p>ML-powered default risk assessment</p>
</div>
""", unsafe_allow_html=True)

# --- API Status ---
try:
    status = requests.get(f"{API_URL}/status", timeout=3).json()
    api_ok = status.get("status") == "available"
except:
    api_ok = False

status_color = "#00cc66" if api_ok else "#ff4444"
status_text = "API Online" if api_ok else "API Offline"
st.markdown(f"""
<div style="text-align:center; margin-bottom:1.5rem;">
    <span style="background:#1a1d26; border:1px solid {status_color}; color:{status_color};
    padding:0.3rem 1rem; border-radius:20px; font-size:0.85rem;">● {status_text}</span>
</div>
""", unsafe_allow_html=True)

# --- Layout ---
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Client Information")

    LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=0, max_value=1000000, value=50000, step=5000)
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

    st.markdown("#### Payment Status")
    pay_options = {
        "-2 (No consumption)": -2, "-1 (Paid in full)": -1,
        "0 (Revolving credit)": 0, "1 (1 month delay)": 1,
        "2 (2 month delay)": 2, "3 (3 month delay)": 3,
        "4+ (4+ month delay)": 4
    }
    PAY_0 = st.selectbox("Payment Status — September (PAY_0)", list(pay_options.keys()), index=2)
    PAY_2 = st.selectbox("Payment Status — August (PAY_2)", list(pay_options.keys()), index=2)
    PAY_3 = st.selectbox("Payment Status — July (PAY_3)", list(pay_options.keys()), index=2)

    st.markdown("#### Bill & Payment Amounts")
    BILL_AMT1 = st.number_input("Bill Amount — September (NT$)", value=5000, step=500)
    BILL_AMT2 = st.number_input("Bill Amount — August (NT$)", value=3000, step=500)
    PAY_AMT1 = st.number_input("Payment Made — September (NT$)", min_value=0, value=2000, step=500)
    PAY_AMT2 = st.number_input("Payment Made — August (NT$)", min_value=0, value=1500, step=500)

    st.markdown('</div>', unsafe_allow_html=True)
    predict_btn = st.button("Run Risk Assessment")

with col_result:
    if predict_btn:
        if not api_ok:
            st.error("API is offline. Start the FastAPI server first.")
        else:
            payload = {
                "LIMIT_BAL": LIMIT_BAL, "AGE": AGE,
                "PAY_0": pay_options[PAY_0],
                "PAY_2": pay_options[PAY_2],
                "PAY_3": pay_options[PAY_3],
                "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2,
                "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2
            }
            with st.spinner("Analysing..."):
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload, timeout=10).json()
                    risk = res["risk"]
                    prob = res["probability_of_default"]
                    pred = res["prediction"]

                    if risk == "HIGH":
                        st.markdown(f"""
                        <div class="result-high">
                            <div class="result-label">Default Risk</div>
                            <div class="result-risk-high">⚠ HIGH RISK</div>
                            <div class="result-prob">Probability of Default: <strong>{prob:.1%}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-low">
                            <div class="result-label">Default Risk</div>
                            <div class="result-risk-low">✓ LOW RISK</div>
                            <div class="result-prob">Probability of Default: <strong>{prob:.1%}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Prediction</div><div class="metric-value">{"Default" if pred==1 else "No Default"}</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Probability</div><div class="metric-value">{prob:.1%}</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{risk}</div></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:4rem 2rem; color:#555;">
            <div style="font-size:3rem;">📊</div>
            <div style="margin-top:1rem; font-size:1rem;">Fill in client details and click<br><strong style="color:#7c3aed;">Run Risk Assessment</strong></div>
        </div>
        """, unsafe_allow_html=True)
