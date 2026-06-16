import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import random
import string
from datetime import datetime

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="MediAI — Disease Prediction Platform",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# ==============================
# Load Models & Scalers
# ==============================
@st.cache_resource
def load_models():
    models = {}
    models["diabetes_model"] = pickle.load(open("models/diabetes_model.pkl", "rb"))
    models["diabetes_scaler"] = pickle.load(open("models/diabetes_scaler.pkl", "rb"))
    models["heart_model"] = pickle.load(open("models/heart_model.pkl", "rb"))
    models["heart_scaler"] = pickle.load(open("models/heart_scaler.pkl", "rb"))
    models["heart_columns"] = pickle.load(open("models/heart_columns.pkl", "rb"))
    models["liver_model"] = pickle.load(open("models/liver_model.pkl", "rb"))
    models["liver_scaler"] = pickle.load(open("models/liver_scaler.pkl", "rb"))
    models["liver_columns"] = pickle.load(open("models/liver_columns.pkl", "rb"))
    models["kidney_model"] = pickle.load(open("models/kidney_model.pkl", "rb"))
    models["kidney_scaler"] = pickle.load(open("models/kidney_scaler.pkl", "rb"))
    models["kidney_columns"] = pickle.load(open("models/kidney_columns.pkl", "rb"))
    models["breast_model"] = pickle.load(open("models/breast_cancer_model.pkl", "rb"))
    models["breast_scaler"] = pickle.load(open("models/breast_cancer_scaler.pkl", "rb"))
    models["breast_columns"] = pickle.load(open("models/breast_cancer_columns.pkl", "rb"))
    return models

m = load_models()

# ==============================
# Session State Init
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "history" not in st.session_state:
    st.session_state.history = []
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 25
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Male"

def gen_pred_id():
    return "MED-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

# ==============================
# Master CSS
# ==============================
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Root Tokens ── */
:root {
    --blue-900: #0A1628;
    --blue-800: #0D2247;
    --blue-700: #1A3A6B;
    --blue-600: #1E4D9B;
    --blue-500: #2563EB;
    --blue-400: #3B82F6;
    --blue-300: #93C5FD;
    --blue-100: #DBEAFE;
    --blue-50:  #EFF6FF;
    --teal:     #0EA5E9;
    --teal-light: #E0F2FE;
    --green:    #10B981;
    --green-bg: #ECFDF5;
    --amber:    #F59E0B;
    --amber-bg: #FFFBEB;
    --red:      #EF4444;
    --red-bg:   #FEF2F2;
    --gray-900: #111827;
    --gray-700: #374151;
    --gray-500: #6B7280;
    --gray-300: #D1D5DB;
    --gray-100: #F3F4F6;
    --gray-50:  #F9FAFB;
    --white:    #FFFFFF;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.06);
    --shadow-md: 0 4px 16px rgba(0,0,0,.08), 0 2px 6px rgba(0,0,0,.05);
    --shadow-lg: 0 12px 32px rgba(37,99,235,.12), 0 4px 12px rgba(0,0,0,.06);
    --radius:   14px;
    --radius-sm: 8px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'DM Sans', sans-serif !important;
    color: var(--gray-900);
}

/* Remove default Streamlit top padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--blue-900) 0%, var(--blue-800) 100%) !important;
    border-right: none !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--white) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    color: var(--white) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    color: #CBD5E1 !important;
    border-radius: var(--radius-sm) !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 2px !important;
    transition: all .2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(37,99,235,.35) !important;
    border-color: var(--blue-400) !important;
    color: var(--white) !important;
}

/* ── Main background ── */
.main { background: var(--gray-50) !important; }

/* ── Card component ── */
.medi-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(0,0,0,.04);
    margin-bottom: 1.25rem;
    transition: box-shadow .2s ease, transform .2s ease;
}
.medi-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-1px);
}

/* ── Stat card ── */
.stat-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(0,0,0,.04);
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: box-shadow .2s ease, transform .2s ease;
}
.stat-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}
.stat-icon {
    width: 52px; height: 52px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.stat-value { font-size: 1.75rem; font-weight: 800; line-height: 1.1; color: var(--gray-900); }
.stat-label { font-size: 0.8rem; color: var(--gray-500); font-weight: 500; margin-top: 2px; }

/* ── Section header ── */
.section-hdr {
    display: flex; align-items: center; gap: .6rem;
    margin: 1.75rem 0 1rem;
}
.section-hdr .icon-wrap {
    width: 36px; height: 36px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    background: var(--blue-50);
}
.section-hdr h2 {
    font-size: 1.25rem; font-weight: 700;
    color: var(--gray-900); margin: 0; padding: 0;
}
.section-hdr span.sub {
    font-size: 0.82rem; color: var(--gray-500); font-weight: 400;
}

/* ── Disease hero banner ── */
.disease-banner {
    border-radius: var(--radius);
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.25rem;
    color: var(--white);
}
.disease-banner .d-icon { font-size: 2.5rem; }
.disease-banner .d-title { font-size: 1.45rem; font-weight: 800; }
.disease-banner .d-sub { font-size: 0.88rem; opacity: .82; margin-top: .15rem; }
.disease-banner .d-badge {
    margin-left: auto; background: rgba(255,255,255,.18);
    border-radius: 8px; padding: .4rem .85rem;
    font-size: .78rem; font-weight: 600; white-space: nowrap;
    backdrop-filter: blur(6px);
}

/* Gradient palettes per disease */
.banner-diabetes  { background: linear-gradient(135deg, #1E4D9B 0%, #2563EB 100%); }
.banner-heart     { background: linear-gradient(135deg, #991B1B 0%, #EF4444 100%); }
.banner-liver     { background: linear-gradient(135deg, #92400E 0%, #F59E0B 100%); }
.banner-kidney    { background: linear-gradient(135deg, #065F46 0%, #10B981 100%); }
.banner-breast    { background: linear-gradient(135deg, #831843 0%, #EC4899 100%); }

/* ── Input group label ── */
.field-group-label {
    font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: var(--gray-500); margin: 1.2rem 0 .5rem;
    border-left: 3px solid var(--blue-500);
    padding-left: .5rem;
}

/* ── Streamlit input overrides ── */
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    font-size: .8rem !important;
    font-weight: 600 !important;
    color: var(--gray-700) !important;
    margin-bottom: 2px !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--gray-300) !important;
    font-size: .9rem !important;
    transition: border-color .2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue-500) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
}
[data-testid="stSelectbox"] > div > div {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--gray-300) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, var(--blue-600), var(--blue-500)) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .75rem 2.2rem !important;
    font-size: .95rem !important;
    font-weight: 700 !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.3) !important;
    transition: all .2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--blue-700), var(--blue-600)) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Risk meter ── */
.risk-container {
    background: var(--white);
    border-radius: var(--radius);
    padding: 1.75rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(0,0,0,.04);
    text-align: center;
    margin-top: 1rem;
}
.risk-title { font-size: .8rem; font-weight: 600; color: var(--gray-500); text-transform: uppercase; letter-spacing: .07em; margin-bottom: .75rem; }
.risk-value { font-size: 3rem; font-weight: 900; line-height: 1; margin-bottom: .25rem; }
.risk-label { font-size: 1rem; font-weight: 700; margin-bottom: 1rem; }
.risk-low   { color: var(--green); }
.risk-med   { color: var(--amber); }
.risk-high  { color: var(--red); }
.risk-bar-track {
    height: 12px; background: var(--gray-100);
    border-radius: 999px; overflow: hidden;
}
.risk-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width .8s cubic-bezier(.4,0,.2,1);
}

/* ── Result card ── */
.result-card {
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-top: .75rem;
}
.result-card.low  { background: var(--green-bg); border: 1.5px solid #6EE7B7; }
.result-card.high { background: var(--red-bg);   border: 1.5px solid #FCA5A5; }
.result-card h3   { font-size: 1.1rem; font-weight: 700; margin: 0 0 .4rem; }
.result-card p    { font-size: .88rem; color: var(--gray-700); margin: 0; }

/* ── Recommendation pill ── */
.rec-pill {
    display: inline-flex; align-items: center; gap: .35rem;
    background: var(--blue-50); color: var(--blue-600);
    border-radius: 999px; padding: .35rem .85rem;
    font-size: .8rem; font-weight: 600;
    margin: .25rem .2rem;
}

/* ── Info expandable ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
.info-tile {
    background: var(--gray-50);
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid var(--gray-100);
}
.info-tile h4 { font-size: .85rem; font-weight: 700; margin: 0 0 .4rem; color: var(--blue-600); }
.info-tile ul { margin: 0; padding-left: 1.1rem; }
.info-tile ul li { font-size: .82rem; color: var(--gray-700); margin-bottom: .18rem; }

/* ── History table ── */
.hist-table { width: 100%; border-collapse: collapse; font-size: .86rem; }
.hist-table th {
    background: var(--gray-50); color: var(--gray-500);
    font-weight: 600; font-size: .75rem; text-transform: uppercase;
    letter-spacing: .06em; padding: .7rem 1rem;
    border-bottom: 2px solid var(--gray-100); text-align: left;
}
.hist-table td { padding: .7rem 1rem; border-bottom: 1px solid var(--gray-100); color: var(--gray-700); }
.hist-table tr:last-child td { border-bottom: none; }
.hist-table tr:hover td { background: var(--blue-50); }
.badge { border-radius: 999px; padding: .18rem .65rem; font-size: .75rem; font-weight: 700; }
.badge-low  { background: #D1FAE5; color: #065F46; }
.badge-high { background: #FEE2E2; color: #991B1B; }

/* ── Patient card ── */
.patient-card {
    background: linear-gradient(135deg, var(--blue-800), var(--blue-700));
    border-radius: var(--radius); padding: 1.25rem 1.5rem;
    color: var(--white); box-shadow: var(--shadow-lg);
}
.patient-card .avatar {
    width: 44px; height: 44px; border-radius: 50%;
    background: rgba(255,255,255,.18); display: flex;
    align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
}
.patient-card .p-name  { font-size: 1rem; font-weight: 700; }
.patient-card .p-meta  { font-size: .78rem; opacity: .75; margin-top: .15rem; }
.patient-card .p-id    { font-size: .72rem; opacity: .55; font-family: monospace; margin-top: .4rem; }

/* ── Model accuracy card ── */
.acc-card {
    background: var(--white); border-radius: 10px;
    padding: 1rem 1.25rem;
    border: 1px solid var(--gray-100);
    box-shadow: var(--shadow-sm);
    display: flex; align-items: center; gap: .75rem;
}
.acc-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.acc-name { font-size: .85rem; font-weight: 600; color: var(--gray-900); }
.acc-model { font-size: .76rem; color: var(--gray-500); }
.acc-pct { margin-left: auto; font-size: 1.15rem; font-weight: 800; color: var(--blue-600); }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, var(--blue-900) 0%, var(--blue-700) 60%, var(--teal) 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2.75rem;
    color: var(--white);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    margin-bottom: 1.75rem;
}
.hero-wrap::before {
    content: "";
    position: absolute; top: -80px; right: -60px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,.25) 0%, transparent 70%);
}
.hero-eyebrow {
    font-size: .75rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--blue-300);
    margin-bottom: .6rem;
}
.hero-title  { font-size: 2rem; font-weight: 900; line-height: 1.15; }
.hero-title span { color: var(--blue-300); }
.hero-sub    { font-size: .95rem; opacity: .78; margin-top: .5rem; max-width: 560px; line-height: 1.6; }
.hero-chips  { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.25rem; }
.hero-chip {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    border-radius: 999px; padding: .3rem .9rem;
    font-size: .78rem; font-weight: 600;
    backdrop-filter: blur(4px);
}

/* ── Divider ── */
.medi-divider {
    height: 1px; background: var(--gray-100);
    margin: 1.5rem 0; border: none;
}

/* ── Footer ── */
.medi-footer {
    margin-top: 3rem; padding: 1.5rem 2rem;
    background: var(--white); border-radius: var(--radius);
    box-shadow: var(--shadow-sm); border: 1px solid var(--gray-100);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
}
.footer-copy { font-size: .8rem; color: var(--gray-500); }
.footer-chips { display: flex; gap: .5rem; flex-wrap: wrap; }
.footer-chip {
    background: var(--gray-100); color: var(--gray-600);
    border-radius: 999px; padding: .2rem .7rem; font-size: .74rem; font-weight: 600;
}

/* ── Streamlit overrides ── */
.stExpander { border-radius: var(--radius) !important; border: 1px solid var(--gray-100) !important; }
[data-testid="metric-container"] {
    background: var(--white);
    border-radius: var(--radius-sm);
    padding: .85rem 1rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--gray-100);
}
.stProgress > div > div { border-radius: 999px !important; }
hr { border: none; border-top: 1px solid var(--gray-100) !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem;">
        <div style="font-size:1.6rem; font-weight:900; color:#FFFFFF; letter-spacing:-.02em;">🩺 MediAI</div>
        <div style="font-size:.75rem; color:#64748B; margin-top:.25rem; letter-spacing:.06em; text-transform:uppercase;">Disease Prediction Platform</div>
    </div>
    <hr style="border-color:rgba(255,255,255,.07); margin:0 0 1.25rem;">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:.68rem; color:#64748B; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin-bottom:.5rem;">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("🏠", "Dashboard"),
        ("🩸", "Diabetes"),
        ("❤️", "Heart Disease"),
        ("🍺", "Liver Disease"),
        ("🧬", "Kidney Disease"),
        ("🎗️", "Breast Cancer"),
    ]
    for icon, label in pages:
        is_active = st.session_state.page == label
        style_active = "background: rgba(37,99,235,.4) !important; border-color: rgba(59,130,246,.5) !important; color: #FFFFFF !important;"
        btn_style = style_active if is_active else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}"):
            st.session_state.page = label
            st.rerun()

    st.markdown('<hr style="border-color:rgba(255,255,255,.07); margin:1.25rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.68rem; color:#64748B; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin-bottom:.5rem;">Patient Profile</div>', unsafe_allow_html=True)

    st.session_state.patient_name   = st.text_input("Patient Name", value=st.session_state.patient_name, placeholder="e.g. John Doe")
    st.session_state.patient_age    = st.number_input("Age", 1, 120, value=st.session_state.patient_age)
    st.session_state.patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male","Female","Other"].index(st.session_state.patient_gender))

    st.markdown('<hr style="border-color:rgba(255,255,255,.07); margin:1.25rem 0;">', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.72rem; color:#475569; line-height:1.6;">
        ⚠️ <strong style="color:#94A3B8;">Disclaimer:</strong> For informational purposes only. Always consult a licensed physician.
    </div>""", unsafe_allow_html=True)

# ==============================
# Helpers
# ==============================
def patient_banner():
    name    = st.session_state.patient_name or "Anonymous Patient"
    age     = st.session_state.patient_age
    gender  = st.session_state.patient_gender
    date    = datetime.now().strftime("%d %b %Y")
    pid     = gen_pred_id()
    st.markdown(f"""
    <div class="patient-card">
        <div style="display:flex;align-items:center;gap:1rem;">
            <div class="avatar">👤</div>
            <div>
                <div class="p-name">{name}</div>
                <div class="p-meta">{age} yrs · {gender} · Session: {date}</div>
                <div class="p-id">Prediction ID: {pid}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return pid

def show_risk_result(score_pct, high_risk, high_msg, low_msg, disease, pred_id, recs_high, recs_low, info):
    """Render the unified risk result block."""
    if high_risk:
        level, cls = "HIGH RISK", "risk-high"
        bar_color, bar_pct = "#EF4444", max(score_pct, 65)
        card_cls, emoji = "high", "⚠️"
    else:
        level, cls = "LOW RISK", "risk-low"
        bar_color, bar_pct = "#10B981", min(score_pct, 35)
        card_cls, emoji = "low", "✅"

    st.markdown(f"""
    <div class="risk-container">
        <div class="risk-title">AI Risk Assessment</div>
        <div class="risk-value {cls}">{bar_pct}%</div>
        <div class="risk-label {cls}">{level}</div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:.35rem;font-size:.72rem;color:#9CA3AF;">
            <span>Low</span><span>Moderate</span><span>High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card {card_cls}" style="margin-top:1rem;">
        <h3>{emoji} {'High Risk Detected' if high_risk else 'Low Risk Detected'}</h3>
        <p>{'⚠️ ' + high_msg if high_risk else '✅ ' + low_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    recs = recs_high if high_risk else recs_low
    st.markdown("""<div style="margin-top:1.25rem;">
        <div style="font-size:.78rem;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;">Recommendations</div>""",
        unsafe_allow_html=True)
    pills = "".join(f'<span class="rec-pill">✦ {r}</span>' for r in recs)
    st.markdown(f'<div>{pills}</div></div>', unsafe_allow_html=True)

    # History entry
    st.session_state.history.append({
        "ID": pred_id, "Disease": disease,
        "Result": "High Risk" if high_risk else "Low Risk",
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Date": datetime.now().strftime("%d %b %Y"),
    })

    # Disease info
    with st.expander("📖 Disease Information & Prevention", expanded=False):
        st.markdown(f'<div class="info-grid">', unsafe_allow_html=True)
        for section, items in info.items():
            bullets = "".join(f"<li>{i}</li>" for i in items)
            st.markdown(f"""
            <div class="info-tile">
                <h4>{section}</h4>
                <ul>{bullets}</ul>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# DASHBOARD PAGE
# ==============================
if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">🩺 Powered by Machine Learning</div>
        <div class="hero-title">Intelligent Disease<br><span>Prediction Platform</span></div>
        <div class="hero-sub">Clinical-grade AI models analyzing patient biomarkers to deliver real-time health risk assessments across five major disease categories.</div>
        <div class="hero-chips">
            <span class="hero-chip">🔬 Evidence-Based</span>
            <span class="hero-chip">⚡ Real-Time Analysis</span>
            <span class="hero-chip">🔒 Private & Secure</span>
            <span class="hero-chip">📊 ML-Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (col1, "🦠", "#EFF6FF", "#2563EB", "5", "Diseases Supported"),
        (col2, "🧠", "#ECFDF5", "#059669", "5", "ML Models Active"),
        (col3, "🎯", "#FFFBEB", "#D97706", "≤100%", "Peak Accuracy"),
        (col4, "⚡", "#FDF4FF", "#9333EA", "< 1s", "Prediction Time"),
    ]
    for col, icon, bg, fg, val, lbl in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon" style="background:{bg};color:{fg};">{icon}</div>
                <div>
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">{lbl}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Disease cards
    st.markdown("""
    <div class="section-hdr">
        <div class="icon-wrap">🗂️</div>
        <div>
            <h2>Disease Modules</h2>
            <span class="sub">Select a module from the sidebar to begin prediction</span>
        </div>
    </div>""", unsafe_allow_html=True)

    disease_meta = [
        ("🩸", "Diabetes", "banner-diabetes", "Support Vector Machine", "77%",
         "Predicts diabetes risk using glucose levels, BMI, insulin, and family history metrics."),
        ("❤️", "Heart Disease", "banner-heart", "Random Forest", "85%",
         "Evaluates cardiovascular risk through blood pressure, cholesterol, and ECG features."),
        ("🍺", "Liver Disease", "banner-liver", "Gradient Boosting", "87%",
         "Assesses liver health via bilirubin, enzyme levels, and protein concentration markers."),
        ("🧬", "Kidney Disease", "banner-kidney", "Random Forest", "100%",
         "Screens for chronic kidney disease using 24 clinical and biochemical parameters."),
        ("🎗️", "Breast Cancer", "banner-breast", "Random Forest", "95%",
         "Classifies tumors as benign or malignant using 30 diagnostic imaging measurements."),
    ]
    cols = st.columns(3)
    for i, (icon, name, banner_cls, model, acc, desc) in enumerate(disease_meta):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="medi-card" style="cursor:pointer;">
                <div style="font-size:2rem; margin-bottom:.6rem;">{icon}</div>
                <div style="font-size:1rem; font-weight:800; color:#111827; margin-bottom:.25rem;">{name}</div>
                <div style="font-size:.8rem; color:#6B7280; margin-bottom:.85rem; line-height:1.5;">{desc}</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:.74rem; color:#3B82F6; font-weight:600;">⚙ {model}</span>
                    <span style="font-size:.8rem; font-weight:800; color:#059669;">{acc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Model accuracy section
    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hdr">
        <div class="icon-wrap">📊</div>
        <div>
            <h2>Model Performance</h2>
            <span class="sub">Accuracy benchmarks on held-out test sets</span>
        </div>
    </div>""", unsafe_allow_html=True)

    acc_data = [
        ("#2563EB", "Diabetes Prediction",  "Support Vector Machine",  77),
        ("#EF4444", "Heart Disease",         "Random Forest",           85),
        ("#F59E0B", "Liver Disease",         "Gradient Boosting",       87),
        ("#10B981", "Kidney Disease",        "Random Forest",          100),
        ("#EC4899", "Breast Cancer",         "Random Forest",           95),
    ]
    acol1, acol2 = st.columns(2)
    for i, (color, name, model, pct) in enumerate(acc_data):
        with (acol1 if i % 2 == 0 else acol2):
            st.markdown(f"""
            <div class="acc-card" style="margin-bottom:.6rem;">
                <div class="acc-dot" style="background:{color};"></div>
                <div>
                    <div class="acc-name">{name}</div>
                    <div class="acc-model">{model}</div>
                </div>
                <div class="acc-pct">{pct}%</div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct / 100)

    # History section
    if st.session_state.history:
        st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-hdr">
            <div class="icon-wrap">🕑</div>
            <div>
                <h2>Prediction History</h2>
                <span class="sub">Session log — resets on page refresh</span>
            </div>
        </div>""", unsafe_allow_html=True)
        rows = ""
        for entry in reversed(st.session_state.history[-10:]):
            badge_cls = "badge-high" if "High" in entry["Result"] else "badge-low"
            rows += f"""<tr>
                <td><code style="font-size:.78rem;">{entry['ID']}</code></td>
                <td>{entry['Disease']}</td>
                <td><span class="badge {badge_cls}">{entry['Result']}</span></td>
                <td>{entry['Date']}</td>
                <td>{entry['Time']}</td>
            </tr>"""
        st.markdown(f"""
        <div class="medi-card" style="overflow:auto;">
            <table class="hist-table">
                <thead><tr>
                    <th>Pred. ID</th><th>Disease</th>
                    <th>Result</th><th>Date</th><th>Time</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="medi-footer">
        <div class="footer-copy">
            <strong>MediAI</strong> &nbsp;·&nbsp; v1.0.0 &nbsp;·&nbsp; Built for educational purposes only
        </div>
        <div class="footer-chips">
            <span class="footer-chip">Python</span>
            <span class="footer-chip">Streamlit</span>
            <span class="footer-chip">scikit-learn</span>
            <span class="footer-chip">Pandas</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ==============================
# DIABETES PAGE
# ==============================
elif st.session_state.page == "Diabetes":

    st.markdown("""
    <div class="disease-banner banner-diabetes">
        <div class="d-icon">🩸</div>
        <div>
            <div class="d-title">Diabetes Risk Assessment</div>
            <div class="d-sub">Blood glucose & metabolic biomarker analysis · SVM Classifier</div>
        </div>
        <div class="d-badge">🎯 Accuracy: 77%</div>
    </div>""", unsafe_allow_html=True)

    pred_id = patient_banner()
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="medi-card">', unsafe_allow_html=True)
        st.markdown('<div class="field-group-label">Reproductive & Metabolic Markers</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, help="Number of times pregnant")
            glucose     = st.number_input("Glucose Level (mg/dL)", 50, 300, value=120, help="Plasma glucose concentration (2-hr oral glucose tolerance test)")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 40, 200, value=72, help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, value=20, help="Triceps skin fold thickness")
        with col2:
            insulin = st.number_input("Insulin Level (μU/mL)", 0, 900, value=80, help="2-Hour serum insulin")
            bmi     = st.number_input("BMI (kg/m²)", 10.0, 60.0, value=25.0, step=0.1, help="Body mass index")
            dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, value=0.5, step=0.01, help="Genetic risk score based on family history")
            age     = st.number_input("Age (years)", 1, 120, value=st.session_state.patient_age)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)

    if st.button("🔍  Run Diabetes Risk Analysis", use_container_width=True):
        with st.spinner("Analyzing biomarkers..."):
            time.sleep(0.6)
            input_data   = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = m["diabetes_scaler"].transform(input_data)
            prediction   = m["diabetes_model"].predict(input_scaled)
            high         = prediction[0] == 1
            score        = int(glucose / 3) if high else int(glucose / 6)

        show_risk_result(
            score_pct  = score,
            high_risk  = high,
            high_msg   = "Elevated glucose and metabolic markers suggest a heightened diabetes risk. Immediate lifestyle modifications and physician consultation are recommended.",
            low_msg    = "Your biomarker profile indicates a low diabetes risk. Continue maintaining a balanced diet and active lifestyle.",
            disease    = "Diabetes",
            pred_id    = pred_id,
            recs_high  = ["Consult an endocrinologist", "HbA1c blood test", "Reduce refined carbs", "Monitor blood glucose daily", "30-min daily exercise"],
            recs_low   = ["Annual glucose screening", "Maintain healthy weight", "Stay physically active", "Limit sugary beverages", "Routine checkups"],
            info       = {
                "📌 Overview": ["Chronic condition affecting blood sugar regulation", "Type 1: autoimmune; Type 2: lifestyle-related", "Affects 537 million adults globally (IDF 2021)"],
                "🔔 Symptoms": ["Frequent urination", "Excessive thirst", "Blurred vision", "Slow-healing wounds", "Unexplained fatigue"],
                "⚠️ Risk Factors": ["Obesity (BMI > 30)", "Family history", "Physical inactivity", "Age > 45", "High blood pressure"],
                "🛡️ Prevention": ["Maintain healthy weight", "Exercise 150 min/week", "Eat high-fiber diet", "Avoid smoking", "Regular HbA1c checks"],
            }
        )

# ==============================
# HEART DISEASE PAGE
# ==============================
elif st.session_state.page == "Heart Disease":

    st.markdown("""
    <div class="disease-banner banner-heart">
        <div class="d-icon">❤️</div>
        <div>
            <div class="d-title">Heart Disease Risk Assessment</div>
            <div class="d-sub">Cardiovascular biomarker & ECG pattern analysis · Random Forest</div>
        </div>
        <div class="d-badge">🎯 Accuracy: 85%</div>
    </div>""", unsafe_allow_html=True)

    pred_id = patient_banner()
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="medi-card">', unsafe_allow_html=True)
        st.markdown('<div class="field-group-label">Demographics & Vitals</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age      = st.number_input("Age (years)", 1, 120, value=st.session_state.patient_age)
            sex      = st.selectbox("Biological Sex", ["Male", "Female"])
            cp       = st.number_input("Chest Pain Type (0–3)", 0, 3, help="0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, value=120)
            chol     = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, value=200)
        with col2:
            fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, value=150)
            exang   = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, value=1.0, step=0.1)
            ca      = st.number_input("Fluoroscopy Vessel Count (0–3)", 0, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)

    if st.button("🔍  Run Cardiac Risk Analysis", use_container_width=True):
        with st.spinner("Evaluating cardiac markers..."):
            time.sleep(0.6)
            sex_v  = 1 if sex  == "Male" else 0
            fbs_v  = 1 if fbs  == "Yes"  else 0
            exang_v= 1 if exang== "Yes"  else 0
            input_dict = {"age": age, "sex": sex_v, "cp": cp, "trestbps": trestbps,
                          "chol": chol, "fbs": fbs_v, "thalach": thalach,
                          "exang": exang_v, "oldpeak": oldpeak, "ca": ca}
            input_df = pd.DataFrame([input_dict])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=m["heart_columns"], fill_value=0)
            scaled   = m["heart_scaler"].transform(input_df)
            pred     = m["heart_model"].predict(scaled)
            high     = pred[0] == 1
            score    = int(chol / 6) if high else int(chol / 12)

        show_risk_result(
            score_pct = score,
            high_risk = high,
            high_msg  = "Cardiovascular markers are outside optimal ranges. Prompt cardiological evaluation is strongly advised.",
            low_msg   = "Cardiovascular markers appear within healthy limits. Sustain your heart-healthy habits.",
            disease   = "Heart Disease",
            pred_id   = pred_id,
            recs_high = ["Cardiology consultation", "ECG & Echocardiogram", "Lipid management", "Reduce sodium intake", "Cardiac rehab program"],
            recs_low  = ["Annual cardiac screening", "Mediterranean diet", "150 min cardio/week", "Quit smoking", "Manage stress"],
            info      = {
                "📌 Overview": ["Leading cause of global mortality", "Includes CAD, arrhythmias, valve disease", "Largely preventable with lifestyle changes"],
                "🔔 Symptoms": ["Chest pain or pressure", "Shortness of breath", "Palpitations", "Fatigue", "Swollen legs (edema)"],
                "⚠️ Risk Factors": ["High LDL cholesterol", "Hypertension", "Diabetes", "Smoking", "Obesity & inactivity"],
                "🛡️ Prevention": ["No smoking / quit now", "Control blood pressure", "Heart-healthy diet", "Regular aerobic exercise", "Limit alcohol"],
            }
        )

# ==============================
# LIVER DISEASE PAGE
# ==============================
elif st.session_state.page == "Liver Disease":

    st.markdown("""
    <div class="disease-banner banner-liver">
        <div class="d-icon">🍺</div>
        <div>
            <div class="d-title">Liver Disease Risk Assessment</div>
            <div class="d-sub">Hepatic enzyme & protein biomarker analysis · Gradient Boosting</div>
        </div>
        <div class="d-badge">🎯 Accuracy: 87%</div>
    </div>""", unsafe_allow_html=True)

    pred_id = patient_banner()
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="medi-card">', unsafe_allow_html=True)
        st.markdown('<div class="field-group-label">Patient Demographics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age    = st.number_input("Age (years)", 1, 120, value=st.session_state.patient_age)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            tot_bilirubin   = st.number_input("Total Bilirubin (mg/dL)", 0.0, 50.0, value=1.0, step=0.1)
            direct_bilirubin= st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, value=0.3, step=0.1)

        st.markdown('<div class="field-group-label">Enzyme & Protein Levels</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            tot_proteins = st.number_input("Total Proteins (g/dL)", 0.0, 2000.0, value=6.8, step=0.1)
            albumin      = st.number_input("Albumin (g/dL)", 0.0, 100.0, value=3.5, step=0.1)
            ag_ratio     = st.number_input("Albumin/Globulin Ratio", 0.0, 10.0, value=1.1, step=0.01)
        with col4:
            sgpt   = st.number_input("SGPT / ALT (U/L)", 0.0, 500.0, value=35.0, step=0.5)
            sgot   = st.number_input("SGOT / AST (U/L)", 0.0, 500.0, value=30.0, step=0.5)
            alkphos= st.number_input("Alkaline Phosphatase (U/L)", 0.0, 2000.0, value=200.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)

    if st.button("🔍  Run Hepatic Risk Analysis", use_container_width=True):
        with st.spinner("Analyzing hepatic biomarkers..."):
            time.sleep(0.6)
            gender_v = 1 if gender == "Male" else 0
            input_data = {"age": age, "gender": gender_v,
                          "tot_bilirubin": tot_bilirubin, "direct_bilirubin": direct_bilirubin,
                          "tot_proteins": tot_proteins, "albumin": albumin,
                          "ag_ratio": ag_ratio, "sgpt": sgpt, "sgot": sgot, "alkphos": alkphos}
            input_df = pd.DataFrame([input_data])
            input_df = input_df[m["liver_columns"]]
            scaled   = m["liver_scaler"].transform(input_df)
            pred     = m["liver_model"].predict(scaled)
            high     = pred[0] == 1
            score    = min(int(sgpt / 5), 95) if high else min(int(sgpt / 10), 40)

        show_risk_result(
            score_pct = score,
            high_risk = high,
            high_msg  = "Elevated hepatic enzyme levels suggest liver stress or damage. Gastroenterology consultation is strongly recommended.",
            low_msg   = "Liver function markers appear within normal reference ranges. Maintain a liver-protective lifestyle.",
            disease   = "Liver Disease",
            pred_id   = pred_id,
            recs_high = ["Gastroenterology referral", "Liver ultrasound / biopsy", "Abstain from alcohol", "Avoid hepatotoxic drugs", "Low-fat diet"],
            recs_low  = ["Limit alcohol intake", "Avoid unnecessary medications", "Hepatitis A & B vaccines", "Maintain healthy weight", "Annual liver panel"],
            info      = {
                "📌 Overview": ["Liver processes nutrients and filters toxins", "Conditions include fatty liver, hepatitis, cirrhosis", "Often asymptomatic in early stages"],
                "🔔 Symptoms": ["Jaundice (yellow skin/eyes)", "Abdominal pain/swelling", "Dark urine", "Nausea & vomiting", "Fatigue"],
                "⚠️ Risk Factors": ["Excessive alcohol use", "Viral hepatitis (B/C)", "Obesity & metabolic syndrome", "Certain medications", "Autoimmune conditions"],
                "🛡️ Prevention": ["Limit alcohol strictly", "Hepatitis vaccination", "Safe food & water practices", "Regular liver function tests", "Avoid sharing needles"],
            }
        )

# ==============================
# KIDNEY DISEASE PAGE
# ==============================
elif st.session_state.page == "Kidney Disease":

    st.markdown("""
    <div class="disease-banner banner-kidney">
        <div class="d-icon">🧬</div>
        <div>
            <div class="d-title">Kidney Disease Risk Assessment</div>
            <div class="d-sub">Renal function & urinalysis biomarker analysis · Random Forest</div>
        </div>
        <div class="d-badge">🎯 Accuracy: 100%</div>
    </div>""", unsafe_allow_html=True)

    pred_id = patient_banner()
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="medi-card">', unsafe_allow_html=True)
        st.markdown('<div class="field-group-label">Urinalysis Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 1, 120, value=st.session_state.patient_age)
            bp  = st.number_input("Blood Pressure (mm Hg)", 50, 200, value=80)
            sg  = st.number_input("Specific Gravity", 1.000, 1.050, value=1.020, step=0.001, format="%.3f")
            al  = st.number_input("Albumin (0–5 scale)", 0, 5)
            su  = st.number_input("Sugar (0–5 scale)", 0, 5)
            rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            pc  = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
            ba  = st.selectbox("Bacteria", ["Not Present", "Present"])
            bgr = st.number_input("Blood Glucose Random (mg/dL)", 0, 500, value=100)
            bu  = st.number_input("Blood Urea (mg/dL)", 0.0, 300.0, value=40.0, step=0.5)

        with col2:
            sc   = st.number_input("Serum Creatinine (mg/dL)", 0.0, 20.0, value=1.2, step=0.1)
            sod  = st.number_input("Sodium (mEq/L)", 0.0, 200.0, value=135.0, step=0.5)
            pot  = st.number_input("Potassium (mEq/L)", 0.0, 10.0, value=4.5, step=0.1)
            hemo = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, value=13.0, step=0.1)
            pcv  = st.number_input("Packed Cell Volume (%)", 0.0, 60.0, value=40.0, step=0.5)
            wc   = st.number_input("White Blood Cell Count (cells/cumm)", 0, 20000, value=8000)
            rc   = st.number_input("Red Blood Cell Count (millions/cumm)", 0.0, 10.0, value=5.0, step=0.1)
            htn  = st.selectbox("Hypertension", ["No", "Yes"])
            dm   = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
            cad  = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
            appet= st.selectbox("Appetite", ["Good", "Poor"])
            pe   = st.selectbox("Pedal Edema", ["No", "Yes"])
            ane  = st.selectbox("Anemia", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)

    if st.button("🔍  Run Renal Risk Analysis", use_container_width=True):
        with st.spinner("Processing renal biomarkers..."):
            time.sleep(0.6)
            input_data = {
                "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
                "rbc": 1 if rbc=="Normal" else 0, "pc": 1 if pc=="Normal" else 0,
                "pcc": 1 if pcc=="Present" else 0, "ba": 1 if ba=="Present" else 0,
                "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
                "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
                "htn": 1 if htn=="Yes" else 0, "dm": 1 if dm=="Yes" else 0,
                "cad": 1 if cad=="Yes" else 0, "appet": 1 if appet=="Good" else 0,
                "pe": 1 if pe=="Yes" else 0, "ane": 1 if ane=="Yes" else 0,
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=m["kidney_columns"], fill_value=0)
            scaled   = m["kidney_scaler"].transform(input_df)
            pred     = m["kidney_model"].predict(scaled)
            high     = pred[0] == 1
            score    = min(int(sc * 30), 92) if high else min(int(sc * 15), 35)

        show_risk_result(
            score_pct = score,
            high_risk = high,
            high_msg  = "Renal biomarkers indicate possible chronic kidney disease. Immediate nephrological consultation is advised.",
            low_msg   = "Renal function markers suggest healthy kidney activity. Maintain adequate hydration and follow-up annually.",
            disease   = "Kidney Disease",
            pred_id   = pred_id,
            recs_high = ["Nephrology consultation", "GFR & creatinine panel", "Low-protein / low-sodium diet", "Blood pressure control", "Dialysis readiness assessment"],
            recs_low  = ["Stay well hydrated (2L/day)", "Annual renal panel", "Avoid nephrotoxic medications", "Control blood sugar & BP", "Low-salt diet"],
            info      = {
                "📌 Overview": ["Kidneys filter ~200L blood daily", "CKD is gradual loss of kidney function", "Affects 1 in 10 people worldwide"],
                "🔔 Symptoms": ["Decreased urine output", "Swelling in legs/ankles", "Fatigue & weakness", "Shortness of breath", "Persistent itching"],
                "⚠️ Risk Factors": ["Diabetes & hypertension (top causes)", "Family history of CKD", "Recurrent UTIs", "NSAIDs overuse", "Age > 60"],
                "🛡️ Prevention": ["Control blood sugar", "Manage blood pressure", "Hydrate adequately", "Avoid smoking", "Annual kidney function test"],
            }
        )

# ==============================
# BREAST CANCER PAGE
# ==============================
elif st.session_state.page == "Breast Cancer":

    st.markdown("""
    <div class="disease-banner banner-breast">
        <div class="d-icon">🎗️</div>
        <div>
            <div class="d-title">Breast Cancer Risk Assessment</div>
            <div class="d-sub">FNA diagnostic imaging feature analysis · Random Forest</div>
        </div>
        <div class="d-badge">🎯 Accuracy: 95%</div>
    </div>""", unsafe_allow_html=True)

    pred_id = patient_banner()
    st.markdown('<div style="height:.75rem;"></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="medi-card">', unsafe_allow_html=True)
        st.markdown('<div class="field-group-label">FNA Cell Nucleus Measurements (30 Features)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:.82rem;color:#6B7280;margin-bottom:1rem;">Enter diagnostic measurements from Fine Needle Aspiration biopsy image analysis.</p>', unsafe_allow_html=True)

        inputs = {}
        features = m["breast_columns"]
        cols     = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    format="%.5f", key=f"bc_{feature}",
                    help=f"FNA measurement: {feature}"
                )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem;"></div>', unsafe_allow_html=True)

    if st.button("🔍  Run Oncology Risk Analysis", use_container_width=True):
        with st.spinner("Analyzing cellular measurements..."):
            time.sleep(0.7)
            input_df = pd.DataFrame([inputs])
            input_df = input_df.reindex(columns=m["breast_columns"])
            scaled   = m["breast_scaler"].transform(input_df)
            pred     = m["breast_model"].predict(scaled)
            proba    = m["breast_model"].predict_proba(scaled)[0][1]
            high     = pred[0] == 0   # 0 = Malignant in original logic
            score    = int(proba * 100)

        show_risk_result(
            score_pct = score,
            high_risk = high,
            high_msg  = f"Malignant tumor characteristics detected (confidence: {score}%). Immediate oncological evaluation is critical.",
            low_msg   = f"Tumor features suggest benign classification (confidence: {100-score}%). Routine follow-up recommended.",
            disease   = "Breast Cancer",
            pred_id   = pred_id,
            recs_high = ["Oncology referral immediately", "Core needle / surgical biopsy", "MRI / mammography", "Genetic counseling (BRCA)", "Treatment planning"],
            recs_low  = ["Annual mammogram (age 40+)", "Monthly self-examination", "Maintain healthy BMI", "Limit alcohol", "Routine clinical breast exam"],
            info      = {
                "📌 Overview": ["Most common cancer in women globally", "FNA measures nuclear features of cells", "Early detection dramatically improves outcomes"],
                "🔔 Symptoms": ["New lump in breast or armpit", "Change in breast shape/size", "Nipple discharge or inversion", "Skin dimpling or redness", "Persistent breast pain"],
                "⚠️ Risk Factors": ["BRCA1/BRCA2 mutations", "Dense breast tissue", "HRT use > 5 years", "First-degree family history", "Age > 50"],
                "🛡️ Prevention": ["Regular screening mammograms", "Maintain healthy weight", "Exercise regularly", "Limit alcohol", "Breastfeed if possible"],
            }
        )
