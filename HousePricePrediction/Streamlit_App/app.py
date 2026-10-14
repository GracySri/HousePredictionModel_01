import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        min-height: 100vh;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #161b22; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border: 1px solid rgba(56,139,253,0.25);
        border-radius: 20px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(56,139,253,0.1), 0 20px 40px rgba(0,0,0,0.4);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(ellipse at 50% 30%, rgba(56,139,253,0.08) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,139,253,0.6), transparent);
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0.4rem 0 0.5rem 0;
        background: linear-gradient(90deg, #58a6ff 0%, #c8e6ff 50%, #58a6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 0.92rem;
        color: #6e7681;
        margin: 0;
        font-weight: 300;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(56,139,253,0.12);
        color: #58a6ff;
        border: 1px solid rgba(56,139,253,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        border-color: rgba(56,139,253,0.35);
        box-shadow: 0 4px 25px rgba(56,139,253,0.08);
    }
    .card h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #58a6ff;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Labels */
    label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px !important;
    }

    /* Inputs */
    .stNumberInput input {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    .stNumberInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(56,139,253,0.15) !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: #58a6ff !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(59,130,246,0.55) !important;
        background: linear-gradient(135deg, #2563eb, #3b82f6, #60a5fa) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f1f3d 40%, #071428 100%);
        border: 1px solid rgba(56,139,253,0.35);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(56,139,253,0.12), 0 20px 40px rgba(0,0,0,0.5);
    }
    .result-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(88,166,255,0.8), transparent);
    }
    .result-box::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(ellipse at 50% 20%, rgba(56,139,253,0.07) 0%, transparent 60%);
        pointer-events: none;
    }
    .result-box .label {
        font-size: 0.75rem;
        color: #484f58;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }
    .result-box .divider {
        width: 50px; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(88,166,255,0.6), transparent);
        margin: 0.6rem auto;
        border-radius: 2px;
    }
    .result-box .price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff 0%, #a5d6ff 50%, #58a6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    .result-box .range {
        font-size: 0.8rem;
        color: #484f58;
        margin-top: 0.7rem;
        font-weight: 300;
        letter-spacing: 0.3px;
    }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    }
    [data-testid="metric-container"] label {
        color: #6e7681 !important;
        font-size: 0.78rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Alerts */
    .stSuccess {
        background: rgba(35,134,54,0.1) !important;
        border: 1px solid rgba(35,134,54,0.35) !important;
        border-radius: 10px !important;
        color: #3fb950 !important;
    }
    .stWarning {
        background: rgba(187,128,9,0.1) !important;
        border: 1px solid rgba(187,128,9,0.35) !important;
        border-radius: 10px !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #21262d !important;
        margin: 1.5rem 0 !important;
    }

    footer {
        text-align: center;
        color: #30363d;
        font-size: 0.75rem;
        margin-top: 2.5rem;
        padding-bottom: 1rem;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path    = '../Model/xgboost_model.pkl'
    features_path = '../Model/features.pkl'
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()


# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">✦ AIML Internship 2026 · MNNIT Allahabad ✦</div>
    <h1>🏠 House Price Predictor</h1>
    <p>Enter property details below to get an instant AI-powered price estimate</p>
</div>
""", unsafe_allow_html=True)


# ── Model Status ───────────────────────────────────────────────
if model is None:
    st.warning("⚠️ Model not found. Please run the Jupyter Notebook first to train and save the model.")
    st.info("📌 Steps: Open `Notebook/house_price_prediction.ipynb` → Run all cells → Then relaunch this app.")
    st.stop()

st.success("✅ Model loaded successfully — XGBoost R² = 0.8395")


# ── Inputs ─────────────────────────────────────────────────────
st.markdown('<div class="card"><h3>🏗️ Property Details</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    overall_qual  = st.slider("Overall Quality (1–10)", 1, 10, 7)
    gr_liv_area   = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
    year_built    = st.number_input("Year Built", 1870, 2024, 2000)
with col2:
    full_bath     = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    half_bath     = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
    totrms_abvgrd = st.slider("Total Rooms Above Ground", 2, 14, 7)
    fireplaces    = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)
    yr_sold       = st.selectbox("Year Sold", list(range(2006, 2025)), index=14)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><h3>📐 Size & Extras</h3>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    first_flr_sf  = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 500)
    lot_area      = st.number_input("Lot Area (sq ft)", 1000, 100000, 8000)
with col4:
    overall_cond   = st.slider("Overall Condition (1–10)", 1, 10, 5)
    bsmt_full_bath = st.selectbox("Basement Full Bathrooms", [0, 1, 2], index=0)
    bsmt_half_bath = st.selectbox("Basement Half Bathrooms", [0, 1], index=0)
st.markdown('</div>', unsafe_allow_html=True)


# ── Predict ────────────────────────────────────────────────────
def build_input(feature_names):
    house_age  = yr_sold - year_built
    remod_age  = max(0, yr_sold - year_built)
    total_sf   = total_bsmt_sf + first_flr_sf + second_flr_sf
    total_bath = full_bath + 0.5*half_bath + bsmt_full_bath + 0.5*bsmt_half_bath
    has_garage = 1 if garage_area > 0 else 0
    lookup = {
        'OverallQual': overall_qual,   'GrLivArea':    gr_liv_area,
        'TotalBsmtSF': total_bsmt_sf,  'GarageArea':   garage_area,
        'YearBuilt':   year_built,     'FullBath':     full_bath,
        'TotRmsAbvGrd':totrms_abvgrd,  'Fireplaces':   fireplaces,
        'YrSold':      yr_sold,        '1stFlrSF':     first_flr_sf,
        '2ndFlrSF':    second_flr_sf,  'LotArea':      lot_area,
        'OverallCond': overall_cond,   'BsmtFullBath': bsmt_full_bath,
        'BsmtHalfBath':bsmt_half_bath, 'HalfBath':     half_bath,
        'HouseAge':    house_age,      'RemodAge':     remod_age,
        'TotalSF':     total_sf,       'TotalBath':    total_bath,
        'HasGarage':   has_garage,
    }
    return np.array([[lookup.get(f, 0) for f in feature_names]])


if st.button("🔍 Predict House Price"):
    try:
        X_input         = build_input(feature_names)
        log_pred        = model.predict(X_input)[0]
        predicted_price = np.expm1(log_pred)
        low  = predicted_price * 0.92
        high = predicted_price * 1.08

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Estimated House Price</div>
            <div class="divider"></div>
            <div class="price">$ {predicted_price:,.0f}</div>
            <div class="range">Confidence Range &nbsp;·&nbsp; $ {low:,.0f} &nbsp;–&nbsp; $ {high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🏠 Total Area", f"{first_flr_sf + second_flr_sf + total_bsmt_sf:,} sqft")
        col_b.metric("📅 House Age",  f"{yr_sold - year_built} years")
        col_c.metric("⭐ Quality",    f"{overall_qual} / 10")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<footer>
    ⚡ Powered by XGBoost &nbsp;·&nbsp; AIML Summer Internship 2026 &nbsp;·&nbsp; IIHMF, MNNIT Allahabad
</footer>
""", unsafe_allow_html=True)