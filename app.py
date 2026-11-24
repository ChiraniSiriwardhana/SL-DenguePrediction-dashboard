"""Sri Lanka weekly dengue forecasting dashboard (Streamlit, dark theme).

Pages (sidebar sections)
  Forecast            Latest Forecast (forecast_next_week.csv, all districts) · Dengue Predictor (what-if tool) · Latest vs Predictor
  Data                Dataset overview
  Models              All models · Feature importance · Hybrid model · RF vs XGBoost · Best model & why
  Research & info     Research · About

Files (repository root, or in a folder called data/):
  required : predictions.csv, model_comparison.csv, risk_thresholds.csv, forecast_next_week.csv,
             sri_lanka_districts.geojson, Sri_Lanka_Dengue_Dataset.csv
  Dengue Predictor : rf_model.joblib, predictor_params.json, district_week_profile.csv, whatif_backtest.csv,
             plus the model files of the stacking members that have a weight above 0 (e.g. catboost_model.cbm)
  optional : feature_importance.csv, training_time.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sri Lanka Dengue Forecast", page_icon="🦟", layout="wide")

BASE = Path(__file__).parent
MODELS = ["Hybrid ARIMA + RF", "ARIMA", "Linear Regression", "Naive", "Random Forest",
          "XGBoost", "LightGBM", "CatBoost", "Stacking"]
RISK_COLORS = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}
RISK_ICON = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
RF_COLOR, XGB_COLOR, OTHER_COLOR, BEST_COLOR = "#2DD4BF", "#FB923C", "#5B7DB1", "#22C55E"
AVG = "Average of all models"
FAMILY = {"Naive": "Baseline", "Linear Regression": "Baseline", "ARIMA": "Time series", "Random Forest": "Bagging",
          "XGBoost": "Boosting", "LightGBM": "Boosting", "CatBoost": "Boosting", "Stacking": "Stacking",
          "Hybrid ARIMA + RF": "Hybrid"}
FAMILY_COLORS = {"Baseline": "#9CA3AF", "Time series": "#4DA3FF", "Bagging": "#2DD4BF", "Boosting": "#FB923C",
                 "Stacking": "#EAB308", "Hybrid": "#A855F7"}
EDGES = [5, 20, 50]                                   # prediction-size bands used for forecast ranges
BANDS, BAND_LABELS = [-1, 10, 50, 200, 1e9], ["0–10", "11–50", "51–200", "over 200"]
TRAIN_END = pd.Timestamp("2020-12-28")
RAIN, TMAX, TMIN, WIND = "Total Precipitation (mm)", "Avg Max Temp (°C)", "Avg Min Temp (°C)", "Avg Wind Speed (km/h)"
DROP_COLS = ["Total Rain (mm)", "Avg Sunrise Time", "Avg Sunset Time"]
# Training MAE of ARIMA and of the hybrid (printed in notebook Section 14)
HYBRID_TRAIN_MAE = {"ARIMA": 10.646, "Hybrid ARIMA + RF": 8.081}

FEATURE_GROUPS = {
    "Case history": ["Number_of_Cases", "cases_lag1", "cases_lag2", "cases_lag4", "cases_lag8", "cases_lag12",
                     "cases_roll4_mean"],
    "Current weather": [RAIN, TMAX, TMIN, WIND],
    "Rainfall history": ["rain_lag2", "rain_lag4", "rain_lag6", "rain_lag8", "rain_lag10",
                         "rain_sum4", "rain_sum8", "rain_sum12"],
    "Temperature history": ["maxtemp_lag4", "maxtemp_lag8"],
    "Season": ["Week_sin", "Week_cos"],
    "District": ["District_code"],
    "ARIMA forecast": ["arima_pred"],
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
GROUP_COLORS = {"Case history": "#4DA3FF", "Current weather": "#F59E0B", "Rainfall history": "#2DD4BF",
                "Temperature history": "#FB923C", "Season": "#A78BFA", "District": "#9CA3AF",
                "ARIMA forecast": "#F43F5E"}


# ------------------------------------------------------------------
# Styling
st.markdown("""
<style>

/* Import Poppins Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
/* Fix white strip at top/bottom edges while scrolling */
/* Remove white strip at top/bottom */
html, body {
    background-color: #0B1220 !important;
    overscroll-behavior: none;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background-color: #0B1220 !important;
}

[data-testid="stDecoration"] {
    display: none;
}
.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* KPI Cards */
.kpi {
    background: #E3EDF7;
    border: 1px solid #C5D7E8;
    border-bottom: 3px solid #9BB8D3;
    border-radius: 14px;
    padding: 14px 18px;
    height: 100%;
    box-shadow: 0 4px 10px rgba(30, 50, 70, 0.10);
}

.kpi .label {
    font-size: .72rem;
    color: #52677F;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.kpi .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.3;
    color: #193653;
}

.kpi .sub {
    font-size: .8rem;
    color: #52677F;
}

/* Pills */
.pill {
    display: inline-block;
    padding: 0 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.3rem;
}

/* Note */
.note {
    background: #F4EBDD;
    border-left: 4px solid #D6A96C;
    border-radius: 8px;
    padding: 10px 14px;
    color: #654F35;
    font-size: .9rem;
    margin-bottom: .8rem;
}

/* General Cards */
.card {
    background: #EAE5F4;
    border: 1px solid #D5CBE8;
    border-bottom: 3px solid #B9A9D5;
    border-radius: 14px;
    padding: 16px 20px;
    height: 100%;
    box-shadow: 0 4px 10px rgba(50, 40, 70, 0.10);
}

.card h4 {
    margin: 0 0 .4rem 0;
    font-size: 1rem;
    color: #51416D;
}

.card p {
    margin: 0;
    font-size: .9rem;
    color: #625875;
}

/* Tags */
.tag {
    font-size: .7rem;
    padding: 1px 8px;
    border-radius: 999px;
    border: 1px solid #B9A9D5;
    color: #51416D;
}

/* Flow Cards */
.flow {
    display: flex;
    gap: 10px;
    align-items: stretch;
    flex-wrap: wrap;
    margin: .4rem 0 1rem 0;
}

.flow .box {
    flex: 1;
    min-width: 170px;
    background: #DFEEE9;
    border: 1px solid #C3DDD3;
    border-bottom: 3px solid #9FC7B7;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 4px 10px rgba(30, 60, 50, 0.10);
}

.flow .box b {
    color: #205B4B;
}

.flow .box span {
    color: #52766A;
    font-size: .85rem;
}

.flow .arrow {
    align-self: center;
    color: #6B9C87;
    font-size: 1.6rem;
}

/* Card Hover Effect */
.kpi:hover,
.card:hover,
.flow .box:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(30, 50, 70, 0.15);
    transition: all 0.2s ease;
}

/* Main Dashboard Header */
.hero-header {
    background: linear-gradient(110deg, #14532D, #228B5A);
    border: 1px solid #2F855A;
    border-radius: 18px;
    padding: 36px 40px;
    width: 100%;
    height: auto;
    min-height: 210px;
    box-sizing: border-box;
    overflow: visible;
    display: block;
    margin: 0 0 32px 0;
}

.hero-header h1 {
    font-family: 'Poppins', sans-serif;
    color: #FFFFFF;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.35;
    letter-spacing: 0.3px;
}

.hero-header p {
    font-family: 'Poppins', sans-serif;
    color: #DCFCE7;
    font-size: 1.08rem;
    font-weight: 400;
    margin: 12px 0 0 0;
    line-height: 1.6;
}

/* Responsive Header */
@media (max-width: 768px) {
    .hero-header {
        padding: 20px;
        min-height: 130px;
    }

    .hero-header h1 {
        font-size: 1.6rem;
    }

    .hero-header p {
        font-size: 0.9rem;
    }
}

/* Sidebar Navigation */
.stSidebar [data-testid="stSidebarNav"] a {
    font-size: 1.08rem;
    font-weight: 600;
}

.stSidebar [data-testid="stSidebarNav"] span {
    font-size: 1.05rem;
}

.stSidebar [data-testid="stSidebarNav"] section {
    font-size: 1.1rem;
    font-weight: 700;
}

/* Sidebar Acknowledgement */
.sidebar-footer {
    border-top: none;
    margin-top: 4px;
    padding-top: 6px;
    color: #B8C9BE;
    font-size: 0.78rem;
    line-height: 1.6;
    overflow-wrap: break-word;
}

.sidebar-footer strong {
    color: #D9FBE5;
    font-weight: 700;
}

.sidebar-footer .credit {
    margin-top: 8px;
}

/* Page Titles */
.stApp [data-testid="stMain"] h1 {
    font-size: 2.15rem !important;
    line-height: 1.3 !important;
}
/* Best Model and Why Cards Only */
.card.best-model-card {
    background: #0B1220 !important;
    border: 1px solid #263247 !important;
    border-bottom: 3px solid #263247 !important;
    box-shadow: none;
}

.card.best-model-card h4 {
    color: #E5E7EB !important;
}

.card.best-model-card p {
    color: #AAB4C5 !important;
}

.card.best-model-card .tag {
    border-color: #52677F;
    color: #C9D6EC;
}
/* ===== Larger content text (headers and hero banner are not changed) ===== */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] td,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] th {
    font-size: 1.1rem;
    line-height: 1.7;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    font-size: 1rem !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p {
    font-size: 1.05rem !important;
}
[data-baseweb="select"] div,
[data-baseweb="input"] input {
    font-size: 1.05rem !important;
}
button[data-baseweb="tab"] p {
    font-size: 1.1rem !important;
}
.stButton button p,
.stDownloadButton button p,
[data-testid="stFormSubmitButton"] button p {
    font-size: 1.05rem !important;
}
[data-testid="stAlert"] p {
    font-size: 1.05rem !important;
}

/* custom cards and boxes */
.note { font-size: 1.05rem; line-height: 1.6; }
.card h4 { font-size: 1.15rem; }
.card p { font-size: 1.05rem; line-height: 1.6; }
.kpi .label { font-size: .8rem; }
.kpi .sub { font-size: .95rem; }
.flow .box b { font-size: 1.1rem; }
.flow .box span { font-size: 1rem; }
.tag { font-size: .8rem; }
.sidebar-footer { font-size: .85rem; }

/* keep the hero banner text at its original size */
[data-testid="stMain"] .hero-header p {
    font-size: 1.08rem !important;
    line-height: 1.6 !important;
}

</style>
""", unsafe_allow_html=True)


def render_app_header():
    st.markdown("""
    <div class="hero-header">
        <h1>🦟 Sri Lanka Dengue Risk Prediction</h1>
        <p>Weekly District-Level Dengue Forecasting Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


def kpi(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value"{style}>{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def card(title, text, tag="", extra_class=""):
    tag_html = f' <span class="tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="card {extra_class}"><h4>{title}{tag_html}</h4><p>{text}</p></div>',
        unsafe_allow_html=True
    )


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), font=dict(color="#C9D6EC"))
    fig.update_xaxes(gridcolor="#1F2B45")
    fig.update_yaxes(gridcolor="#1F2B45")
    return fig


DISCLAIMER = ("⚠️ This is a model-based estimate from a university research project. It is <b>not</b> an official "
              "health warning and does not confirm an outbreak. For official information, contact the Ministry of "
              "Health or the National Dengue Control Unit.")


# ------------------------------------------------------------------ data loading
def find(name):
    for p in (BASE / name, BASE / "outputs" / name, BASE / "models" / name, BASE / "data" / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


@st.cache_data
def load_results():
    pred = pd.read_csv(find("predictions.csv"), parse_dates=["Week_Start_Date", "Target_Week"])
    comp = pd.read_csv(find("model_comparison.csv"))
    thr = pd.read_csv(find("risk_thresholds.csv"))
    fc = pd.read_csv(find("forecast_next_week.csv"), parse_dates=["Last_data_week", "Forecast_week"])
    with open(find("sri_lanka_districts.geojson"), encoding="utf-8") as f:
        geo = json.load(f)
    comp["Model"] = (comp["Model"].str.replace(" (Tuned)", "", regex=False)
                     .str.replace(" (convex weights)", "", regex=False))
    return pred, comp, thr, fc, geo



def load_optional(name):
    try:
        return pd.read_csv(find(name))
    except FileNotFoundError:
        return None


@st.cache_data
def load_raw():
    d = pd.read_csv(find("Sri_Lanka_Dengue_Dataset.csv"))
    d["Week_Start_Date"] = pd.to_datetime(d["Week_Start_Date"], format="%m/%d/%Y")
    d["Week_End_Date"] = pd.to_datetime(d["Week_End_Date"], format="%m/%d/%Y")
    return d.sort_values(["District", "Week_Start_Date"]).reset_index(drop=True)


@st.cache_data
def load_model_dataset():
    """The model dataset of notebook Section 8 (cleaned data + lag/rolling/season features + target)."""
    f = load_raw().drop(columns=DROP_COLS).copy()
    g = f.groupby("District")
    for lag in [1, 2, 4, 8, 12]:
        f[f"cases_lag{lag}"] = g["Number_of_Cases"].shift(lag)
    f["cases_roll4_mean"] = g["Number_of_Cases"].transform(lambda s: s.shift(1).rolling(4).mean())
    for lag in [2, 4, 6, 8, 10]:
        f[f"rain_lag{lag}"] = g[RAIN].shift(lag)
    for w in [4, 8, 12]:
        f[f"rain_sum{w}"] = g[RAIN].transform(lambda s: s.shift(1).rolling(w).sum())
    for lag in [4, 8]:
        f[f"maxtemp_lag{lag}"] = g[TMAX].shift(lag)
    f["target_next_week_cases"] = g["Number_of_Cases"].shift(-1)
    f["Week_sin"] = np.sin(2 * np.pi * f["Week"] / 52)
    f["Week_cos"] = np.cos(2 * np.pi * f["Week"] / 52)
    f["District_code"] = f["District"].astype("category").cat.codes
    required = [c for fs in FEATURE_GROUPS.values() for c in fs if c in f.columns] + ["target_next_week_cases"]
    return f.dropna(subset=required).reset_index(drop=True)


@st.cache_data
def lag_stats():
    """Within-district autocorrelation of cases and Spearman correlation of lagged weather (notebook Section 7.7)."""
    d = load_raw()
    ac, rain_c, temp_c = [], [], []
    series = {k: g for k, g in d.groupby("District")}
    for lag in range(1, 13):
        ac.append(np.mean([g["Number_of_Cases"].autocorr(lag) for g in series.values()]))
    for lag in range(0, 13):
        rain_c.append(np.mean([g["Number_of_Cases"].corr(g[RAIN].shift(lag), method="spearman") for g in series.values()]))
        temp_c.append(np.mean([g["Number_of_Cases"].corr(g[TMAX].shift(lag), method="spearman") for g in series.values()]))
    return ac, rain_c, temp_c


try:
    pred, comp, thr, fc, geo = load_results()
except FileNotFoundError as err:
    st.error(f"File not found: **{err}**. Upload the result files from the notebook (Section 15) and "
             "`sri_lanka_districts.geojson` to the GitHub repository, next to `app.py`.")
    st.stop()

model_cols = [m for m in MODELS if m in pred.columns]
fc_models = [m for m in model_cols if m in fc.columns]
comp = comp[comp["Model"].isin(model_cols)].set_index("Model")
best_model = comp["MAE (Test)"].idxmin()
pred = pred.merge(thr, on="District", how="left")
fc = fc.merge(thr, on="District", how="left")
pred[AVG] = pred[model_cols].mean(axis=1)          # simple average of all models
val = pred[pred["Split"] == "Validation"]
test = pred[pred["Split"] == "Test"]
train_time = load_optional("training_time.csv")
TT = {} if train_time is None else dict(zip(train_time["Model"], train_time["Training_time_sec"]))


# ------------------------------------------------------------------ helpers
def risk_of(values, p50, p80):
    v = np.asarray(values, dtype=float)
    return np.where(v <= p50, "Low", np.where(v <= p80, "Medium", "High"))


def interval_band(model):
    """10th / 90th percentile of validation errors (actual - predicted) in each prediction-size band."""
    res = (val["Actual_next_week"] - val[model]).values
    band_of = np.digitize(val[model].values, EDGES)
    out = {}
    for k in range(4):
        r = res[band_of == k]
        if len(r) < 30:
            r = res
        out[k] = (float(np.percentile(r, 10)), float(np.percentile(r, 90)))
    return out


def apply_interval(p, band):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    k = np.digitize(p, EDGES)
    lo = np.clip(p + np.array([band[i][0] for i in k]), 0, None)
    hi = np.clip(p + np.array([band[i][1] for i in k]), 0, None)
    return lo, hi


def test_coverage(model):
    lo, hi = apply_interval(test[model].values, interval_band(model))
    a = test["Actual_next_week"].values
    return 100 * float(np.mean((a >= lo) & (a <= hi)))


def risk_accuracy(frame, model):
    a = risk_of(frame["Actual_next_week"], frame["p50"], frame["p80"])
    p = risk_of(frame[model], frame["p50"], frame["p80"])
    return 100 * float(np.mean(a == p))


def model_metrics(split):
    """MAE, RMSE, R2, MAPE (zero-case weeks excluded) and training time for every model."""
    f = pred[pred["Split"] == split]
    a = f["Actual_next_week"].values.astype(float)
    nz = a > 0
    rows = []
    for m in model_cols:
        p = f[m].values.astype(float)
        e = a - p
        rows.append({"Model": m, "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                     "R²": 1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2),
                     "MAPE %": 100 * np.mean(np.abs(e[nz] / a[nz])),
                     "Training time (s)": TT.get(m, np.nan)})
    return pd.DataFrame(rows), 100 * (1 - nz.mean())


def band_matrix(frame, models, metric="MAE"):
    t = frame.copy()
    t["Band"] = pd.cut(t["Actual_next_week"], BANDS, labels=BAND_LABELS)
    rows = []
    for m in models:
        e = t["Actual_next_week"] - t[m]
        v = e.abs().groupby(t["Band"], observed=True).mean() if metric == "MAE" \
            else (e ** 2).groupby(t["Band"], observed=True).mean() ** 0.5
        rows.append(v.rename(m))
    return pd.DataFrame(rows).reindex(columns=BAND_LABELS)


def metric_bar(df, col, lower_better, height=300, log_x=False):
    d = df.dropna(subset=[col]).sort_values(col, ascending=lower_better)
    best = d["Model"].iloc[0]
    d = d.assign(Colour=np.where(d["Model"] == best, "Best", "Other"))
    fig = px.bar(d, x=col, y="Model", orientation="h", text=col, color="Colour",
                 color_discrete_map={"Best": BEST_COLOR, "Other": OTHER_COLOR})
    fig.update_traces(texttemplate="%{text:.3g}")
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title=col,
                      yaxis=dict(categoryorder="array", categoryarray=list(d["Model"])[::-1]),
                      title=f"{col} ({'lower' if lower_better else 'higher'} is better)")
    if log_x:
        fig.update_xaxes(type="log")
    return style(fig, height)


def risk_map(m, selected):
    """Latest Forecast map: every district coloured by the risk level of its forecast."""
    kw = dict(geojson=geo, locations="District", featureidkey="properties.District", color="Risk",
              color_discrete_map=RISK_COLORS, category_orders={"Risk": ["Low", "Medium", "High"]},
              hover_name="District", hover_data={"Forecast": ":.0f", "Last week": ":.0f", "Risk": False, "District": False},
              zoom=6.1, center={"lat": 7.85, "lon": 80.75}, opacity=0.9, height=560)
    outline = dict(geojson=geo, locations=[selected], z=[1], featureidkey="properties.District",
                   colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                   marker=dict(line=dict(color="white", width=3)), hoverinfo="skip")
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(m, map_style="carto-darkmatter", **kw)
        fig.add_trace(go.Choroplethmap(**outline))
    else:
        fig = px.choropleth_mapbox(m, mapbox_style="carto-darkmatter", **kw)
        fig.add_trace(go.Choroplethmapbox(**outline))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D6EC"),
                      legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(18,27,46,0.75)", title_text=""))
    return fig


def risk_map_single(district, level, label):
    """Dengue Predictor map: all districts dark grey; the chosen district is coloured by its risk level."""
    names = sorted(thr["District"])
    m = pd.DataFrame({"District": names})
    m["State"] = np.where(m["District"] == district, level, "Other districts")
    m["Label"] = np.where(m["District"] == district, label, m["District"])
    cmap = dict(RISK_COLORS, **{"Other districts": "#1B2742", "Waiting for prediction": "#3B5B92"})
    kw = dict(geojson=geo, locations="District", featureidkey="properties.District", color="State",
              color_discrete_map=cmap, category_orders={"State": ["Low", "Medium", "High", "Waiting for prediction", "Other districts"]},
              hover_name="Label", hover_data={"State": False, "District": False, "Label": False},
              zoom=6.1, center={"lat": 7.85, "lon": 80.75}, opacity=0.92, height=540)
    outline = dict(geojson=geo, locations=[district], z=[1], featureidkey="properties.District",
                   colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                   marker=dict(line=dict(color="white", width=3)), hoverinfo="skip")
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(m, map_style="carto-darkmatter", **kw)
        fig.add_trace(go.Choroplethmap(**outline))
    else:
        fig = px.choropleth_mapbox(m, mapbox_style="carto-darkmatter", **kw)
        fig.add_trace(go.Choroplethmapbox(**outline))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D6EC"),
                      legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(18,27,46,0.75)", title_text=""))
    return fig


# ------------------------------------------------------------------ saved models (used only by the Dengue Predictor)
@st.cache_resource
def _load_params():
    try:
        with open(find("predictor_params.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_resource
def _load_joblib(fname):
    try:
        return joblib.load(find(fname))
    except FileNotFoundError:
        return None


@st.cache_resource
def _load_xgb():
    try:
        import xgboost as xgb_lib
        booster = xgb_lib.Booster()
        booster.load_model(str(find("xgboost_model.json")))
        return booster
    except (FileNotFoundError, ImportError):
        return None


@st.cache_resource
def _load_lgbm():
    try:
        import lightgbm as lgb_lib
        return lgb_lib.Booster(model_file=str(find("lightgbm_model.txt")))
    except (FileNotFoundError, ImportError):
        return None


@st.cache_resource
def _load_cat():
    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor()
        cat.load_model(str(find("catboost_model.cbm")))
        return cat
    except (FileNotFoundError, ImportError):
        return None


class Models:
    """Each attribute loads its file on first use and keeps it in memory afterwards (None if the file is missing)."""
    params = property(lambda self: _load_params())
    rf = property(lambda self: _load_joblib("rf_model.joblib"))
    xgb = property(lambda self: _load_xgb())
    lgbm = property(lambda self: _load_lgbm())
    cat = property(lambda self: _load_cat())


M = Models()


def stack_weights():
    """Stacking weights above zero. Only these models are loaded, so a member with weight 0 needs no file."""
    w = (M.params or {}).get("stacking_weights", {})
    return {m: float(v) for m, v in w.items() if float(v) > 1e-9}


def feature_models(feats, district, want):
    """Forecast of the models that use the 24 features. Only the models in `want` (and what they need) are loaded."""
    X = pd.DataFrame([feats])
    sw = stack_weights()
    need = set(want) | (set(sw) if "Stacking" in want else set())
    out = {}
    if "Random Forest" in need and M.rf is not None:
        out["Random Forest"] = float(M.rf.predict(X[list(M.rf.feature_names_in_)])[0])
    if "XGBoost" in need and M.xgb is not None:
        import xgboost as xgb_lib
        names = M.xgb.feature_names
        dm = xgb_lib.DMatrix(X[list(names)] if names else X)
        n = int((M.params or {}).get("xgb_n_trees", 0))         # trees up to the best iteration (early stopping)
        out["XGBoost"] = float((M.xgb.predict(dm, iteration_range=(0, n)) if n else M.xgb.predict(dm))[0])
    if "LightGBM" in need and M.lgbm is not None:
        out["LightGBM"] = float(M.lgbm.predict(X[M.lgbm.feature_name()])[0])
    if "CatBoost" in need and M.cat is not None:
        out["CatBoost"] = float(M.cat.predict(X[list(M.cat.feature_names_)])[0])
    if "Linear Regression" in need and M.params and "linear" in M.params:
        lin = M.params["linear"]
        row = dict.fromkeys(lin["columns"], 0.0)
        for k, v in feats.items():
            if k in row:
                row[k] = float(v)
        if f"District_{district}" in row:
            row[f"District_{district}"] = 1.0
        out["Linear Regression"] = float(np.dot([row[c] for c in lin["columns"]], lin["coef"]) + lin["intercept"])
    if "Stacking" in want and sw and all(m in out for m in sw):
        out["Stacking"] = float(sum(w * out[m] for m, w in sw.items()))
    return {m: max(out[m], 0.0) for m in want if m in out}


# ------------------------------------------------------------------ "Dengue Predictor": district-week profile

SRC_INPUT = "✍️ You entered"
SRC_CALC = "🧮 Calculated from your cases"
SRC_RAIN = "🌧️ District-week average × rainfall setting"
SRC_MED = "📊 District-week median"
SRC_MEAN = "📊 District-week average"
SRC_FIXED = "📅 Calculated from district and week"


@st.cache_data
def load_profile():
    """Median cases and mean weather for every district and week of the year (all years, from the notebook)."""
    try:
        p = pd.read_csv(find("district_week_profile.csv"))
    except FileNotFoundError:
        return None
    return {(r.District, int(r.Week)): {"cases_median": float(r.cases_median), "rain_mean": float(r.rain_mean),
                                        "tmax_mean": float(r.tmax_mean), "tmin_mean": float(r.tmin_mean),
                                        "wind_mean": float(r.wind_mean)} for r in p.itertuples()}


@st.cache_data
def load_backtest():
    try:
        return pd.read_csv(find("whatif_backtest.csv"))
    except FileNotFoundError:
        return None


# NOTE: profile_value and whatif_features are copied word for word into the notebook export cell, so the backtest
# in whatif_backtest.csv uses exactly the logic of this page.
def profile_value(prof, district, week, k, col):
    """Profile value for the week k weeks before `week` (negative k = after). Weeks wrap around a 52-week year;
    week 53 is treated as week 52."""
    w = (min(int(week), 52) - k - 1) % 52 + 1
    return float(prof[(district, w)][col])


def whatif_features(district, codes, prof, week, c0, c1, c2, c3, c4, rain_scale):
    """The 24 model features from five case counts, a rainfall setting and the district-week profile."""
    pv = lambda k, col: profile_value(prof, district, week, k, col)
    r = {k: rain_scale * pv(k, "rain_mean") for k in range(13)}
    feats = {
        "Number_of_Cases": float(c0), "cases_lag1": float(c1), "cases_lag2": float(c2), "cases_lag4": float(c4),
        "cases_lag8": pv(8, "cases_median"), "cases_lag12": pv(12, "cases_median"),
        "cases_roll4_mean": (c1 + c2 + c3 + c4) / 4.0,
        RAIN: r[0], TMAX: pv(0, "tmax_mean"), TMIN: pv(0, "tmin_mean"), WIND: pv(0, "wind_mean"),
        "rain_lag2": r[2], "rain_lag4": r[4], "rain_lag6": r[6], "rain_lag8": r[8], "rain_lag10": r[10],
        "rain_sum4": sum(r[k] for k in range(1, 5)), "rain_sum8": sum(r[k] for k in range(1, 9)),
        "rain_sum12": sum(r[k] for k in range(1, 13)),
        "maxtemp_lag4": pv(4, "tmax_mean"), "maxtemp_lag8": pv(8, "tmax_mean"),
        "Week_sin": float(np.sin(2 * np.pi * week / 52)), "Week_cos": float(np.cos(2 * np.pi * week / 52)),
        "District_code": codes[district],
    }
    sources = {"Number_of_Cases": SRC_INPUT, "cases_lag1": SRC_INPUT, "cases_lag2": SRC_INPUT, "cases_lag4": SRC_INPUT,
               "cases_lag8": SRC_MED, "cases_lag12": SRC_MED, "cases_roll4_mean": SRC_CALC,
               RAIN: SRC_RAIN, TMAX: SRC_MEAN, TMIN: SRC_MEAN, WIND: SRC_MEAN,
               "rain_lag2": SRC_RAIN, "rain_lag4": SRC_RAIN, "rain_lag6": SRC_RAIN, "rain_lag8": SRC_RAIN,
               "rain_lag10": SRC_RAIN, "rain_sum4": SRC_RAIN, "rain_sum8": SRC_RAIN, "rain_sum12": SRC_RAIN,
               "maxtemp_lag4": SRC_MEAN, "maxtemp_lag8": SRC_MEAN,
               "Week_sin": SRC_FIXED, "Week_cos": SRC_FIXED, "District_code": SRC_FIXED}
    return feats, sources



# ================================================================== PAGE: LATEST FORECAST 
def page_latest_forecast():
    st.title("Latest-Week Dengue Forecast")
    st.caption(f"Forecast for the week starting {fc['Forecast_week'].iloc[0]:%d %b %Y} · based on reported cases up to "
               f"the week starting {fc['Last_data_week'].iloc[0]:%d %b %Y}")
    st.markdown('<div class="note">⚠️ This is a model-based estimate from a university research project. '
                'It is <b>not</b> an official health warning and does not confirm an outbreak. For official '
                'information, contact the Ministry of Health or the National Dengue Control Unit.</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.3, 1, 1])
    audience = f1.radio("I am a", ["Member of the public", "Health authority / researcher"],
                        horizontal=True, key="lf_audience")
    district = f2.selectbox("District", sorted(fc["District"]), key="lf_district")
    model = f3.selectbox("Model", fc_models,
                         index=fc_models.index(best_model) if best_model in fc_models else 0, key="lf_model")
    st.caption(f"Default model: the one with the lowest test MAE ({best_model}).")
    public = audience.startswith("Member")

    row = fc[fc["District"] == district].iloc[0]
    predicted = float(row[model])
    last = float(row["Current_week_cases"])
    level = risk_of([predicted], row["p50"], row["p80"])[0]
    lo_a, hi_a = apply_interval([predicted], interval_band(model))
    lo, hi = float(lo_a[0]), float(hi_a[0])
    color = RISK_COLORS[level]
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Cases last week", f"{last:.0f}", district)
    with c2:
        kpi("Forecast next week", f"{predicted:.0f}", f"{model} · likely {lo:.0f} – {hi:.0f}")
    with c3:
        pct = f"{100 * (predicted - last) / last:+.0f}% vs last week" if last > 0 else "no cases last week"
        kpi("Change", f"{predicted - last:+.0f}", pct)
    with c4:
        kpi("Risk level",
            f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}">{level}</span>',
            f"Low ≤ {row['p50']:.0f} · Medium ≤ {row['p80']:.0f} cases (district's own history)")
    st.write("")

    m = fc[["District", model, "Current_week_cases", "p50", "p80"]].rename(
        columns={model: "Forecast", "Current_week_cases": "Last week"})
    m["Risk"] = risk_of(m["Forecast"], m["p50"], m["p80"])

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("District risk map")
        st.plotly_chart(risk_map(m, district), width="stretch")
        st.caption("Colours compare each district's forecast with its own usual weekly cases (2007–2020). "
                   "The white outline is the selected district.")
    with right:
        st.subheader("All districts" if not public else "Districts with the highest forecast")
        table = m.sort_values("Forecast", ascending=False).copy()
        table["Risk"] = table["Risk"].map(lambda r: f"{RISK_ICON[r]} {r}")
        table = table[["District", "Last week", "Forecast", "Risk"]]
        shown = table.head(10) if public else table
        st.dataframe(shown, hide_index=True, height=520, width="stretch",
                     column_config={"Last week": st.column_config.NumberColumn(format="%d"),
                                    "Forecast": st.column_config.NumberColumn(format="%d")})
        if not public:
            st.download_button("⬇ Download forecast table (CSV)", table.to_csv(index=False).encode("utf-8"),
                               file_name="dengue_forecast_next_week.csv", mime="text/csv")

    st.subheader(f"Reported vs predicted cases: {district}")
    hist = pred[pred["District"] == district].sort_values("Target_Week").tail(52)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Target_Week"], y=hist["Actual_next_week"], name="Reported cases",
                             line=dict(color="#E6EDF7", width=3)))
    fig.add_trace(go.Scatter(x=hist["Target_Week"], y=hist[model], name=f"Predicted ({model})",
                             line=dict(color="#4DA3FF", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=[row["Forecast_week"]], y=[predicted], mode="markers", name="Next-week forecast",
                             marker=dict(size=12, color="#F43F5E"),
                             error_y=dict(type="data", symmetric=False, array=[hi - predicted],
                                          arrayminus=[predicted - lo], color="#F43F5E")))
    fig.update_layout(yaxis_title="Cases per week")
    st.plotly_chart(style(fig, 380), width="stretch")

    if public:
        trend = "higher than" if predicted > last * 1.1 else "lower than" if predicted < last * 0.9 else "about the same as"
        st.markdown(f"**In simple words:** the model expects about **{predicted:.0f}** reported dengue cases in "
                    f"{district} next week, which is {trend} last week ({last:.0f}). The most likely range is "
                    f"{lo:.0f} to {hi:.0f}. The level **{level}** compares this with what is usual for {district}.")
        
    else:
        st.subheader("How reliable is this forecast?")
        sub = test[test["District"] == district]
        d_mae = float(np.mean(np.abs(sub["Actual_next_week"] - sub[model])))
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            kpi("Typical error, all districts", f"{comp.loc[model, 'MAE (Test)']:.1f}", "test MAE, cases per week")
        with r2:
            kpi(f"Typical error, {district}", f"{d_mae:.1f}", "test MAE, cases per week")
        with r3:
            kpi("Range coverage (test)", f"{test_coverage(model):.0f}%", "target 80%; lower = range too narrow")
        with r4:
            kpi("Risk-level accuracy (test)", f"{risk_accuracy(test, model):.0f}%", "forecast and actual in same level")
        st.caption("The range is built from the validation-period errors of this model, so it is a guide, not a "
                   "guarantee. Errors grow when cases are high.")


# ================================================================== PAGE: DENGUE PREDICTOR (what-if tool)
WHATIF_MODELS = ["Random Forest", "Stacking", "Linear Regression"]
RAIN_OPTIONS = {50: "50% · much drier", 75: "75% · drier", 100: "100% · normal", 150: "150% · wetter", 200: "200% · much wetter"}


def page_whatif():
    raw = load_raw()
    prof = load_profile()
    districts = sorted(raw["District"].unique())
    codes = {d: i for i, d in enumerate(districts)}

    st.title("Dengue Predictor")
    st.caption("Choose a district and a week of the year, enter the recent dengue cases and how wet it has been, "
               "and get the expected cases for the **following week**.")
    
    if prof is None:
        st.error("`district_week_profile.csv` was not found. Run the export cell at the end of the notebook "
                 "and upload `district_week_profile.csv`, `whatif_backtest.csv`, `predictor_params.json` and the "
                 "model files to the GitHub repository.")
        return

    # ---------------- step 1: where and when (outside the form, so the suggested values follow the choice)
    st.markdown("#### 1 · Where and when")
    s1, s2, s3 = st.columns([1.1, 0.9, 1.6])
    district = s1.selectbox("District", districts, key="wi_district")
    latest_week = int(raw.loc[raw["Week_Start_Date"].idxmax(), "Week"])
    week = int(s2.number_input("Week of the year (1–53)", min_value=1, max_value=53, value=latest_week, key="wi_week"))
    typical = [profile_value(prof, district, week, k, "cases_median") for k in range(5)]
    month = (pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=min(week, 52) - 1)).strftime("%B")
    with s3:
        kpi(f"Usual cases in {district}, week {week}", f"{typical[0]:.0f}", f"median of 2007–2024 · around {month}")
    st.write("")

    with st.form("whatif"):
        st.markdown("#### 2 · Recent dengue cases")
        labels = ["This week", "1 week ago", "2 weeks ago", "3 weeks ago", "4 weeks ago"]
        cols = st.columns(5)
        cases = [cols[i].number_input(labels[i], min_value=0, value=int(round(typical[i])), step=1,
                                      key=f"wi_c{i}_{district}_{week}") for i in range(5)]
        st.caption("Boxes start at the usual (median) cases for this district and week. Replace them with the "
                   "reported numbers.")
        st.markdown("#### 3 · Recent rainfall")
        rain_pct = st.select_slider("Recent rainfall vs normal", options=list(RAIN_OPTIONS), value=100,
                                    format_func=lambda v: RAIN_OPTIONS[v], key=f"wi_rain_{district}",
                                    help="Rain of the last 12 weeks, compared with the usual (average) rain of this "
                                         "district in the same weeks of the year. It scales this week's rain, the "
                                         "rain 2–10 weeks ago and the 4, 8 and 12-week rainfall totals.")
        st.markdown("#### 4 · Model")
        model = st.radio("Model", WHATIF_MODELS, horizontal=True, key="wi_model",
                         help="ARIMA and the hybrid model are not offered here because they need the full weekly "
                              "case history of the district.")
        w = stack_weights()
        stack_txt = ("A weighted mix of " + ", ".join(f"{m} {v:.1f}" for m, v in w.items())
                     + " (weights found in the notebook).") if w else "A weighted mix of tree models."
        info = {"Random Forest": "Many decision trees averaged. It has the best validation MAE of the tree models.",
                "Stacking": stack_txt,
                "Linear Regression": "A simple weighted sum of the inputs. Fast, and it had the lowest test RMSE of all models."}
        st.caption(" · ".join(f"**{m}:** {t}" for m, t in info.items()))
        submitted = st.form_submit_button("Predict next week", type="primary")

    if submitted:
        st.session_state["wi_done"] = (district, week)
    t = thr.set_index("District").loc[district]
    if st.session_state.get("wi_done") != (district, week):
        left, right = st.columns([1.15, 1])
        with left:
            st.plotly_chart(risk_map_single(district, "Waiting for prediction", district), width="stretch")
        with right:
            note("Fill in steps 2 to 4 and press <b>Predict next week</b>. The expected cases, the risk level and the "
                 f"colour of <b>{district}</b> on the map will appear here.")
        return

    c0, c1, c2, c3, c4 = (float(x) for x in cases)
    feats, sources = whatif_features(district, codes, prof, week, c0, c1, c2, c3, c4, rain_pct / 100.0)
    preds = feature_models(feats, district, [model])
    if model not in preds:
        st.error(f"The saved files for **{model}** were not found. Upload `rf_model.joblib`, `predictor_params.json` "
                 "and the model files of the stacking members (for example `catboost_model.cbm`) exported by the "
                 "notebook.")
        return
    p = preds[model]
    level = str(risk_of([p], t["p50"], t["p80"])[0])
    color = RISK_COLORS[level]
    usual_next = profile_value(prof, district, week, -1, "cases_median")
    bt = load_backtest()
    row = bt[bt["Model"] == model].iloc[0] if bt is not None and (bt["Model"] == model).any() else None

    ds = load_model_dataset()
    train_max = ds[(ds["District"] == district) & (ds["Week_Start_Date"] <= TRAIN_END)]["Number_of_Cases"].max()
    if max(cases) > train_max:
        st.warning(f"A case number you entered ({max(cases):.0f}) is larger than any week in {district} during "
                   f"2007–2020 ({train_max:.0f}). Tree models cannot predict beyond what they have seen, so a very "
                   "large outbreak will probably be under-estimated.")
    if c0 + c1 + c2 + c3 + c4 > 0 and max(cases) > 4 * typical[0] + 10:
        st.info("You entered much higher cases than usual for this week. Cases 8 and 12 weeks ago (and the "
                "temperatures) still come from a typical week, so this input is partly inconsistent and the "
                "forecast is less reliable.")

    st.markdown("#### 5 · Result")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Expected cases next week", f"{p:.0f}", f"{model} · {district}")
    with k2:
        kpi("Risk level",
            f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}">{level}</span>',
            f"Low ≤ {t['p50']:.0f} · Medium ≤ {t['p80']:.0f} cases (district's own history)")
    with k3:
        ratio = f"{p / usual_next:.1f}× the usual" if usual_next > 0 else "usual week had no cases"
        kpi("Compared with a usual week", ratio, f"usual next week: {usual_next:.0f} cases")
    with k4:
        kpi("Typical error with these inputs", "n/a" if row is None else f"± {row['MAE_whatif']:.0f}",
            "test MAE of the backtest, cases per district-week")
    st.write("")

    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(risk_map_single(district, level, f"{district}: {level} risk (about {p:.0f} cases)"),
                        width="stretch")
        st.caption("Only the selected district is coloured. The colour is the risk level compared with this "
                   "district's usual weekly cases in 2007–2020, not an official threshold.")
    with right:
        n_in = sum(v in (SRC_INPUT, SRC_CALC) for v in sources.values())
        n_fill = sum(v in (SRC_RAIN, SRC_MED, SRC_MEAN) for v in sources.values())
        n_fix = sum(v == SRC_FIXED for v in sources.values())
        st.markdown(f"**How the model input was built.** {n_in} of the 24 features come from your cases, "
                    f"**{n_fill} were filled with district averages** (rainfall features scaled by your rainfall "
                    f"setting) and {n_fix} were calculated from the district and week.")
        tbl = pd.DataFrame({"Feature": list(feats), "Value": [round(float(v), 3) for v in feats.values()],
                            "Source": [sources[f] for f in feats]})
        st.dataframe(tbl, hide_index=True, height=420, width="stretch")

    st.markdown("#### How much accuracy do these few inputs cost?")
    if bt is None:
        st.info("Upload `whatif_backtest.csv` (export cell of the notebook) to see the backtest.")
    else:
        long = bt.melt(id_vars="Model", value_vars=["MAE_full", "MAE_whatif"], var_name="Inputs", value_name="MAE")
        long["Inputs"] = long["Inputs"].map({"MAE_full": "All 24 features (actual values)", "MAE_whatif": "Only the inputs of this page"})
        fig = px.bar(long, x="Model", y="MAE", color="Inputs", barmode="group",
                     color_discrete_map={"All 24 features (actual values)": "#2DD4BF", "Only the inputs of this page": "#F59E0B"})
        fig.update_layout(title="Test MAE (2023 to July 2024): full features vs this page's inputs", yaxis_title="MAE")
        st.plotly_chart(style(fig, 340), width="stretch")
        if row is not None:
            pct = 100 * (row["MAE_whatif"] / row["MAE_full"] - 1)
            st.markdown(f"**With only these inputs, the test MAE of {model} goes from {row['MAE_full']:.2f} to "
                        f"{row['MAE_whatif']:.2f} ({pct:+.0f}%).**")
        st.caption("Backtest: for every test week the actual cases of the last 5 weeks were entered, rainfall was set "
                   "to 100% of normal, and all other features came from the district-week averages of 2007–2020 "
                   "(no test data used).")

    st.markdown("#### Please remember")
    r1, r2, r3 = st.columns(3)
    with r1:
        card("Cases drive the result", "The prediction mostly depends on the case numbers you enter, so its accuracy "
             "depends on how accurate those numbers are.", "Important")
    with r2:
        card("Weather moves it little", "In this study the models rely mostly on recent cases, so the rainfall setting "
             "is expected to change the result only slightly.", "Expected")
    with r3:
        card("Week numbers wrap around", "Weeks before week 1 count back from week 52, and week 53 uses week 52's "
             "usual values. This is an approximation.", "Approximation")
 #================================================================== PAGE: LATEST VS PREDICTOR
def page_compare():
    st.title("Latest Forecast vs Dengue Predictor")
    st.caption("Two pages, one set of trained models. This page explains what is the same, what is different, "
               "and why the Dengue Predictor asks for so few inputs.")

    note("<b>Short answer:</b> Both pages use the <b>same trained models</b>. The only difference is the "
         "<b>data that goes in</b>. Most of the 24 features are not columns of the dataset. They are calculated "
         "from it (lags, rolling totals, averages). For the Latest Forecast we have the full dataset, so the "
         "notebook can easily calculate all 24 features and feed them to the models. A normal user calculating all these features by hand is not practical. So the Dengue "
         "Predictor asks for a few numbers and fills the rest with averages.")

    # ---------------- side-by-side
    st.subheader("1 · What is the difference")
    side = pd.DataFrame([
        ("Where the prediction is calculated", "In the notebook (Google Colab). The app only reads the saved file "
                                               "`forecast_next_week.csv`.",
         "Live in the app. The saved model files are loaded and run when you press the button."),
        ("Input data", "Real data of the last week in the dataset. All 24 features are real values.",
         "Values typed by the user (5 weeks of cases, a rainfall setting). 16 of the 24 features are averages."),
        ("Districts", "All 25 at once", "One district at a time"),
        ("Week", "The week after the last data week (July 2024)", "Any week of the year the user chooses"),
        ("Models", "Naive, ARIMA, Random Forest, XGBoost, LightGBM, CatBoost, Stacking, Hybrid",
         "Random Forest, Stacking, Linear Regression (ARIMA and Hybrid need the full weekly history)"),
        ("Can the user change the inputs?", "No", "Yes "),
        ("Accuracy", "Best available: every input is real", "Lower: many inputs are averages (see the backtest on that page)"),
        ("Purpose", "Show the forecast for the latest data", "Let users try scenarios"),
    ], columns=["", "Latest Forecast", "Dengue Predictor"])
    st.dataframe(side, hide_index=True, width="stretch")

    # ---------------- flows
    st.subheader("2 · How each page gets its number")
    st.markdown("**Latest Forecast**: the prediction is made in the notebook and imported into the dashboard.")
    st.markdown(
        '<div class="flow">'
        '<div class="box"><b>1 · Notebook</b><br><span>Models are trained on data up to 2020, tuned on 2021–2022 '
        'and tested on 2023 to July 2024.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>2 · Last data week</b><br><span>The 24 real features are built for the last week '
        'in the dataset, for every district.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>3 · Predict</b><br><span>The tuned models predict the following week. No retraining '
        'happens here.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>4 · CSV → dashboard</b><br><span>Results are saved as <code>forecast_next_week.csv</code> '
        'and the app only displays them.</span></div></div>', unsafe_allow_html=True)
    st.markdown("**Dengue Predictor**: the same models, but the app builds the input row itself.")
    st.markdown(
        '<div class="flow">'
        '<div class="box"><b>1 · User input</b><br><span>District, week of the year, cases this week and the 4 '
        'weeks before, rainfall setting.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>2 · Fill the gaps</b><br><span>The other features come from district-week averages '
        '(<code>district_week_profile.csv</code>).</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>3 · 24 features</b><br><span>One complete row of 24 numbers is built.</span></div>'
        '<div class="arrow">➜</div>'
        '<div class="box"><b>4 · Saved model</b><br><span>The model file runs live in the app and returns the '
        'expected cases for the next week.</span></div></div>', unsafe_allow_html=True)

    # ---------------- one week ahead
    st.subheader("3 · Why only one week ahead?")
    c1, c2 = st.columns(2)
    with c1:
        card("The models learned one task",
             "Every model was trained to answer: <i>given the cases and weather up to this week, how many cases "
             "will there be next week?</i> The target in the notebook is next week's cases. They were never "
             "trained to predict 2, 3 or 4 weeks ahead.", "Training")
    with c2:
        card("The inputs for later weeks do not exist yet",
             "The models rely mostly on recent cases (this week, last week, the average of the last 4 weeks because of the life cycle of a mosquito). "
             "For a forecast several weeks after the data ends, those recent weeks have not happened, so their "
             "values are unknown. Using predicted values as inputs is possible but errors add up each step, and "
             "it was not trained or tested in this study.", "Limitation")
   

    # ---------------- dataset columns
    st.subheader("4 · What is in the dataset")
    raw = load_raw()
    st.markdown(f"The dataset has **{raw.shape[1]} columns** and **{len(raw):,} rows** "
                f"(25 districts × {raw.groupby('District').size().iloc[0]} weeks). "
                "Only a few columns go into the models directly.")
    cols = pd.DataFrame([
        ("District", "Name of the district", "Used (becomes `District_code`)"),
        ("Number_of_Cases", "Reported dengue cases in the week", "Used (also the target, shifted by one week)"),
        ("Week_Start_Date, Week_End_Date", "Start and end date of the week", "Used only to order and split the data in time"),
        ("Week", "Week number of the year", "Used (becomes `Week_sin` and `Week_cos`)"),
        ("Avg Max Temp (°C)", "Average maximum temperature", "Used"),
        ("Avg Min Temp (°C)", "Average minimum temperature", "Used"),
        ("Total Precipitation (mm)", "Total rainfall in the week", "Used"),
        ("Avg Wind Speed (km/h)", "Average wind speed", "Used"),
        ("Total Rain (mm)", "Identical to Total Precipitation", "Removed (duplicate)"),
        ("Avg Sunrise Time, Avg Sunset Time", "Sunrise and sunset", "Removed in cleaning"),
        ("Avg Apparent Max / Min Temp (°C)", "Apparent temperatures", "Not used (overlap with real temperatures)"),
        ("Max Wind Gusts (km/h)", "Strongest wind gusts", "Not used (overlaps with wind speed)"),
        ("Avg Daylight Duration (hours)", "Length of daylight", "Not used (almost purely seasonal)"),
        ("Weather Code, Month, Year", "Weather category, month, year", "Not used"),
    ], columns=["Dataset column", "Meaning", "Role in the models"])
    st.dataframe(cols, hide_index=True, width="stretch")

    # ---------------- 24 features
    st.subheader("5 · The 24 features the models use")
    note("<b>Important:</b> most of the 24 features are <b>not columns of the dataset</b>. They are "
         "<b>calculated</b> in the notebook from the dataset: lags (values from earlier weeks), rolling totals and "
         "averages, and a circular version of the week number. All calculations are done separately for each "
         "district and use only past weeks.")
    feats = pd.DataFrame([
        ("Number_of_Cases", "Case history", "Cases this week", "Taken directly from the dataset", "You enter"),
        ("cases_lag1", "Case history", "Cases 1 week ago", "Number_of_Cases moved back 1 week", "You enter"),
        ("cases_lag2", "Case history", "Cases 2 weeks ago", "Number_of_Cases moved back 2 weeks", "You enter"),
        ("cases_lag4", "Case history", "Cases 4 weeks ago", "Number_of_Cases moved back 4 weeks", "You enter"),
        ("cases_lag8", "Case history", "Cases 8 weeks ago", "Number_of_Cases moved back 8 weeks", "Average (district-week median)"),
        ("cases_lag12", "Case history", "Cases 12 weeks ago", "Number_of_Cases moved back 12 weeks", "Average (district-week median)"),
        ("cases_roll4_mean", "Case history", "Average cases of the 4 weeks before this week",
         "Mean of the cases 1, 2, 3 and 4 weeks ago (this week not included)", "Calculated from your cases"),
        ("Total Precipitation (mm)", "Current weather", "Rainfall this week", "Taken directly from the dataset",
         "Average × rainfall setting"),
        ("Avg Max Temp (°C)", "Current weather", "Max temperature this week", "Taken directly from the dataset", "Average"),
        ("Avg Min Temp (°C)", "Current weather", "Min temperature this week", "Taken directly from the dataset", "Average"),
        ("Avg Wind Speed (km/h)", "Current weather", "Wind speed this week", "Taken directly from the dataset", "Average"),
        ("rain_lag2", "Rainfall history", "Rainfall 2 weeks ago", "Rainfall moved back 2 weeks", "Average × rainfall setting"),
        ("rain_lag4", "Rainfall history", "Rainfall 4 weeks ago", "Rainfall moved back 4 weeks", "Average × rainfall setting"),
        ("rain_lag6", "Rainfall history", "Rainfall 6 weeks ago", "Rainfall moved back 6 weeks", "Average × rainfall setting"),
        ("rain_lag8", "Rainfall history", "Rainfall 8 weeks ago", "Rainfall moved back 8 weeks", "Average × rainfall setting"),
        ("rain_lag10", "Rainfall history", "Rainfall 10 weeks ago", "Rainfall moved back 10 weeks", "Average × rainfall setting"),
        ("rain_sum4", "Rainfall history", "Total rain of the previous 4 weeks", "Sum of rainfall 1 to 4 weeks ago (this week not included)",
         "Average × rainfall setting"),
        ("rain_sum8", "Rainfall history", "Total rain of the previous 8 weeks", "Sum of rainfall 1 to 8 weeks ago (this week not included)",
         "Average × rainfall setting"),
        ("rain_sum12", "Rainfall history", "Total rain of the previous 12 weeks", "Sum of rainfall 1 to 12 weeks ago (this week not included)",
         "Average × rainfall setting"),
        ("maxtemp_lag4", "Temperature history", "Max temperature 4 weeks ago", "Max temperature moved back 4 weeks", "Average"),
        ("maxtemp_lag8", "Temperature history", "Max temperature 8 weeks ago", "Max temperature moved back 8 weeks", "Average"),
        ("Week_sin", "Season", "Week of the year as a circle (sine part)", "sin(2π × Week / 52)", "Calculated from the week"),
        ("Week_cos", "Season", "Week of the year as a circle (cosine part)", "cos(2π × Week / 52)", "Calculated from the week"),
        ("District_code", "District", "Number that stands for the district", "Districts in alphabetical order, numbered from 0",
         "Calculated from the district"),
    ], columns=["Feature", "Group", "Meaning", "How it is calculated", "In the Dengue Predictor"])
    st.dataframe(feats, hide_index=True, width="stretch", height=600)
    st.caption("The Hybrid ARIMA + RF model adds one more input, the ARIMA forecast (`arima_pred`), which needs "
               "the full weekly history of the district. The Naive model simply repeats this week's cases.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Typed by the user", "4", "cases now, 1, 2 and 4 weeks ago")
    with k2:
        kpi("Calculated from the user's cases", "1", "average of the previous 4 weeks")
    with k3:
        kpi("Calculated from district and week", "3", "Week_sin, Week_cos, District_code")
    with k4:
        kpi("Filled with averages", "16", "weather and older case lags")

    # ---------------- why averages
    st.subheader("6 · Why the Dengue Predictor uses averages")
    c1, c2 = st.columns(2)
    with c1:
        card("Asking for all 24 values is not practical",
             "Most features are calculated from weeks of history: rainfall 2, 4, 6, 8 and 10 weeks ago, 4, 8 and "
             "12-week rainfall totals, temperatures 4 and 8 weeks ago, and so on. A user would have to look up "
             "and calculate all of them by hand, which most people cannot do. Recent case counts are what a user "
             "is most likely to know.", "Usability")
    with c2:
        card("A simplified tool for this assignment",
             "For this assignment the Dengue Predictor therefore asks for only a few inputs and fills the rest with "
             "district-week averages. The Latest Forecast page, which uses all "
             "24 real features, is the full-accuracy result.", "Design choice")
    



# ================================================================== PAGE: DATASET
def page_dataset():
    raw = load_raw()
    clean = raw.drop(columns=DROP_COLS)
    st.title("Data Overview")
    st.caption("Weekly dengue cases and weather for the 25 districts of Sri Lanka, 2007 to July 2024.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Records", f"{len(raw):,}", "district-weeks")
    with c2:
        kpi("Districts", raw["District"].nunique(), f"{raw.groupby('District').size().iloc[0]} weeks each")
    with c3:
        kpi("Period", f"{raw['Week_Start_Date'].min():%Y}–{raw['Week_Start_Date'].max():%Y}",
            f"to {raw['Week_Start_Date'].max():%d %b %Y}")
    with c4:
        kpi("Columns", f"{raw.shape[1]} → {clean.shape[1]}", "raw → cleaned")
    with c5:
        kpi("Model features", len(FEATURE_GROUPS["Case history"] + FEATURE_GROUPS["Current weather"]
                                   + FEATURE_GROUPS["Rainfall history"] + FEATURE_GROUPS["Temperature history"]
                                   + FEATURE_GROUPS["Season"] + FEATURE_GROUPS["District"]), "after feature engineering")
    st.write("")

    tabs = st.tabs(["Outbreaks", "Distribution (log scale)", "Data quality", "Why lags?", "Cleaned & lagged data"])

    # ---------------- outbreaks
    with tabs[0]:
        weekly = raw.groupby("Week_Start_Date")["Number_of_Cases"].sum().reset_index()
        fig = px.area(weekly, x="Week_Start_Date", y="Number_of_Cases")
        fig.update_traces(line_color="#4DA3FF", fillcolor="rgba(77,163,255,0.25)")
        fig.update_layout(title="National weekly dengue cases (all 25 districts)", xaxis_title="", yaxis_title="Cases per week")
        st.plotly_chart(style(fig, 340), width="stretch")
        yearly = raw.groupby("Year")["Number_of_Cases"].sum()
        top = yearly.sort_values(ascending=False)
        top3 = top.head(3)
        share = 100 * top3.sum() / yearly.sum()
        fig = px.bar(yearly.reset_index(), x="Year", y="Number_of_Cases")
        fig.update_traces(marker_color=np.where(yearly.index.isin(top3.index), "#EF4444", OTHER_COLOR))
        fig.update_layout(title="Total cases per year (red = the three biggest years)", yaxis_title="Cases per year")
        st.plotly_chart(style(fig, 320), width="stretch")
        note(f"<b>Why this matters:</b> three of the {len(yearly)} years ({', '.join(str(y) for y in sorted(top3.index))}) "
             f"contain <b>{share:.0f}%</b> of all recorded cases, although they are only {100 * 3 / len(yearly):.0f}% of the years. "
             "The data are shaped by a few large outbreaks, so there are relatively few extreme weeks to learn from "
             "and the models have to be judged on how they handle them. 2024 only covers January to July.")

    # ---------------- distribution
    with tabs[1]:
        y = raw["Number_of_Cases"]
        a, b = st.columns(2)
        with a:
            fig = px.histogram(y, nbins=60)
            fig.update_traces(marker_color="#4DA3FF")
            fig.update_yaxes(type="log")
            fig.update_layout(title="Raw cases (frequency on a log axis)", xaxis_title="Cases in a district-week",
                              yaxis_title="Number of district-weeks (log)", showlegend=False)
            st.plotly_chart(style(fig, 340), width="stretch")
        with b:
            fig = px.histogram(np.log1p(y), nbins=50)
            fig.update_traces(marker_color="#2DD4BF")
            fig.update_layout(title="log(1 + cases)", xaxis_title="log(1 + cases)",
                              yaxis_title="Number of district-weeks", showlegend=False)
            st.plotly_chart(style(fig, 340), width="stretch")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Zero-case weeks", f"{100 * (y == 0).mean():.1f}%", "district-weeks")
        with k2:
            kpi("10 cases or fewer", f"{100 * (y <= 10).mean():.0f}%", "of district-weeks")
        with k3:
            kpi("Over 200 cases", f"{100 * (y > 200).mean():.1f}%", "of district-weeks")
        with k4:
            kpi("Largest week", f"{int(y.max()):,}", f"median {y.median():.0f}")
        st.markdown("**How to read this.** On the raw scale almost everything is squeezed against zero and a few "
                    "huge outbreak weeks stretch the axis, so the shape cannot be seen. The log scale compresses "
                    "those large values; `log(1 + cases)` is used instead of `log(cases)` because many weeks have "
                    "zero cases and `log(0)` does not exist.")
        st.caption("The log scale is used only for viewing. The models are trained on the original case counts, "
                   "and the large values are kept because they are real outbreaks.")

    # ---------------- data quality
    with tabs[2]:
        gaps = sorted(raw.groupby("District")["Week_Start_Date"].diff().dropna().dt.days.unique().tolist())
        checks = pd.DataFrame([
            ("Missing values", int(raw.isna().sum().sum()), "✅"),
            ("Duplicate rows", int(raw.duplicated().sum()), "✅"),
            ("Duplicate district-week records", int(raw.duplicated(subset=["District", "Week_Start_Date"]).sum()), "✅"),
            ("Districts", raw["District"].nunique(), "✅"),
            ("Weeks per district", ", ".join(str(x) for x in raw.groupby("District").size().unique()), "✅"),
            ("Gap between weeks (days)", ", ".join(str(x) for x in gaps), "✅" if gaps == [7] else "⚠️"),
            ("Total Rain identical to Total Precipitation", bool((raw["Total Precipitation (mm)"] == raw["Total Rain (mm)"]).all()), "⚠️ redundant"),
            ("Share of zero-case weeks", f"{100 * (raw['Number_of_Cases'] == 0).mean():.1f}%", "⚠️ affects MAPE"),
        ], columns=["Check", "Result", "Status"])
        st.dataframe(checks.astype({"Result": str}), hide_index=True, width="stretch")
        r1 = raw[TMAX].corr(raw["Avg Apparent Max Temp (°C)"])
        r2 = raw[TMIN].corr(raw["Avg Apparent Min Temp (°C)"])
        r3 = raw[WIND].corr(raw["Max Wind Gusts (km/h)"])
        g1, g2 = st.columns(2)
        with g1:
            card("Redundant columns removed",
                 "<b>Total Rain (mm)</b> is identical to Total Precipitation. <b>Sunrise and sunset times</b> are "
                 "removed as well. This turns 19 columns into 16.", "Cleaning")
            st.write("")
            card("Zero cases are common",
                 f"{100 * (raw['Number_of_Cases'] == 0).mean():.0f}% of district-weeks report no cases. Percentage "
                 "errors such as MAPE divide by the actual value, so they break down for zeros. MAE, RMSE and R² "
                 "are therefore the main metrics.", "Finding")
        with g2:
            card("Highly overlapping weather columns",
                 f"Apparent temperatures correlate with the real ones (r = {r1:.2f} and {r2:.2f}), and wind gusts "
                 f"with wind speed (r = {r3:.2f}). Daylight duration is almost purely seasonal. These are left out "
                 "of the models to avoid duplicated information.", "Finding")
            st.write("")
            card("Outliers are kept",
                 "The extreme weeks are real outbreaks, not errors, so they are investigated rather than deleted.",
                 "Decision")

    # ---------------- why lags
    with tabs[3]:
        ac, rain_c, temp_c = lag_stats()
        st.markdown("A single weekly record only describes **one week**, but next week's cases depend on the recent "
                    "past. A **lag** of *k* weeks is the value observed *k* weeks earlier, added to each row as a "
                    "new column. Three patterns in the data justify the lags we created:")
        r1, r2, r3 = st.columns(3)
        with r1:
            card("① Cases repeat from week to week",
                 f"The average correlation between this week's and last week's cases is <b>{ac[0]:.2f}</b>. "
                 "So we add case lags (1, 2, 4, 8, 12 weeks) and a 4-week average.", "Persistence")
            fig = px.bar(x=list(range(1, 13)), y=ac)
            fig.update_traces(marker_color="#4DA3FF")
            fig.update_layout(xaxis_title="Lag (weeks)", yaxis_title="Autocorrelation", title="Cases vs earlier cases")
            st.plotly_chart(style(fig, 280), width="stretch")
        with r2:
            peak = int(np.argmax(rain_c))
            card("② Rain acts with a delay",
                 f"Same-week rainfall barely matters (ρ = {rain_c[0]:.2f}), but the link grows with the lag and "
                 f"peaks around <b>{peak} weeks</b> (ρ = {rain_c[peak]:.2f}). So we add rainfall lags (2–10 weeks) "
                 "and rolling totals (4, 8, 12 weeks).", "Delayed effect")
            fig = px.bar(x=list(range(0, 13)), y=rain_c)
            fig.update_traces(marker_color="#2DD4BF")
            fig.update_layout(xaxis_title="Rainfall lag (weeks)", yaxis_title="Spearman ρ", title="Cases vs earlier rainfall")
            st.plotly_chart(style(fig, 280), width="stretch")
        with r3:
            card("③ The season repeats",
                 "Cases rise and fall with the year (monsoon periods). We add the week of the year as a circular "
                 "pair (sin, cos) so that week 52 and week 1 are close together.", "Season")
            wk = raw.groupby(["Week_Start_Date"])["Number_of_Cases"].sum().reset_index()
            wk["Week of year"] = wk["Week_Start_Date"].dt.isocalendar().week.astype(int)
            seas = wk.groupby("Week of year")["Number_of_Cases"].mean().reset_index()
            fig = px.line(seas, x="Week of year", y="Number_of_Cases")
            fig.update_traces(line_color="#A78BFA")
            fig.update_layout(yaxis_title="Average national cases", title="Average cases by week of the year")
            st.plotly_chart(style(fig, 280), width="stretch")
        st.caption("These are associations, not proof of cause. The 8–9 week peak is longer than the mosquito life "
                   "cycle alone would suggest and may partly reflect shared seasonality.")

        st.markdown("#### What a lag looks like")
        ds = load_model_dataset()
        d_pick = st.selectbox("District", sorted(ds["District"].unique()), key="lag_district")
        ex = ds[ds["District"] == d_pick].iloc[40:48][["Week_Start_Date", "Number_of_Cases", "cases_lag1",
                                                       "cases_lag2", "cases_roll4_mean", "target_next_week_cases"]]
        ex = ex.rename(columns={"Number_of_Cases": "Cases this week", "cases_lag1": "Cases 1 week ago",
                                "cases_lag2": "Cases 2 weeks ago", "cases_roll4_mean": "Average of previous 4 weeks",
                                "target_next_week_cases": "TARGET: cases next week"})
        st.dataframe(ex, hide_index=True, width="stretch")
        st.caption("Each row is one district-week. The model sees the left-hand columns and tries to predict the "
                   "last column. Notice that 'cases 1 week ago' is just the previous row's 'cases this week'.")

        st.markdown("#### The 24 model features")
        groups = pd.DataFrame([
            ("Case history", "Number_of_Cases, cases_lag1, 2, 4, 8, 12, cases_roll4_mean", "Cases persist (autocorrelation 0.85)"),
            ("Current weather", "Rainfall, max temp, min temp, wind speed (this week)", "Same-week weather, weak link"),
            ("Rainfall history", "rain_lag2, 4, 6, 8, 10 and rain_sum4, 8, 12", "Rain acts after a delay (peak 8–9 weeks)"),
            ("Temperature history", "maxtemp_lag4, maxtemp_lag8", "Weak, changing link, kept for completeness"),
            ("Season", "Week_sin, Week_cos", "Monsoon and yearly pattern"),
            ("District", "District_code", "Districts have very different case levels"),
        ], columns=["Group", "Features", "Why"])
        st.dataframe(groups, hide_index=True, width="stretch")
        st.caption("All lag and rolling features use only past information; rolling sums and means exclude the "
                   "current week. Apparent temperatures, wind gusts, daylight duration and the year are not used.")

    # ---------------- data tables
    with tabs[4]:
        which = st.radio("Dataset", ["Raw (19 columns)", "Cleaned (16 columns)", "Model dataset with lags (24 features + target)"],
                         horizontal=True)
        if which.startswith("Raw"):
            data, fname = raw, "dengue_raw.csv"
        elif which.startswith("Cleaned"):
            data, fname = clean, "dengue_cleaned.csv"
        else:
            data, fname = load_model_dataset(), "dengue_model_dataset_with_lags.csv"
        picks = st.multiselect("Filter districts (optional)", sorted(raw["District"].unique()))
        view = data[data["District"].isin(picks)] if picks else data
        st.caption(f"{len(view):,} rows × {view.shape[1]} columns" + (" (rows with a full set of lags)"
                                                                         if which.startswith("Model") else ""))
        st.dataframe(view.head(500), hide_index=True, width="stretch", height=420)
        st.download_button("⬇ Download this dataset (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name=fname, mime="text/csv")


# ================================================================== PAGE: ALL MODELS
def page_all_models():
    st.title("All models compared")
    st.caption("Same data, same chronological split: trained up to 2020, validation 2021–2022, test 2023-01 to 2024-07.")
    split = st.radio("Evaluate on", ["Test", "Validation"], horizontal=True)
    mt, zero_pct = model_metrics(split)

    show = mt.sort_values("MAE").reset_index(drop=True)
    show.insert(0, "Rank (MAE)", range(1, len(show) + 1))
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "R²": st.column_config.NumberColumn(format="%.3f"),
                                "MAPE %": st.column_config.NumberColumn(format="%.1f"),
                                "Training time (s)": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"MAPE is calculated only on weeks with at least one actual case ({zero_pct:.0f}% of the "
               f"{split.lower()} weeks have zero cases and are left out). It is very sensitive to weeks with only "
               "1–2 cases, so read MAE and RMSE first. Naive and Linear Regression train in well under a second.")

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(metric_bar(mt, "MAE", True), width="stretch")
    with g2:
        st.plotly_chart(metric_bar(mt, "RMSE", True), width="stretch")
    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(metric_bar(mt, "R²", False), width="stretch")
    with g4:
        st.plotly_chart(metric_bar(mt, "MAPE %", True), width="stretch")
    if mt["Training time (s)"].notna().any():
        tt = mt.dropna(subset=["Training time (s)"]).copy()
        tt = tt[tt["Training time (s)"] > 0]
        st.plotly_chart(metric_bar(tt, "Training time (s)", True, height=320, log_x=True), width="stretch")
        st.caption("Training time is on a log axis and was measured in the notebook run on the best settings. "
                   "Stacking = base models + 5 chronological folds; Hybrid = ARIMA + the residual Random Forest.")
    else:
        st.info("Add `training_time.csv` to the repository to see the training-time chart.")

    st.subheader("Where do the errors happen?")
    st.caption("Mean error in groups of weeks by the size of the actual cases next week. Larger outbreaks are "
               "harder: every model's error grows with the number of cases.")
    metric = st.radio("Error measure", ["MAE", "RMSE"], horizontal=True)
    frame = pred[pred["Split"] == split]
    mat = band_matrix(frame, model_cols, metric)
    fig = px.imshow(mat, text_auto=".1f", aspect="auto", color_continuous_scale="Magma",
                    labels=dict(x="Actual cases next week (per district-week)", y="", color=metric))
    fig.update_layout(title=f"{metric} by size of the week's cases")
    st.plotly_chart(style(fig, 420), width="stretch")
    share = (pd.cut(frame["Actual_next_week"], BANDS, labels=BAND_LABELS).value_counts(normalize=True)
             .reindex(BAND_LABELS) * 100)
    st.caption("Share of district-weeks in each group: " + ", ".join(f"{k} {v:.1f}%" for k, v in share.items()) + ".")
    best_by_band = mat.idxmin()
    st.markdown("**Best model in each group:** " + " · ".join(f"{b}: **{best_by_band[b]}**" for b in BAND_LABELS))

# ================================================================== PAGE: HYBRID
def page_hybrid():
    st.title("Hybrid model: ARIMA + Random Forest")
    st.markdown(
        '<div class="flow">'
        '<div class="box"><b>1 · ARIMA</b><br><span>Learns the week-to-week persistence of cases in each district '
        '(linear time-series model).</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>2 · Residuals</b><br><span>What ARIMA got wrong: actual next-week cases minus the ARIMA '
        'forecast, on the training years.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>3 · Random Forest</b><br><span>Learns to predict those residuals from the lags, weather, '
        'season and the ARIMA forecast itself.</span></div><div class="arrow">➜</div>'
        '<div class="box"><b>4 · Hybrid forecast</b><br><span>ARIMA forecast + predicted residual, never below '
        'zero.</span></div></div>', unsafe_allow_html=True)
    st.caption("Idea: ARIMA is good at simple persistence but cannot use weather or season; the Random Forest can use "
               "many inputs but is weaker at pure persistence. The hybrid lets each do what it is good at.")

    if not {"ARIMA", "Hybrid ARIMA + RF", "Random Forest"} <= set(comp.index):
        st.warning("ARIMA, Random Forest or Hybrid results are missing from the result files.")
        return
    h, ar, rf = comp.loc["Hybrid ARIMA + RF"], comp.loc["ARIMA"], comp.loc["Random Forest"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Test MAE: hybrid", f"{h['MAE (Test)']:.2f}", f"ARIMA {ar['MAE (Test)']:.2f} · RF {rf['MAE (Test)']:.2f}")
    with c2:
        kpi("Test RMSE: hybrid", f"{h['RMSE (Test)']:.2f}", f"ARIMA {ar['RMSE (Test)']:.2f} · RF {rf['RMSE (Test)']:.2f}")
    with c3:
        kpi("Test R²: hybrid", f"{h['R2 (Test)']:.3f}", f"ARIMA {ar['R2 (Test)']:.3f} · RF {rf['R2 (Test)']:.3f}")
    with c4:
        t_h = TT.get("Hybrid ARIMA + RF")
        kpi("Training time: hybrid", "n/a" if t_h is None else f"{t_h:.0f} s",
            "ARIMA " + (f"{TT['ARIMA']:.0f} s" if "ARIMA" in TT else "n/a") + " + residual RF")
    st.write("")

    gain_tr = 100 * (1 - HYBRID_TRAIN_MAE["Hybrid ARIMA + RF"] / HYBRID_TRAIN_MAE["ARIMA"])
    gain_v = 100 * (1 - h["MAE (Validation)"] / ar["MAE (Validation)"])
    gain_t = 100 * (1 - h["MAE (Test)"] / ar["MAE (Test)"])
    a, b = st.columns(2)
    with a:
        fig = px.bar(pd.DataFrame({"Period": ["Training", "Validation", "Test"], "Gain": [gain_tr, gain_v, gain_t]}),
                     x="Period", y="Gain", text="Gain")
        fig.update_traces(marker_color=["#5B7DB1", "#2DD4BF", "#F59E0B"], texttemplate="%{text:.1f}%")
        fig.update_layout(title="How much the Random Forest lowers ARIMA's MAE", yaxis_title="MAE reduction (%)",
                          xaxis_title="")
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Training MAE (ARIMA 10.65, hybrid 8.08) is taken from the notebook.")
    with b:
        mat = band_matrix(test, ["ARIMA", "Random Forest", "Hybrid ARIMA + RF"], "MAE").reset_index(names="Model")
        long = mat.melt(id_vars="Model", var_name="Actual cases next week", value_name="MAE")
        fig = px.bar(long, x="Actual cases next week", y="MAE", color="Model", barmode="group",
                     category_orders={"Actual cases next week": BAND_LABELS},
                     color_discrete_map={"ARIMA": "#9CA3AF", "Random Forest": RF_COLOR, "Hybrid ARIMA + RF": "#A78BFA"})
        fig.update_layout(title="Test error by size of the week's cases")
        st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("See it yourself")
    districts = sorted(test["District"].unique())
    d = st.selectbox("District", districts, index=districts.index("Colombo") if "Colombo" in districts else 0)
    sub = test[test["District"] == d].sort_values("Target_Week")
    fig = go.Figure()
    for name, col, w in [("Reported", "Actual_next_week", 3), ("ARIMA", "ARIMA", 2), ("Hybrid ARIMA + RF", "Hybrid ARIMA + RF", 2)]:
        fig.add_trace(go.Scatter(x=sub["Target_Week"], y=sub[col], name=name,
                                 line=dict(color={"Reported": "#E6EDF7", "ARIMA": "#9CA3AF", "Hybrid ARIMA + RF": "#A78BFA"}[name], width=w)))
    fig.update_layout(yaxis_title="Cases per week", title=f"{d}: test period")
    st.plotly_chart(style(fig, 380), width="stretch")

    st.subheader("What the results mean")
    g1, g2 = st.columns(2)
    with g1:
        card("The idea works on validation", f"The hybrid lowers ARIMA's MAE by {gain_v:.1f}% on the validation years and "
             f"has the best validation MAE of all nine models ({h['MAE (Validation)']:.2f}).", "Measured")
        st.write("")
        card("The gain shrinks on new years",
             f"The reduction is {gain_tr:.0f}% on the training years, {gain_v:.1f}% on validation and only "
             f"{gain_t:.1f}% on test. The Random Forest partly learns patterns of the training years that do not carry "
             "over to the test years, which have many more cases.", "Measured + hypothesis")
    with g2:
        worse = "worse" if h["RMSE (Test)"] > ar["RMSE (Test)"] else "better"
        card("Large errors are not reduced",
             f"The test RMSE of the hybrid ({h['RMSE (Test)']:.2f}) is {worse} than that of ARIMA "
             f"({ar['RMSE (Test)']:.2f}), so the hybrid does not help in the biggest outbreak weeks.", "Measured")
        st.write("")
        card("Limitations",
             "The residuals used to train the Random Forest come from ARIMA predictions on the same years ARIMA was "
             "fitted on. ARIMA(1,0,0) has very few parameters (its training MAE 10.65 is close to its validation MAE "
             "10.76), so this effect is probably small, but it was not checked with out-of-fold residuals. "
             "Differences between the top models were not tested for statistical significance.")
   

# ================================================================== PAGE: RF vs XGBOOST
def page_rf_xgb():
    st.title("Random Forest vs XGBoost")
    st.caption("Bagging versus boosting on the same data and features. Models were trained on data up to 2020; "
               "validation = 2021–2022, test = 2023-01 to 2024-07.")
    rf, xg = comp.loc["Random Forest"], comp.loc["XGBoost"]

    def versus(col, label, lower_better, fmt):
        a, b = float(rf[col]), float(xg[col])
        rf_wins = (a < b) if lower_better else (a > b)
        value = (f'<span style="color:{RF_COLOR}">{fmt.format(a)}</span> '
                 f'<span style="color:#8FA3C0;font-size:1rem">vs</span> '
                 f'<span style="color:{XGB_COLOR}">{fmt.format(b)}</span>')
        kpi(label, value, f"better: {'Random Forest' if rf_wins else 'XGBoost'}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        versus("MAE (Test)", "Test MAE", True, "{:.2f}")
    with c2:
        versus("RMSE (Test)", "Test RMSE", True, "{:.2f}")
    with c3:
        versus("R2 (Test)", "Test R²", False, "{:.3f}")
    with c4:
        versus("Accuracy % (Test)", "Test accuracy %", False, "{:.1f}")
    st.markdown(f'<span style="color:{RF_COLOR}">■ Random Forest</span> &nbsp; '
                f'<span style="color:{XGB_COLOR}">■ XGBoost</span>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("All metrics")
        rows = []
        for col in ["MAE (Validation)", "MAE (Test)", "RMSE (Validation)", "RMSE (Test)",
                    "R2 (Validation)", "R2 (Test)", "Accuracy % (Validation)", "Accuracy % (Test)"]:
            lower = col.startswith(("MAE", "RMSE"))
            better = "Random Forest" if (rf[col] < xg[col]) == lower else "XGBoost"
            rows.append({"Metric": col, "Random Forest": round(float(rf[col]), 3),
                         "XGBoost": round(float(xg[col]), 3), "Better": better})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=350)
        st.caption("Lower is better for MAE and RMSE; higher is better for R² and accuracy.")
    with right:
        st.subheader("Errors side by side")
        long = pd.DataFrame([{"Measure": c.replace(" (Validation)", " · validation").replace(" (Test)", " · test"),
                              "Model": mname, "Value": float(comp.loc[mname, c])}
                             for c in ["MAE (Validation)", "MAE (Test)", "RMSE (Validation)", "RMSE (Test)"]
                             for mname in ["Random Forest", "XGBoost"]])
        fig = px.bar(long, x="Measure", y="Value", color="Model", barmode="group",
                     color_discrete_map={"Random Forest": RF_COLOR, "XGBoost": XGB_COLOR})
        fig.update_layout(yaxis_title="Error (cases per week, lower is better)", xaxis_title="")
        st.plotly_chart(style(fig, 350), width="stretch")

    st.info(" note: in the notebook the Random Forest was tuned over seven configurations and XGBoost over "
            "three, so the tuning effort was not identical. The differences between the tree-based models are small. "
            "All nine models are compared on the **All models** page.")

    st.subheader("See it yourself: actual vs predicted")
    districts = sorted(test["District"].unique())
    district = st.selectbox("District", districts, index=districts.index("Colombo") if "Colombo" in districts else 0)
    sub = test[test["District"] == district].sort_values("Target_Week")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Target_Week"], y=sub["Actual_next_week"], name="Reported",
                             line=dict(color="#E6EDF7", width=3)))
    fig.add_trace(go.Scatter(x=sub["Target_Week"], y=sub["Random Forest"], name="Random Forest",
                             line=dict(color=RF_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=sub["Target_Week"], y=sub["XGBoost"], name="XGBoost",
                             line=dict(color=XGB_COLOR, width=2)))
    fig.update_layout(yaxis_title="Cases per week", title=f"{district}: test period")
    st.plotly_chart(style(fig, 400), width="stretch")


# ================================================================== PAGE: BEST MODEL & WHY
def page_best():
    st.title("Best model and why")
    t = comp.copy()
    best_mae, best_rmse, best_r2 = t["MAE (Test)"].idxmin(), t["RMSE (Test)"].idxmin(), t["R2 (Test)"].idxmax()
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Lowest test MAE", best_mae, f"{t.loc[best_mae, 'MAE (Test)']:.2f} cases per week")
    with c2:
        kpi("Lowest test RMSE", best_rmse, f"{t.loc[best_rmse, 'RMSE (Test)']:.2f}")
    with c3:
        kpi("Highest test R²", best_r2, f"{t.loc[best_r2, 'R2 (Test)']:.3f}")
    st.write("")

    ranked = t["MAE (Test)"].sort_values()
    spread = 100 * (ranked.iloc[2] / ranked.iloc[0] - 1)
    naive_txt = ""
    if "Naive" in t.index:
        n_rank = int(list(ranked.index).index("Naive")) + 1
        naive_txt = (f" The simple Naive forecast (next week = this week) has a test MAE of "
                     f"{t.loc['Naive', 'MAE (Test)']:.2f} and ranks {n_rank} of {len(ranked)}.")
 

    board = t[["MAE (Test)", "RMSE (Test)", "R2 (Test)", "Accuracy % (Test)"]].copy()
    board.insert(0, "Rank (MAE)", board["MAE (Test)"].rank().astype(int))
    board["Gap to best MAE %"] = 100 * (board["MAE (Test)"] / board["MAE (Test)"].min() - 1)
    st.dataframe(board.sort_values("Rank (MAE)").round(3), width="stretch")

    st.subheader("What the stored predictions show (test period)")
    compare = list(dict.fromkeys([m for m in ["Random Forest", "XGBoost", "Naive", best_mae] if m in model_cols]))
    tt = test.copy()
    tt["Size"] = pd.cut(tt["Actual_next_week"], BANDS, labels=BAND_LABELS)
    share = tt["Size"].value_counts(normalize=True).reindex(BAND_LABELS) * 100

    e1, e2 = st.columns(2)
    with e1:
        rows = []
        for mname in compare:
            ae = (tt["Actual_next_week"] - tt[mname]).abs().groupby(tt["Size"], observed=True).mean()
            rows += [{"Actual cases next week": str(s), "Model": mname, "MAE": float(v)} for s, v in ae.items()]
        fig = px.bar(pd.DataFrame(rows), x="Actual cases next week", y="MAE", color="Model", barmode="group",
                     category_orders={"Actual cases next week": BAND_LABELS},
                     color_discrete_map={"Random Forest": RF_COLOR, "XGBoost": XGB_COLOR, "Naive": "#9CA3AF",
                                         best_mae: "#A78BFA"})
        fig.update_layout(title="Error by size of the week's cases")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Share of district-weeks in each group: " +
                   ", ".join(f"{lab} {share[lab]:.0f}%" for lab in BAND_LABELS) + ".")
    with e2:
        bias = pd.DataFrame({"Model": model_cols,
                             "Predicted ÷ actual": [float(test[m].sum() / test["Actual_next_week"].sum())
                                                    for m in model_cols]})
        fig = px.bar(bias, x="Model", y="Predicted ÷ actual", color="Model",
                     color_discrete_map={m: (RF_COLOR if m == "Random Forest" else XGB_COLOR if m == "XGBoost"
                                             else OTHER_COLOR) for m in model_cols})
        fig.add_hline(y=1, line_dash="dash", line_color="#E6EDF7")
        fig.update_layout(title="Total predicted cases ÷ total actual cases", showlegend=False, xaxis_title="")
        fig.update_yaxes(range=[min(0.8, bias.iloc[:, 1].min() - 0.03), max(1.2, bias.iloc[:, 1].max() + 0.03)])
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Below the dashed line = the model under-predicts overall; above = it over-predicts.")

    a = test["Actual_next_week"].values
    ae_rf, ae_xg = np.abs(a - test["Random Forest"].values), np.abs(a - test["XGBoost"].values)
    win = 100 * float(np.mean(ae_rf < ae_xg))
    lose = 100 * float(np.mean(ae_rf > ae_xg))
    top = a >= np.quantile(a, 0.95)
    sq = lambda p: float((((a - p) ** 2)[top]).sum() / ((a - p) ** 2).sum() * 100)
    s1, s2, s3 = st.columns(3)
    with s1:
        kpi("Random Forest closer than XGBoost", f"{win:.0f}%", f"of district-weeks (XGBoost closer in {lose:.0f}%)")
    with s2:
        kpi("Squared error from the top 5% weeks", f"{sq(test['Random Forest'].values):.0f}%",
            f"Random Forest · XGBoost {sq(test['XGBoost'].values):.0f}%")
    with s3:
        kpi("Test weeks / districts", f"{test['Target_Week'].nunique()} / {test['District'].nunique()}",
            "2023-01 to 2024-07")
    st.caption("RMSE squares the errors, so a few very large outbreak weeks dominate it. When most of the squared "
               "error comes from the top 5% weeks, the RMSE mainly measures how well a model handles outbreaks.")

    st.subheader("Why do the models behave this way?")
    st.caption("Measured = shown by the notebook or the charts above. Hypothesis = a likely explanation that the "
               "study did not test directly.")
    if {"Hybrid ARIMA + RF", "ARIMA"} <= set(comp.index):
        h, ar = comp.loc["Hybrid ARIMA + RF"], comp.loc["ARIMA"]
        gain_v = 100 * (1 - h["MAE (Validation)"] / ar["MAE (Validation)"])
        gain_t = 100 * (1 - h["MAE (Test)"] / ar["MAE (Test)"])
        rmse_txt = "worse" if h["RMSE (Test)"] > ar["RMSE (Test)"] else "better"
        hybrid_text = (f"ARIMA captures week-to-week persistence and a Random Forest corrects its errors. The hybrid "
                       f"lowers the MAE of ARIMA by {gain_v:.1f}% on validation but only {gain_t:.1f}% on test, and its "
                       f"test RMSE is {rmse_txt} than ARIMA's. The correction probably learns some patterns of the "
                       "training years that do not carry over.")
    else:
        hybrid_text = "The hybrid model was not found in the result files."
    g1, g2 = st.columns(2)
    with g1:
        card("1. Dengue cases repeat from week to week",
             "The average lag-1 autocorrelation is 0.85, and the feature importances show that the current and "
             "previous weeks' cases dominate the predictions. That is why the Naive forecast is already hard to beat.",
             "Measured",
             extra_class="best-model-card"
    
            )
        st.write("")
        card("3. Trees cannot go beyond what they have seen",
             "Random Forest and boosting models predict averages of training cases, so in new, much larger outbreaks "
             "they tend to under-predict. The test period has more cases (mean 59.4 vs 32.1 in training); the bias "
             "chart above shows whether the models under-predict overall.", "Hypothesis",
             extra_class="best-model-card"
             )
        st.write("")
        card("5. Stacking added little",
             "The four base models make almost the same predictions (correlation 0.974–0.995), so combining them "
             "adds little. The first Ridge stacking failed because its weights learned from out-of-fold predictions "
             "(which under-predicted the 2017 outbreak) were scaled up (sum 1.29) and then over-predicted. "
             "With weights constrained to sum to 1 the test MAE improved from 21.2 to 13.8.", "Measured",
             extra_class="best-model-card"
             )
    with g2:
        card("2. Weather adds only a little",
             "Weather features have weak correlations with cases, with rainfall linked more strongly 8–9 weeks "
             "earlier (Spearman 0.18), and in the feature importance of the notebook the case features dominate. "
             "A model without weather was not tested, so the exact benefit of weather is unknown.", "Measured",
             extra_class="best-model-card"
             )
        st.write("")
        card("4. XGBoost is worse in the big weeks",
             f"XGBoost has a similar MAE to the Random Forest but a clearly higher validation RMSE "
             f"({comp.loc['XGBoost', 'RMSE (Validation)']:.1f} vs {comp.loc['Random Forest', 'RMSE (Validation)']:.1f}), "
             "which suggests larger errors in a few outbreak weeks. It was also tuned over fewer configurations. "
             "Which of these reasons matters more was not tested.", "Measured + hypothesis",
             extra_class="best-model-card"
             )
        st.write("")
        card("6. The hybrid helps mainly where it was trained", hybrid_text, "Measured + hypothesis",
        extra_class="best-model-card"
        )


# ================================================================== PAGE: RESEARCH
def page_research():
    st.title("Research: comparing ensemble methods")

    st.subheader("1 · Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost)")
    st.markdown("""
| | **Random Forest** (bagging) | **Gradient boosting** (XGBoost, LightGBM, CatBoost) |
|---|---|---|
| How the trees are built | Many deep trees, each trained independently on a random sample of rows and features, then averaged | Many shallow trees, built one after another; each new tree fits the errors the previous trees left |
| Main effect | Averaging reduces variance (instability) | Sequential correction reduces bias; shrinkage and subsampling control variance |
| Tuning | Reasonable results with few settings (trees, depth, features per split) | More settings: learning rate, depth, number of trees (early stopping), regularisation |
| Overfitting | Adding more trees does not by itself cause overfitting | Too many trees overfit, so early stopping on a validation set is used |
| Speed | Trees can be built in parallel, but hundreds of deep trees take time | Histogram-based methods are very fast |
| Special features | – | **XGBoost:** regularised objective. **LightGBM:** histogram bins and leaf-wise growth. **CatBoost:** ordered boosting and built-in handling of categorical features |
""")
    four = [m for m in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"] if m in comp.index]
    tab = comp.loc[four, ["MAE (Validation)", "MAE (Test)", "RMSE (Validation)", "RMSE (Test)", "R2 (Test)"]].copy()
    tab["Training time (s)"] = [TT.get(m, np.nan) for m in four]
    st.markdown("**What this study found (same features, same split):**")
    st.dataframe(tab.reset_index().rename(columns={"Model": "Model", "R2 (Test)": "R² (Test)"}), hide_index=True,
                 width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                                  ["MAE (Validation)", "MAE (Test)", "RMSE (Validation)", "RMSE (Test)", "Training time (s)"]})
    a, b = st.columns(2)
    with a:
        long = tab.reset_index().melt(id_vars="Model", value_vars=["MAE (Validation)", "MAE (Test)"], var_name="Set", value_name="MAE")
        fig = px.bar(long, x="Model", y="MAE", color="Set", barmode="group",
                     color_discrete_map={"MAE (Validation)": "#2DD4BF", "MAE (Test)": "#F59E0B"})
        fig.update_layout(title="MAE of the four tree ensembles")
        st.plotly_chart(style(fig, 320), width="stretch")
    with b:
        spread = 100 * (tab["MAE (Test)"].max() / tab["MAE (Test)"].min() - 1)
        t_rf, t_x = TT.get("Random Forest"), TT.get("XGBoost")
        speed = f" Random Forest took {t_rf:.0f} s to train and XGBoost {t_x:.1f} s." if t_rf and t_x else ""
        st.markdown(f"- The four tree ensembles are within **{spread:.1f}%** of each other on test MAE.\n"
                    f"- The Random Forest has the best validation MAE and RMSE; the boosting models were tuned over "
                    f"fewer configurations (3 each vs 7).\n"
                    f"- Boosting models trained far faster.{speed}\n"
                    "- None of them beats the simple Naive forecast on test MAE, because next week's cases are "
                    "mostly this week's cases.")
        st.caption("Differences are small and were not tested for statistical significance.")

    st.subheader("2 · Strengths and limitations: tabular vs image/text data")
    st.markdown("""
| | **Tabular data** (like this study) | **Image and text data** |
|---|---|---|
| **Strengths of RF / boosting** | Handle numbers, counts and categories together; need no scaling; capture non-linear effects and interactions; train quickly on a CPU; give feature importance. | Fast, simple baselines on hand-made or pre-trained features, for example word counts (TF-IDF) or embeddings from a neural network |
| **Limitations** | Cannot predict beyond the range seen in training (a problem for new outbreaks); time-series need hand-made lag features; correlated features share importance; no built-in uncertainty (here the ranges come from past errors) | Single pixels or single words are not meaningful features; trees ignore spatial layout and word order; features must be extracted first. Deep networks (CNNs, Transformers) learn their own features from raw data and generally do better |
""")
    note("Our study used tabular data only. The image and text points come from the literature and were not tested here.")

    st.subheader("3 · Hybridization suggested in the course")
    h1, h2 = st.columns(2)
    with h1:
        card("RF + ARIMA (time series) ✅ done",
             "ARIMA models the week-to-week persistence and a Random Forest corrects its errors. See the Hybrid page "
             "for the results: a small gain on validation that mostly disappears on the test years.", "Implemented")
    with h2:
        card("RF + ANN (tabular regression) ⏳ not done",
             "A neural network could learn the Random Forest's residuals, or be one member of a stacking/weighted "
             "ensemble. It would need scaled "
             "inputs and out-of-fold residuals, and was left as future work.", "Future work")

    st.subheader("References")
    st.markdown("""
 - rajaththomson. (n.d.). *Sri Lanka dengue fever analysis* [Dataset]. GitHub. https://github.com/rajaththomson/sri-lanka-dengue-fever-analysis

""")


# ================================================================== PAGE: ABOUT
def page_about():
    st.title("About and limitations")

    # Custom styling for section headings
    st.markdown("""
    <style>
    .about-section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 12px;
        color: inherit;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section-title">Project</div>

    *Comparative Analysis of Ensemble Learning and Hybrid Models for Dengue Case Prediction in Sri Lanka.*

    Weekly dengue cases and weather data (22,900 records, 25 districts, 2007–2024) were used to forecast the number of
    reported cases in the **next week**. Bagging (Random Forest), boosting (XGBoost, LightGBM, CatBoost), stacking and a
    hybrid ARIMA + Random Forest model were compared with a Naive forecast, Linear Regression and ARIMA.

    <div class="about-section-title">How the data were split</div>

    Chronologically, never at random: training up to 2020, validation 2021–2022, test
    2023-01 to 2024-07. Hyperparameters were chosen on the validation set. The convex-weight stacking version was
    added after the first (Ridge) stacking performed poorly on the test set, and both are reported in the notebook.

    <div class="about-section-title">How to read the numbers</div>

    - **MAE**: the average number of cases by which a forecast is wrong (lower is better).
    - **RMSE**: like MAE, but large errors in outbreak weeks count more (lower is better).
    - **R²**: the share of the variation in next-week cases that the model explains (higher is better).
    - **MAPE**: the average percentage error, calculated only on weeks with at least one case.
    - **Accuracy %**: 100 × (1 − MAE ÷ average actual cases). It is a rescaled MAE and ranks models like MAE.
    - **Risk level**: *Low* if the cases are at or below the district's median of the training period, *Medium* up to
      its 80th percentile, *High* above that. It is relative to each district's own history, not an official threshold.
    - **Forecast range**: built from the validation-period errors of each model (10th–90th percentile). The test
      coverage shown for health authorities tells how often actual cases fell inside it.

    <div class="about-section-title">Limitations</div>

    - The data end in July 2024. This is a demonstration, not a live surveillance system.
    - The Dengue Predictor fills 16 of the 24 model features with district-week averages, so it is a what-if tool, not a
      validated forecast. Its accuracy was measured with a backtest (shown on that page).
    - Weather has only a small effect on the forecasts, so changing it moves the results little.
    - Forecasts use last week's reported cases; reporting delays would reduce accuracy.
    - The models predict *reported* cases, not true infections, and large outbreaks are hard to predict.
    - The test period has more cases than the training period (distribution shift), so test errors are larger.
    - Differences between the best models are small and were not tested for statistical significance.
    - Nothing on this dashboard is an official health warning.

    <div class="about-section-title">Credits</div>

    Data: Sri Lanka weekly dengue and weather data 2007–2024 (source repository:
    rajaththomson/sri-lanka-dengue-fever-analysis). District boundaries: Natural Earth (public domain), simplified.

    <div class="about-section-title">Tools & Links</div>

    - **Training:** Google Colab, scikit-learn, XGBoost, LightGBM, CatBoost
    - **Dashboard:** Streamlit, Plotly
    - **Code Repository:** [SL Dengue Prediction Dashboard](https://github.com/ChiraniSiriwardhana/SL-DenguePrediction-dashboard)
    - **Google Colab Notebook:** [Open Training Notebook](https://colab.research.google.com/drive/1BkcAJhWFWFsqvuumIJKepReC2PNs3FQH#scrollTo=gs8xIUZWNusf)
    - **Dataset Source:** [Sri Lanka Dengue Fever Analysis](https://github.com/rajaththomson/sri-lanka-dengue-fever-analysis)
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ navigation (sections in the sidebar)
NAV = {
    "Forecast": [
        st.Page(page_latest_forecast, title="Latest Forecast", icon="🔮", url_path="latest-forecast", default=True),
        st.Page(page_whatif, title="Dengue Predictor", icon="🧪", url_path="dengue-predictor"),
        st.Page(page_compare, title="Latest vs Predictor", icon="🆚", url_path="latest-vs-predictor"),
    ],
    "Data": [
        st.Page(page_dataset, title="Dataset", icon="📊", url_path="dataset"),
    ],
    "Models": [
        st.Page(page_all_models, title="All models", icon="📈", url_path="all-models"),
        st.Page(page_hybrid, title="Hybrid model", icon="🔀", url_path="hybrid-model"),
        st.Page(page_rf_xgb, title="RF vs XGBoost", icon="⚔️", url_path="rf-vs-xgboost"),
        st.Page(page_best, title="Best model & why", icon="🏆", url_path="best-model"),
    ],
    "Research & info": [
        st.Page(page_research, title="Research", icon="🔬", url_path="research"),
        st.Page(page_about, title="About", icon="ℹ️", url_path="about"),
    ],
}
current_page = st.navigation(NAV)

# Sidebar acknowledgement
st.sidebar.markdown(
    """
<div class="sidebar-footer">
    <div>
        <strong>AI Assistance:</strong>
        Claude (Anthropic) supported code generation and explanations.
        Dataset selection, model training, deployment and review of results by:
        <strong>Siriwardhana C.R.H.P.E.R.</strong>
    </div>
</div>
""",
    unsafe_allow_html=True
)

# Header on every page
render_app_header()

# Display selected page
current_page.run()

