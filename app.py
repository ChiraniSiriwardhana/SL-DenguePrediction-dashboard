"""Sri Lanka weekly dengue forecasting dashboard (Streamlit, dark theme).

The dashboard only DISPLAYS results that were produced in the notebook (Section 15):
    predictions.csv, model_comparison.csv, risk_thresholds.csv, forecast_next_week.csv
plus sri_lanka_districts.geojson (district boundaries, used for the map).
It does not train any model.
"""
import json
from pathlib import Path

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
RF_COLOR, XGB_COLOR, OTHER_COLOR = "#2DD4BF", "#FB923C", "#5B7DB1"
EDGES = [5, 20, 50]                      # prediction-size bands used for the forecast ranges

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
.kpi {background:#121B2E; border:1px solid #223152; border-radius:14px; padding:14px 18px; height:100%;}
.kpi .label {font-size:.72rem; color:#8FA3C0; text-transform:uppercase; letter-spacing:.08em;}
.kpi .value {font-size:2rem; font-weight:700; line-height:1.3; color:#E6EDF7;}
.kpi .sub {font-size:.8rem; color:#8FA3C0;}
.pill {display:inline-block; padding:0 14px; border-radius:999px; font-weight:600; font-size:1.3rem;}
.note {background:#1A2540; border-left:4px solid #4DA3FF; border-radius:8px; padding:10px 14px;
       color:#C9D6EC; font-size:.9rem; margin-bottom:.8rem;}
.card {background:#121B2E; border:1px solid #223152; border-radius:14px; padding:16px 20px; height:100%;}
.card h4 {margin:0 0 .4rem 0; font-size:1rem; color:#E6EDF7;}
.card p {margin:0; font-size:.9rem; color:#B7C6DE;}
.tag {font-size:.7rem; padding:1px 8px; border-radius:999px; border:1px solid #3A4C73; color:#8FA3C0;}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value"{style}>{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def card(title, text, tag=""):
    tag_html = f' <span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="card"><h4>{title}{tag_html}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), font=dict(color="#C9D6EC"))
    fig.update_xaxes(gridcolor="#1F2B45")
    fig.update_yaxes(gridcolor="#1F2B45")
    return fig


# ------------------------------------------------------------------ data
def find(name):
    for p in (BASE / name, BASE / "data" / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


@st.cache_data
def load_data():
    pred = pd.read_csv(find("predictions.csv"), parse_dates=["Week_Start_Date", "Target_Week"])
    comp = pd.read_csv(find("model_comparison.csv"))
    thr = pd.read_csv(find("risk_thresholds.csv"))
    fc = pd.read_csv(find("forecast_next_week.csv"), parse_dates=["Last_data_week", "Forecast_week"])
    with open(find("sri_lanka_districts.geojson"), encoding="utf-8") as f:
        geo = json.load(f)
    comp["Model"] = (comp["Model"].str.replace(" (Tuned)", "", regex=False)
                     .str.replace(" (convex weights)", "", regex=False))
    return pred, comp, thr, fc, geo


try:
    pred, comp, thr, fc, geo = load_data()
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
val = pred[pred["Split"] == "Validation"]
test = pred[pred["Split"] == "Test"]


# ------------------------------------------------------------------ helpers
def risk_of(values, p50, p80):
    v = np.asarray(values, dtype=float)
    return np.where(v <= p50, "Low", np.where(v <= p80, "Medium", "High"))


def interval_band(model):
    """10th and 90th percentile of the validation errors (actual - predicted) for each prediction-size band."""
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


def risk_map(m, selected):
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


# ------------------------------------------------------------------ sidebar
PAGES = ["🔮 Prediction", "⚔️ RF vs XGBoost", "🏆 Best model & why", "ℹ️ About & limitations"]
st.sidebar.markdown("## 🦟 Dengue Forecast")
st.sidebar.caption("Sri Lanka · weekly cases · 25 districts")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.divider()

# ================================================================== PAGE 1: PREDICTION
if page == PAGES[0]:
    audience = st.sidebar.radio("I am a", ["Member of the public", "Health authority / researcher"])
    district = st.sidebar.selectbox("District", sorted(fc["District"]))
    model = st.sidebar.selectbox("Model", fc_models,
                                 index=fc_models.index(best_model) if best_model in fc_models else 0)
    st.sidebar.caption(f"Default model: the one with the lowest test MAE ({best_model}).")
    public = audience.startswith("Member")

    row = fc[fc["District"] == district].iloc[0]
    predicted = float(row[model])
    last = float(row["Current_week_cases"])
    level = risk_of([predicted], row["p50"], row["p80"])[0]
    lo_a, hi_a = apply_interval([predicted], interval_band(model))
    lo, hi = float(lo_a[0]), float(hi_a[0])
    color = RISK_COLORS[level]

    st.title("Next-week dengue forecast")
    st.caption(f"Forecast for the week starting {row['Forecast_week']:%d %b %Y} · based on reported cases up to "
               f"the week starting {row['Last_data_week']:%d %b %Y}")
    st.markdown('<div class="note">⚠️ This is a model-based estimate from a university research project. '
                'It is <b>not</b> an official health warning and does not confirm an outbreak. For official '
                'information, contact the Ministry of Health or the National Dengue Control Unit.</div>',
                unsafe_allow_html=True)

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
        st.info("General advice: regularly empty and clean containers that hold standing water (pots, tyres, "
                "gutters), because dengue mosquitoes breed there.")
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

# ================================================================== PAGE 2: RF vs XGBOOST
elif page == PAGES[1]:
    st.title("Random Forest vs XGBoost")
    st.caption("Bagging versus boosting on the same data and features. Models were trained on data up to 2020; "
               "validation = 2021–2022, test = 2023-01 to 2024-07.")
    rf, xg = comp.loc["Random Forest"], comp.loc["XGBoost"]

    def versus(col, label, lower_better, fmt):
        a, b = float(rf[col]), float(xg[col])
        rf_wins = (a < b) if lower_better else (a > b)
        winner = "Random Forest" if rf_wins else "XGBoost"
        value = (f'<span style="color:{RF_COLOR}">{fmt.format(a)}</span> '
                 f'<span style="color:#8FA3C0;font-size:1rem">vs</span> '
                 f'<span style="color:{XGB_COLOR}">{fmt.format(b)}</span>')
        kpi(label, value, f"better: {winner}")

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

    st.info("Fairness note: in the notebook the Random Forest was tuned over seven configurations and XGBoost over "
            "three, so the tuning effort was not identical. The differences between the tree-based models are small.")

    st.subheader("See it yourself: actual vs predicted")
    district = st.selectbox("District", sorted(test["District"].unique()),
                            index=sorted(test["District"].unique()).index("Colombo")
                            if "Colombo" in set(test["District"]) else 0)
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

    st.subheader("All models ranked")
    metric = st.selectbox("Rank by", ["MAE (Test)", "RMSE (Test)", "R2 (Test)", "Accuracy % (Test)",
                                      "MAE (Validation)", "RMSE (Validation)", "R2 (Validation)",
                                      "Accuracy % (Validation)"])
    asc = metric.startswith(("MAE", "RMSE"))
    ranked = comp[metric].sort_values(ascending=not asc).reset_index()
    ranked.columns = ["Model", "Value"]
    ranked["Colour"] = ranked["Model"].map(lambda x: x if x in ("Random Forest", "XGBoost") else "Other models")
    fig = px.bar(ranked, x="Value", y="Model", orientation="h", color="Colour", text="Value",
                 color_discrete_map={"Random Forest": RF_COLOR, "XGBoost": XGB_COLOR, "Other models": OTHER_COLOR})
    fig.update_traces(texttemplate="%{text:.3f}")
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(ranked["Model"])[::-1]),
                      xaxis_title=metric, yaxis_title="")
    st.plotly_chart(style(fig, 420), width="stretch")

# ================================================================== PAGE 3: BEST MODEL & WHY
elif page == PAGES[2]:
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
    st.markdown(
        f"### Verdict\n**{ranked.index[0]}** has the lowest test MAE ({ranked.iloc[0]:.2f}), but the best three models "
        f"are within **{spread:.1f}%** of each other, and different models win on different metrics "
        f"({best_rmse} on RMSE, {best_r2} on R²).{naive_txt} So no ensemble model is *clearly* better than the "
        "simple baselines. The notebook did not run a significance test, so small gaps should not be over-interpreted.")

    board = t[["MAE (Test)", "RMSE (Test)", "R2 (Test)", "Accuracy % (Test)"]].copy()
    board.insert(0, "Rank (MAE)", board["MAE (Test)"].rank().astype(int))
    board["Gap to best MAE %"] = 100 * (board["MAE (Test)"] / board["MAE (Test)"].min() - 1)
    st.dataframe(board.sort_values("Rank (MAE)").round(3), width="stretch")

    # ---------- evidence computed from predictions.csv
    st.subheader("What the stored predictions show (test period)")
    compare = [m for m in ["Random Forest", "XGBoost", "Naive", best_mae] if m in model_cols]
    compare = list(dict.fromkeys(compare))
    tt = test.copy()
    bins, labels = [-1, 10, 50, 200, 1e9], ["0–10", "11–50", "51–200", "over 200"]
    tt["Size"] = pd.cut(tt["Actual_next_week"], bins, labels=labels)
    share = tt["Size"].value_counts(normalize=True).reindex(labels) * 100

    e1, e2 = st.columns(2)
    with e1:
        rows = []
        for mname in compare:
            ae = (tt["Actual_next_week"] - tt[mname]).abs().groupby(tt["Size"], observed=True).mean()
            rows += [{"Actual cases next week": str(s), "Model": mname, "MAE": float(v)} for s, v in ae.items()]
        fig = px.bar(pd.DataFrame(rows), x="Actual cases next week", y="MAE", color="Model", barmode="group",
                     category_orders={"Actual cases next week": labels},
                     color_discrete_map={"Random Forest": RF_COLOR, "XGBoost": XGB_COLOR, "Naive": "#9CA3AF",
                                         best_mae: "#A78BFA"})
        fig.update_layout(title="Error by size of the week's cases")
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Share of district-weeks in each group: " +
                   ", ".join(f"{lab} {share[lab]:.0f}%" for lab in labels) + ".")
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

    # ---------- explanations
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
             "Measured")
        st.write("")
        card("3. Trees cannot go beyond what they have seen",
             "Random Forest and boosting models predict averages of training cases, so in new, much larger outbreaks "
             "they tend to under-predict. The test period has more cases (mean 59.4 vs 32.1 in training); the bias "
             "chart above shows whether the models under-predict overall.", "Hypothesis")
        st.write("")
        card("5. Stacking added little",
             "The four base models make almost the same predictions (correlation 0.974–0.995), so combining them "
             "adds little. The first Ridge stacking failed because its weights learned from out-of-fold predictions "
             "(which under-predicted the 2017 outbreak) were scaled up (sum 1.29) and then over-predicted. "
             "With weights constrained to sum to 1 the test MAE improved from 21.2 to 13.8.", "Measured")
    with g2:
        card("2. Weather adds only a little",
             "Weather features have weak correlations with cases, with rainfall linked more strongly 8–9 weeks "
             "earlier (Spearman 0.18), and in the feature importance of the notebook the case features dominate. "
             "A model without weather was not tested, so the exact benefit of weather is unknown.", "Measured")
        st.write("")
        card("4. XGBoost is worse in the big weeks",
             f"XGBoost has a similar MAE to the Random Forest but a clearly higher validation RMSE "
             f"({comp.loc['XGBoost', 'RMSE (Validation)']:.1f} vs {comp.loc['Random Forest', 'RMSE (Validation)']:.1f}), "
             "which suggests larger errors in a few outbreak weeks. It was also tuned over fewer configurations. "
             "Which of these reasons matters more was not tested.", "Measured + hypothesis")
        st.write("")
        card("6. The hybrid helps mainly where it was trained",
             hybrid_text, "Measured + hypothesis")

# ================================================================== PAGE 4: ABOUT
else:
    st.title("About and limitations")
    st.markdown("""
**Project.** *Comparative Analysis of Ensemble Learning and Hybrid Models for Dengue Case Prediction in Sri Lanka.*
Weekly dengue cases and weather data (22,900 records, 25 districts, 2007–2024) were used to forecast the number of
reported cases in the **next week**. Bagging (Random Forest), boosting (XGBoost, LightGBM, CatBoost), stacking and a
hybrid ARIMA + Random Forest model were compared with a Naive forecast, Linear Regression and ARIMA.

**How the data were split.** Chronologically, never at random: training up to 2020, validation 2021–2022, test
2023-01 to 2024-07. Hyperparameters were chosen on the validation set. The convex-weight stacking version was
added after the first (Ridge) stacking performed poorly on the test set, and both are reported in the notebook.

**How to read the numbers**
- **MAE**: the average number of cases by which a forecast is wrong (lower is better).
- **RMSE**: like MAE, but large errors in outbreak weeks count more (lower is better).
- **R²**: the share of the variation in next-week cases that the model explains (higher is better).
- **Accuracy %**: 100 × (1 − MAE ÷ average actual cases). It is a rescaled MAE and ranks models like MAE.
- **Risk level**: *Low* if the cases are at or below the district's median of the training period, *Medium* up to
  its 80th percentile, *High* above that. It is relative to each district's own history, not an official threshold.
- **Forecast range**: built from the validation-period errors of each model (10th–90th percentile). The test
  coverage shown for health authorities tells how often actual cases fell inside it.

**Limitations**
- The data end in July 2024. This is a demonstration, not a live surveillance system.
- Forecasts use last week's reported cases; reporting delays would reduce accuracy.
- The models predict *reported* cases, not true infections, and large outbreaks are hard to predict.
- The test period has more cases than the training period (distribution shift), so test errors are larger.
- Differences between the best models are small and were not tested for statistical significance.
- Nothing on this dashboard is an official health warning.

**Credits.** District boundaries: Natural Earth (public domain), simplified for this dashboard.
""")
