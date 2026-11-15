import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tdui import common as C
from trendlab import core

ctx = C.get_ctx()
st.title("Compare and link regions", anchor=False)

# ----------------------------------------------------------------------------
# Linked regions: fires upwind -> haze in Singapore
# ----------------------------------------------------------------------------
st.subheader("Do fires upwind make Singapore hazy?", anchor=False)
st.caption("Singapore's haze (aerosol optical depth) against fire detections in Riau / South Sumatra and Central "
           "Kalimantan, with the seasonal cycle removed from both so the shared dry season does not fake a link.")
links = C.region_links(ctx.period)
by = links["by_source"]
if by:
    fig = go.Figure()
    for (name, ll), colr in zip(by.items(), (C.ORANGE, C.BLUE)):
        lags = list(ll["by_lag"])
        fig.add_trace(go.Bar(x=[f"{k} mo" for k in lags], y=[ll["by_lag"][k] for k in lags], name=name,
                             marker=dict(color=colr, cornerradius=3), hovertemplate="%{x}: r = %{y:+.2f}<extra></extra>"))
    C.style(fig, 300)
    fig.update_layout(barmode="group", bargap=0.35, hovermode="closest", yaxis_title="Correlation (r)",
                      xaxis_title="Fires lead the haze by", title=dict(text="How strongly fires track Singapore's haze, by lead time",
                                                                       font=dict(size=13, color=C.INK2)))
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        C.show(fig, alt="Correlation between upwind fires and Singapore haze at leads of 0 to 3 months")
    reg = links.get("regression")
    with right:
        if reg:
            fig = go.Figure()
            fig.add_hline(y=0, line_color=C.AXIS, line_width=1)
            fig.add_trace(go.Scatter(x=reg["actual"].index, y=reg["actual"], mode="lines", name="Observed haze anomaly",
                                     line=dict(color=C.BLUE, width=1.6), hovertemplate="%{y:+.3f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=reg["fitted"].index, y=reg["fitted"], mode="lines", name="Explained by upwind fires",
                                     line=dict(color=C.ORANGE, width=2), hovertemplate="%{y:+.3f}<extra></extra>"))
            C.style(fig, 300)
            fig.update_layout(title=dict(text="Singapore haze anomaly: observed vs explained by fires",
                                         font=dict(size=13, color=C.INK2)))
            C.show(fig, alt="Singapore haze anomaly observed and as explained by upwind fires")
    if reg:
        sig = [k for k, p in reg["p"].items() if p < ctx.alpha]
        parts = ", ".join(f"{k} (p {C.fmt_p(reg['p'][k])})" for k in reg["coef"])
        adds = ("Both regions add information of their own" if len(sig) == len(reg["p"]) else
                f"Only {', '.join(sig)} adds information of its own" if sig else "Neither region is significant on its own")
        lags = {k: v["lag"] for k, v in by.items()}
        when = ("The link is strongest in the same month: smoke reaches Singapore within days."
                if all(v == 0 for v in lags.values()) else
                "Best leads: " + ", ".join(f"{k} {v} month(s)" for k, v in lags.items()) + ".")
        st.success(f"Upwind fires explain **{reg['r2']:.0%}** of Singapore's month-to-month haze swings "
                   f"({reg['n']} months). {adds}: {parts}. {when}", icon=":material/air:", title="Linked")
        st.caption("Regression of Singapore's AOD anomaly on both regions' log fire-count anomalies at their best lag, "
                   "with autocorrelation-robust (HAC) p-values. Wind direction, rain and fires outside these two boxes "
                   "explain the rest.")

# ----------------------------------------------------------------------------
# Compare the same variable across regions
# ----------------------------------------------------------------------------
st.divider()
st.subheader("Same variable, three regions", anchor=False)
st.caption("The same variable can trend differently in different places.")
opts = [c for c in core.VARIABLES if c not in core.DRIVERS]
if st.session_state.get("cmp_var") not in opts:
    st.session_state["cmp_var"] = "vpd_kpa"
cvar = st.selectbox("Variable", opts, format_func=C.label, key="cmp_var", persist_state="session")
rows, series = [], {}
for lab_, nm in C.BUNDLED.items():
    k, b = C.bundled_raw(nm)
    try:
        rr = C.variable_detail(k, b, cvar, ctx.alpha, ctx.period)
    except Exception:
        continue
    if rr["n"] < 36:
        continue
    series[lab_] = rr
    rows.append({"Region": lab_, "Slope / decade": rr["sk_slope_per_decade"], "% / decade": rr["pct_per_decade"],
                 "Seasonal Kendall p": rr["sk_p"], "ENSO-adjusted p": rr["adj_p"], "Verdict": rr["verdict"]})
if series:
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=list(series))
    for i, (lab_, rr) in enumerate(series.items(), 1):
        a = rr["_anom"]
        fig.add_hline(y=0, line_color=C.AXIS, line_width=1, row=i, col=1)
        fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", line=dict(color=C.BLUE, width=1.6), name="Anomaly",
                                 showlegend=i == 1, hovertemplate="%{y:+.3g}<extra></extra>"), i, 1)
        fig.add_trace(go.Scatter(x=a.index, y=rr["_sen_intercept"] + rr["sen_slope_per_decade"] / 10 * core._years(a.index),
                                 mode="lines", line=dict(color=C.ORANGE, width=2.5), name="Sen trend", showlegend=i == 1,
                                 hovertemplate="%{y:+.3g}<extra></extra>"), i, 1)
    C.style(fig, 200 * len(series) + 60)
    fig.update_annotations(font=dict(size=13, color=C.INK2), x=0, xanchor="left")
    fig.update_yaxes(matches="y")
    C.show(fig, alt=f"{C.label(cvar)} anomaly and trend in each region")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt=f"Trend in {C.label(cvar)} by region",
                 column_config={"Slope / decade": st.column_config.NumberColumn(format="%+.3g"),
                                "% / decade": st.column_config.NumberColumn(format="%+.1f"),
                                "Seasonal Kendall p": st.column_config.NumberColumn(format="%.4f"),
                                "ENSO-adjusted p": st.column_config.NumberColumn(format="%.4f")})
else:
    st.info("No region has enough data for this variable.", icon=":material/info:")
