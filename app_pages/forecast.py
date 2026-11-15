import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tdui import common as C
from trendlab import core, models

ctx = C.get_ctx()
ss = st.session_state
st.title("Models and forecast", anchor=False)
st.caption("Every model forecasts each year from 2011 on using only the years before it (rolling-origin validation), "
           "and only information available at forecast time: no future El Nino values. The best one makes the forecast.")

c1, c2 = st.columns([1.4, 1], vertical_alignment="bottom")
with c1:
    fvar = C.pick_variable(ctx, "fvar_ml", "Variable to forecast", default="fire_count")
with c2:
    lead = st.segmented_control("Forecast issued", [1, 3, 6], key="fc_lead", default=3, required=True,
                                persist_state="session", format_func=lambda k: f"{k} month{'s' if k > 1 else ''} ahead",
                                help="How far ahead each forecast is made. Three months gives time to prepare for a fire season.")
if ctx.period:
    st.caption("The model comparison always uses the full record, not the study period, so every model sees the same years.")

lb = C.leaderboard(ctx.name, ctx.key, ctx.raw, fvar, lead)
table = lb["table"]
rec = models.recommend(table)
spec = models.BY_KEY[rec["key"]]
lab, unit = core.meta(fvar)[0], core.meta(fvar)[1]

if rec["key"] == "climatology":
    st.warning(rec["reason"], icon=":material/balance:", title="Recommended: seasonal average")
else:
    st.success(rec["reason"], icon=":material/emoji_events:", title=f"Recommended: {spec.name}")

# ----------------------------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------------------------
st.subheader(f"Leaderboard: {lab}, {lead} month{'s' if lead > 1 else ''} ahead", anchor=False)
t = table.sort_values("Skill vs seasonal average")
colors = [C.FAMILY_COLOR[models.BY_KEY[k].family] for k in t["key"]]
fig = go.Figure(go.Bar(
    x=t["Skill vs seasonal average"] * 100, y=t["Model"], orientation="h",
    marker=dict(color=colors, cornerradius=4, line=dict(color=[C.INK if k == rec["key"] else "rgba(0,0,0,0)" for k in t["key"]],
                                                         width=2.5)),
    customdata=np.stack([t["Years beating seasonal average"], t["Years tested"], t["Type"]], axis=1),
    hovertemplate="%{y}<br>%{x:+.0f}% vs seasonal average<br>won %{customdata[0]} of %{customdata[1]} years"
                  "<br>type: %{customdata[2]}<extra></extra>"))
fig.add_vline(x=0, line_color=C.INK2, line_width=1)
C.style(fig, 90 + 36 * len(t), legend=False, left=220)
fig.update_layout(hovermode="closest", bargap=0.35, xaxis_title="Error reduction vs the seasonal average (%), higher is better")
C.show(fig, alt=f"Forecast skill of each model for {lab}")
st.caption("Colours: grey = baseline, blue = linear, green = count regression, orange = tree ensemble, "
           "purple = neural network. The outlined bar is the recommended model.")

st.dataframe(
    table.drop(columns=["key"]), hide_index=True, width="stretch", alt=f"Model leaderboard for {lab}",
    column_config={
        "MAE": st.column_config.NumberColumn("Mean abs. error", format="%.3g", help=f"In {unit}. Lower is better."),
        "RMSE": st.column_config.NumberColumn(format="%.3g"),
        "Skill vs seasonal average": st.column_config.NumberColumn(format="percent"),
        "Dry-season skill": st.column_config.NumberColumn("Jul-Oct skill", format="percent",
                                                          help="Skill in the fire season only."),
        "Years beating seasonal average": st.column_config.NumberColumn("Years won"),
        "How it works": st.column_config.TextColumn(width="large"),
    })

# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
preds = lb["preds"]
show_keys = [rec["key"]] + ([rec["top"]] if rec.get("top") and rec["top"] != rec["key"] else [])
with st.container(border=True):
    st.markdown(f"**Backtest: every year from {lb['years'][0]} predicted without seeing it**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds.index, y=preds["actual"], mode="lines", name="Observed",
                             line=dict(color=C.INK2, width=2), hovertemplate="%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=preds.index, y=preds["climatology"], mode="lines", name="Seasonal average",
                             line=dict(color=C.MUTED, width=1.6, dash="dot"), hovertemplate="%{y:.3g}<extra></extra>"))
    for k, colr in zip(show_keys, (C.ORANGE, C.BLUE)):
        if k != "climatology":
            fig.add_trace(go.Scatter(x=preds.index, y=preds[k], mode="lines", name=models.BY_KEY[k].name,
                                     line=dict(color=colr, width=2), hovertemplate="%{y:.3g}<extra></extra>"))
    C.style(fig, 360)
    fig.update_yaxes(title=f"{lab} ({unit})")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05))
    C.show(fig, alt=f"Backtest of {lab}: observed against the models' predictions")

    if rec["key"] != "climatology":
        e = (preds[rec["key"]] - preds["actual"]).abs().groupby(preds["year"]).mean()
        ec = (preds["climatology"] - preds["actual"]).abs().groupby(preds["year"]).mean()
        gain = (1 - e / ec) * 100
        fig = go.Figure(go.Bar(x=gain.index, y=gain.values, marker=dict(color=[C.AQUA if v > 0 else C.STATUS["critical"]
                                                                              for v in gain.values], cornerradius=3),
                               hovertemplate="%{x}: %{y:+.0f}%<extra></extra>"))
        fig.add_hline(y=0, line_color=C.INK2, line_width=1)
        C.style(fig, 240, legend=False)
        fig.update_layout(hovermode="closest", bargap=0.3, yaxis_title="% better than average",
                          title=dict(text=f"{spec.name}: error reduction in each test year (green = beat the seasonal average)",
                                     font=dict(size=13, color=C.INK2)))
        C.show(fig, alt=f"Year-by-year skill of {spec.name}")

# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------
st.subheader("Forecast", anchor=False)
opts = list(table["key"])
if ss.get("_fc_rec_for") != (fvar, lead) or ss.get("fc_model") not in opts:
    ss["fc_model"] = rec["key"]
    ss["_fc_rec_for"] = (fvar, lead)
mkey = st.selectbox("Model", opts, key="fc_model", format_func=lambda k: models.BY_KEY[k].name
                    + ("  (recommended)" if k == rec["key"] else ""))
with st.spinner(f"Fitting {models.BY_KEY[mkey].name} for each of the next {ctx.horizon} months..."):
    f = C.ml_forecast(ctx.key, ctx.raw, fvar, mkey, ctx.horizon)
fc, hist = f["forecast"], f["history"]
sk = float(table.set_index("key").loc[mkey, "Skill vs seasonal average"])
m = st.columns(4)
C.tile(m[0], "Peak forecast", f"{C.num(fc['forecast'].max())} {unit}", f"{fc['forecast'].idxmax():%b %Y}")
C.tile(m[1], "Normal for that month", f"{C.num(fc.loc[fc['forecast'].idxmax(), 'climatology'])} {unit}")
C.tile(m[2], "Historical 90th percentile", f"{C.num(f['p90'])} {unit}")
C.tile(m[3], f"Backtest skill ({lead} mo ahead)", f"{sk:+.0%}", "above 0 beats the seasonal average")

fig = go.Figure()
h = hist.iloc[-84:]
fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(217,89,38,0.18)", name="80% range", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=h.index, y=h, mode="lines", line=dict(color=C.BLUE, width=2), name="Observed",
                         hovertemplate="%{y:.3g}<extra></extra>"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["climatology"], mode="lines", line=dict(color=C.MUTED, width=2, dash="dot"),
                         name="Normal for the month", hovertemplate="%{y:.3g}<extra></extra>"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], mode="lines+markers", line=dict(color=C.ORANGE, width=2.5),
                         marker=dict(size=6), name=f"Forecast ({f['model']})", hovertemplate="%{y:.3g}<extra></extra>"))
fig.add_hline(y=f["p90"], line_color=C.MUTED, line_width=1, annotation_text="90th percentile",
              annotation_font_color=C.MUTED, annotation_position="top left")
C.style(fig, 420)
fig.update_yaxes(title=f"{lab} ({unit})")
C.show(fig, alt=f"{ctx.horizon}-month forecast of {lab} with an 80% range")
st.caption(f"One model is fitted for each lead time (1 to {ctx.horizon} months), each using only data up to "
           f"{hist.index[-1]:%B %Y}. The 80% range comes from this model's own errors at each lead over "
           f"{f['error_years'][0]}-{f['error_years'][-1]}, so it is as wide as the model has earned.")
if sk <= 0.05:
    st.warning("This model does not clearly beat the seasonal average for this lead. Read the forecast as "
               "'roughly normal', not as a prediction of a specific spike.", icon=":material/warning:")

# ----------------------------------------------------------------------------
# More detail
# ----------------------------------------------------------------------------
with st.expander("What does gradient boosting rely on?", icon=":material/account_tree:", on_change="rerun") as ex:
    if ex.open:
        imp = C.importance(ctx.key, ctx.raw, fvar, lead)
        fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h",
                               marker=dict(color=C.ORANGE, cornerradius=3), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        C.style(fig, 360, legend=False, left=280)
        fig.update_layout(hovermode="closest", xaxis_title="Drop in accuracy when this input is shuffled")
        C.show(fig, alt=f"Inputs the gradient-boosting model relies on for {lab}")
        st.caption("Permutation importance on the full record. Large values = the model leans on that input. "
                   "Correlated inputs share credit, so treat the ranking as a guide.")

old = ctx.res["forecasts"].get(fvar)
if old:
    with st.expander("ENSO scenario model (used by the alerts)", icon=":material/thunderstorm:"):
        fo = old["forecast"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=old["history"].index[-84:], y=old["history"].iloc[-84:], mode="lines", name="Observed",
                                 line=dict(color=C.BLUE, width=2)))
        fig.add_trace(go.Scatter(x=fo.index, y=fo["forecast"], mode="lines", name="Scenario",
                                 line=dict(color=C.ORANGE, width=2)))
        fig.add_trace(go.Scatter(x=fo.index, y=fo["climatology"], mode="lines", name="Normal for the month",
                                 line=dict(color=C.MUTED, width=2, dash="dot")))
        C.show(C.style(fig, 320), alt=f"ENSO-driven scenario for {lab}")
        st.caption(f"{old['model']}. Its backtest is given the real Nino 3.4 and DMI, so it shows what the drivers "
                   f"explain (skill {old['skill'].get('skill', np.nan):+.0%}), not what could be forecast in advance. "
                   "Driver values past the end of the index file decay toward zero.")

with st.expander("Why not an LSTM or other deep learning?", icon=":material/help:"):
    st.markdown(
        "- **Too little data.** Each region has about 300 months, and the events that matter (strong El Nino dry seasons) "
        "number about five. A recurrent network would overfit or collapse to the seasonal average.\n"
        "- **The neural network here stands in for deep learning.** A small 3-seed MLP ensemble on the same inputs "
        "is in the leaderboard, scored the same way, so you can see how a neural model does.\n"
        "- **Explainability matters.** The audience is students and the public; a regression or a tree model can be "
        "explained, and the recommender prefers the simpler model when the scores are close.\n"
        "- **When it would make sense:** training on every grid pixel (thousands of series) rather than three regional "
        "averages, and even then neighbouring pixels are strongly correlated.")
