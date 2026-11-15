import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tdui import common as C
from trendlab import advisor as A
from trendlab import core

ctx = C.get_ctx()
st.title("Explore trends", anchor=False)
st.caption(f"{ctx.label} · {ctx.period_text} · {len(ctx.cols)} variables")

var = C.pick_variable(ctx, "var", "What do you want to investigate?", default="vpd_kpa")
r = C.variable_detail(ctx.key, ctx.raw, var, ctx.alpha, ctx.period)
lab, unit, definition, logged = core.meta(var)

with st.container(border=True):
    C.verdict_badge(r["verdict"])
    st.markdown(A.plain_conclusion(r, ctx.where, ctx.period_text, ctx.alpha))
    st.caption(definition + ("  Analysed on a log(1 + count) scale because counts are extremely skewed." if logged else ""))

m = st.columns(4)
C.tile(m[0], "Verdict", r["verdict"].capitalize(), f"seasonal Kendall p {C.fmt_p(r['sk_p'])}")
C.tile(m[1], "Rate of change", f"{r['sk_slope_per_decade']:+.3g} {r['unit']} / decade",
       f"{A.pct_per_decade(r):+.1f}% per decade")
C.tile(m[2], "Total change over the record", f"{r['total_change']:+.3g} {r['unit']}", ctx.period_text)
C.tile(m[3], "After removing ENSO / IOD", r["verdict_adjusted"].capitalize(), f"p {C.fmt_p(r['adj_p'])}")

df = C.sliced(ctx.key, ctx.raw, ctx.period)
anom = r["_anom"]
line = r["_sen_intercept"] + r["sen_slope_per_decade"] / 10 * core._years(anom.index)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=(f"{lab} as measured ({unit})", "Seasonal cycle removed (anomaly), with trend line"))
fig.add_trace(go.Scatter(x=df.index, y=df[var], mode="lines", line=dict(color=C.BLUE, width=2), name="Measured",
                         hovertemplate="%{y:.3g}<extra></extra>"), 1, 1)
fig.add_hline(y=0, line_color=C.AXIS, line_width=1, row=2, col=1)
fig.add_trace(go.Scatter(x=anom.index, y=anom, mode="lines", line=dict(color=C.BLUE, width=2), name="Anomaly",
                         hovertemplate="%{y:+.3g}<extra></extra>"), 2, 1)
fig.add_trace(go.Scatter(x=anom.index, y=line, mode="lines", line=dict(color=C.ORANGE, width=2.5), name="Sen trend",
                         hovertemplate="%{y:+.3g}<extra></extra>"), 2, 1)
C.style(fig, 520)
fig.update_annotations(font=dict(size=13, color=C.INK2), x=0, xanchor="left")
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
C.show(fig, alt=f"{lab} over time, measured and as an anomaly with its trend line")

st.subheader("Is it significant?", anchor=False)
tests = pd.DataFrame([
    {"Test": "Seasonal Kendall (serial-correlation corrected)", "What it asks": "Within each calendar month, do later years rank higher?",
     "Slope / decade": r["sk_slope_per_decade"], "Statistic": f"z = {r['sk_z']:+.2f}", "p-value": C.fmt_p(r["sk_p"])},
    {"Test": "Mann-Kendall on anomalies (Hamed-Rao corrected)", "What it asks": "Is there a monotonic rise or fall, linear or not?",
     "Slope / decade": r["sen_slope_per_decade"], "Statistic": f"z = {r['mk_z']:+.2f}", "p-value": C.fmt_p(r["mk_p"])},
    {"Test": "t-test on the linear slope (AR(1)-adjusted)", "What it asks": "Is the straight-line slope different from zero?",
     "Slope / decade": r["ols_slope_per_decade"], "Statistic": f"t = {r['ols_t']:+.2f}, n_eff = {r['ols_n_eff']:.0f}",
     "p-value": C.fmt_p(r["ols_p"])},
    {"Test": "t-test controlling for Nino 3.4 and DMI", "What it asks": "Is there still a trend once El Nino / IOD swings are explained?",
     "Slope / decade": r["adj_slope_per_decade"], "Statistic": f"R2 = {r['adj_r2']:.2f}", "p-value": C.fmt_p(r["adj_p"])},
])
st.dataframe(tests, hide_index=True, width="stretch", alt=f"Four significance tests for {lab}",
             column_config={"Slope / decade": st.column_config.NumberColumn(format="%+.3g")})
st.caption(f"95% confidence interval on the Sen slope: {r['sen_lo']:+.3g} to {r['sen_hi']:+.3g} {r['unit']} per decade. "
           f"Lag-1 autocorrelation of residuals {r['ar1']:.2f}, so {r['n']} months behave like about {r['ols_n_eff']:.0f} "
           "independent ones. " + " ".join(f"{d} leads by {v['lag']} month(s), r = {v['r']:+.2f}."
                                           for d, v in r["driver_lags"].items()))

pm = pd.DataFrame(r["_per_month"])
fig = go.Figure(go.Bar(x=C.MONTHS, y=pm["slope_per_decade"], marker=dict(color=C.BLUE, cornerradius=4, line=dict(width=0)),
                       marker_pattern_shape=["" if p < ctx.alpha else "/" for p in pm["p"]],
                       customdata=pm["p"].map(C.fmt_p), hovertemplate="%{x}: %{y:+.3g} per decade<br>p %{customdata}<extra></extra>"))
C.style(fig, 280, legend=False)
fig.update_layout(title=dict(text=f"Which months are changing? Slope per decade by calendar month (hatched = not significant at {ctx.alpha})",
                             font=dict(size=13, color=C.INK2)), hovermode="closest", bargap=0.35)
fig.add_hline(y=0, line_color=C.AXIS, line_width=1)
C.show(fig, alt=f"Trend in {lab} for each calendar month")

st.subheader("All variables at once", anchor=False)
st.caption("Each dot is the trend per decade in units of that variable's normal month-to-month spread, so different "
           "variables share one axis. Bars are 95% confidence intervals; filled dots are significant.")
table = ctx.table
t = table.assign(std_slope=table["sen_slope_per_decade"] / table["anom_std"],
                 lo=table["sen_lo"] / table["anom_std"], hi=table["sen_hi"] / table["anom_std"]).sort_values("std_slope")
is_sig = t["sk_p"] < ctx.alpha
fig = go.Figure()
fig.add_vline(x=0, line_color=C.AXIS, line_width=1)
for flag, nm, colr, sym in [(True, "Significant", C.BLUE, "circle"), (False, "Not significant", C.MUTED, "circle-open")]:
    d = t[is_sig == flag]
    fig.add_trace(go.Scatter(
        x=d["std_slope"], y=d["label"], mode="markers", name=nm,
        marker=dict(color=colr, size=11, symbol=sym, line=dict(width=2, color=colr)),
        error_x=dict(type="data", symmetric=False, array=d["hi"] - d["std_slope"], arrayminus=d["std_slope"] - d["lo"],
                     color=colr, thickness=2, width=0),
        customdata=np.stack([d["sk_slope_per_decade"], d["unit"], d["sk_p"].map(C.fmt_p)], axis=1),
        hovertemplate="%{y}<br>%{customdata[0]:+.3g} %{customdata[1]} per decade<br>seasonal Kendall p %{customdata[2]}<extra></extra>"))
C.style(fig, 110 + 34 * len(t), left=240)
fig.update_layout(hovermode="closest", xaxis_title="Trend per decade (standard deviations of the monthly anomaly)")
C.show(fig, alt="Standardised trend of every variable with confidence intervals")

view = table[["label", "unit", "sk_slope_per_decade", "pct_per_decade", "sk_p", "mk_p", "ols_p", "adj_p", "verdict"]].sort_values("sk_p")
view.columns = ["Variable", "Units", "Slope / decade", "% / decade", "Seasonal Kendall p", "Mann-Kendall p", "t-test p",
                "ENSO-adjusted p", "Verdict"]
st.dataframe(view, hide_index=True, width="stretch", alt="Trend results for every variable", column_config={
    "Slope / decade": st.column_config.NumberColumn(format="%+.3g"), "% / decade": st.column_config.NumberColumn(format="%+.1f"),
    **{k: st.column_config.NumberColumn(format="%.4f") for k in ["Seasonal Kendall p", "Mann-Kendall p", "t-test p", "ENSO-adjusted p"]}})
st.caption("Soil moisture appears twice (mm and volume fraction): they are the same quantity in different units. "
           "With 11 variables tested, about one 'significant' result in 20 can be chance at p = 0.05.")
