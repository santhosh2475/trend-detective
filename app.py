"""Earth Systems Trend Detective: Streamlit dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trendlab import alerts as al
from trendlab import core, grid, ml, pipeline

st.set_page_config(page_title="Trend Detective", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BUNDLED = {"Central Kalimantan (fire hotspot)": "central_kalimantan",
           "Riau / South Sumatra (fire hotspot)": "riau_south_sumatra",
           "Singapore (where haze arrives)": "singapore_receptor"}

# palette (validated categorical order; status colours reserved for alerts)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATUS = {"critical": "#d03b3b", "warning": "#fab219", "info": "#898781"}
ICON = {"critical": "🔴", "warning": "🟠", "info": "⚪"}
DIVERGING = [[0, "#184f95"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a37f"], [1, "#b8431a"]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style(fig, height=380, legend=True, left=70):
    fig.update_layout(height=height, margin=dict(l=left, r=20, t=64, b=50), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=13), hovermode="x unified",
                      showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right", font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def tile(col, label, value, sub=""):
    """Stat tile that wraps instead of truncating (st.metric cuts long text)."""
    col.markdown(f"<div style='font-size:0.85rem;color:{INK2}'>{label}</div>"
                 f"<div style='font-size:1.45rem;font-weight:600;line-height:1.25;color:{INK}'>{value}</div>"
                 f"<div style='font-size:0.8rem;color:{MUTED};margin-bottom:0.8rem'>{sub}</div>", unsafe_allow_html=True)


def num(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def fmt_p(p):
    return "n/a" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


# ----------------------------------------------------------------------------
# Cached work
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running the pipeline...")
def run_pipeline(key: str, raw: bytes, name: str, alpha: float, contamination: float, horizon: int):
    return pipeline.run(io.BytesIO(raw), name, alpha, contamination, horizon)


@st.cache_data(show_spinner=False)
def variable_detail(key: str, raw: bytes, col: str, alpha: float):
    df, _ = core.load_csv(io.BytesIO(raw), core.load_indices())
    return core.analyse_variable(df, col, alpha)


@st.cache_resource(show_spinner=False)
def grid_index():
    return grid.find_grids(DATA)


@st.cache_resource(show_spinner="Loading grid...")
def grid_var(var: str):
    return grid.load_var(grid_index()[var], var)


@st.cache_data(show_spinner="Testing every pixel (seasonal Kendall)...")
def grid_trends(var: str):
    return grid.pixel_trends(grid_var(var))


@st.cache_resource(show_spinner=False)
def coast():
    return grid.land_mask(DATA)


# ----------------------------------------------------------------------------
# Sidebar: data source
# ----------------------------------------------------------------------------
st.sidebar.title("Trend Detective")
st.sidebar.caption("Find what is changing, where, by how much, and whether it is significant.")
source = st.sidebar.radio("Data", ["Bundled region", "Upload a CSV"], key="source")
if source == "Bundled region":
    label = st.sidebar.selectbox("Where do you want to investigate?", list(BUNDLED), key="region")
    name = BUNDLED[label]
    raw = (DATA / f"{name}.csv").read_bytes()
else:
    up = st.sidebar.file_uploader("Monthly CSV (a date column + numeric columns)", type="csv")
    if up is None:
        st.info("Upload a CSV with one row per month. The first column (or a column named `date`) must be the date; "
                "every other numeric column is analysed. Nino 3.4 and DMI are added automatically from the bundled index file.")
        st.stop()
    raw, name, label = up.getvalue(), Path(up.name).stem, up.name
with st.sidebar.expander("Settings"):
    alpha = st.select_slider("Significance level", [0.01, 0.05, 0.10], 0.05)
    contamination = st.slider("Expected share of unusual months", 0.01, 0.15, 0.05, 0.01)
    horizon = st.slider("Forecast horizon (months)", 3, 18, 12)

key = hashlib.sha1(raw).hexdigest()
try:
    res = run_pipeline(key, raw, name, alpha, contamination, horizon)
except Exception as e:
    st.error(f"Could not process this file: {e}")
    st.stop()
df, table, q = res["data"], res["table"], res["quality"]
cols = res["columns"]
if not cols:
    st.error("No numeric column with at least 36 months of data was found.")
    st.stop()

# toast new alerts once per dataset/settings
seen = st.session_state.setdefault("toasted", set())
for a in res["alerts"]:
    if a["severity"] == "critical" and (key, a["id"]) not in seen:
        st.toast(a["title"], icon="🔴")
        seen.add((key, a["id"]))

n_crit = sum(a["severity"] == "critical" for a in res["alerts"])
tabs = st.tabs(["Start", "Trends", "Compare regions", "Map", "Patterns", "Forecast",
                f"Alerts ({len(res['alerts'])})", "Pipeline & export"])

# ----------------------------------------------------------------------------
# Start
# ----------------------------------------------------------------------------
with tabs[0]:
    st.header(f"{label}")
    st.caption(f"{q['start']} to {q['end']} · {q['n_years']} years · {len(cols)} variables · "
               f"{len(q['missing_rows_inserted'])} missing month(s) inserted as blanks")
    sig = table[table["sk_p"] < alpha]
    c = st.columns(4)
    c[0].metric("Variables tested", len(table))
    c[1].metric("Significant trends", len(sig))
    c[2].metric("Unusual months flagged", int(res["outliers"]["outlier"].sum()) if len(res["outliers"]) else 0)
    c[3].metric("Critical alerts", n_crit)
    for w in q["warnings"]:
        st.warning(w)

    st.subheader("What is changing")
    st.caption("Each dot is the trend per decade, in units of that variable's normal month-to-month spread, so "
               "different variables can share one axis. Bars are 95% confidence intervals. Filled = significant.")
    t = table.assign(std_slope=table["sen_slope_per_decade"] / table["anom_std"],
                     lo=table["sen_lo"] / table["anom_std"], hi=table["sen_hi"] / table["anom_std"]).sort_values("std_slope")
    is_sig = t["sk_p"] < alpha
    fig = go.Figure()
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    for flag, nm, colr, sym in [(True, "Significant", BLUE, "circle"), (False, "Not significant", MUTED, "circle-open")]:
        d = t[is_sig == flag]
        fig.add_trace(go.Scatter(
            x=d["std_slope"], y=d["label"], mode="markers", name=nm,
            marker=dict(color=colr, size=11, symbol=sym, line=dict(width=2, color=colr)),
            error_x=dict(type="data", symmetric=False, array=d["hi"] - d["std_slope"], arrayminus=d["std_slope"] - d["lo"],
                         color=colr, thickness=2, width=0),
            customdata=np.stack([d["sk_slope_per_decade"], d["unit"], d["sk_p"].map(fmt_p)], axis=1),
            hovertemplate="%{y}<br>%{customdata[0]:+.3g} %{customdata[1]} per decade<br>seasonal Kendall p %{customdata[2]}<extra></extra>"))
    style(fig, 110 + 34 * len(t), left=240)
    fig.update_layout(hovermode="closest", xaxis_title="Trend per decade (standard deviations of the monthly anomaly)")
    show(fig)

    st.subheader("Try a demonstration")
    demos = [("Is the air getting thirstier?", "Central Kalimantan (fire hotspot)", "vpd_kpa"),
             ("Are fires becoming more frequent?", "Riau / South Sumatra (fire hotspot)", "fire_count"),
             ("Is Singapore's haze getting worse?", "Singapore (where haze arrives)", "aod")]

    def _demo(region, var):
        st.session_state.update(source="Bundled region", region=region, var=var)

    for col_, (title, region, var) in zip(st.columns(3), demos):
        col_.button(title, on_click=_demo, args=(region, var), width="stretch")
    st.caption("Pick one, then open the Trends tab.")

    with st.expander("What do the variables mean?"):
        st.dataframe(pd.DataFrame([{"Column": c_, "Name": core.meta(c_)[0], "Units": core.meta(c_)[1],
                                    "Plain meaning": core.meta(c_)[2]} for c_ in cols + core.DRIVERS]),
                     hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# Trends
# ----------------------------------------------------------------------------
with tabs[1]:
    if st.session_state.get("var") not in cols:
        st.session_state["var"] = "vpd_kpa" if "vpd_kpa" in cols else cols[0]
    var = st.selectbox("What do you want to investigate?", cols, key="var",
                       format_func=lambda c_: f"{core.meta(c_)[0]}  ({c_})")
    r = variable_detail(key, raw, var, alpha)
    lab, unit, definition, logged = core.meta(var)
    st.caption(definition + ("  Analysed on a log(1 + count) scale because counts are extremely skewed." if logged else ""))

    m = st.columns(4)
    tile(m[0], "Verdict", r["verdict"].capitalize(), f"seasonal Kendall p {fmt_p(r['sk_p'])}")
    tile(m[1], "Rate of change", f"{r['sk_slope_per_decade']:+.3g} {r['unit']} / decade", f"{r['pct_per_decade']:+.1f}% of the mean per decade")
    tile(m[2], "Total change over the record", f"{r['total_change']:+.3g} {r['unit']}", f"{q['start']} to {q['end']}")
    tile(m[3], "After removing ENSO / IOD", r["verdict_adjusted"].capitalize(), f"p {fmt_p(r['adj_p'])}")

    anom = r["_anom"]
    yrs = core._years(anom.index)
    line = r["_sen_intercept"] + r["sen_slope_per_decade"] / 10 * yrs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"{lab} as measured ({unit})", "Seasonal cycle removed (anomaly), with trend line"))
    fig.add_trace(go.Scatter(x=df.index, y=df[var], mode="lines", line=dict(color=BLUE, width=2), name="Measured",
                             hovertemplate="%{y:.3g}<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line_color=AXIS, line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=anom.index, y=anom, mode="lines", line=dict(color=BLUE, width=2), name="Anomaly",
                             hovertemplate="%{y:+.3g}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=anom.index, y=line, mode="lines", line=dict(color=ORANGE, width=2), name="Sen trend",
                             hovertemplate="%{y:+.3g}<extra></extra>"), 2, 1)
    style(fig, 520)
    fig.update_annotations(selector=dict(xref="x domain"), font=dict(size=13, color=INK2), x=0, xanchor="left")
    fig.update_annotations(font=dict(size=13, color=INK2), x=0, xanchor="left")
    show(fig)

    st.subheader("Is it significant?")
    tests = pd.DataFrame([
        {"Test": "Seasonal Kendall (serial-correlation corrected)", "What it asks": "Within each calendar month, do later years rank higher?",
         "Slope / decade": r["sk_slope_per_decade"], "Statistic": f"z = {r['sk_z']:+.2f}", "p-value": fmt_p(r["sk_p"])},
        {"Test": "Mann-Kendall on anomalies (Hamed-Rao corrected)", "What it asks": "Is there a monotonic rise or fall, linear or not?",
         "Slope / decade": r["sen_slope_per_decade"], "Statistic": f"z = {r['mk_z']:+.2f}", "p-value": fmt_p(r["mk_p"])},
        {"Test": "t-test on the linear slope (AR(1)-adjusted)", "What it asks": "Is the straight-line slope different from zero?",
         "Slope / decade": r["ols_slope_per_decade"], "Statistic": f"t = {r['ols_t']:+.2f}, n_eff = {r['ols_n_eff']:.0f}", "p-value": fmt_p(r["ols_p"])},
        {"Test": "t-test controlling for Nino 3.4 and DMI", "What it asks": "Is there still a trend once El Nino / IOD swings are explained?",
         "Slope / decade": r["adj_slope_per_decade"], "Statistic": f"R2 = {r['adj_r2']:.2f}", "p-value": fmt_p(r["adj_p"])},
    ])
    st.dataframe(tests, hide_index=True, width="stretch", column_config={"Slope / decade": st.column_config.NumberColumn(format="%+.3g")})
    st.caption(f"95% confidence interval on the Sen slope: {r['sen_lo']:+.3g} to {r['sen_hi']:+.3g} {r['unit']} per decade. "
               f"Lag-1 autocorrelation of residuals {r['ar1']:.2f}, so {r['n']} months behave like about {r['ols_n_eff']:.0f} independent ones. "
               + " ".join(f"{d} leads by {v['lag']} month(s), r = {v['r']:+.2f}." for d, v in r["driver_lags"].items()))

    pm = pd.DataFrame(r["_per_month"])
    fig = go.Figure(go.Bar(x=MONTHS, y=pm["slope_per_decade"], marker=dict(color=BLUE, cornerradius=4, line=dict(width=0)),
                           marker_pattern_shape=["" if p < alpha else "/" for p in pm["p"]],
                           customdata=pm["p"].map(fmt_p), hovertemplate="%{x}: %{y:+.3g} per decade<br>p %{customdata}<extra></extra>"))
    style(fig, 280, legend=False)
    fig.update_layout(title=dict(text=f"Which months are changing? Slope per decade by calendar month (hatched = not significant at {alpha})",
                                 font=dict(size=13, color=INK2)), hovermode="closest", bargap=0.35)
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    show(fig)

    st.subheader("All variables")
    view = table[["label", "unit", "sk_slope_per_decade", "pct_per_decade", "sk_p", "mk_p", "ols_p", "adj_p", "verdict"]].sort_values("sk_p")
    view.columns = ["Variable", "Units", "Slope / decade", "% / decade", "Seasonal Kendall p", "Mann-Kendall p", "t-test p", "ENSO-adjusted p", "Verdict"]
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Slope / decade": st.column_config.NumberColumn(format="%+.3g"), "% / decade": st.column_config.NumberColumn(format="%+.1f"),
        **{k: st.column_config.NumberColumn(format="%.4f") for k in ["Seasonal Kendall p", "Mann-Kendall p", "t-test p", "ENSO-adjusted p"]}})

# ----------------------------------------------------------------------------
# Compare regions
# ----------------------------------------------------------------------------
with tabs[2]:
    st.caption("The same variable can trend differently in different places. This compares the three bundled regions.")
    cvar = st.selectbox("Variable", [c_ for c_ in core.VARIABLES if c_ not in core.DRIVERS], index=8,
                        format_func=lambda c_: core.meta(c_)[0], key="cmp_var")
    rows, series = [], {}
    for lab_, nm in BUNDLED.items():
        b = (DATA / f"{nm}.csv").read_bytes()
        try:
            rr = variable_detail(hashlib.sha1(b).hexdigest(), b, cvar, alpha)
        except Exception:
            continue
        series[lab_] = rr
        rows.append({"Region": lab_, "Slope / decade": rr["sk_slope_per_decade"], "% / decade": rr["pct_per_decade"],
                     "Seasonal Kendall p": rr["sk_p"], "ENSO-adjusted p": rr["adj_p"], "Verdict": rr["verdict"]})
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=list(series))
    for i, (lab_, rr) in enumerate(series.items(), 1):
        a = rr["_anom"]
        fig.add_hline(y=0, line_color=AXIS, line_width=1, row=i, col=1)
        fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", line=dict(color=BLUE, width=2), name="Anomaly",
                                 showlegend=i == 1, hovertemplate="%{y:+.3g}<extra></extra>"), i, 1)
        fig.add_trace(go.Scatter(x=a.index, y=rr["_sen_intercept"] + rr["sen_slope_per_decade"] / 10 * core._years(a.index),
                                 mode="lines", line=dict(color=ORANGE, width=2), name="Sen trend", showlegend=i == 1,
                                 hovertemplate="%{y:+.3g}<extra></extra>"), i, 1)
    style(fig, 200 * len(series) + 60)
    fig.update_annotations(selector=dict(xref="x domain"), font=dict(size=13, color=INK2), x=0, xanchor="left")
    fig.update_annotations(font=dict(size=13, color=INK2), x=0, xanchor="left")
    fig.update_yaxes(matches="y")
    show(fig)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Slope / decade": st.column_config.NumberColumn(format="%+.3g"), "% / decade": st.column_config.NumberColumn(format="%+.1f"),
        "Seasonal Kendall p": st.column_config.NumberColumn(format="%.4f"), "ENSO-adjusted p": st.column_config.NumberColumn(format="%.4f")})

# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
with tabs[3]:
    gi = grid_index()
    if not gi:
        st.info("No NetCDF grids found. Put gldas_grid.nc, imerg_grid.nc and aod_grid.nc in the data/ folder.")
    else:
        c1, c2, c3 = st.columns([2, 2, 3])
        gvar = c1.selectbox("Gridded variable", list(gi), index=list(gi).index("vpd_kpa") if "vpd_kpa" in gi else 0,
                            format_func=lambda c_: core.meta(c_)[0], key="gvar")
        mask_mode = c2.radio("Show", ["All pixels", "Significant (p < 0.05)", "Significant after FDR"], key="mask")
        tr = grid_trends(gvar)
        slope = tr["slope"].copy()
        if mask_mode.startswith("Significant (p"):
            slope[~tr["sig"]] = np.nan
        elif mask_mode.endswith("FDR"):
            slope[~tr["sig_fdr"]] = np.nan
        valid = ~np.isnan(tr["z"])
        up_, dn_ = (tr["sig"] & (tr["slope"] > 0)).sum() / valid.sum(), (tr["sig"] & (tr["slope"] < 0)).sum() / valid.sum()
        c3.markdown(f"**Where is it changing?** {up_:.0%} of pixels show a significant increase and {dn_:.0%} a significant "
                    f"decrease (p < 0.05). {tr['sig_fdr'].sum() / valid.sum():.0%} survive the false-discovery correction across the map.")
        lim = float(np.nanpercentile(np.abs(tr["slope"]), 98)) or 1.0
        fig = go.Figure(go.Heatmap(z=slope, x=tr["lon"], y=tr["lat"], zmin=-lim, zmax=lim, colorscale=DIVERGING,
                                   colorbar=dict(title=dict(text=f"{core.meta(gvar)[1]}<br>per decade", font=dict(size=12)), thickness=14),
                                   customdata=tr["p"], hovertemplate="%{y:.2f}, %{x:.2f}<br>%{z:+.3g} per decade<br>p %{customdata:.3f}<extra></extra>"))
        cm = coast()
        if cm is not None:
            fig.add_trace(go.Contour(z=cm[2], x=cm[1], y=cm[0], contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                                     line=dict(color=INK2, width=1), showscale=False, hoverinfo="skip"))
        box = st.session_state.get("box")
        for nm, (la0, la1, lo0, lo1) in grid.REGIONS.items():
            fig.add_shape(type="rect", x0=lo0, x1=lo1, y0=la0, y1=la1, line=dict(color=INK, width=1.5))
            fig.add_annotation(x=lo0, y=la1, text=nm.replace("_", " "), showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(size=11, color=INK))
        style(fig, 520, legend=False)
        fig.update_layout(hovermode="closest", xaxis_title="Longitude", yaxis_title="Latitude")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        show(fig)
        st.caption("Orange = increasing, blue = decreasing. Per-pixel seasonal Kendall test with seasonal Sen slope. "
                   "Black boxes are the regions behind the CSVs; the thin line is the coast.")

        st.subheader("Test your own box")
        lat = st.slider("Latitude range", float(tr["lat"].min()), float(tr["lat"].max()), (-3.5, -1.5), 0.25)
        lon = st.slider("Longitude range", float(tr["lon"].min()), float(tr["lon"].max()), (112.0, 115.0), 0.25)
        s = grid.region_series(grid_var(gvar), (lat[0], lat[1], lon[0], lon[1]))
        if s.notna().sum() < 36:
            st.info("That box has no data for this variable (GLDAS is land-only).")
        else:
            one, _ = core.load_csv(io.StringIO(s.rename(gvar).rename_axis("date").to_csv()), core.load_indices())
            rb = core.analyse_variable(one, gvar, alpha)
            b = st.columns(4)
            tile(b[0], "Verdict", rb["verdict"].capitalize())
            tile(b[1], "Rate of change", f"{rb['sk_slope_per_decade']:+.3g} {rb['unit']} / decade")
            tile(b[2], "Seasonal Kendall p", fmt_p(rb["sk_p"]))
            tile(b[3], "ENSO-adjusted p", fmt_p(rb["adj_p"]))
            a = rb["_anom"]
            fig = go.Figure()
            fig.add_hline(y=0, line_color=AXIS, line_width=1)
            fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", line=dict(color=BLUE, width=2), name="Anomaly",
                                     hovertemplate="%{y:+.3g}<extra></extra>"))
            fig.add_trace(go.Scatter(x=a.index, y=rb["_sen_intercept"] + rb["sen_slope_per_decade"] / 10 * core._years(a.index),
                                     mode="lines", line=dict(color=ORANGE, width=2), name="Sen trend", hovertemplate="%{y:+.3g}<extra></extra>"))
            show(style(fig, 300))

# ----------------------------------------------------------------------------
# Patterns: PCA + Elliptic Envelope
# ----------------------------------------------------------------------------
with tabs[4]:
    feats = st.multiselect("Variables to combine", cols, res["features"], format_func=lambda c_: core.meta(c_)[0])
    if len(feats) < 2:
        st.info("Pick at least two variables.")
    else:
        pca = ml.pca_modes(df, feats)
        out = ml.elliptic_outliers(df, feats, contamination)
        left, right = st.columns(2)
        if pca:
            with left:
                st.subheader("Which variables move together? (PCA)")
                ev = pca["explained"]
                st.caption(f"PC1 explains {ev[0]:.0%} of all month-to-month variability, PC2 {ev[1]:.0%}. "
                           + " ".join(f"PC1 correlates {v['PC1']:+.2f} with {d}." for d, v in pca["driver_corr"].items()))
                ld = pca["loadings"]["PC1"].sort_values()
                fig = go.Figure(go.Bar(x=ld.values, y=[core.meta(c_)[0] for c_ in ld.index], orientation="h",
                                       marker=dict(color=[ORANGE if v > 0 else BLUE for v in ld.values], cornerradius=4),
                                       hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
                fig.add_vline(x=0, line_color=AXIS, line_width=1)
                style(fig, 330, legend=False, left=220)
                fig.update_layout(hovermode="closest", xaxis_title="PC1 loading", bargap=0.4)
                show(fig)
            with right:
                st.subheader("PC1 over time")
                st.caption("One combined index of the dominant pattern. Spikes are the months when everything lined up.")
                sc = pca["scores"]["PC1"].asfreq("MS")
                fig = go.Figure()
                fig.add_hline(y=0, line_color=AXIS, line_width=1)
                fig.add_trace(go.Scatter(x=sc.index, y=sc, mode="lines", line=dict(color=BLUE, width=2), name="PC1",
                                         hovertemplate="%{y:+.2f}<extra></extra>"))
                show(style(fig, 330, legend=False))
        if len(out):
            st.subheader("Unusual months (Elliptic Envelope)")
            st.caption("A robust covariance ellipse is fitted to the anomalies of all selected variables. The distance says how far "
                       "a month's combination of conditions sits from normal; months above the line are flagged.")
            thr = out["threshold"].iloc[0]
            d = out["distance"].asfreq("MS")
            flagged = out[out["outlier"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.index, y=d, mode="lines", line=dict(color=BLUE, width=2), name="Distance from normal",
                                     hovertemplate="%{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=flagged.index, y=flagged["distance"], mode="markers", name="Flagged month",
                                     marker=dict(color=ORANGE, size=9, line=dict(color="#fcfcfb", width=2)),
                                     customdata=np.stack([[core.meta(c_)[0] for c_ in flagged["main_driver"]], flagged["main_z"]], axis=1),
                                     hovertemplate="%{customdata[0]} %{customdata[1]:+.1f} sigma<extra></extra>"))
            if not np.isnan(thr):
                fig.add_hline(y=thr, line_color=MUTED, line_width=1, line_dash="dot",
                              annotation_text="flag threshold", annotation_font_color=MUTED, annotation_position="top left")
            style(fig, 340)
            fig.update_yaxes(type="log", title="Mahalanobis distance (log scale)",
                             range=[np.log10(d.min() * 0.8), np.log10(d.max() * 1.6)])
            show(fig)
            tbl = flagged.sort_values("distance", ascending=False).head(15)
            st.dataframe(pd.DataFrame({"Month": tbl.index.strftime("%Y-%m"), "Distance": tbl["distance"].round(1),
                                       "Largest contributor": [core.meta(c_)[0] for c_ in tbl["main_driver"]],
                                       "Its anomaly (sigma)": tbl["main_z"].round(1),
                                       "Nino 3.4": df["nino34"].reindex(tbl.index).round(2).values if "nino34" in df else np.nan}),
                         hide_index=True, width="stretch")
            by_year = flagged.groupby(flagged.index.year).size()
            st.caption("Flagged months by year: " + ", ".join(f"{y} ({n})" for y, n in by_year.items()) + ".")

# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------
with tabs[5]:
    if not res["forecasts"]:
        st.info("No forecast was produced for this dataset.")
    else:
        fvar = st.selectbox("Variable", list(res["forecasts"]), format_func=lambda c_: core.meta(c_)[0], key="fvar")
        f = res["forecasts"][fvar]
        fc, hist, sk = f["forecast"], f["history"], f["skill"]
        m = st.columns(4)
        tile(m[0], "Peak forecast", f"{num(fc['forecast'].max())} {core.meta(fvar)[1]}", f"{fc['forecast'].idxmax():%b %Y}")
        tile(m[1], "Historical 90th percentile", num(f["p90"]))
        tile(m[2], "Backtest skill vs climatology", f"{sk.get('skill', np.nan):+.0%}" if sk else "n/a", "above 0 beats the seasonal average")
        tile(m[3], "Backtest months", sk.get("months", 0))
        if sk and sk["skill"] <= 0.1:
            st.warning("On the hold-out period this model did no better than simply using the average seasonal cycle. "
                       "Read the forecast as a what-if scenario under the predicted El Nino, not as a prediction.")
        fig = go.Figure()
        h = hist.iloc[-84:]
        fig.add_trace(go.Scatter(x=fc.index, y=fc["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(235,104,52,0.18)", name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h.index, y=h, mode="lines", line=dict(color=BLUE, width=2), name="Observed",
                                 hovertemplate="%{y:.3g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["climatology"], mode="lines", line=dict(color=MUTED, width=2, dash="dot"),
                                 name="Normal for the month", hovertemplate="%{y:.3g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], mode="lines", line=dict(color=ORANGE, width=2), name="Forecast",
                                 hovertemplate="%{y:.3g}<extra></extra>"))
        fig.add_hline(y=f["p90"], line_color=MUTED, line_width=1, annotation_text="90th percentile", annotation_font_color=MUTED,
                      annotation_position="top left")
        style(fig, 420)
        fig.update_yaxes(title=f"{core.meta(fvar)[0]} ({core.meta(fvar)[1]})")
        show(fig)
        st.caption(f"Model: {f['model']}. " + " ".join(f"{d} enters at a {v['lag']}-month lead (r = {v['r']:+.2f})."
                                                      for d, v in f["driver_lags"].items())
                   + " Driver values past the end of the index file decay toward zero.")
        if f["backtest"] is not None:
            with st.expander("Backtest: how did it do on the last 36 months it never saw?"):
                bt = f["backtest"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt.index, y=bt["actual"], mode="lines", line=dict(color=BLUE, width=2), name="Observed"))
                fig.add_trace(go.Scatter(x=bt.index, y=bt["forecast"], mode="lines", line=dict(color=ORANGE, width=2), name="Forecast"))
                fig.add_trace(go.Scatter(x=bt.index, y=bt["climatology"], mode="lines", line=dict(color=MUTED, width=2, dash="dot"),
                                         name="Normal for the month"))
                show(style(fig, 320))
                st.caption("The backtest is given the real Nino 3.4 and DMI for those months, so it measures how much the drivers "
                           "explain, which is the best case for a real forecast.")

# ----------------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------------
with tabs[6]:
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Rules")
        zt = st.slider("Latest month is extreme beyond (sigma)", 1.5, 4.0, 2.0, 0.5)
        fq = st.slider("Forecast exceeds historical percentile", 0.75, 0.99, 0.90, 0.01)
        nw = st.slider("El Nino watch when Nino 3.4 reaches", 0.5, 2.0, 1.0, 0.1)
        kinds = st.multiselect("Alert types", ["trend", "extreme", "outlier", "forecast", "enso"],
                               ["trend", "extreme", "outlier", "forecast", "enso"])
    alerts = [a for a in al.build_alerts(df, table, res["outliers"], res["forecasts"], core.load_indices(),
                                         {"alpha": alpha, "z_threshold": zt, "forecast_quantile": fq, "nino_watch": nw}, name)
              if a["kind"] in kinds]
    with left:
        st.subheader(f"{len(alerts)} alert(s) for {name}")
        for a in alerts:
            with st.container(border=True):
                st.markdown(f"{ICON[a['severity']]} **{a['severity'].capitalize()} · {a['title']}**")
                st.caption(a["detail"])
        if not alerts:
            st.success("Nothing to flag with the current rules.")
    with right:
        st.subheader("Send")
        log_path = RESULTS / name / "alerts_log.jsonl"
        hook = st.text_input("Slack / Discord / Teams webhook URL", type="password")
        if st.button("Send to webhook", disabled=not (hook and alerts)):
            try:
                st.success(al.send_webhook(alerts, hook, name))
            except Exception as e:
                st.error(f"Webhook failed: {e}")
        mail = st.text_input("Email address (needs SMTP_HOST / SMTP_USER / SMTP_PASSWORD set)")
        if st.button("Send email", disabled=not (mail and alerts)):
            try:
                st.success(al.send_email(alerts, mail, name))
            except Exception as e:
                st.error(f"Email failed: {e}")
        if st.button("Write to alert log"):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            new = al.log_alerts(alerts, log_path)
            st.success(f"{len(new)} new alert(s) logged, {len(alerts) - len(new)} already seen.")
        st.download_button("Download alerts (JSON)", json.dumps(alerts, indent=2), f"{name}_alerts.json")

# ----------------------------------------------------------------------------
# Pipeline & export
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("What happened to your file")
    steps = [
        ("1. Validate", f"Parsed dates, averaged {q['duplicate_months_averaged']} duplicate month(s), dropped {q['unparseable_dates']} bad date(s)."),
        ("2. Complete the calendar", f"Reindexed to {q['n_months']} months; inserted {len(q['missing_rows_inserted'])} blank row(s)"
         + (f" ({', '.join(q['missing_rows_inserted'][:6])})." if q["missing_rows_inserted"] else ".")),
        ("3. Remove the seasonal cycle", "Subtracted each calendar month's mean to get anomalies. Counts are log-transformed first."),
        ("4. Test for trends", "Seasonal Kendall, Mann-Kendall, Sen slope with 95% CI, slope t-test, and a t-test controlling for ENSO / IOD."),
        ("5. Find unusual months", f"Elliptic Envelope on {len(res['features'])} variables; PCA for the dominant pattern."),
        ("6. Forecast", f"{len(res['forecasts'])} variable(s), {horizon} months ahead, with a 36-month backtest."),
        ("7. Alert", f"{len(res['alerts'])} alert(s) raised from the rules."),
    ]
    for title, text in steps:
        st.markdown(f"**{title}** · {text}")
    with st.expander("Missing values per column"):
        st.dataframe(pd.Series(q["missing_per_column"], name="missing months").to_frame(), width="stretch")

    st.subheader("Download")
    d = st.columns(4)
    d[0].download_button("Report (Markdown)", pipeline.report_markdown(res), f"{name}_report.md")
    d[1].download_button("Trend table (CSV)", table.to_csv(index=False), f"{name}_trend_table.csv")
    if len(res["outliers"]):
        d[2].download_button("Unusual months (CSV)", res["outliers"].to_csv(), f"{name}_outliers.csv")
    if res["forecasts"]:
        d[3].download_button("Forecasts (CSV)", pd.concat({c_: f_["forecast"] for c_, f_ in res["forecasts"].items()},
                                                          names=["variable", "date"]).to_csv(), f"{name}_forecast.csv")
    st.subheader("Run it without the dashboard")
    st.code("python -m trendlab.pipeline your_file.csv --out results/your_file\n"
            "python -m trendlab.pipeline your_file.csv --webhook https://hooks.slack.com/...   # notify on new alerts", language="bash")
