import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tdui import common as C
from trendlab import ml

ctx = C.get_ctx()
st.title("Patterns", anchor=False)
st.caption("Which variables move together, and which months were unusual for their combination of conditions.")
res, df = ctx.res, ctx.df

if st.session_state.get("pat_feats") is None or not set(st.session_state["pat_feats"]) <= set(ctx.cols):
    st.session_state["pat_feats"] = res["features"]
# Chips, like the design. (A multiselect draws white tag text on the primary colour, which is white in this theme.)
feats = st.pills("Variables to combine", ctx.cols, selection_mode="multi", format_func=C.label, key="pat_feats",
                 persist_state="session")
if len(feats) < 2:
    st.info("Pick at least two variables.", icon=":material/info:")
    st.stop()

pca = ml.pca_modes(df, feats)
out = ml.elliptic_outliers(df, feats, ctx.contamination)
left, right = st.columns(2)
if pca:
    with left:
        st.subheader("Which variables move together? (PCA)", anchor=False)
        ev = pca["explained"]
        st.caption(f"PC1 explains {ev[0]:.0%} of all month-to-month variability, PC2 {ev[1]:.0%}. "
                   + " ".join(f"PC1 correlates {v['PC1']:+.2f} with {d}." for d, v in pca["driver_corr"].items()))
        ld = pca["loadings"]["PC1"].sort_values()
        fig = go.Figure(go.Bar(x=ld.values, y=[C.label(c) for c in ld.index], orientation="h",
                               marker=dict(color=[C.ORANGE if v > 0 else C.BLUE for v in ld.values], cornerradius=4),
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.add_vline(x=0, line_color=C.AXIS, line_width=1)
        C.style(fig, 330, legend=False, left=220)
        fig.update_layout(hovermode="closest", xaxis_title="PC1 loading", bargap=0.4)
        C.show(fig, alt="Loadings of the first principal component")
    with right:
        st.subheader("PC1 over time", anchor=False)
        st.caption("One combined index of the dominant pattern. Spikes are the months when everything lined up.")
        sc = pca["scores"]["PC1"].asfreq("MS")
        fig = go.Figure()
        fig.add_hline(y=0, line_color=C.AXIS, line_width=1)
        fig.add_trace(go.Scatter(x=sc.index, y=sc, mode="lines", line=dict(color=C.BLUE, width=2), name="PC1",
                                 hovertemplate="%{y:+.2f}<extra></extra>"))
        C.show(C.style(fig, 330, legend=False), alt="First principal component score over time")

if len(out):
    st.subheader("Unusual months (Elliptic Envelope)", anchor=False)
    st.caption("A robust covariance ellipse is fitted to the anomalies of all selected variables. The distance says how far "
               "a month's combination of conditions sits from normal; months above the line are flagged.")
    thr = out["threshold"].iloc[0]
    d = out["distance"].asfreq("MS")
    flagged = out[out["outlier"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d, mode="lines", line=dict(color=C.BLUE, width=2), name="Distance from normal",
                             hovertemplate="%{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=flagged.index, y=flagged["distance"], mode="markers", name="Flagged month",
                             marker=dict(color=C.ORANGE, size=9, line=dict(color=C.SURFACE, width=2)),
                             customdata=np.stack([[C.label(c) for c in flagged["main_driver"]], flagged["main_z"]], axis=1),
                             hovertemplate="%{customdata[0]} %{customdata[1]:+.1f} sigma<extra></extra>"))
    if not np.isnan(thr):
        fig.add_hline(y=thr, line_color=C.MUTED, line_width=1, line_dash="dot",
                      annotation_text="flag threshold", annotation_font_color=C.MUTED, annotation_position="top left")
    C.style(fig, 340)
    fig.update_yaxes(type="log", title="Mahalanobis distance (log scale)",
                     range=[np.log10(d.min() * 0.8), np.log10(d.max() * 1.6)])
    C.show(fig, alt="Distance of each month from normal conditions, with flagged months")
    tbl = flagged.sort_values("distance", ascending=False).head(15)
    st.dataframe(pd.DataFrame({"Month": tbl.index.strftime("%Y-%m"), "Distance": tbl["distance"].round(1),
                               "Largest contributor": [C.label(c) for c in tbl["main_driver"]],
                               "Its anomaly (sigma)": tbl["main_z"].round(1),
                               "Nino 3.4": df["nino34"].reindex(tbl.index).round(2).values if "nino34" in df else np.nan}),
                 hide_index=True, width="stretch", alt="Most unusual months and what drove them")
    by_year = flagged.groupby(flagged.index.year).size()
    st.caption("Flagged months by year: " + ", ".join(f"{y} ({n})" for y, n in by_year.items()) + ".")
