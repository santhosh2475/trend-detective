import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tdui import common as C
from trendlab import core, grid

ctx = C.get_ctx()
st.title("Globe", anchor=False)
st.caption("Drag the globe to spin it and scroll to zoom. Black boxes are the regions behind the CSVs.")

gi = C.grid_index()
if not gi:
    st.info("No NetCDF grids found. Put gldas_grid.nc, imerg_grid.nc or aod_grid.nc in the data/ folder.",
            icon=":material/public_off:")
    st.stop()

CITIES = {"Singapore": (1.35, 103.82), "Palangka Raya": (-2.21, 113.92), "Pekanbaru": (0.51, 101.45),
          "Palembang": (-2.99, 104.76), "Jakarta": (-6.21, 106.85), "Kuala Lumpur": (3.14, 101.69),
          "Pontianak": (-0.03, 109.34)}
FIRE_REGIONS = ["riau_south_sumatra", "central_kalimantan"]
SCALE = 4.6


def geo_layout(fig, height=640):
    fig.update_geos(projection_type="orthographic", projection_rotation=dict(lon=107, lat=1),
                    projection_scale=SCALE, showland=True, landcolor="#efeee9", showocean=True, oceancolor="#dde8f2",
                    showcountries=True, countrycolor="#a9a8a0", countrywidth=0.6, coastlinecolor="#6d6c66",
                    coastlinewidth=0.8, showlakes=False, resolution=50, showframe=False, bgcolor="rgba(0,0,0,0)",
                    lataxis=dict(showgrid=True, gridcolor="#c9d6e3", dtick=10),
                    lonaxis=dict(showgrid=True, gridcolor="#c9d6e3", dtick=10))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C.INK2, size=13), showlegend=False)
    return fig


def overlays(fig):
    for nm, (la0, la1, lo0, lo1) in grid.REGIONS.items():
        fig.add_trace(go.Scattergeo(lon=[lo0, lo1, lo1, lo0, lo0], lat=[la0, la0, la1, la1, la0], mode="lines",
                                    line=dict(color=C.INK, width=1.6), hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lon=[v[1] for v in CITIES.values()], lat=[v[0] for v in CITIES.values()],
                                text=list(CITIES), mode="markers+text", textposition="top center",
                                textfont=dict(size=11, color=C.INK), marker=dict(size=5, color=C.INK),
                                hovertemplate="%{text}<extra></extra>"))


gvars = list(gi)
c1, c2, c3 = st.columns([1.2, 1.6, 1.4], vertical_alignment="bottom")
gvar = c1.selectbox("Gridded variable", gvars, index=gvars.index("vpd_kpa") if "vpd_kpa" in gvars else 0,
                    format_func=C.label, key="gvar", persist_state="session")
mode = c2.segmented_control("Show", ["Time-lapse by year", "Trend per pixel"], key="globe_mode",
                            default="Time-lapse by year", required=True, persist_state="session")
unit = core.meta(gvar)[1]
short_unit = "AOD" if unit == "unitless" else unit

# ----------------------------------------------------------------------------
# Time-lapse
# ----------------------------------------------------------------------------
if mode == "Time-lapse by year":
    season = c3.segmented_control("Months", ["Fire season (Aug-Oct)", "Whole year"], key="globe_season",
                                  default="Fire season (Aug-Oct)", required=True, persist_state="session")
    months = (8, 9, 10) if season.startswith("Fire") else tuple(range(1, 13))
    years, lat, lon, cube = C.grid_seasonal(gvar, months)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    ok = np.isfinite(cube).any(axis=0)
    vmax = float(np.nanpercentile(cube, 98))
    fires = {n: C.region_annual(n, "fire_count", months) for n in FIRE_REGIONS}
    fmax = max((float(s.max()) for s in fires.values() if len(s)), default=1.0) or 1.0
    centers = {n: ((grid.REGIONS[n][0] + grid.REGIONS[n][1]) / 2, (grid.REGIONS[n][2] + grid.REGIONS[n][3]) / 2)
               for n in FIRE_REGIONS}
    span = f"{C.MONTHS[months[0] - 1]}-{C.MONTHS[months[-1] - 1]}"

    def bubbles(y):
        counts = [float(fires[n].get(y, 0.0)) for n in FIRE_REGIONS]
        return dict(size=[10 + 52 * np.sqrt(c / fmax) for c in counts]), counts

    st.caption(f"Each frame is one year's {span} average of {C.label(gvar).lower()}. Orange bubbles are total fire "
               f"detections in the two fire regions over the same months. Press play.")
    pix = dict(lon=LON[ok], lat=LAT[ok], mode="markers", hovertemplate="%{lat:.2f}, %{lon:.2f}<br>%{marker.color:.3g} "
               + short_unit + "<extra></extra>")
    mk = dict(symbol="square", size=15, colorscale=C.SEQUENTIAL, cmin=0, cmax=vmax, opacity=0.82, line=dict(width=0),
              colorbar=dict(title=dict(text=f"{short_unit}<br>{span}", font=dict(size=12)), thickness=12, len=0.6, x=0.98))
    b0, c0 = bubbles(int(years[0]))
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(**pix, marker={**mk, "color": cube[0][ok]}))
    bub = dict(lon=[centers[n][1] for n in FIRE_REGIONS], lat=[centers[n][0] for n in FIRE_REGIONS], mode="markers",
               hovertemplate="%{text}: %{customdata:,.0f} fire detections<extra></extra>",
               text=[C.SHORT[n] for n in FIRE_REGIONS])
    fig.add_trace(go.Scattergeo(**bub, customdata=c0, marker=dict(**b0, color="rgba(235,104,52,0.55)",
                                                                   line=dict(color=C.ORANGE, width=2))))
    overlays(fig)
    fig.frames = [go.Frame(name=str(int(y)), traces=[0, 1],
                           data=[go.Scattergeo(marker={**mk, "color": cube[i][ok]}),
                                 go.Scattergeo(customdata=bubbles(int(y))[1],
                                               marker=dict(**bubbles(int(y))[0], color="rgba(235,104,52,0.55)",
                                                           line=dict(color=C.ORANGE, width=2)))])
                  for i, y in enumerate(years)]
    geo_layout(fig, 660)
    anim = dict(frame=dict(duration=650, redraw=True), transition=dict(duration=0), fromcurrent=True)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.04, xanchor="left", yanchor="bottom",
                          showactive=False, bgcolor="#ffffff", bordercolor=C.AXIS, font=dict(color=C.INK),
                          buttons=[dict(label="Play", method="animate", args=[None, anim]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.17, len=0.78, y=0.02, yanchor="bottom", pad=dict(t=0, b=0),
                      currentvalue=dict(prefix="Year ", font=dict(size=15, color=C.INK)),
                      steps=[dict(label=str(int(y)), method="animate",
                                  args=[[str(int(y))], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                            transition=dict(duration=0))]) for y in years])])
    C.show(fig, alt=f"Animated globe of {span} {C.label(gvar).lower()} by year with fire bubbles", key="globe_lapse")

    # the story in one chart: fires upwind vs haze downwind
    sg = C.region_annual("singapore_receptor", "aod", months)
    if len(sg):
        fig = go.Figure()
        for n, colr in zip(FIRE_REGIONS, (C.ORANGE, "#f0a37f")):
            s = fires[n]
            fig.add_trace(go.Bar(x=s.index, y=s.values, name=f"Fires, {C.SHORT[n]}", marker=dict(color=colr, cornerradius=3),
                                 hovertemplate="%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sg.index, y=sg.values, name="Haze over Singapore (AOD)", yaxis="y2", mode="lines+markers",
                                 line=dict(color=C.BLUE, width=2.5), marker=dict(size=6), hovertemplate="%{y:.2f}<extra></extra>"))
        C.style(fig, 330)
        fig.update_layout(barmode="stack", bargap=0.3, title=dict(text=f"Fires upwind and haze downwind, {span} each year",
                                                                  font=dict(size=13, color=C.INK2)),
                          yaxis=dict(title="Fire detections"),
                          yaxis2=dict(title="AOD", overlaying="y", side="right", showgrid=False, rangemode="tozero"))
        C.show(fig, alt="Fire detections in Sumatra and Kalimantan against haze over Singapore, by year")
        total = sum(fires[n].reindex(sg.index).fillna(0) for n in FIRE_REGIONS)
        r = np.corrcoef(total, sg.values)[0, 1]
        idx = core.load_indices()
        drv = idx[idx.index.month.isin(list(months))].groupby(lambda d: d.year).mean()

        def tag(y):
            t = [k for k, ok_ in (("El Nino", drv.loc[y, "nino34"] > 0.5), ("positive IOD", drv.loc[y, "dmi"] > 0.4))
                 if y in drv.index and ok_]
            return f"{y} ({', '.join(t)})" if t else str(y)

        top = total.sort_values(ascending=False).head(4).index
        st.caption(f"Year to year, {span} haze over Singapore correlates r = {r:+.2f} with fires in the two regions. "
                   f"The four biggest fire years: {', '.join(tag(int(y)) for y in sorted(top))}. "
                   f"(El Nino = Nino 3.4 above +0.5, positive IOD = DMI above +0.4, averaged over {span}.)")

# ----------------------------------------------------------------------------
# Trend per pixel
# ----------------------------------------------------------------------------
else:
    mask_mode = c3.segmented_control("Pixels", ["All", "Significant", "After FDR"], key="globe_mask", default="All",
                                     required=True, persist_state="session",
                                     help="FDR = Benjamini-Hochberg false-discovery control across all pixels.")
    tr = C.grid_trends(gvar)
    LAT, LON = np.meshgrid(tr["lat"], tr["lon"], indexing="ij")
    slope = tr["slope"].copy()
    if mask_mode == "Significant":
        slope[~tr["sig"]] = np.nan
    elif mask_mode == "After FDR":
        slope[~tr["sig_fdr"]] = np.nan
    ok = ~np.isnan(slope)
    valid = ~np.isnan(tr["z"])
    up_, dn_ = (tr["sig"] & (tr["slope"] > 0)).sum() / valid.sum(), (tr["sig"] & (tr["slope"] < 0)).sum() / valid.sum()
    st.markdown(f"**Where is it changing?** {up_:.0%} of pixels show a significant increase and {dn_:.0%} a significant "
                f"decrease (p < 0.05). {tr['sig_fdr'].sum() / valid.sum():.0%} survive the false-discovery correction. "
                "**Click a pixel** to test it on its own.")
    lim = float(np.nanpercentile(np.abs(tr["slope"]), 98)) or 1.0
    fig = go.Figure(go.Scattergeo(
        lon=LON[ok], lat=LAT[ok], mode="markers",
        marker=dict(symbol="square", size=15, color=slope[ok], colorscale=C.DIVERGING, cmin=-lim, cmax=lim, opacity=0.85,
                    line=dict(width=0), colorbar=dict(title=dict(text=f"{short_unit}<br>per decade", font=dict(size=12)),
                                                      thickness=12, len=0.6, x=0.98)),
        customdata=tr["p"][ok], hovertemplate="%{lat:.2f}, %{lon:.2f}<br>%{marker.color:+.3g} per decade<br>"
                                              "p %{customdata:.3f}<extra></extra>"))
    overlays(fig)
    geo_layout(fig, 640)
    event = C.show(fig, alt=f"Globe of per-pixel trends in {C.label(gvar).lower()}", key="globe_trend",
                   on_select="rerun", selection_mode="points")
    st.caption("Orange = increasing, blue = decreasing. Per-pixel seasonal Kendall test with seasonal Sen slope.")

    pts = event.selection.points if event and event.selection else []
    da = C.grid_var(gvar)
    if pts:
        p0 = pts[0]
        la, lo = p0.get("lat"), p0.get("lon")
        if la is None and p0.get("point_index") is not None:
            la, lo = LAT[ok][p0["point_index"]], LON[ok][p0["point_index"]]
        s = da.sel(lat=la, lon=lo, method="nearest").to_series()
        s.index = pd.DatetimeIndex(s.index)
        title = f"Pixel at {float(la):.2f}, {float(lo):.2f}"
    else:
        with st.expander("Or test your own box", icon=":material/crop_free:", expanded=True):
            lat_r = st.slider("Latitude range", float(tr["lat"].min()), float(tr["lat"].max()), (-3.5, -1.5), 0.25, key="box_lat")
            lon_r = st.slider("Longitude range", float(tr["lon"].min()), float(tr["lon"].max()), (112.0, 115.0), 0.25, key="box_lon")
        s = grid.region_series(da, (lat_r[0], lat_r[1], lon_r[0], lon_r[1]))
        title = f"Box {lat_r[0]:.2f} to {lat_r[1]:.2f} N, {lon_r[0]:.2f} to {lon_r[1]:.2f} E"
    s = s.asfreq("MS") if len(s) else s
    if s.notna().sum() < 36:
        st.info("No data there for this variable (GLDAS is land-only).", icon=":material/info:")
    else:
        one, _ = core.load_csv(io.StringIO(s.rename(gvar).rename_axis("date").to_csv()), core.load_indices())
        rb = core.analyse_variable(one, gvar, ctx.alpha)
        st.subheader(title, anchor=False)
        b = st.columns(4)
        C.tile(b[0], "Verdict", rb["verdict"].capitalize())
        C.tile(b[1], "Rate of change", f"{rb['sk_slope_per_decade']:+.3g} {rb['unit']} / decade")
        C.tile(b[2], "Seasonal Kendall p", C.fmt_p(rb["sk_p"]))
        C.tile(b[3], "ENSO-adjusted p", C.fmt_p(rb["adj_p"]))
        C.show(C.trend_figure(rb, 300), alt=f"Anomaly and trend for the selected {title.lower()}")
