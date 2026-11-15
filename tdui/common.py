"""Shared UI pieces: palette, chart helpers, cached computations and the sidebar context."""
from __future__ import annotations

import hashlib
import html
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trendlab import advisor, core, grid, ml, models, pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BUNDLED = {"Central Kalimantan (fire hotspot)": "central_kalimantan",
           "Riau / South Sumatra (fire hotspot)": "riau_south_sumatra",
           "Singapore (where haze arrives)": "singapore_receptor"}
SHORT = {"central_kalimantan": "Central Kalimantan", "riau_south_sumatra": "Riau / South Sumatra",
         "singapore_receptor": "Singapore"}

# The app is locked to the Strata design's dark mode (.streamlit/config.toml), so every chart colour here
# is the dark-surface step. Palette: validated categorical order, dark steps; status colours reserved for alerts.
BLUE, ORANGE, AQUA, PURPLE = "#3987e5", "#d95926", "#199e70", "#9085e9"
# neutrals follow the Strata dark tokens (ink, ink-soft, chip grey, 6% grid, 15% hairline on #0a0a0a)
INK, INK2, MUTED, GRID, AXIS = "#f5f5f5", "#a3a3a3", "#787878", "#191919", "#2f2f2f"
MUTED_TEXT = "#8f8f8f"  # muted text that still passes 4.5:1 on the page background
CONTEXT = "#858585"     # a series shown only as context under the accent (dimmer than ink-soft so it recedes)
SURFACE, PANEL = "#0a0a0a", "#111111"  # page background (for mark rings) and raised panels (hover labels, buttons)
STATUS = {"critical": "#e5554a", "warning": "#e89d4a", "info": MUTED}
# fonts are loaded by .streamlit/config.toml; charts reuse them so axis text matches the page
FONT_UI, FONT_HEAD, FONT_MONO = "Jura, Geist, sans-serif", "Geist, sans-serif", "'Geist Mono', monospace"
# on a dark surface the "nothing" end of each scale is the dark one, so it recedes into the background
DIVERGING = [[0, "#9ec5f4"], [0.25, "#3987e5"], [0.5, "#383835"], [0.75, "#d95926"], [1, "#f6b48f"]]
SEQUENTIAL = [[0, "#2a1a10"], [0.25, "#6b2c12"], [0.5, "#b8481c"], [0.75, "#ec7d45"], [1, "#fbd3a6"]]
FAMILY_COLOR = {"baseline": MUTED, "linear": BLUE, "count": AQUA, "tree": ORANGE, "neural": PURPLE}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------------
# Formatting and chart helpers
# ----------------------------------------------------------------------------
def style(fig, height=380, legend=True, left=70):
    fig.update_layout(height=height, margin=dict(l=left, r=20, t=64, b=50), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT_UI, color=INK2, size=12),
                      title_font=dict(family=FONT_HEAD, color=INK, size=15), hovermode="x unified",
                      hoverlabel=dict(bgcolor=PANEL, bordercolor=AXIS, font=dict(family=FONT_MONO, size=11, color=INK)),
                      showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right",
                                                     font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig, alt: str, key: str | None = None, **kw):
    return st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, alt=alt, key=key, **kw)


def tile(col, label, value, sub=""):
    """Stat tile that wraps instead of truncating (st.metric cuts long text)."""
    col.markdown(f"<div style='font-size:0.85rem;color:{INK2}'>{label}</div>"
                 f"<div style='font-family:{FONT_HEAD};font-size:1.6rem;font-weight:300;line-height:1.25;color:{INK}'>{value}</div>"
                 f"<div style='font-size:0.8rem;color:{MUTED_TEXT};margin-bottom:0.8rem'>{sub}</div>", unsafe_allow_html=True)


def num(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def fmt_p(p):
    return "n/a" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def label(c: str) -> str:
    return core.meta(c)[0]


def verdict_badge(verdict: str):
    if verdict.startswith("significant increasing"):
        st.badge("Significant increase", icon=":material/trending_up:", color="orange")
    elif verdict.startswith("significant decreasing"):
        st.badge("Significant decrease", icon=":material/trending_down:", color="blue")
    elif verdict.startswith("weak"):
        st.badge("Weak signal, could be chance", icon=":material/help:", color="yellow")
    else:
        st.badge("No clear trend", icon=":material/trending_flat:", color="gray")


def trend_figure(r: dict, height=300, title: str | None = None):
    """Anomaly with its Sen trend line."""
    a = r["_anom"]
    line = r["_sen_intercept"] + r["sen_slope_per_decade"] / 10 * core._years(a.index)
    fig = go.Figure()
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    # design: data in a receding grey as context, the finding (trend) in the single warm accent
    fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", line=dict(color=CONTEXT, width=1.25), name="Anomaly",
                             hovertemplate="%{y:+.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.index, y=line, mode="lines", line=dict(color=ORANGE, width=2.25), name="Trend",
                             hovertemplate="%{y:+.3g}<extra></extra>"))
    style(fig, height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family=FONT_HEAD, size=14, color=INK2)))
    return fig


# ----------------------------------------------------------------------------
# Strata design pieces: headline number, dial, seasonal calendar, month ring, 12-month outlook
# ----------------------------------------------------------------------------
def _minus(s: str) -> str:
    return s.replace("-", "−")  # typographic minus, as in the design's numerals


def _grey(t: float) -> str:
    """0 -> a step above the page, 1 -> ink: the design's monochrome ramp (low values recede into the dark)."""
    t = 0.0 if not np.isfinite(t) else min(1.0, max(0.0, float(t)))
    v = round(0x22 + t * (0xe8 - 0x22))
    return f"#{v:02x}{v:02x}{v:02x}"


def hero_number(value: str, caption: str, size: str = "clamp(56px, 8vw, 120px)"):
    """The design's headline reading: one very large hairline number over a small spaced caption."""
    st.html(f"<div style='font-family:{FONT_HEAD};font-weight:100;font-size:{size};line-height:1;"
            f"letter-spacing:-0.02em;color:{INK};margin:0.4rem 0 0.5rem'>{html.escape(_minus(value))}</div>"
            f"<div style='font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;color:{MUTED_TEXT}'>"
            f"{html.escape(caption)}</div>")


def dial(r: float, name: str):
    """Design dial: a ring of ticks and a straight needle. The needle sweeps from -1 (lower left) through
    0 (top) to +1 (lower right), so it reads as a correlation at a glance; the number below is exact."""
    size, c = 116, 58
    parts = []
    for i in range(48):
        a, major = math.radians(i * 7.5), i % 6 == 0
        r1, r2 = 54, (46 if major else 50)
        parts.append(f"<line x1='{c + r1 * math.sin(a):.1f}' y1='{c - r1 * math.cos(a):.1f}' "
                     f"x2='{c + r2 * math.sin(a):.1f}' y2='{c - r2 * math.cos(a):.1f}' stroke='{INK2 if major else MUTED}' "
                     f"stroke-width='{1.2 if major else 0.8}' stroke-linecap='round'/>")
    for sign, txt in ((-1, "−"), (1, "+")):
        a = math.radians(sign * 135)
        parts.append(f"<text x='{c + 34 * math.sin(a):.1f}' y='{c - 34 * math.cos(a) + 4:.1f}' text-anchor='middle' "
                     f"font-size='12' fill='{MUTED_TEXT}' font-family='sans-serif'>{txt}</text>")
    a = math.radians(max(-1.0, min(1.0, float(r))) * 135)
    parts.append(f"<line x1='{c}' y1='{c}' x2='{c + 38 * math.sin(a):.1f}' y2='{c - 38 * math.cos(a):.1f}' "
                 f"stroke='{INK}' stroke-width='1.6' stroke-linecap='round'/>")
    parts.append(f"<circle cx='{c}' cy='{c}' r='2.4' fill='{INK}'/>")
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
           f"{''.join(parts)}</svg>")
    with st.container(horizontal_alignment="center", gap="small"):
        st.image(svg, width=size, alt=f"Dial for {name}: correlation {r:+.2f}")
        st.html(f"<div style='text-align:center;font-family:{FONT_HEAD};font-weight:300;font-size:32px;line-height:1;"
                f"color:{INK}'>{_minus(f'{r:+.2f}')}</div>"
                f"<div style='text-align:center;font-size:0.75rem;color:{MUTED_TEXT};margin-top:6px'>correlation</div>")


def season_calendar_figure(df: pd.DataFrame, cols: list[str]):
    """Design 'pollen calendar': one thin bar per variable across the year, brighter where that variable is
    high for its own range. Variables are ordered by their peak month, so the seasons cascade down."""
    clim = df[cols].groupby(df.index.month).mean().reindex(range(1, 13))
    span = (clim.max() - clim.min()).replace(0, np.nan)
    norm = ((clim - clim.min()) / span).fillna(0.5)
    order = sorted(cols, key=lambda c: (int(np.nanargmax(clim[c].values)) if clim[c].notna().any() else 0))
    xs = np.linspace(0.5, 12.5, 145)
    ext_x = np.arange(0, 14)  # Dec, Jan..Dec, Jan: wraps the year so the bar fades smoothly across New Year
    z, text = [], []
    for c in order:
        ext = np.r_[norm[c].iloc[-1], norm[c].values, norm[c].iloc[0]]
        z.append(np.interp(xs, ext_x, ext))
        months = np.clip(np.rint(xs).astype(int), 1, 12)
        text.append([f"{MONTHS[m - 1]} average {num(clim[c].iloc[m - 1])} {core.meta(c)[1]}" for m in months])
    row_h, bar_h = 30, 5
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=[label(c) for c in order], zmin=0, zmax=1, showscale=False,
                               colorscale=[[0, _grey(0)], [1, _grey(1)]], ygap=row_h - bar_h, text=text,
                               hovertemplate="%{y}<br>%{text}<extra></extra>"))
    style(fig, height=56 + row_h * len(order), legend=False, left=10)
    fig.update_layout(margin=dict(l=10, r=215, t=40, b=10), hovermode="closest")  # r: room for names on the right
    fig.update_xaxes(range=[0.5, 12.5], tickvals=[1, 3, 5, 7, 9, 11], ticktext=MONTHS[0::2], side="top",
                     showgrid=False, showline=False, ticks="")
    fig.update_yaxes(side="right", autorange="reversed", showgrid=False, showline=False, ticks="", ticklabelstandoff=12)
    return fig


def month_ring_figure(s: pd.Series, name: str, unit: str, height=330):
    """Design 'year at a glance' ring: twelve month segments shaded by the variable's average for that month,
    with the peak month in the centre."""
    clim = s.groupby(s.index.month).mean().reindex(range(1, 13))
    span = clim.max() - clim.min()
    norm = ((clim - clim.min()) / span) if span else clim * 0 + 0.5
    peak = int(clim.idxmax())
    theta = [i * 30 for i in range(12)]
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=[1] * 12, theta=theta, width=[30] * 12,
                              marker=dict(color=[_grey(v) for v in norm], line=dict(color=SURFACE, width=3)),
                              customdata=[[MONTHS[i], num(v)] for i, v in enumerate(clim)],
                              hovertemplate="%{customdata[0]}: %{customdata[1]} " + unit + "<extra></extra>"))
    fig.add_trace(go.Scatterpolar(r=[0.5] * 12, theta=theta, mode="text", text=[m[0] for m in MONTHS], hoverinfo="skip",
                                  textfont=dict(family=FONT_UI, size=13,
                                                color=[SURFACE if v > 0.55 else INK2 for v in norm])))
    style(fig, height=height, legend=False, left=0)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      polar=dict(hole=0.6, bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1]),
                                 angularaxis=dict(visible=False, rotation=90, direction="clockwise")),
                      annotations=[dict(text=MONTHS[peak - 1], x=0.5, y=0.54, xref="paper", yref="paper", showarrow=False,
                                        font=dict(family=FONT_HEAD, size=44, color=INK)),
                                   dict(text="peak month", x=0.5, y=0.38, xref="paper", yref="paper", showarrow=False,
                                        font=dict(family=FONT_UI, size=12, color=MUTED_TEXT))])
    return fig


def outlook_figure(forecasts: dict):
    """Design '7-day forecast' grid, stretched to the 12-month outlook: one row per forecast variable, one
    column per month. Filled = above its normal for that month, open = below, dotted ring = within 10%.
    Size grows with the gap from normal."""
    first = next(iter(forecasts.values()))["forecast"]
    xs = [f"{d:%b}" for d in first.index]
    groups = {"Above normal": ("circle", []), "Near normal": ("circle-open-dot", []), "Below normal": ("circle-open", [])}
    rows = []
    for c, f in forecasts.items():
        fc = f["forecast"]
        rows.append(label(c))
        unit = core.meta(c)[1]
        for d, x in zip(fc.index, xs):
            val, norm_ = fc.at[d, "forecast"], fc.at[d, "climatology"]
            gap = (val / norm_ - 1) if norm_ else 0.0
            key = "Above normal" if gap > 0.1 else ("Below normal" if gap < -0.1 else "Near normal")
            groups[key][1].append((x, label(c), 9 + 13 * min(abs(gap), 1.0),
                                   f"{label(c)}, {d:%b %Y}<br>forecast {num(val)} {unit}<br>normal {num(norm_)} {unit} "
                                   f"({gap:+.0%})"))
    fig = go.Figure()
    for name, (symbol, pts) in groups.items():
        if pts:
            x, y, size, hover = zip(*pts)
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, text=hover, hovertemplate="%{text}<extra></extra>",
                                     marker=dict(symbol=symbol, size=size, color=INK if symbol == "circle" else INK2,
                                                 line=dict(width=1.2, color=INK if symbol == "circle" else INK2))))
    style(fig, height=100 + 48 * len(rows), legend=True, left=10)
    fig.update_layout(margin=dict(l=215, r=10, t=40, b=50), hovermode="closest",  # l: room for variable names
                      legend=dict(orientation="h", y=-0.04, yanchor="top", x=0, xanchor="left"))
    fig.update_xaxes(categoryorder="array", categoryarray=xs, side="top", showgrid=False, showline=False, ticks="")
    fig.update_yaxes(categoryorder="array", categoryarray=rows[::-1], showgrid=False, showline=False, ticks="")
    return fig


# ----------------------------------------------------------------------------
# Cached work
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running the pipeline...", max_entries=32)
def run_pipeline(key: str, raw: bytes, name: str, alpha: float, contamination: float, horizon: int,
                 period: tuple[int, int] | None):
    return pipeline.run(io.BytesIO(raw), name, alpha, contamination, horizon, period=period)


@st.cache_data(show_spinner=False, max_entries=16)
def load_frame(key: str, raw: bytes) -> pd.DataFrame:
    return core.load_csv(io.BytesIO(raw), core.load_indices())[0]


def sliced(key: str, raw: bytes, period: tuple[int, int] | None) -> pd.DataFrame:
    df = load_frame(key, raw)
    return df.loc[f"{period[0]}-01-01":f"{period[1]}-12-01"] if period else df


@st.cache_data(show_spinner=False, max_entries=256)
def variable_detail(key: str, raw: bytes, col: str, alpha: float, period: tuple[int, int] | None = None):
    return core.analyse_variable(sliced(key, raw, period), col, alpha)


@st.cache_data(show_spinner=False, max_entries=64)
def driver_ranking(key: str, raw: bytes, target: str, period: tuple[int, int] | None):
    df = sliced(key, raw, period)
    return advisor.rank_drivers(df, target, core.analysis_columns(df))


def bundled_raw(name: str) -> tuple[str, bytes]:
    b = (DATA / f"{name}.csv").read_bytes()
    return hashlib.sha1(b).hexdigest(), b


@st.cache_data(show_spinner=False)
def region_links(period: tuple[int, int] | None):
    frames = {n: sliced(*bundled_raw(n), period) for n in BUNDLED.values()}
    return advisor.linked_regions(frames["singapore_receptor"],
                                  {SHORT[n]: frames[n] for n in ("riau_south_sumatra", "central_kalimantan")})


def leaderboard(name: str, key: str, raw: bytes, target: str, lead: int):
    """Model comparison, read from the disk cache when it exists, else computed with a progress bar."""
    cdir = models.cache_dir_for(RESULTS, name, raw)
    if (cdir / f"lead{lead}_{target}.csv").exists():
        return _leaderboard_cached(name, key, raw, target, lead)
    bar = st.progress(0.0, text="Testing every model on each year it never saw...")
    lb = models.cached_leaderboard(load_frame(key, raw), target, lead, cdir,
                                   progress=lambda f, y: bar.progress(f, text=f"Predicting {y} from the years before it..."))
    bar.empty()
    return lb


@st.cache_data(show_spinner=False, max_entries=64)
def _leaderboard_cached(name, key, raw, target, lead):
    return models.cached_leaderboard(load_frame(key, raw), target, lead, models.cache_dir_for(RESULTS, name, raw))


def has_leaderboard(name: str, raw: bytes, target: str, lead: int) -> bool:
    return (models.cache_dir_for(RESULTS, name, raw) / f"lead{lead}_{target}.csv").exists()


@st.cache_data(show_spinner=False, max_entries=32)
def ml_forecast(key: str, raw: bytes, target: str, model_key: str, horizon: int):
    return models.forecast(load_frame(key, raw), target, model_key, horizon)


@st.cache_data(show_spinner="Measuring which inputs the model relies on...", max_entries=32)
def importance(key: str, raw: bytes, target: str, lead: int):
    return models.feature_importance(load_frame(key, raw), target, lead)


@st.cache_resource(show_spinner=False)
def grid_index():
    return grid.find_grids(DATA)


@st.cache_resource(show_spinner="Loading grid...")
def grid_var(var: str):
    return grid.load_var(grid_index()[var], var)


@st.cache_data(show_spinner="Testing every pixel (seasonal Kendall)...")
def grid_trends(var: str):
    return grid.pixel_trends(grid_var(var))


@st.cache_data(show_spinner="Building the time-lapse...")
def grid_seasonal(var: str, months: tuple[int, ...]):
    """Per-pixel mean over the chosen months, for every year (year, lat, lon)."""
    da = grid_var(var)
    sel = da.sel(time=da["time.month"].isin(list(months)))
    yearly = sel.groupby("time.year").mean("time")
    return yearly["year"].values, yearly["lat"].values, yearly["lon"].values, yearly.values


@st.cache_data(show_spinner=False)
def region_annual(name: str, col: str, months: tuple[int, ...]):
    _, raw = bundled_raw(name)
    df = load_frame(*bundled_raw(name))
    if col not in df:
        return pd.Series(dtype=float)
    s = df[col][df.index.month.isin(list(months))]
    return s.groupby(s.index.year).sum() if core.meta(col)[3] else s.groupby(s.index.year).mean()


# ----------------------------------------------------------------------------
# Sidebar and shared context
# ----------------------------------------------------------------------------
@dataclass
class Ctx:
    key: str
    raw: bytes
    name: str
    label: str
    bundled: bool
    period: tuple[int, int] | None
    alpha: float
    contamination: float
    horizon: int
    res: dict = field(repr=False)

    @property
    def df(self) -> pd.DataFrame:
        return self.res["data"]

    @property
    def table(self) -> pd.DataFrame:
        return self.res["table"]

    @property
    def cols(self) -> list[str]:
        return self.res["columns"]

    @property
    def where(self) -> str:
        return SHORT.get(self.name, self.label)

    @property
    def period_text(self) -> str:
        q = self.res["quality"]
        return f"{q['start'][:4]}-{q['end'][:4]}"


def sidebar() -> Ctx | None:
    """Data source and settings, rendered on every page. Stores the context in session state."""
    sb = st.sidebar
    sb.caption("Find what is changing, where, by how much, and whether it is significant.")
    ss = st.session_state
    if ss.get("source") not in ("Bundled region", "Upload a CSV"):
        ss["source"] = "Bundled region"
    source = sb.segmented_control("Data", ["Bundled region", "Upload a CSV"], key="source",
                                  required=True, persist_state="session")
    if source == "Upload a CSV":
        # file uploaders cannot persist across pages, so keep the bytes in session state
        up = sb.file_uploader("Monthly CSV (a date column + numeric columns)", type="csv", key="upload")
        if up is not None:
            ss["upload_file"] = (up.getvalue(), up.name)
        if "upload_file" not in ss:
            ss["ctx"] = None
            return None
        raw, fname = ss["upload_file"]
        if up is None:
            sb.caption(f"Using {fname}")
            sb.button("Remove file", icon=":material/close:", key="upload_clear",
                      on_click=lambda: ss.pop("upload_file", None))
        name, lab, bundled = Path(fname).stem, fname, False
    else:
        lab = sb.selectbox("Region", list(BUNDLED), key="region", persist_state="session")
        name = BUNDLED[lab]
        raw, bundled = (DATA / f"{name}.csv").read_bytes(), True
    key = hashlib.sha1(raw).hexdigest()
    try:
        full = load_frame(key, raw)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        st.session_state["ctx"] = None
        return None
    y0, y1 = int(full.index.year.min()), int(full.index.year.max())
    cur = st.session_state.get("period")
    if not cur or cur[0] < y0 or cur[1] > y1 or cur[0] >= cur[1]:
        st.session_state["period"] = (y0, y1)
    period = sb.slider("Study period", y0, y1, key="period", help="Trend tests use only these years.",
                       persist_state="session")
    with sb.expander("Settings", icon=":material/tune:"):
        alpha = st.select_slider("Significance level", [0.01, 0.05, 0.10], 0.05, key="alpha", persist_state="session")
        contamination = st.slider("Expected share of unusual months", 0.01, 0.15, 0.05, 0.01, key="contamination",
                                  persist_state="session")
        horizon = st.slider("Forecast horizon (months)", 3, 12, 12, key="horizon", persist_state="session")
    per = None if tuple(period) == (y0, y1) else tuple(period)
    try:
        res = run_pipeline(key, raw, name, alpha, contamination, horizon, per)
    except Exception as e:
        st.error(f"Could not process this file: {e}")
        st.session_state["ctx"] = None
        return None
    if not res["columns"]:
        st.error("No numeric column with at least 36 months of data was found.")
        st.session_state["ctx"] = None
        return None
    ctx = Ctx(key, raw, name, lab, bundled, per, alpha, contamination, horizon, res)
    st.session_state["ctx"] = ctx
    return ctx


def get_ctx() -> Ctx:
    ctx = st.session_state.get("ctx")
    if ctx is None:
        st.info("Upload a CSV with one row per month in the sidebar. The first column (or a column named `date`) "
                "must be the date; every other numeric column is analysed. Nino 3.4 and DMI are added automatically.",
                icon=":material/upload_file:")
        st.stop()
    return ctx


def pick_variable(ctx: Ctx, key: str, label_text: str = "Variable", default: str | None = None,
                  options: list[str] | None = None, **kw) -> str:
    """Selectbox over the analysable columns that remembers its choice across pages."""
    opts = options or ctx.cols
    if st.session_state.get(key) not in opts:
        st.session_state[key] = default if default in opts else opts[0]
    return st.selectbox(label_text, opts, key=key, format_func=lambda c: f"{label(c)}  ({c})",
                        persist_state="session", **kw)
