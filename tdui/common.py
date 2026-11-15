"""Shared UI pieces: palette, chart helpers, cached computations and the sidebar context."""
from __future__ import annotations

import hashlib
import io
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

# palette (validated categorical order; status colours reserved for alerts)
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATUS = {"critical": "#d03b3b", "warning": "#fab219", "info": "#898781"}
DIVERGING = [[0, "#184f95"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a37f"], [1, "#b8431a"]]
SEQUENTIAL = [[0, "#fbf3e4"], [0.25, "#f6cf8f"], [0.5, "#ef9a4d"], [0.75, "#d0582a"], [1, "#7c2410"]]
FAMILY_COLOR = {"baseline": MUTED, "linear": BLUE, "count": AQUA, "tree": ORANGE, "neural": PURPLE}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------------
# Formatting and chart helpers
# ----------------------------------------------------------------------------
def style(fig, height=380, legend=True, left=70):
    fig.update_layout(height=height, margin=dict(l=left, r=20, t=64, b=50), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=13), hovermode="x unified",
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
                 f"<div style='font-size:1.4rem;font-weight:600;line-height:1.25;color:{INK}'>{value}</div>"
                 f"<div style='font-size:0.8rem;color:{MUTED};margin-bottom:0.8rem'>{sub}</div>", unsafe_allow_html=True)


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
    fig.add_trace(go.Scatter(x=a.index, y=a, mode="lines", line=dict(color=BLUE, width=1.6), name="Anomaly",
                             hovertemplate="%{y:+.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.index, y=line, mode="lines", line=dict(color=ORANGE, width=2.5), name="Trend",
                             hovertemplate="%{y:+.3g}<extra></extra>"))
    style(fig, height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=INK2)))
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
