import streamlit as st

from tdui import common as C
from trendlab import advisor as A
from trendlab import core

ctx = C.get_ctx()
ss = st.session_state

st.title("What is changing, and should we worry?", anchor=False)
st.caption("NASA satellite and model data for Indonesia's peat-fire region and Singapore. "
           "Answer three questions and get a plain-language answer, backed by four statistical tests.")

# ----------------------------------------------------------------------------
# Three questions. They write straight into the sidebar's settings, so every page follows.
# ----------------------------------------------------------------------------
y0, y1 = int(C.load_frame(ctx.key, ctx.raw).index.year.min()), int(C.load_frame(ctx.key, ctx.raw).index.year.max())
ss["q_where"] = ctx.label
ss["q_period"] = ss.get("period", (y0, y1))
if ss.get("concern") not in A.CONCERNS:
    ss["concern"] = "fires"


def _set_where():
    if ss.get("q_where"):
        ss["region"] = ss["q_where"]


def _set_period():
    ss["period"] = ss["q_period"]


def _demo(region: str, concern: str):
    ss.update(source="Bundled region", region=region, concern=concern, period=(y0, y1))


# Design: controls sit straight on the page in stacked rows (no card outline), each chip row at full width
# so every option stays visible.
with st.container(gap="medium"):
    if ctx.bundled:
        st.pills("1. Where?", list(C.BUNDLED), key="q_where", on_change=_set_where, required=True,
                 format_func=lambda s: C.SHORT[C.BUNDLED[s]])
    else:
        st.markdown("**1. Where?**")
        st.caption(f"Your file: {ctx.label}")
    st.pills("2. What worries you?", list(A.CONCERNS), key="concern", required=True, persist_state="session",
             format_func=lambda k: f"{A.CONCERNS[k]['icon']} {A.CONCERNS[k]['label']}")
    st.slider("3. Which years?", y0, y1, key="q_period", on_change=_set_period, width=480)

concern = A.CONCERNS[ss["concern"]]

st.space("small")
st.caption("Or try a demonstration")
demos = [("Is the air getting thirstier?", "Central Kalimantan (fire hotspot)", "heat", ":material/thermostat:"),
         ("Are fires becoming more frequent?", "Riau / South Sumatra (fire hotspot)", "fires", ":material/local_fire_department:"),
         ("Is Singapore's haze getting worse?", "Singapore (where haze arrives)", "haze", ":material/foggy:")]
with st.container(horizontal=True):
    for title, region, cn, icon in demos:
        st.button(title, icon=icon, on_click=_demo, args=(region, cn), key=f"demo_{cn}")

# ----------------------------------------------------------------------------
# Answer
# ----------------------------------------------------------------------------
st.divider()
st.subheader(f"{concern['label']} in {ctx.where}, {ctx.period_text}", anchor=False)
st.caption(concern["why"])

targets = [t for t in concern["targets"] if t in ctx.cols]
if not targets:
    st.info(f"This dataset has no {concern['label'].lower()} variable. Try another concern or region, "
            "or see the AI advisor for NASA datasets you could add.", icon=":material/info:")


def _go(page: str, **state):
    ss.update(state)


for i, t in enumerate(targets[:2]):
    r = C.variable_detail(ctx.key, ctx.raw, t, ctx.alpha, ctx.period)
    if i:
        st.divider()
    # Design headline block: name + status badge on one line, the numbers straight under it,
    # then the explanation. Sections are separated by keylines rather than card outlines.
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"#### {C.label(t)}")
            C.verdict_badge(r["verdict"])
        # design hero reading: the headline number, very large and very thin
        pct = A.pct_per_decade(r)
        C.hero_number(f"{pct:+.1f}%", "change per decade",
                      size="clamp(56px, 8vw, 120px)" if i == 0 else "clamp(44px, 5vw, 76px)")
        st.markdown(A.plain_conclusion(r, ctx.where, ctx.period_text, ctx.alpha))
        st.metric("Chance it is just noise", C.fmt_p(r["sk_p"]),
                  help="Seasonal Kendall p-value, corrected for months that resemble their neighbours.")
        with st.container(horizontal=True):
            if st.button("Explore in detail", icon=":material/show_chart:", key=f"go_tr_{t}",
                         on_click=_go, args=("trends",), kwargs={"var": t}):
                st.switch_page("app_pages/trends.py")
            if st.button("Forecast it", icon=":material/model_training:", key=f"go_fc_{t}",
                         on_click=_go, args=("forecast",), kwargs={"fvar_ml": t}):
                st.switch_page("app_pages/forecast.py")
    with right:
        C.show(C.trend_figure(r, 260, f"{C.label(t)}: seasonal cycle removed, with trend line"),
               alt=f"{C.label(t)} anomaly over time with its trend line", key=f"home_fig_{t}")

    # design breakdown: one dial per companion variable (needle = correlation), then the plain sentences
    ranks = C.driver_ranking(ctx.key, ctx.raw, t, ctx.period)
    if len(ranks):
        top = ranks.head(3)
        st.markdown(f"##### What usually comes with {C.label(t).lower()}?")
        for col, (_, row) in zip(st.columns(len(top), gap="large"), top.iterrows()):
            with col:  # name, then badge, on their own lines so the three dials stay level
                st.markdown(row["Variable"])
                st.badge(str(row["Direction"]).capitalize(), color="gray",
                         icon=":material/north_east:" if row["r"] > 0 else ":material/south_east:")
                C.dial(float(row["r"]), row["Variable"])
        for _, row in top.iterrows():
            st.markdown(f"- {A.driver_sentence(row, t)}")
        st.caption("Correlations of the anomalies (seasonal cycle removed). They show what moves together, "
                   "not what causes what. The AI advisor ranks every variable and suggests a forecast model.")

if ctx.name == "singapore_receptor" and ss["concern"] == "haze":
    links = C.region_links(ctx.period)
    reg = links.get("regression")
    if reg:
        st.info(f"Fires upwind explain about **{reg['r2']:.0%}** of Singapore's month-to-month haze swings "
                f"(Riau and Central Kalimantan together). See Compare and link regions.",
                icon=":material/compare_arrows:", title="Where the haze comes from")

# ----------------------------------------------------------------------------
# The year ahead and the usual year (design: 7-day forecast grid, pollen calendar, year at a glance)
# ----------------------------------------------------------------------------
fcs = ctx.res["forecasts"]
if fcs:
    st.divider()
    span = next(iter(fcs.values()))["forecast"].index
    st.subheader("The next 12 months", anchor=False)
    st.caption(f"Scenario forecasts for {span[0]:%b %Y} to {span[-1]:%b %Y}, each compared with that month's normal. "
               "Filled circles are above normal, open circles below, dotted rings within 10%; bigger means further "
               "from normal. Models and forecast has the full model comparison.")
    C.show(C.outlook_figure(fcs), alt="Grid of the next 12 months comparing each forecast variable with its normal",
           key="home_outlook")

st.divider()
frame = C.sliced(ctx.key, ctx.raw, ctx.period)
main = targets[0] if targets else ctx.cols[0]
cal, ring = st.columns([1.6, 1], gap="large")
with cal:
    st.subheader("Seasonal calendar", anchor=False)
    st.caption("Each variable's average year, brighter where it runs high for itself. "
               "Rows run from the earliest peak to the latest.")
    C.show(C.season_calendar_figure(frame, ctx.cols), alt="Seasonal calendar: each variable's average level by month",
           key="home_calendar")
with ring:
    st.subheader("Year at a glance", anchor=False)
    st.caption(f"{C.label(main)}, average for each month.")
    C.show(C.month_ring_figure(frame[main].dropna(), C.label(main), core.meta(main)[1]),
           alt=f"Ring of twelve months shaded by average {C.label(main).lower()}", key="home_ring")

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
st.divider()
st.subheader("Everything in this dataset at a glance", anchor=False)
table, res = ctx.table, ctx.res
sig = table[table["sk_p"] < ctx.alpha]
# Design breakdown row: bare label + thin number columns, no boxes.
with st.container(horizontal=True, gap="large"):
    st.metric("Variables tested", len(table))
    st.metric("Significant trends", len(sig))
    st.metric("Unusual months flagged", int(res["outliers"]["outlier"].sum()) if len(res["outliers"]) else 0)
    st.metric("Critical alerts", sum(a["severity"] == "critical" for a in res["alerts"]))
for w in res["quality"]["warnings"]:
    st.warning(w, icon=":material/warning:")

with st.expander("What do the variables mean?", icon=":material/menu_book:"):
    st.dataframe(
        [{"Column": c, "Name": core.meta(c)[0], "Units": core.meta(c)[1], "Plain meaning": core.meta(c)[2]}
         for c in ctx.cols + core.DRIVERS],
        hide_index=True, width="stretch", alt="Glossary of variables, units and plain-language meanings")
