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


with st.container(border=True):
    c1, c2, c3 = st.columns([1.3, 1.6, 1], gap="large")
    with c1:
        if ctx.bundled:
            st.pills("1. Where?", list(C.BUNDLED), key="q_where", on_change=_set_where, required=True,
                     format_func=lambda s: C.SHORT[C.BUNDLED[s]])
        else:
            st.markdown("**1. Where?**")
            st.caption(f"Your file: {ctx.label}")
    with c2:
        st.pills("2. What worries you?", list(A.CONCERNS), key="concern", required=True, persist_state="session",
                 format_func=lambda k: f"{A.CONCERNS[k]['icon']} {A.CONCERNS[k]['label']}")
    with c3:
        st.slider("3. Which years?", y0, y1, key="q_period", on_change=_set_period)

concern = A.CONCERNS[ss["concern"]]

st.markdown("**Or try a demonstration**")
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


for t in targets[:2]:
    r = C.variable_detail(ctx.key, ctx.raw, t, ctx.alpha, ctx.period)
    with st.container(border=True):
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.markdown(f"#### {C.label(t)}")
            C.verdict_badge(r["verdict"])
            st.markdown(A.plain_conclusion(r, ctx.where, ctx.period_text, ctx.alpha))
            pct = A.pct_per_decade(r)
            m1, m2 = st.columns(2)
            m1.metric("Change per decade", f"{pct:+.1f}%", border=True)
            m2.metric("Chance it is just noise", C.fmt_p(r["sk_p"]), border=True,
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

    ranks = C.driver_ranking(ctx.key, ctx.raw, t, ctx.period)
    if len(ranks):
        with st.expander(f"What usually comes with {C.label(t).lower()}?", icon=":material/insights:", expanded=True):
            for _, row in ranks.head(3).iterrows():
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
# Overview
# ----------------------------------------------------------------------------
st.divider()
st.subheader("Everything in this dataset at a glance", anchor=False)
table, res = ctx.table, ctx.res
sig = table[table["sk_p"] < ctx.alpha]
with st.container(horizontal=True):
    st.metric("Variables tested", len(table), border=True)
    st.metric("Significant trends", len(sig), border=True)
    st.metric("Unusual months flagged", int(res["outliers"]["outlier"].sum()) if len(res["outliers"]) else 0, border=True)
    st.metric("Critical alerts", sum(a["severity"] == "critical" for a in res["alerts"]), border=True)
for w in res["quality"]["warnings"]:
    st.warning(w, icon=":material/warning:")

with st.expander("What do the variables mean?", icon=":material/menu_book:"):
    st.dataframe(
        [{"Column": c, "Name": core.meta(c)[0], "Units": core.meta(c)[1], "Plain meaning": core.meta(c)[2]}
         for c in ctx.cols + core.DRIVERS],
        hide_index=True, width="stretch", alt="Glossary of variables, units and plain-language meanings")
