"""Earth Systems Trend Detective: Streamlit dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from tdui import common as C

st.set_page_config(page_title="Trend Detective", page_icon=":material/travel_explore:", layout="wide")

PAGES = {
    "": [
        st.Page("app_pages/home.py", title="Start here", icon=":material/explore:", default=True,
                help="Answer three questions (where, what worries you, which years) and get plain-language answers."),
    ],
    "Analyse": [
        st.Page("app_pages/trends.py", title="Explore trends", icon=":material/show_chart:",
                help="Rate of change and significance for any variable, with four tests and a month-by-month view."),
        st.Page("app_pages/globe.py", title="Globe", icon=":material/public:",
                help="Interactive 3D globe of per-pixel trends and a year-by-year haze time-lapse. Click a pixel to test it."),
        st.Page("app_pages/regions.py", title="Compare and link regions", icon=":material/compare_arrows:",
                help="The same variable across regions, and how fires upwind drive haze in Singapore."),
        st.Page("app_pages/patterns.py", title="Patterns", icon=":material/scatter_plot:",
                help="Which variables move together (PCA) and which months were unusual (Elliptic Envelope)."),
    ],
    "Predict": [
        st.Page("app_pages/advisor.py", title="AI advisor", icon=":material/psychology:",
                help="Recommended variables, early-warning signals and NASA datasets to add, plus an optional Claude chat."),
        st.Page("app_pages/forecast.py", title="Models and forecast", icon=":material/model_training:",
                help="Eight models scored on years they never saw; the winner makes the forecast."),
    ],
    "Act": [
        st.Page("app_pages/alerts.py", title="Alerts", icon=":material/notifications:",
                help="Rule-based alerts for trends, extremes, unusual months and forecasts, sent to Slack or email."),
        st.Page("app_pages/export.py", title="Data and export", icon=":material/download:",
                help="What the pipeline did to your file, and downloads of every result."),
    ],
}

page = st.navigation(PAGES, position="sidebar")
st.sidebar.subheader(":material/travel_explore: Trend Detective", anchor=False)
ctx = C.sidebar()

# toast each critical alert once per dataset
if ctx is not None:
    if "toasted" not in st.session_state:
        st.session_state["toasted"] = set()
    for a in ctx.res["alerts"]:
        if a["severity"] == "critical" and (ctx.key, a["id"]) not in st.session_state["toasted"]:
            st.toast(a["title"], icon=":material/error:")
            st.session_state["toasted"].add((ctx.key, a["id"]))

page.run()
