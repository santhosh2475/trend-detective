import pandas as pd
import streamlit as st

from tdui import common as C
from trendlab import pipeline

ctx = C.get_ctx()
st.title("Data and export", anchor=False)
res, q, table, name = ctx.res, ctx.res["quality"], ctx.table, ctx.name

st.subheader("What happened to your file", anchor=False)
steps = [
    ("Validate", f"Parsed dates, averaged {q['duplicate_months_averaged']} duplicate month(s), dropped {q['unparseable_dates']} bad date(s)."),
    ("Complete the calendar", f"Reindexed to {q['n_months']} months; inserted {len(q['missing_rows_inserted'])} blank row(s)"
     + (f" ({', '.join(q['missing_rows_inserted'][:6])})." if q["missing_rows_inserted"] else ".")),
    ("Remove the seasonal cycle", "Subtracted each calendar month's mean to get anomalies. Counts are log-transformed first."),
    ("Test for trends", "Seasonal Kendall, Mann-Kendall, Sen slope with 95% CI, slope t-test, and a t-test controlling for ENSO / IOD."),
    ("Find unusual months", f"Elliptic Envelope on {len(res['features'])} variables; PCA for the dominant pattern."),
    ("Compare forecast models", "Eight models scored by rolling-origin validation (each year predicted from the years before it)."),
    ("Forecast", f"The recommended model, one fit per lead time, {ctx.horizon} months ahead."),
    ("Alert", f"{len(res['alerts'])} alert(s) raised from the rules."),
]
for i, (title, text) in enumerate(steps, 1):
    st.markdown(f"**{i}. {title}** · {text}")
with st.expander("Missing values per column", icon=":material/grid_off:"):
    st.dataframe(pd.Series(q["missing_per_column"], name="missing months").to_frame(), width="stretch",
                 alt="Missing months per column")

st.subheader("Download", anchor=False)
with st.container(horizontal=True):
    st.download_button("Report (Markdown)", pipeline.report_markdown(res), f"{name}_report.md", icon=":material/description:")
    st.download_button("Trend table (CSV)", table.to_csv(index=False), f"{name}_trend_table.csv", icon=":material/table:")
    if len(res["outliers"]):
        st.download_button("Unusual months (CSV)", res["outliers"].to_csv(), f"{name}_outliers.csv", icon=":material/table:")
    if res["forecasts"]:
        st.download_button("Scenario forecasts (CSV)", pd.concat({c: f["forecast"] for c, f in res["forecasts"].items()},
                                                                 names=["variable", "date"]).to_csv(),
                           f"{name}_forecast.csv", icon=":material/table:")

st.subheader("Run it without the dashboard", anchor=False)
st.code("python -m trendlab.pipeline your_file.csv --out results/your_file\n"
        "python -m trendlab.pipeline your_file.csv --webhook https://hooks.slack.com/...   # notify on new alerts\n"
        "python -m trendlab.models your_file.csv --lead 1 3 --save results            # model leaderboards",
        language="bash")
