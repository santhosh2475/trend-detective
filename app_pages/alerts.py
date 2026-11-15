import json

import streamlit as st

from tdui import common as C
from trendlab import alerts as al
from trendlab import core

ctx = C.get_ctx()
st.title("Alerts", anchor=False)
st.caption("Rules turn the results into alerts. Adjust them on the right, then send or log what is raised.")
res, df, name = ctx.res, ctx.df, ctx.name
ICON = {"critical": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}
COLOR = {"critical": "red", "warning": "orange", "info": "gray"}

left, right = st.columns([3, 2], gap="large")
with right:
    with st.container(border=True):
        st.markdown("**Rules**")
        zt = st.slider("Latest month is extreme beyond (sigma)", 1.5, 4.0, 2.0, 0.5, key="al_z", persist_state="session")
        fq = st.slider("Forecast exceeds historical percentile", 0.75, 0.99, 0.90, 0.01, key="al_q", persist_state="session")
        nw = st.slider("El Nino watch when Nino 3.4 reaches", 0.5, 2.0, 1.0, 0.1, key="al_n", persist_state="session")
        kinds = st.pills("Alert types", ["trend", "extreme", "outlier", "forecast", "enso"], selection_mode="multi",
                         default=["trend", "extreme", "outlier", "forecast", "enso"], key="al_kinds",
                         persist_state="session")
alerts = [a for a in al.build_alerts(df, ctx.table, res["outliers"], res["forecasts"], core.load_indices(),
                                     {"alpha": ctx.alpha, "z_threshold": zt, "forecast_quantile": fq, "nino_watch": nw}, name)
          if a["kind"] in (kinds or [])]
with left:
    st.subheader(f"{len(alerts)} alert(s) for {ctx.where}", anchor=False)
    for a in alerts:
        with st.container(border=True):
            st.badge(a["severity"].capitalize(), icon=ICON[a["severity"]], color=COLOR[a["severity"]])
            st.markdown(f"**{a['title']}**")
            st.caption(a["detail"])
    if not alerts:
        st.success("Nothing to flag with the current rules.", icon=":material/check_circle:")
with right:
    with st.container(border=True):
        st.markdown("**Send**")
        log_path = C.RESULTS / name / "alerts_log.jsonl"
        hook = st.text_input("Slack / Discord / Teams webhook URL", type="password", key="al_hook")
        if st.button("Send to webhook", icon=":material/send:", disabled=not (hook and alerts), key="al_send"):
            try:
                st.success(al.send_webhook(alerts, hook, name))
            except Exception as e:
                st.error(f"Webhook failed: {e}")
        mail = st.text_input("Email address (needs SMTP_HOST / SMTP_USER / SMTP_PASSWORD set)", key="al_mail")
        if st.button("Send email", icon=":material/mail:", disabled=not (mail and alerts), key="al_email"):
            try:
                st.success(al.send_email(alerts, mail, name))
            except Exception as e:
                st.error(f"Email failed: {e}")
        with st.container(horizontal=True):
            if st.button("Write to alert log", icon=":material/edit_note:", key="al_log"):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                new = al.log_alerts(alerts, log_path)
                st.success(f"{len(new)} new alert(s) logged, {len(alerts) - len(new)} already seen.")
            st.download_button("Download (JSON)", json.dumps(alerts, indent=2), f"{name}_alerts.json",
                               icon=":material/download:", key="al_dl")
