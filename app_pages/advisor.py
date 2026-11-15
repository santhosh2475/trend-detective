import os

import streamlit as st

from tdui import common as C
from trendlab import advisor as A
from trendlab import core, models

ctx = C.get_ctx()
ss = st.session_state
st.title("AI advisor", anchor=False)
st.caption("Which variables to look at, which model to trust, and which NASA data to add, worked out from your data. "
           "Ask follow-up questions to Claude at the bottom.")

if ss.get("concern") not in A.CONCERNS:
    ss["concern"] = "fires"
st.pills("What are you investigating?", list(A.CONCERNS), key="concern", required=True, persist_state="session",
         format_func=lambda k: f"{A.CONCERNS[k]['icon']} {A.CONCERNS[k]['label']}")
concern = A.CONCERNS[ss["concern"]]
targets = [t for t in concern["targets"] if t in ctx.cols][:2]
lead_pref = 3

# ----------------------------------------------------------------------------
# Recommendations (no API key needed)
# ----------------------------------------------------------------------------
if not targets:
    st.info(f"This dataset has no {concern['label'].lower()} variable. The NASA datasets below would add one.",
            icon=":material/info:")

leaderboards, rankings = {}, {}
for t in targets:
    r = C.variable_detail(ctx.key, ctx.raw, t, ctx.alpha, ctx.period)
    ranks = C.driver_ranking(ctx.key, ctx.raw, t, ctx.period)
    rankings[t] = ranks
    with st.container(border=True):
        st.markdown(f"### :material/target: Look at: {C.label(t)}")
        C.verdict_badge(r["verdict"])
        st.markdown(A.plain_conclusion(r, ctx.where, ctx.period_text, ctx.alpha))

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.markdown("**Variables that travel with it**")
            if len(ranks):
                view = ranks.head(6)[["Variable", "r", "Best lag (months)", "Early-warning lag", "Early-warning r"]]
                st.dataframe(view, hide_index=True, width="stretch", alt=f"Variables most related to {C.label(t)}",
                             column_config={
                                 "r": st.column_config.ProgressColumn("Strength (r)", min_value=-1, max_value=1, format="%+.2f",
                                                                      help="Correlation of the anomalies, -1 to +1."),
                                 "Best lag (months)": st.column_config.NumberColumn("Lag", format="%d mo",
                                                                                     help="How many months it leads by."),
                                 "Early-warning lag": st.column_config.NumberColumn("Warning lag", format="%d mo"),
                                 "Early-warning r": st.column_config.NumberColumn("Warning r", format="%+.2f",
                                                                                  help="Best correlation at a lead of 1+ months."),
                             })
                warn = ranks[ranks["Early-warning r"].abs() >= 0.3].head(2)
                if len(warn):
                    st.caption("Early-warning signals: " + "; ".join(
                        f"{row['Variable'].lower()} {int(row['Early-warning lag'])} month(s) ahead (r = {row['Early-warning r']:+.2f})"
                        for _, row in warn.iterrows()) + ".")
        with right:
            st.markdown("**Model to trust for forecasting it**")
            lead = next((L for L in (lead_pref, 1) if C.has_leaderboard(ctx.name, ctx.raw, t, L)), None)
            if lead is not None:
                lb = C.leaderboard(ctx.name, ctx.key, ctx.raw, t, lead)
                leaderboards[f"{t} ({lead} months ahead)"] = lb["table"]
                rec = models.recommend(lb["table"])
                st.markdown(f":material/emoji_events: **{models.BY_KEY[rec['key']].name}** "
                            f"({lead} month{'s' if lead > 1 else ''} ahead)")
                st.caption(rec["reason"])
            else:
                st.caption(A.model_hint(C.load_frame(ctx.key, ctx.raw), t, ranks))

            def _open_forecast(var=t):
                ss["fvar_ml"] = var

            if st.button("Open the model comparison", icon=":material/model_training:", key=f"adv_fc_{t}",
                         on_click=_open_forecast):
                st.switch_page("app_pages/forecast.py")

links = None
if ctx.bundled and ss["concern"] in ("haze", "fires"):
    links = C.region_links(ctx.period)
    reg = links.get("regression")
    if reg and ss["concern"] == "haze":
        st.info(f"Fires in Riau / South Sumatra and Central Kalimantan explain about **{reg['r2']:.0%}** of Singapore's "
                "month-to-month haze swings. To study haze, look at upwind fires, not just local conditions.",
                icon=":material/compare_arrows:", title="Look upwind too")
    elif reg:
        st.info(f"These fires matter beyond the region: fires in Riau / South Sumatra and Central Kalimantan explain "
                f"about **{reg['r2']:.0%}** of Singapore's month-to-month haze swings.",
                icon=":material/compare_arrows:", title="Downwind impact")

st.markdown(f"#### :material/satellite_alt: NASA data you could add for {concern['label'].lower()}")
cat = [d for d in A.NASA_CATALOG if ss["concern"] in d["concerns"]]
for row_items in (cat[i:i + 3] for i in range(0, len(cat), 3)):
    cols = st.columns(3)
    for col, d in zip(cols, row_items):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{d['name']}**")
            st.caption(f"`{d['product']}` · {d['source']}")
            st.markdown(d["why"])
            st.link_button("Find it on Earthdata", A.earthdata_url(d["product"]), icon=":material/open_in_new:")
st.caption("Add a downloaded variable as a new column in your CSV (one value per month) and upload it in the sidebar.")

# ----------------------------------------------------------------------------
# Claude chat (optional)
# ----------------------------------------------------------------------------
st.divider()
st.subheader(":material/forum: Ask the AI", anchor=False)


def _secret_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


env_key = _secret_key() or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
api_key = env_key
if not env_key:
    with st.container(border=True):
        st.markdown("The chat uses Claude and needs an Anthropic API key. Add `ANTHROPIC_API_KEY` to "
                    "`.streamlit/secrets.toml`, or paste a key here for this session only.")
        api_key = st.text_input("Anthropic API key", type="password", key="anthropic_key", placeholder="sk-ant-...")
st.caption(f"Each question sends a summary of the numbers on these pages (trend results, rankings, leaderboards; "
           f"no raw data) to Anthropic's API and uses your API credits. Model: `{A.MODEL_ID}`.")

if "chat" not in ss:
    ss["chat"] = []
starters = [f"Which variables should I watch to anticipate {concern['label'].lower()} here?",
            "Which forecasting model should I trust, and why?",
            f"What NASA data should I add to study {concern['label'].lower()}?",
            "Explain the most important trend here to a 14-year-old."]


def _queue_starter():
    ss["pending_q"] = ss.get("chat_starter")
    ss["chat_starter"] = None


st.pills("Try asking", starters, key="chat_starter", on_change=_queue_starter, disabled=not api_key)
for msg in ss["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about variables, trends or models", submit_mode="disable", disabled=not api_key,
                       key="chat_box") or ss.pop("pending_q", None)
if prompt and api_key:
    ss["chat"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    for t in ("fire_count", "aod"):
        if t in ctx.cols and t not in rankings:
            rankings[t] = C.driver_ranking(ctx.key, ctx.raw, t, ctx.period)
    for c in ctx.cols:
        for L in (1, 3, 6):
            name = f"{c} ({L} months ahead)"
            if name not in leaderboards and C.has_leaderboard(ctx.name, ctx.raw, c, L):
                leaderboards[name] = C.leaderboard(ctx.name, ctx.key, ctx.raw, c, L)["table"]
    context = A.build_context(ctx.label, ctx.period_text, ctx.table, rankings, leaderboards, links, ctx.cols)
    import anthropic

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(A.stream_answer(ss["chat"], context, api_key))
            ss["chat"].append({"role": "assistant", "content": answer if isinstance(answer, str) else "".join(answer)})
        except anthropic.AuthenticationError:
            ss["chat"].pop()
            st.error("That API key was rejected. Check it and try again.", icon=":material/key_off:")
        except anthropic.RateLimitError:
            ss["chat"].pop()
            st.error("Rate limited by the API. Wait a moment and ask again.", icon=":material/hourglass:")
        except anthropic.APIStatusError as e:
            ss["chat"].pop()
            st.error(f"The API returned an error ({e.status_code}): {e.message}", icon=":material/error:")
        except anthropic.APIConnectionError:
            ss["chat"].pop()
            st.error("Could not reach the Anthropic API. Check your internet connection.", icon=":material/wifi_off:")

if ss["chat"]:
    st.button("Clear the conversation", icon=":material/delete:", on_click=lambda: ss.update(chat=[]), key="chat_clear")
