"""
pages/1_📊_Analytics.py
────────────────────────────────────────────────────────────────
Analytics dashboard — live telemetry from the GitHub Docs Assistant.
Uses altair + pandas (both available in standard Streamlit installs).
────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import altair as alt
import pandas as pd

from configs.settings import load_settings
from agent.session_state import SessionState

st.set_page_config(
    page_title="Analytics — GitHub Docs Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Minimal GitHub-style CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 15px; color: #1F2328; background: #FFFFFF;
}
h1  { font-size: 22px !important; font-weight: 700 !important; }
h3  { font-size: 14px !important; font-weight: 600 !important;
      color: #656D76 !important; text-transform: uppercase !important;
      letter-spacing: 0.6px !important; margin-bottom: 8px !important; }
[data-testid="metric-container"] {
    background: #F6F8FA; border: 1px solid #D0D7DE;
    border-radius: 8px; padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
hr { border: none !important; border-top: 1px solid #D0D7DE !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data (fresh read every page load, no caching of data) ───────────────
@st.cache_resource
def _get_db() -> SessionState:
    """Cache only the DB handle, not the query results."""
    return SessionState(load_settings().sqlite_db_path)


db   = _get_db()
data = db.get_analytics_data()

INTENT_COLORS = {
    "rag_query":          "#0969DA",
    "create_ticket":      "#1A7F37",
    "check_ticket":       "#6E40C9",
    "check_billing":      "#9A6700",
    "out_of_scope":       "#CF222E",
    "prompt_injection":   "#CF222E",
    "close_tickets":      "#8B949E",
    "close_ticket_by_id": "#8B949E",
    "action_in_progress": "#1A7F37",
}

# ── Header ────────────────────────────────────────────────────────────────────
_hc1, _hc2 = st.columns([8, 2])
with _hc1:
    st.markdown("# 📊 Analytics Dashboard")
    st.caption("Live query telemetry · data updates as users interact with the assistant")
with _hc2:
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total       = data["total_queries"]
rag_cnt     = next((x["count"] for x in data["intents"] if x["intent"] == "rag_query"), 0)
ticket_cnt  = next((x["count"] for x in data["intents"] if x["intent"] == "create_ticket"), 0)
blocked_cnt = sum(
    x["count"] for x in data["intents"]
    if x["intent"] in ("out_of_scope", "prompt_injection")
)
gap_cnt = len(data["gaps"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Queries",     total)
k2.metric("Knowledge Queries", rag_cnt)
k3.metric("Tickets Created",   ticket_cnt)
k4.metric("Blocked / OOS",     blocked_cnt)
k5.metric("Knowledge Gaps",    gap_cnt)

# ── Empty state ───────────────────────────────────────────────────────────────
if total == 0:
    st.divider()
    st.html("""
    <div style="text-align:center;padding:48px 24px;color:#656D76;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="font-size:48px;margin-bottom:12px;">💬</div>
      <div style="font-size:16px;font-weight:600;color:#1F2328;margin-bottom:8px;">
        No data yet
      </div>
      <div style="font-size:14px;line-height:1.6;">
        Ask a question in the <strong>Chat</strong> tab to start logging telemetry.<br>
        Analytics update automatically as conversations happen.
      </div>
    </div>
    """)
    st.stop()

st.divider()

# ── Row 1: Intent donut + Confidence line ─────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Intent Distribution")
    if data["intents"]:
        df_i = pd.DataFrame(data["intents"])
        df_i["intent_label"] = df_i["intent"].str.replace("_", " ").str.title()
        df_i["color"] = df_i["intent"].map(lambda x: INTENT_COLORS.get(x, "#8B949E"))

        # Altair donut chart
        donut = (
            alt.Chart(df_i)
            .mark_arc(innerRadius=65, outerRadius=130, stroke="#FFFFFF", strokeWidth=2)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "intent_label:N",
                    scale=alt.Scale(
                        domain=df_i["intent_label"].tolist(),
                        range=df_i["color"].tolist(),
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="right",
                        labelFontSize=12,
                        symbolSize=120,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("intent_label:N", title="Intent"),
                    alt.Tooltip("count:Q",        title="Queries"),
                ],
            )
            .properties(height=280)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(donut, use_container_width=True)
    else:
        st.caption("No intent data yet.")

with col_r:
    st.markdown("### Confidence Over Time")
    if data["confidence_series"]:
        df_c = pd.DataFrame(data["confidence_series"])
        df_c["timestamp"]    = pd.to_datetime(df_c["timestamp"])
        df_c["intent_label"] = df_c["intent"].str.replace("_", " ").str.title()
        df_c["color"]        = df_c["intent"].map(lambda x: INTENT_COLORS.get(x, "#8B949E"))
        df_c = df_c.sort_values("timestamp")

        line = (
            alt.Chart(df_c)
            .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T",    title="Time", axis=alt.Axis(format="%H:%M")),
                y=alt.Y(
                    "confidence:Q",
                    title="Confidence",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color(
                    "intent_label:N",
                    scale=alt.Scale(
                        domain=df_c["intent_label"].unique().tolist(),
                        range=df_c.drop_duplicates("intent_label")["color"].tolist(),
                    ),
                    legend=alt.Legend(title=None, orient="bottom", labelFontSize=11),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T",    title="Time",       format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("confidence:Q",   title="Confidence", format=".0%"),
                    alt.Tooltip("intent_label:N", title="Intent"),
                ],
            )
            .properties(height=280)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#F3F4F6", gridOpacity=1)
        )
        st.altair_chart(line, use_container_width=True)
    else:
        st.caption("Ask a few questions to see the confidence trend.")

st.divider()

# ── Row 2: Top questions + Knowledge gaps ─────────────────────────────────────
col_b_l, col_b_r = st.columns(2)

with col_b_l:
    st.markdown("### Top 5 Questions")
    if data["top_questions"]:
        df_q = pd.DataFrame(data["top_questions"])
        df_q["q_short"] = df_q["question"].str[:52].str.strip() + "…"

        bars = (
            alt.Chart(df_q)
            .mark_bar(color="#0969DA", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("count:Q", title="Times Asked", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("q_short:N", title="", sort="-x",
                         axis=alt.Axis(labelLimit=280, labelFontSize=12)),
                tooltip=[
                    alt.Tooltip("question:N", title="Question"),
                    alt.Tooltip("count:Q",    title="Times Asked"),
                ],
            )
            .properties(height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        st.altair_chart(bars, use_container_width=True)
    else:
        st.caption("No repeated questions logged yet.")

with col_b_r:
    st.markdown("### 🕳️ Knowledge Gaps")
    st.caption("Questions the knowledge base couldn't fully answer")
    if data["gaps"]:
        for _g in data["gaps"]:
            _date = _g.get("timestamp", "")[:10]
            _q    = _g.get("question", "")[:95].replace("<", "&lt;").replace(">", "&gt;")
            st.html(f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                        padding:8px 12px;margin:5px 0;background:#FFEBE9;
                        border-left:3px solid #CF222E;border-radius:0 6px 6px 0;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
              <span style="font-size:13px;color:#1F2328;flex:1;">{_q}</span>
              <span style="font-size:11px;color:#9198A1;white-space:nowrap;
                           margin-left:10px;">{_date}</span>
            </div>
            """)
    else:
        st.html("""
        <div style="padding:20px;text-align:center;background:#DAFBE1;
                    border-radius:8px;border:1px solid #ACEEBB;
                    font-family:-apple-system,sans-serif;">
          <div style="font-size:20px;margin-bottom:6px;">🎉</div>
          <div style="font-size:14px;font-weight:600;color:#1A7F37;">
            No knowledge gaps detected!
          </div>
          <div style="font-size:12px;color:#656D76;margin-top:4px;">
            The knowledge base is answering every question.
          </div>
        </div>
        """)
