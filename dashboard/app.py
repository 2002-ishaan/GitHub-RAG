"""
dashboard/app.py
────────────────────────────────────────────────────────────────
GitHub Documentation Assistant — Streamlit UI

HOW TO RUN:
    streamlit run dashboard/app.py

FEATURES:
    - Chat interface with intent label on every message
    - Multi-turn ticket creation with progress indicator
    - Session panel showing open tickets
    - Cross-session persistence via SQLite
    - Guardrails visible in the UI
────────────────────────────────────────────────────────────────
"""

import re
import sys
import time
import uuid
import difflib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from configs.settings import load_settings, load_prompts, setup_logging
from generation.rag_chain import RAGChain, _linkify_answer, _url_to_label
from agent.intent_router import IntentRouter
from agent.session_state import SessionState
from agent.actions import (
    handle_create_ticket,
    handle_check_ticket,
    handle_check_billing,
    is_ticket_flow_active,
    get_ticket_flow_state,
    get_last_created_ticket,
    handle_close_tickets,
    handle_close_ticket_by_id,
)
from agent.guardrails import get_guardrail_response, handle_insufficient_evidence


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Docs Assistant",
    page_icon="🐙",
    layout="wide",
)

# ── Intent label styling ───────────────────────────────────────────────────────
# Tuple: (label, text_color, background, border_color)
INTENT_LABELS = {
    "rag_query":          ("🔍 Knowledge Query", "#0969DA", "#DDF4FF", "#B6E3FF"),
    "create_ticket":      ("🎫 Create Ticket",   "#1A7F37", "#DAFBE1", "#ACEEBB"),
    "check_ticket":       ("📋 Check Ticket",    "#6E40C9", "#F3EEFF", "#D2B1FA"),
    "check_billing":      ("💳 Billing Check",   "#9A6700", "#FFF8C5", "#E9C46A"),
    "out_of_scope":       ("🚫 Out of Scope",    "#CF222E", "#FFEBE9", "#FFBCB5"),
    "prompt_injection":   ("⚠️ Blocked",         "#CF222E", "#FFEBE9", "#FFBCB5"),
    "action_in_progress": ("🎫 Ticket Flow",     "#1A7F37", "#DAFBE1", "#ACEEBB"),
    "close_tickets":      ("🔒 Close Tickets",   "#CF222E", "#FFEBE9", "#FFBCB5"),
}


def intent_badge(intent: str) -> str:
    label, color, bg, border = INTENT_LABELS.get(
        intent, ("❓ Unknown", "#656D76", "#F6F8FA", "#D0D7DE")
    )
    return (
        f'<span style="'
        f'display:inline-flex;align-items:center;gap:4px;'
        f'background:{bg};color:{color};'
        f'padding:3px 10px;border-radius:20px;'
        f'font-size:11px;font-weight:600;letter-spacing:0.3px;'
        f'border:1px solid {border};'
        f'font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Helvetica,Arial,sans-serif;'
        f'">{label}</span>'
    )


# ── Streaming helpers ──────────────────────────────────────────────────────────

def _render_stream_footer(sources: list, elapsed: float, top_similarity: float) -> None:
    """Render sources + confidence bar below a streamed answer (live view only)."""
    seen  = set()
    cites = []
    for src in sources:
        url = src.source_file
        if url in seen:
            continue
        seen.add(url)
        cites.append(f"- [{_url_to_label(url)}]({url})")

    st.markdown("---")
    if cites:
        st.markdown("**Sources:**\n" + "\n".join(cites))
    confidence_pct = round(min(top_similarity, 1.0) * 100, 1)
    st.progress(
        min(top_similarity, 1.0),
        text=f"Confidence: {confidence_pct}%  ·  Response time: {elapsed:.1f}s",
    )


def _build_stream_response(
    full_answer: str,
    sources: list,
    elapsed: float,
    top_similarity: float,
) -> str:
    """
    Build the stored message string from streaming results.
    Mirrors RAGResponse.formatted_answer() so replayed history renders identically.
    """
    linked = _linkify_answer(full_answer)

    seen  = set()
    cites = []
    for src in sources:
        url = src.source_file
        if url in seen:
            continue
        seen.add(url)
        cites.append(f"- [{_url_to_label(url)}]({url})")

    confidence_pct = round(min(top_similarity, 1.0) * 100, 1)
    return (
        f"{linked}\n\n"
        f"---\n"
        f"**Sources:**\n" + "\n".join(cites) + "\n\n"
        f"*Confidence: {confidence_pct}%  ·  Response time: {elapsed:.1f}s*"
    )


# ── Load resources (cached so they only load once) ─────────────────────────────
@st.cache_resource
def load_resources():
    settings = load_settings()
    setup_logging(settings)
    prompts       = load_prompts()
    rag_chain     = RAGChain(settings)
    intent_router = IntentRouter(settings, prompts)
    session_state = SessionState(settings.sqlite_db_path)
    return settings, prompts, rag_chain, intent_router, session_state


# ── Session initialisation ─────────────────────────────────────────────────────
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ticket_step" not in st.session_state:
        st.session_state.ticket_step = None
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []


# ── Process one user message ───────────────────────────────────────────────────
def process_message(
    user_input: str,
    settings,
    prompts,
    rag_chain,
    intent_router,
    session_state,
) -> tuple[str, str]:
    """
    Route user message and return (response_text, intent_label).
    """
    session_id = st.session_state.session_id

    # ── Check if multi-turn ticket flow is active ──────────────────────────
    if is_ticket_flow_active(session_id):
        response = handle_create_ticket(
            session_id=session_id,
            user_message=user_input,
            session_state=session_state,
            prompts=prompts,
        )
        # Save to history
        session_state.append_to_history(session_id, "user", user_input)
        session_state.append_to_history(session_id, "assistant", response)
        return response, "action_in_progress"

    # ── Classify intent ────────────────────────────────────────────────────
    intent_result = intent_router.classify(user_input)

    # ── Check guardrails ───────────────────────────────────────────────────
    guardrail_response = get_guardrail_response(intent_result, prompts)
    if guardrail_response:
        session_state.append_to_history(session_id, "user", user_input)
        session_state.append_to_history(session_id, "assistant", guardrail_response)
        return guardrail_response, intent_result.intent

    # ── Route to handler ───────────────────────────────────────────────────
    intent = intent_result.intent

    if intent == "create_ticket":
        response = handle_create_ticket(
            session_id=session_id,
            user_message=user_input,
            session_state=session_state,
            prompts=prompts,
        )

    # Version 1: Check ticket only returns ticket status without response generation
    # elif intent == "check_ticket":
    #     response = handle_check_ticket(user_input, session_state)

    # Version 2: Check ticket returns both ticket status and response generation
    elif intent == "check_ticket":
        response = handle_check_ticket(user_input, session_state, rag_chain)

    elif intent == "check_billing":
        response = handle_check_billing(user_input)

    # Added: Close ticket by ID intent
    elif intent == "close_ticket_by_id":
        response = handle_close_ticket_by_id(user_input, session_state)
    
    elif intent == "close_tickets":
        from agent.actions import handle_close_tickets
        response = handle_close_tickets(user_input, session_state)
    

    else:
        # Default: RAG query
        intent = "rag_query"
        rag_response = rag_chain.ask(
            question=user_input,
            session_id=session_id,
            session_state=session_state,
        )
        if rag_response.is_supported:
            response = rag_response.formatted_answer()
        else:
            response = handle_insufficient_evidence(prompts)

    # Save to persistent history
    session_state.append_to_history(session_id, "user", user_input)
    session_state.append_to_history(session_id, "assistant", response)

    return response, intent


# ── Frustration detection ─────────────────────────────────────────────────────
_FRUSTRATION_PATTERNS = [
    r"\bugh\b", r"\bstill\s+not\s+working\b", r"\bstill\s+don.t\s+understand\b",
    r"\bdoesn.t\s+work\b", r"\bnot\s+working\b", r"\bfrustrat\w+\b",
    r"\bstuck\b", r"\bwtf\b", r"\bffs\b", r"\bannoying\b",
    r"\buseless\b", r"\bdoesn.t\s+make\s+sense\b", r"\bwhy\s+isn.t\s+this\b",
]

def _is_frustrated(user_input: str, messages: list) -> bool:
    """Detect frustration via keywords or same question asked twice in recent history."""
    lower = user_input.lower()
    for pat in _FRUSTRATION_PATTERNS:
        if re.search(pat, lower):
            return True
    # Same question repeated (> 75% similarity) in last 6 user messages
    recent_user = [
        m["content"].lower()[:80]
        for m in messages[-6:]
        if m["role"] == "user"
    ]
    for prev in recent_user[:-1]:
        if difflib.SequenceMatcher(None, prev, lower[:80]).ratio() > 0.75:
            return True
    return False


# ── Comparative question detection ────────────────────────────────────────────
_COMPARATIVE_RE = re.compile(
    r"\bcompare\b|\bcomparison\b|\bvs\.?\b|\bversus\b"
    r"|\bdifference\s+between\b|\bwhich\s+is\s+better\b",
    re.IGNORECASE,
)

def _detect_comparative(question: str):
    """Return (True, topic1, topic2) for compare questions; else (False, None, None)."""
    if not _COMPARATIVE_RE.search(question):
        return False, None, None
    for pat in [
        r"(.+?)\s+vs\.?\s+(.+)",
        r"difference\s+between\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+with\s+(.+)",
    ]:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            t1 = m.group(1).strip().rstrip("?.,")
            t2 = m.group(2).strip().rstrip("?.,")
            if t1 and t2:
                return True, t1, t2
    return False, None, None


# ── HTML conversation export ──────────────────────────────────────────────────
def _generate_html_export(messages: list, session_id: str) -> str:
    """Return a self-contained styled HTML report of the conversation."""
    from datetime import datetime as _dt
    rows = ""
    for msg in messages:
        role    = "You" if msg["role"] == "user" else "Assistant"
        css_cls = msg["role"]
        intent  = msg.get("intent", "")
        badge   = (
            f'<span class="badge">{intent.replace("_"," ").title()}</span>'
            if intent and msg["role"] == "assistant" else ""
        )
        body = (
            msg["content"]
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        rows += f"""
<div class="msg {css_cls}">
  <div class="role">{role}{badge}</div>
  <div class="body">{body}</div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>GitHub Docs Assistant — {session_id}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        max-width:820px;margin:0 auto;padding:24px;color:#1F2328;background:#fff}}
  h1{{font-size:20px;font-weight:700;border-bottom:2px solid #0969DA;padding-bottom:12px;margin-bottom:4px}}
  .meta{{color:#656D76;font-size:13px;margin-bottom:24px}}
  .msg{{margin-bottom:12px;padding:12px 16px;border-radius:8px;line-height:1.6}}
  .user{{background:#DDF4FF;border:1px solid #B6E3FF}}
  .assistant{{border-left:3px solid #0969DA;padding-left:14px;background:#FAFAFA}}
  .role{{font-size:11px;font-weight:700;text-transform:uppercase;color:#656D76;margin-bottom:6px}}
  .body{{font-size:14px}}
  .badge{{background:#F6F8FA;border:1px solid #D0D7DE;border-radius:10px;
          padding:1px 8px;font-size:10px;font-weight:600;margin-left:6px;
          color:#0969DA;text-transform:none;vertical-align:middle}}
  .footer{{margin-top:32px;color:#9198A1;font-size:12px;text-align:center;
           border-top:1px solid #D0D7DE;padding-top:12px}}
  @media print{{body{{margin:0;padding:12px}}}}
</style></head>
<body>
<h1>🐙 GitHub Documentation Assistant</h1>
<div class="meta">
  Session: <code>{session_id}</code> &nbsp;·&nbsp;
  Exported: {_dt.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp;
  {len(messages)} messages
</div>
{rows}
<div class="footer">
  Generated by GitHub Docs Assistant &nbsp;·&nbsp; Print this page (Ctrl+P) to save as PDF
</div>
</body></html>"""


# ── New-content diff helper ───────────────────────────────────────────────────

def _get_new_sentences(prev_text: str, new_text: str) -> list:
    """
    Return sentences from new_text that are substantially different from prev_text.
    Used to highlight what's genuinely new in a follow-up answer.
    """
    def _body(t: str) -> str:
        return t.split("---")[0].strip() if "---" in t else t.strip()

    prev_sents = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', _body(prev_text))
        if len(s.strip()) > 15
    ]
    new_sents = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', _body(new_text))
        if len(s.strip()) > 15
    ]

    new_only = []
    for sent in new_sents:
        is_new = not any(
            difflib.SequenceMatcher(None, sent.lower(), ps.lower()).ratio() > 0.55
            for ps in prev_sents
        )
        if is_new:
            new_only.append(sent)
    return new_only


# ── GitHub design-system CSS ───────────────────────────────────────────────────
GITHUB_CSS = """
<style>
/* ── Fonts & global reset ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                 Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    font-size: 15px;
    color: #1F2328;
    background-color: #FFFFFF;
}

/* Prevent layout from zooming with browser zoom */
.block-container { zoom: 1 !important; }

/* ── Hide default Streamlit chrome ───────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }

/* ── Page header (## heading) ─────────────────────────────────────────────── */
h2 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1F2328 !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid #D0D7DE !important;
    margin-bottom: 16px !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: #F6F8FA;
}

/* Sidebar section headers (#### headings) */
h4 {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #9198A1 !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
}

/* ── Chat messages ────────────────────────────────────────────────────────── */
@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    animation: msgFadeIn 0.15s ease-out;
    border: none !important;
    background: transparent !important;
    padding: 6px 0 !important;
}

/* User bubble — right-aligned blue pill */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    background: #DDF4FF !important;
    color: #0969DA !important;
    border: 1px solid #B6E3FF !important;
    border-radius: 12px 12px 2px 12px !important;
    padding: 10px 14px !important;
    display: inline-block !important;
    max-width: 88% !important;
}

/* Assistant bubble — left border accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #0969DA !important;
    padding-left: 12px !important;
    background: transparent !important;
}

/* ── Chat input ───────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border: 1px solid #D0D7DE !important;
    border-radius: 6px !important;
    background: #FFFFFF !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #0969DA !important;
    box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15) !important;
    outline: none !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
/* Default / outlined buttons */
.stButton > button {
    background: #FFFFFF !important;
    color: #1F2328 !important;
    border: 1px solid #D0D7DE !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 5px 12px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: background 0.12s, border-color 0.12s;
}

.stButton > button:hover {
    background: #F3F4F6 !important;
    border-color: #9198A1 !important;
}

/* Primary button (New Conversation) */
.stButton > button[kind="primary"] {
    background: #0969DA !important;
    color: #FFFFFF !important;
    border-color: #0969DA !important;
    font-weight: 600 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #0550AE !important;
    border-color: #0550AE !important;
}

/* Close-ticket danger buttons inside expanders */
div[data-testid="stExpander"] .stButton > button {
    background: #FFEBE9 !important;
    color: #CF222E !important;
    border-color: #FFBCB5 !important;
    font-size: 12px !important;
    padding: 3px 10px !important;
}

div[data-testid="stExpander"] .stButton > button:hover {
    background: #CF222E !important;
    color: #FFFFFF !important;
    border-color: #CF222E !important;
}

/* Follow-up suggestion buttons — inside chat messages */
[data-testid="stChatMessage"] .stButton > button {
    background: transparent !important;
    color: #0969DA !important;
    border: 1px solid #B6E3FF !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 4px 12px !important;
    margin: 2px 0 !important;
}

[data-testid="stChatMessage"] .stButton > button:hover {
    background: #DDF4FF !important;
    border-color: #0969DA !important;
}

/* Example query buttons in sidebar */
.stButton > button[data-testid*="ex_"] {
    text-align: left !important;
    font-size: 12px !important;
    color: #0969DA !important;
    border-color: #B6E3FF !important;
    background: #F6F8FA !important;
}

/* ── Ticket expander cards ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #D0D7DE !important;
    border-radius: 6px !important;
    background: #FFFFFF !important;
    margin-bottom: 8px !important;
}

[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1F2328 !important;
    padding: 8px 12px !important;
}

/* ── Progress bar (confidence meter) ─────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: #0969DA !important;
    border-radius: 4px !important;
}

[data-testid="stProgress"] {
    background: #F3F4F6 !important;
    border-radius: 4px !important;
}

/* ── Horizontal rules ─────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #D0D7DE !important;
    margin: 12px 0 !important;
}

/* ── Caption / muted text ─────────────────────────────────────────────────── */
.stCaption, small {
    color: #9198A1 !important;
    font-size: 12px !important;
}

/* ── Markdown tables (ticket confirmation) ────────────────────────────────── */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}

th {
    background: #F6F8FA !important;
    color: #656D76 !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 6px 12px !important;
    border: 1px solid #D0D7DE !important;
}

td {
    padding: 6px 12px !important;
    border: 1px solid #D0D7DE !important;
    color: #1F2328 !important;
}

tr:nth-child(even) td { background: #F6F8FA !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F6F8FA; }
::-webkit-scrollbar-thumb { background: #D0D7DE; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9198A1; }
</style>
"""


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    init_session()
    st.markdown(GITHUB_CSS, unsafe_allow_html=True)

    try:
        settings, prompts, rag_chain, intent_router, session_state = load_resources()
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.info("Make sure you have set QWEN_API_KEY and QWEN_BASE_URL in your .env file, and run `python -m ingestion.ingest` first.")
        return

    session_id = st.session_state.session_id

    # ── Layout ─────────────────────────────────────────────────────────────
    col_chat, col_side = st.columns([3, 1])

    with col_side:
        # Analytics navigation — always at top
        if st.button("📊 Analytics Dashboard", use_container_width=True, key="nav_analytics"):
            st.switch_page("pages/1_📊_Analytics.py")

        st.divider()

    # Replaced with Ticket panel
        st.markdown("#### 🎫 Open Tickets")
        open_tickets = session_state.get_open_tickets()
        if not open_tickets:
            st.caption("No open tickets. Say *\"Create a support ticket\"* to start.")
        else:
            for ticket in open_tickets:
                pri_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(ticket["priority"], "⚪")
                with st.expander(f"{ticket['ticket_id']} — {ticket['category']}"):
                    st.write(f"**Priority:** {pri_icon} {ticket['priority']}")
                    st.write(f"**Description:** {ticket['description'][:80]}...")
                    st.write(f"**Created:** {ticket['created_at'][:10]}")
                    if st.button(f"🔒 Close {ticket['ticket_id']}", key=f"close_{ticket['ticket_id']}"):
                        result = session_state.close_ticket_by_id(ticket["ticket_id"])
                        st.toast(result["message"])
                        st.rerun()


        st.divider()

        # New session button
        if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.followup_questions = []
            st.rerun()

        st.divider()

        # Past conversations panel
        st.markdown("#### 🕘 Past Conversations")
        past_sessions = session_state.list_sessions(limit=10)
        # Exclude the current active session
        past_sessions = [s for s in past_sessions if s["session_id"] != session_id][:5]

        if not past_sessions:
            st.caption("No past conversations yet.")
        else:
            for s in past_sessions:
                preview = s["first_message"][:40] + ("…" if len(s["first_message"]) > 40 else "")
                date    = s["created_at"][:10]
                col_label, col_load, col_del, col_rep = st.columns([4, 1, 1, 1])
                with col_label:
                    st.markdown(f"**`{s['session_id']}`** · {date}")
                    st.caption(preview or "*no messages*")
                with col_load:
                    if st.button("↗", key=f"load_{s['session_id']}", help="Load conversation"):
                        loaded = [
                            {"role": m["role"], "content": m["content"], "intent": None}
                            for m in s["history"]
                        ]
                        st.session_state.messages   = loaded
                        st.session_state.session_id = s["session_id"]
                        st.session_state.pop("replay_mode", None)
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{s['session_id']}", help="Delete"):
                        session_state.delete_session(s["session_id"])
                        st.rerun()
                with col_rep:
                    if st.button("▶", key=f"rep_{s['session_id']}", help="Replay conversation"):
                        st.session_state["replay_mode"]    = True
                        st.session_state["replay_history"] = s["history"]
                        st.session_state["replay_idx"]     = 0
                        st.rerun()

        st.divider()
        st.markdown("#### Try these:")
        examples = [
            "How do I create a private repo?",
            "Create a support ticket",
            "Check billing for alice",
            "How do I set up 2FA?",
            "Check ticket TKT-001",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state.messages.append({"role": "user", "content": ex, "intent": None})
                with st.spinner("Thinking..."):
                    response, intent = process_message(
                        ex, settings, prompts, rag_chain, intent_router, session_state
                    )
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "intent": intent
                })
                st.rerun()

    with col_chat:
        # Header row — title + export button on the right
        _hcol_title, _hcol_export = st.columns([7, 1])
        with _hcol_title:
            st.markdown("## 🐙 GitHub Documentation Assistant")
            st.caption(
                "Ask anything about GitHub — repositories, billing, authentication, "
                "security, organizations. I can also create and check support tickets."
            )
        with _hcol_export:
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            if st.session_state.get("messages"):
                _html_export = _generate_html_export(st.session_state.messages, session_id)
                st.download_button(
                    "📄",
                    data=_html_export.encode("utf-8"),
                    file_name=f"conversation_{session_id}.html",
                    mime="text/html",
                    key="export_download",
                    help="Export conversation as HTML (print to PDF from browser)",
                )

        # ── Replay mode — renders instead of normal chat when active ──────
        if st.session_state.get("replay_mode"):
            _rh   = st.session_state.get("replay_history", [])
            _ridx = st.session_state.get("replay_idx", 0)
            _rtot = len(_rh)
            _pct  = round((_ridx / max(_rtot, 1)) * 100, 1)

            # Custom GitHub-styled progress header
            st.html(f"""
            <div style="background:#F6F8FA;border:1px solid #D0D7DE;border-radius:8px;
                        padding:12px 18px;margin-bottom:10px;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:8px;">
                <span style="font-size:13px;font-weight:600;color:#1F2328;">
                  ▶&nbsp; Conversation Replay
                </span>
                <span style="font-size:12px;color:#656D76;font-weight:500;">
                  Message {_ridx} of {_rtot}
                  &nbsp;·&nbsp; {_pct:.0f}% complete
                </span>
              </div>
              <div style="background:#D0D7DE;border-radius:6px;height:6px;overflow:hidden;">
                <div style="background:#0969DA;height:6px;width:{_pct}%;
                            border-radius:6px;transition:width 0.4s ease;"></div>
              </div>
            </div>
            """)

            # Control buttons on same row
            _rb1, _rb2 = st.columns([1, 7])
            with _rb1:
                if st.button("⏹ Stop", key="replay_stop"):
                    for _k in ("replay_mode", "replay_history", "replay_idx"):
                        st.session_state.pop(_k, None)
                    st.rerun()
            with _rb2:
                if _ridx >= _rtot:
                    if st.button("↩ Replay again", key="replay_again"):
                        st.session_state["replay_idx"] = 0
                        st.rerun()

            for _ri in range(min(_ridx, _rtot)):
                _rm = _rh[_ri]
                with st.chat_message(_rm["role"]):
                    st.markdown(_rm.get("content", ""))

            if _ridx < _rtot:
                time.sleep(0.7)
                st.session_state["replay_idx"] = _ridx + 1
                st.rerun()
            else:
                st.success("✅ Replay complete!")
            st.stop()   # don't render normal chat while in replay mode

        # ── Ticket flow progress bar ───────────────────────────────────────
        if is_ticket_flow_active(session_id):
            _ts = get_ticket_flow_state(session_id)
            if _ts:
                _step_names  = ["Category", "Description", "Priority"]
                _step_map    = {
                    "awaiting_category":    1,
                    "awaiting_description": 2,
                    "awaiting_priority":    3,
                    "review":               4,
                }
                _cur = _step_map.get(_ts.step, 1)

                _dots = []
                for _i, _lbl in enumerate(_step_names, 1):
                    if _i < _cur:
                        _dots.append(
                            f'<span style="color:#1A7F37;font-weight:600;">'
                            f'✓ {_lbl}</span>'
                        )
                    elif _i == _cur:
                        _dots.append(
                            f'<span style="color:#0969DA;font-weight:700;">'
                            f'● {_lbl}</span>'
                        )
                    else:
                        _dots.append(
                            f'<span style="color:#9198A1;">○ {_lbl}</span>'
                        )

                if _cur == 4:
                    _title    = "Review &amp; Confirm"
                    _dots_str = (
                        '<span style="color:#1A7F37;font-weight:600;">'
                        '✓ Category &nbsp;·&nbsp; ✓ Description &nbsp;·&nbsp; ✓ Priority</span>'
                        '&nbsp;·&nbsp;'
                        '<span style="color:#0969DA;font-weight:700;">● Review</span>'
                    )
                else:
                    _title    = f"Step {_cur} of 3 &nbsp;·&nbsp; {_step_names[_cur - 1]}"
                    _dots_str = " &nbsp;·&nbsp; ".join(_dots)

                st.html(f"""
                <div style="background:#F6F8FA;border:1px solid #D0D7DE;border-radius:8px;
                            padding:10px 16px;margin-bottom:12px;
                            display:flex;align-items:center;justify-content:space-between;
                            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                    <div style="font-size:13px;font-weight:600;color:#1F2328;">
                        🎫 Creating Support Ticket &nbsp;·&nbsp; {_title}
                    </div>
                    <div style="font-size:12px;">{_dots_str}</div>
                </div>
                """)

        # Display chat history
        _prev_assistant_content = None  # track previous assistant message for diff
        for _msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("intent"):
                    # st.html() renders inline HTML reliably inside st.chat_message
                    st.html(intent_badge(msg["intent"]))

                _is_last = (_msg_idx == len(st.session_state.messages) - 1)

                # Render message content — ticket confirmation gets a visual card
                _is_confirmation = "✅ **Support ticket created!**" in msg["content"]
                if _is_confirmation:
                    # Parse ticket info from the stored markdown table
                    _tid_m = re.search(r'\|\s*Ticket ID\s*\|\s*\*\*(\S+)\*\*', msg["content"])
                    _cat_m = re.search(r'\|\s*Category\s*\|\s*(.+?)\s*\|', msg["content"])
                    _pri_m = re.search(r'\|\s*Priority\s*\|\s*(.+?)\s*\|', msg["content"])
                    _tid   = _tid_m.group(1) if _tid_m else "TKT-???"
                    _cat   = _cat_m.group(1).strip() if _cat_m else "—"
                    _pri   = _pri_m.group(1).strip() if _pri_m else "Medium"
                    _pe    = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(_pri, "⚪")
                    _pc    = {"High": "#CF222E", "Medium": "#9A6700", "Low": "#1A7F37"}.get(_pri, "#1F2328")
                    st.html(f"""
                    <div style="border:1.5px solid #ACEEBB;border-radius:10px;
                                background:#DAFBE1;padding:16px 20px;margin:4px 0;
                                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                      <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                        <span style="font-size:22px;">✅</span>
                        <span style="font-size:15px;font-weight:700;color:#1A7F37;">
                          Support Ticket Created!
                        </span>
                      </div>
                      <div style="background:#FFFFFF;border-radius:8px;border:1px solid #D0D7DE;
                                  padding:14px 16px;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:10px;padding-bottom:10px;
                                    border-bottom:1px solid #F3F4F6;">
                          <span style="color:#656D76;font-size:11px;font-weight:600;
                                       text-transform:uppercase;letter-spacing:0.5px;">Ticket ID</span>
                          <span style="background:#F6F8FA;border:1px solid #D0D7DE;
                                       border-radius:6px;padding:2px 10px;
                                       font-family:monospace;font-size:14px;
                                       font-weight:700;color:#1F2328;">{_tid}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:8px;">
                          <span style="color:#656D76;font-size:11px;font-weight:600;
                                       text-transform:uppercase;letter-spacing:0.5px;">Category</span>
                          <span style="font-size:13px;color:#1F2328;">{_cat}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:8px;">
                          <span style="color:#656D76;font-size:11px;font-weight:600;
                                       text-transform:uppercase;letter-spacing:0.5px;">Priority</span>
                          <span style="font-size:13px;color:{_pc};font-weight:600;">
                            {_pe} {_pri}
                          </span>
                        </div>
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                          <span style="color:#656D76;font-size:11px;font-weight:600;
                                       text-transform:uppercase;letter-spacing:0.5px;">Status</span>
                          <span style="background:#DAFBE1;color:#1A7F37;border:1px solid #ACEEBB;
                                       border-radius:12px;padding:2px 10px;font-size:12px;
                                       font-weight:600;">Open</span>
                        </div>
                      </div>
                      <div style="margin-top:10px;color:#656D76;font-size:12px;">
                        Check status: <em>"Check ticket {_tid}"</em>
                      </div>
                    </div>
                    """)
                    # Balloons on first render after creation (flag set in non-streaming path)
                    if st.session_state.pop("new_ticket_animation", False):
                        st.balloons()
                else:
                    st.markdown(msg["content"])

                # ── "What's new" highlight for follow-up answers ───────────────
                # When there's a prior assistant answer to compare against, show
                # new sentences highlighted green so the memory feature is visible.
                if (
                    _is_last
                    and msg["role"] == "assistant"
                    and _prev_assistant_content is not None
                ):
                    new_sents = _get_new_sentences(_prev_assistant_content, msg["content"])
                    if new_sents:
                        st.html(
                            '<div style="margin-top:10px;padding:10px 14px;'
                            'background:#DAFBE1;border-left:3px solid #1A7F37;'
                            'border-radius:0 6px 6px 0;">'
                            '<div style="font-size:11px;font-weight:600;color:#1A7F37;'
                            'letter-spacing:0.3px;margin-bottom:6px;">✨ NEW INFORMATION</div>'
                            + "".join(
                                f'<p style="margin:3px 0;font-size:13px;color:#1F2328;">{s}</p>'
                                for s in new_sents
                            )
                            + "</div>"
                        )

                # ── RAG Explainability Panel ───────────────────────────────
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("🔍 Why this answer?", expanded=False):
                        st.caption(f"{len(msg['sources'])} chunks retrieved from the knowledge base")
                        for _ei, _esrc in enumerate(msg["sources"], 1):
                            _sc     = float(_esrc.get("score", 0))
                            _sc_pct = f"{_sc:.0%}"
                            _sc_col = (
                                "#1A7F37" if _sc > 0.70 else
                                "#9A6700" if _sc > 0.50 else "#9198A1"
                            )
                            _etext  = (
                                _esrc.get("text", "")[:280]
                                .replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            _eurl   = _esrc.get("source_file", "")
                            _elabel = _esrc.get("label", _url_to_label(_eurl))
                            st.html(f"""
                            <div style="padding:10px 14px;margin:5px 0;
                                        border:1px solid #D0D7DE;border-radius:6px;
                                        background:#F6F8FA;font-family:-apple-system,sans-serif;">
                              <div style="display:flex;justify-content:space-between;
                                          align-items:center;margin-bottom:5px;">
                                <span style="font-size:12px;font-weight:600;color:#1F2328;">
                                  Chunk {_ei} &nbsp;·&nbsp; {_elabel}
                                </span>
                                <span style="background:{_sc_col}22;color:{_sc_col};
                                             border:1px solid {_sc_col}55;border-radius:10px;
                                             padding:1px 9px;font-size:11px;font-weight:600;">
                                  {_sc_pct} match
                                </span>
                              </div>
                              <div style="font-size:12px;color:#656D76;line-height:1.55;">
                                {_etext}…
                              </div>
                              <div style="margin-top:6px;font-size:11px;">
                                <a href="{_eurl}" target="_blank"
                                   style="color:#0969DA;text-decoration:none;">
                                  {_eurl[:70]}
                                </a>
                              </div>
                            </div>
                            """)

                # Track latest assistant message for the next diff
                if msg["role"] == "assistant":
                    _prev_assistant_content = msg["content"]

                # Follow-up suggestion buttons live HERE — in the messages loop —
                # so they are always rendered on every rerun and Streamlit can
                # correctly process click events. Buttons inside a transient
                # `if user_input:` block disappear on the next rerun before
                # Streamlit can match the click, so the event gets silently lost.
                if (
                    _is_last
                    and msg["role"] == "assistant"
                    and st.session_state.get("followup_questions")
                ):
                    st.markdown("---")
                    st.markdown("💡 **You might also want to ask:**")
                    for _fi, _fq in enumerate(st.session_state.followup_questions):
                        if st.button(f'→  "{_fq}"', key=f"fq_{_fi}"):
                            st.session_state.followup_questions = []
                            st.session_state["pending_followup"] = _fq
                            st.rerun()

                # ── Ticket flow interactive UI ─────────────────────────────
                # Render category/priority/review buttons for the last assistant
                # message when a ticket flow is active. Buttons live HERE (in the
                # always-rendered messages loop) so click events are never lost.
                if _is_last and msg["role"] == "assistant" and is_ticket_flow_active(session_id):
                    _ts = get_ticket_flow_state(session_id)
                    if _ts:

                        if _ts.step == "awaiting_category":
                            st.markdown("**Pick a category:**")
                            _cats = [
                                ("🔐", "Account & Authentication"),
                                ("💳", "Billing & Payments"),
                                ("📁", "Repository Issues"),
                                ("👥", "Organizations & Teams"),
                                ("🔒", "Security & 2FA"),
                                ("💬", "Other"),
                            ]
                            _c1, _c2 = st.columns(2)
                            for _ci, (_icon, _cat_name) in enumerate(_cats):
                                _col = _c1 if _ci % 2 == 0 else _c2
                                with _col:
                                    if st.button(
                                        f"{_icon}  {_cat_name}",
                                        key=f"cat_{_ci}",
                                        use_container_width=True,
                                    ):
                                        st.session_state["pending_ticket_input"] = _cat_name.lower()
                                        st.rerun()

                        elif _ts.step == "awaiting_priority":
                            st.markdown("**Choose priority:**")
                            _pris = [
                                ("🔴", "High",   "Blocking my work completely"),
                                ("🟡", "Medium", "Causing problems but workable"),
                                ("🟢", "Low",    "Minor inconvenience"),
                            ]
                            for _pemoji, _plabel, _pdesc in _pris:
                                if st.button(
                                    f"{_pemoji}  **{_plabel}** — {_pdesc}",
                                    key=f"pri_{_plabel}",
                                    use_container_width=True,
                                ):
                                    st.session_state["pending_ticket_input"] = _plabel.lower()
                                    st.rerun()

                        elif _ts.step == "review":
                            _rc1, _rc2, _rc3 = st.columns(3)
                            with _rc1:
                                if st.button(
                                    "✅  Confirm & Submit",
                                    key="review_confirm",
                                    use_container_width=True,
                                    type="primary",
                                ):
                                    st.session_state["pending_ticket_input"] = "yes"
                                    st.rerun()
                            with _rc2:
                                if st.button(
                                    "✏️  Edit",
                                    key="review_edit",
                                    use_container_width=True,
                                ):
                                    st.session_state["pending_ticket_input"] = "edit"
                                    st.rerun()
                            with _rc3:
                                if st.button(
                                    "❌  Cancel",
                                    key="review_cancel",
                                    use_container_width=True,
                                ):
                                    st.session_state["pending_ticket_input"] = "cancel"
                                    st.rerun()

        # ── Resolve pending inputs from button clicks ──────────────────────
        # Ticket flow buttons (category/priority/review) take top priority,
        # then follow-up suggestion clicks, then typed input.
        _pending_ticket   = st.session_state.pop("pending_ticket_input", None)
        _pending_followup = st.session_state.pop("pending_followup", None)
        _chat_typed       = st.chat_input("Ask a GitHub question or request an action...")
        user_input        = _pending_ticket or _pending_followup or _chat_typed

        if user_input:
            # Clear any previous follow-up suggestions (new question is being asked)
            st.session_state.followup_questions = []
            # Show user message immediately
            st.session_state.messages.append({
                "role": "user", "content": user_input, "intent": None
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                # ── Frustration detection (before routing) ─────────────────────
                if (
                    not is_ticket_flow_active(session_id)
                    and _is_frustrated(user_input, st.session_state.messages[:-1])
                ):
                    _frust_msg = (
                        "😕 It sounds like you might be stuck. I want to make sure you get the help you need.\n\n"
                        "Would you like me to **create a support ticket** so a GitHub specialist can look into this? "
                        "Just say *\"yes, create a ticket\"* — or go ahead and ask your question and I'll try again."
                    )
                    st.html(intent_badge("rag_query"))
                    st.markdown(_frust_msg)
                    session_state.append_to_history(session_id, "user", user_input)
                    session_state.append_to_history(session_id, "assistant", _frust_msg)
                    session_state.log_query(session_id, "rag_query", 0.5, user_input, is_gap=False)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _frust_msg, "intent": "rag_query"}
                    )
                    st.rerun()

                # ── Comparative question detection ─────────────────────────────
                _is_comp, _ctopic1, _ctopic2 = _detect_comparative(user_input)
                if _is_comp and _ctopic1 and _ctopic2 and not is_ticket_flow_active(session_id):
                    st.html(intent_badge("rag_query"))
                    st.markdown(f"**Comparing:** {_ctopic1} vs {_ctopic2}")
                    _cc1, _cc2 = st.columns(2)
                    _cr1 = _cr2 = None
                    with _cc1:
                        st.markdown(f"### {_ctopic1}")
                        with st.spinner(f"Searching {_ctopic1}…"):
                            _cr1 = rag_chain.ask(
                                f"Tell me about {_ctopic1} on GitHub",
                                session_id, session_state,
                            )
                        st.markdown(_cr1.answer if _cr1 and _cr1.is_supported else "_No results found._")
                    with _cc2:
                        st.markdown(f"### {_ctopic2}")
                        with st.spinner(f"Searching {_ctopic2}…"):
                            _cr2 = rag_chain.ask(
                                f"Tell me about {_ctopic2} on GitHub",
                                session_id, session_state,
                            )
                        st.markdown(_cr2.answer if _cr2 and _cr2.is_supported else "_No results found._")
                    _comp_stored = (
                        f"**{_ctopic1}:** {_cr1.answer if _cr1 and _cr1.is_supported else 'N/A'}\n\n"
                        f"**{_ctopic2}:** {_cr2.answer if _cr2 and _cr2.is_supported else 'N/A'}"
                    )
                    session_state.append_to_history(session_id, "user", user_input)
                    session_state.append_to_history(session_id, "assistant", _comp_stored)
                    session_state.log_query(session_id, "rag_query", 0.9, user_input, is_gap=False)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _comp_stored, "intent": "rag_query"}
                    )
                    st.rerun()

                # ── Decide whether to use the streaming RAG path ───────────────
                _use_streaming = False
                _intent_result = None
                _NON_RAG_INTENTS = {
                    "create_ticket", "check_ticket", "check_billing",
                    "close_tickets", "close_ticket_by_id",
                    "out_of_scope", "prompt_injection",
                }

                if not is_ticket_flow_active(session_id):
                    _intent_result = intent_router.classify(user_input)
                    _guardrail     = get_guardrail_response(_intent_result, prompts)
                    if not _guardrail and _intent_result.intent not in _NON_RAG_INTENTS:
                        _use_streaming = True

                if _use_streaming:
                    # ── Streaming RAG path ─────────────────────────────────────
                    intent      = "rag_query"
                    response    = handle_insufficient_evidence(prompts)  # safe default
                    full_answer = ""   # populated on successful stream
                    _answered   = False

                    st.html(intent_badge(intent))

                    stream_setup = None
                    try:
                        stream_setup = rag_chain.ask_streaming(
                            question=user_input,
                            session_id=session_id,
                            session_state=session_state,
                        )
                    except Exception as _exc:
                        logger.warning(f"ask_streaming setup failed: {_exc} — falling back")

                    if stream_setup is not None:
                        try:
                            full_answer = st.write_stream(stream_setup.token_gen)
                            elapsed     = time.time() - stream_setup.start_time
                            top_sim     = stream_setup.metadata.get("top_similarity", 0.0)

                            if (
                                "INSUFFICIENT_EVIDENCE" in full_answer.upper()
                                or len(full_answer.strip()) < 20
                            ):
                                st.markdown(handle_insufficient_evidence(prompts))
                            else:
                                _render_stream_footer(stream_setup.sources, elapsed, top_sim)
                                response  = _build_stream_response(
                                    full_answer, stream_setup.sources, elapsed, top_sim
                                )
                                _answered = True

                        except Exception as _exc:
                            logger.warning(f"Streaming interrupted: {_exc} — falling back")
                            try:
                                _fb = rag_chain.ask(user_input, session_id, session_state)
                                response = (
                                    _fb.formatted_answer()
                                    if _fb.is_supported
                                    else handle_insufficient_evidence(prompts)
                                )
                            except Exception:
                                response = handle_insufficient_evidence(prompts)
                            st.markdown(response)
                    else:
                        st.markdown(response)

                    # ── Follow-up suggestions ──────────────────────────────────
                    # Generate suggestions and store in session state.
                    # Buttons are rendered in the messages loop above (not here),
                    # so they survive reruns and click events are never lost.
                    if _answered and full_answer:
                        with st.spinner("Generating suggestions..."):
                            _followups = rag_chain.suggest_followups(user_input, full_answer)
                        st.session_state.followup_questions = _followups
                    else:
                        st.session_state.followup_questions = []

                    # ── Serialise sources for explainability panel ────────────
                    _src_data = []
                    if stream_setup and _answered:
                        _src_data = [
                            {
                                "text":        src.text[:280],
                                "source_file": src.source_file,
                                "score":       float(src.similarity_score),
                                "label":       _url_to_label(src.source_file),
                            }
                            for src in stream_setup.sources
                        ]

                    # ── Log to analytics ──────────────────────────────────────
                    _conf = _intent_result.confidence if _intent_result else 0.8
                    session_state.log_query(
                        session_id, "rag_query", _conf, user_input,
                        is_gap=(not _answered),
                    )

                    # Persist turn then rerun — messages loop will render the
                    # follow-up buttons alongside the new assistant message.
                    session_state.append_to_history(session_id, "user", user_input)
                    session_state.append_to_history(session_id, "assistant", response)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response,
                        "intent": intent, "sources": _src_data,
                    })
                    st.rerun()

                else:
                    # ── Non-streaming path (tickets, billing, guardrails) ──────
                    with st.spinner("Thinking..."):
                        response, intent = process_message(
                            user_input, settings, prompts,
                            rag_chain, intent_router, session_state,
                        )

                    # Check if a ticket was just created so we can animate it
                    _new_tkt = get_last_created_ticket()
                    if _new_tkt:
                        st.session_state["new_ticket_animation"] = True

                    # Log to analytics
                    session_state.log_query(session_id, intent, 0.92, user_input, is_gap=False)

                    st.html(intent_badge(intent))
                    st.markdown(response)

                    # Save and rerun for the non-streaming path.
                    st.session_state.messages.append({
                        "role": "assistant", "content": response, "intent": intent
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
