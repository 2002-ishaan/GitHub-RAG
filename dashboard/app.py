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
    handle_register_user,
    handle_upgrade_plan,
    is_ticket_flow_active,
    is_register_flow_active,
    get_ticket_flow_state,
    get_last_created_ticket,
    handle_close_tickets,
    handle_close_ticket_by_id,
    handle_list_accounts,
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
    "register_user":      ("👤 Register User",   "#0969DA", "#DDF4FF", "#B6E3FF"),
    "upgrade_plan":       ("⚡ Upgrade Plan",    "#9A6700", "#FFF8C5", "#E9C46A"),
    "list_accounts":      ("📋 List Accounts",   "#0969DA", "#DDF4FF", "#B6E3FF"),
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
        session_state.append_to_history(session_id, "user", user_input)
        session_state.append_to_history(session_id, "assistant", response)
        return response, "action_in_progress"

    # ── Check if multi-turn register flow is active ────────────────────────
    if is_register_flow_active(session_id):
        response = handle_register_user(
            session_id=session_id,
            user_message=user_input,
            session_state=session_state,
            prompts=prompts,
        )
        session_state.append_to_history(session_id, "user", user_input)
        session_state.append_to_history(session_id, "assistant", response)
        return response, "register_user"

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
        response = handle_check_billing(user_input, session_state)

    elif intent == "register_user":
        response = handle_register_user(
            session_id=session_id,
            user_message=user_input,
            session_state=session_state,
            prompts=prompts,
        )

    elif intent == "upgrade_plan":
        response = handle_upgrade_plan(user_input, session_state)

    # Added: Close ticket by ID intent
    elif intent == "close_ticket_by_id":
        response = handle_close_ticket_by_id(user_input, session_state)
    
    elif intent == "close_tickets":
        from agent.actions import handle_close_tickets
        response = handle_close_tickets(user_input, session_state)

    elif intent == "list_accounts":
        response = handle_list_accounts(session_state)

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

# Action commands are legitimately repeated (e.g. "check billing" before and after an
# upgrade). Never treat them as "repeated question" frustration signals.
_ACTION_COMMAND_RE = re.compile(
    r'\b(check|view|see)\s+(my\s+)?(billing|plan|subscription)\b'
    r'|\bcheck\s+(ticket|tkt)\b'
    r'|\b(upgrade|downgrade|change|switch)\s+\w'
    r'|\b(register|add|create|onboard)\s+(a\s+)?(new\s+)?(user|account)\b'
    r'|\blist\s+(all\s+)?(accounts?|users?)\b'
    r'|\bclose\s+(all\s+)?tickets?\b',
    re.IGNORECASE,
)

def _is_frustrated(user_input: str, messages: list) -> bool:
    """Detect frustration via keywords or same question asked twice in recent history."""
    lower = user_input.lower()
    for pat in _FRUSTRATION_PATTERNS:
        if re.search(pat, lower):
            return True
    # Action commands (billing, upgrade, register, etc.) are naturally repeated —
    # skip the similarity check so they never get misrouted as frustrated RAG queries.
    if _ACTION_COMMAND_RE.search(user_input):
        return False
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
def _generate_pdf(messages: list, session_id: str) -> bytes:
    """
    Generate a PDF of the conversation using fpdf2.
    - Markdown is converted to readable text (headers, tables, bullets preserved).
    - Intent badges shown as text labels.
    - User messages: blue-tinted box. Assistant messages: left-accent block.
    """
    import re as _re
    from fpdf import FPDF
    from datetime import datetime as _dt

    INTENT_LABELS = {
        "rag_query":          "Knowledge Query",
        "create_ticket":      "Create Ticket",
        "check_ticket":       "Check Ticket",
        "check_billing":      "Billing Check",
        "out_of_scope":       "Out of Scope",
        "prompt_injection":   "Blocked",
        "action_in_progress": "Ticket Flow",
        "close_tickets":      "Close Tickets",
        "close_ticket_by_id": "Close Ticket",
    }

    INTENT_RGB = {
        "rag_query":          (9,  105, 218),
        "create_ticket":      (26, 127, 55),
        "check_ticket":       (110, 64, 201),
        "check_billing":      (154, 103, 0),
        "out_of_scope":       (207, 34,  46),
        "prompt_injection":   (207, 34,  46),
        "action_in_progress": (26, 127,  55),
        "close_tickets":      (139, 148, 158),
        "close_ticket_by_id": (139, 148, 158),
    }

    def _strip_emoji(t: str) -> str:
        """
        Remove any character outside Latin-1 (0x00-0xFF).
        Helvetica is a core PDF font limited to that range.
        """
        return ''.join(c if ord(c) < 256 else '' for c in t)

    def _md_to_plain(text: str) -> str:
        """
        Convert markdown content to clean plain text that reads well in a PDF.
        Preserves headings (as ALLCAPS), bullet points, tables (grid), and separators.
        """
        lines_out = []
        in_table  = False
        table_buf = []

        def _flush_table(buf: list):
            """Convert collected | pipe | rows into aligned text columns."""
            rows = []
            for row in buf:
                cells = [c.strip() for c in row.strip('|').split('|')]
                rows.append(cells)
            # filter out separator rows (---)
            rows = [r for r in rows if not all(set(c) <= {'-', ' '} for c in r)]
            if not rows:
                return []
            col_w = [max(len(r[i]) if i < len(r) else 0 for r in rows)
                     for i in range(max(len(r) for r in rows))]
            out = []
            for ri, row in enumerate(rows):
                line = '  '.join(
                    (row[i] if i < len(row) else '').ljust(col_w[i])
                    for i in range(len(col_w))
                )
                out.append(line.rstrip())
                if ri == 0:
                    out.append('-' * min(sum(col_w) + 2 * len(col_w), 90))
            return out

        for raw_line in text.split('\n'):
            line = raw_line.rstrip()

            # Table rows
            if line.startswith('|'):
                in_table = True
                table_buf.append(line)
                continue
            elif in_table:
                lines_out.extend(_flush_table(table_buf))
                table_buf = []
                in_table = False

            stripped = line.strip()

            if not stripped:
                lines_out.append('')
                continue

            # Headings
            if stripped.startswith('### '):
                lines_out.append(stripped[4:].upper())
            elif stripped.startswith('## '):
                lines_out.append(stripped[3:].upper())
            elif stripped.startswith('# '):
                lines_out.append(stripped[2:].upper())
            # Horizontal rule
            elif _re.match(r'^-{3,}$', stripped):
                lines_out.append('-' * 60)
            # Bullet points
            elif stripped.startswith(('- ', '* ')):
                bullet_text = stripped[2:]
                bullet_text = _re.sub(r'\*\*(.+?)\*\*', r'\1', bullet_text)
                bullet_text = _re.sub(r'\*(.+?)\*',   r'\1', bullet_text)
                bullet_text = _re.sub(r'`(.+?)`',     r'\1', bullet_text)
                lines_out.append(f'  • {bullet_text}')
            # Numbered list
            elif _re.match(r'^\d+\.\s', stripped):
                item = _re.sub(r'^\d+\.\s+', '', stripped)
                item = _re.sub(r'\*\*(.+?)\*\*', r'\1', item)
                lines_out.append(f'  {item}')
            # Normal text — strip inline markdown
            else:
                cleaned = _re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                cleaned = _re.sub(r'\*(.+?)\*',   r'\1', cleaned)
                cleaned = _re.sub(r'`(.+?)`',     r'\1', cleaned)
                cleaned = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
                lines_out.append(cleaned)

        if in_table and table_buf:
            lines_out.extend(_flush_table(table_buf))

        # Collapse 3+ blank lines to 1
        result, prev_blank = [], False
        for ln in lines_out:
            if ln == '':
                if not prev_blank:
                    result.append('')
                prev_blank = True
            else:
                result.append(ln)
                prev_blank = False

        return '\n'.join(result).strip()

    # ── Build PDF ─────────────────────────────────────────────────────────────
    class ChatPDF(FPDF):
        def header(self):
            pass  # custom header below

    pdf = ChatPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    L  = pdf.l_margin   # left margin
    PW = pdf.w - L - pdf.r_margin  # printable width

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(31, 35, 40)
    pdf.cell(0, 9, "GitHub Documentation Assistant", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(101, 109, 118)
    pdf.cell(0, 5,
        f"Session: {session_id}   |   "
        f"Exported: {_dt.now().strftime('%Y-%m-%d %H:%M')}   |   "
        f"{len(messages)} messages",
        ln=True,
    )
    pdf.ln(2)
    pdf.set_draw_color(208, 215, 222)
    pdf.set_line_width(0.4)
    pdf.line(L, pdf.get_y(), L + PW, pdf.get_y())
    pdf.ln(5)

    # ── Messages ──────────────────────────────────────────────────────────────
    for msg in messages:
        role    = msg["role"]
        intent  = msg.get("intent") or ""
        content = msg.get("content") or ""

        plain = _strip_emoji(_md_to_plain(content))

        if role == "user":
            # ── User bubble ───────────────────────────────────────────────────
            pdf.set_fill_color(221, 244, 255)
            pdf.set_draw_color(182, 227, 255)
            pdf.set_text_color(9, 105, 218)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_line_width(0.3)
            pdf.cell(0, 6, "  YOU", ln=True, fill=True, border=1)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(31, 35, 40)
            pdf.set_fill_color(240, 248, 255)
            pdf.set_draw_color(182, 227, 255)
            pdf.multi_cell(0, 5.5, plain, fill=True, border="LRB")

        else:
            # ── Assistant header with intent label ────────────────────────────
            r, g, b = INTENT_RGB.get(intent, (101, 109, 118))
            label   = INTENT_LABELS.get(intent, intent.replace("_", " ").title() if intent else "")

            pdf.set_fill_color(246, 248, 250)
            pdf.set_draw_color(208, 215, 222)
            pdf.set_text_color(101, 109, 118)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_line_width(0.3)

            header_str = "  ASSISTANT"
            if label:
                header_str += f"   [{label}]"
            pdf.cell(0, 6, header_str, ln=True, fill=True, border=1)

            # Left-accent body block
            y_start = pdf.get_y()
            pdf.set_left_margin(L + 4)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(31, 35, 40)
            pdf.multi_cell(PW - 4, 5.5, plain, border=0)
            pdf.set_left_margin(L)
            y_end = pdf.get_y()

            # Draw colored left accent line
            pdf.set_draw_color(r, g, b)
            pdf.set_line_width(1.2)
            pdf.line(L, y_start, L, y_end)
            pdf.set_line_width(0.3)

        pdf.ln(3)
        pdf.set_draw_color(230, 232, 235)
        pdf.line(L, pdf.get_y(), L + PW, pdf.get_y())
        pdf.ln(4)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(145, 152, 161)
    pdf.cell(0, 5, "Generated by GitHub Docs Assistant", align="C", ln=True)

    return bytes(pdf.output())


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


# ── Apple design-system CSS ────────────────────────────────────────────────────
GITHUB_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS  (Apple.com aesthetic — 8-px grid, SF Pro)
   ─────────────────────────────────────────────────────────────────────────
   background  #FAFAFA   surface     #FFFFFF   border      #E5E5EA
   accent      #0071E3   accentHover #0077ED
   textPrimary #1D1D1F   textSec     #6E6E73   textMuted   #AEAEB2
   success     #34C759   warning     #FF9500   danger      #FF3B30
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Global reset & fonts ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #1D1D1F !important;
    background-color: #FAFAFA !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Text selection */
::selection {
    background: rgba(0, 113, 227, 0.20) !important;
    color: #0071E3 !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; display: none !important; }
.stDeployButton { display: none !important; }

/* ── App shell ────────────────────────────────────────────────────────── */
.stApp { background-color: #FAFAFA !important; }
.block-container {
    padding-top: 24px !important;
    padding-bottom: 32px !important;
    max-width: 100% !important;
}

/* Remove stray Streamlit box-shadows and borders on containers */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    box-shadow: none !important;
}

/* ── Custom scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #E5E5EA; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #AEAEB2; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #6E6E73; }

/* ═══════════════════════════════════════════════════════════════════════
   LEFT SIDEBAR
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E5EA !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #FFFFFF !important;
}

/* Sidebar nav items (buttons rendered inside sidebar) */
[data-testid="stSidebar"] .stButton > button {
    font-size: 13px !important;
    color: #1D1D1F !important;
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    text-align: left !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: background 0.12s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #F2F2F7 !important;
    color: #1D1D1F !important;
    border: none !important;
}

[data-testid="stSidebar"] .stButton > button:focus {
    background: #F2F2F7 !important;
    color: #0071E3 !important;
    font-weight: 500 !important;
    border: none !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   RIGHT SIDEBAR  (col_side — a normal st.column, not stSidebar)
   ═══════════════════════════════════════════════════════════════════════ */

/* Section header labels (h4 used as section headers in sidebar) */
h4 {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6E6E73 !important;
    margin-top: 20px !important;
    margin-bottom: 8px !important;
    padding: 0 !important;
    border: none !important;
}

/* Ticket expander cards */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1D1D1F !important;
    padding: 12px 14px !important;
    background: #FFFFFF !important;
}

[data-testid="stExpander"] summary:hover {
    background: #F9F9F9 !important;
}

/* Ticket priority left borders via data attribute on the expander container.
   We use class selectors on paragraphs inside since Streamlit doesn't expose
   priority as a data attribute. Target the first bold text color via :has. */
[data-testid="stExpander"]:has(p:contains("High")) {
    border-left: 3px solid #FF3B30 !important;
}
[data-testid="stExpander"]:has(p:contains("Medium")) {
    border-left: 3px solid #FF9500 !important;
}
[data-testid="stExpander"]:has(p:contains("Low")) {
    border-left: 3px solid #34C759 !important;
}

/* Ticket content typography */
[data-testid="stExpander"] [data-testid="stMarkdown"] code {
    font-family: "SF Mono", "Fira Code", monospace !important;
    font-size: 12px !important;
    color: #0071E3 !important;
    background: #F2F2F7 !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
}

/* Close-ticket danger buttons inside expanders */
[data-testid="stExpander"] .stButton > button {
    background: transparent !important;
    color: #FF3B30 !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 3px 10px !important;
    transition: all 0.12s !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] .stButton > button:hover {
    background: #FF3B30 !important;
    color: #FFFFFF !important;
    border-color: #FF3B30 !important;
}

/* Conversation list rows */
[data-testid="stVerticalBlock"] hr {
    border: none !important;
    border-bottom: 1px solid #F2F2F7 !important;
    margin: 0 !important;
}

/* Session ID chips (rendered as inline code in conversation list) */
[data-testid="stMarkdown"] code {
    font-family: "SF Mono", "Fira Code", monospace !important;
    font-size: 11px !important;
    background: #F2F2F7 !important;
    color: #0071E3 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    border: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   CHAT AREA
   ═══════════════════════════════════════════════════════════════════════ */
@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(3px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    animation: msgFadeIn 0.14s ease-out !important;
    border: none !important;
    background: transparent !important;
    padding: 6px 0 !important;
    box-shadow: none !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    background: #EBF5FF !important;
    border: 1px solid #B6D9FF !important;
    border-radius: 12px 12px 2px 12px !important;
    padding: 10px 14px !important;
    display: inline-block !important;
    max-width: 70% !important;
    margin-left: auto !important;
    font-size: 14px !important;
    color: #1D1D1F !important;
    line-height: 1.6 !important;
}

/* Assistant message — left blue accent bar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #0071E3 !important;
    padding-left: 14px !important;
    border-radius: 0 !important;
    background: transparent !important;
}

/* Avatar circles */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    flex-shrink: 0 !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: #0071E3 !important;
    color: #FFFFFF !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: #F2F2F7 !important;
    color: #0071E3 !important;
}

/* ── Chat input bar ────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    color: #1D1D1F !important;
    box-shadow: none !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #AEAEB2 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.12) !important;
    outline: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════════ */

/* Default / outlined buttons (example queries, action buttons) */
.stButton > button {
    background: #FFFFFF !important;
    color: #1D1D1F !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    font-weight: 400 !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
    transition: border-color 0.12s, color 0.12s !important;
}

.stButton > button:hover {
    border-color: #0071E3 !important;
    color: #0071E3 !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
}

/* Primary buttons (New Conversation, Analytics Dashboard) */
.stButton > button[kind="primary"],
.stButton > button[data-testid*="new_conv"],
.stButton > button[data-testid*="analytics"] {
    background: #0071E3 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
    transition: background 0.12s, transform 0.12s !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid*="new_conv"]:hover,
.stButton > button[data-testid*="analytics"]:hover {
    background: #0077ED !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* Follow-up suggestion chips (inside chat messages) */
[data-testid="stChatMessage"] .stButton > button {
    background: #FFFFFF !important;
    color: #0071E3 !important;
    border: 1px solid #B6D9FF !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 4px 12px !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
    transition: background 0.12s, border-color 0.12s !important;
}

[data-testid="stChatMessage"] .stButton > button:hover {
    background: #EBF5FF !important;
    border-color: #0071E3 !important;
    color: #0071E3 !important;
    transform: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   TYPOGRAPHY & MARKDOWN
   ═══════════════════════════════════════════════════════════════════════ */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-weight: 600 !important;
    color: #1D1D1F !important;
    margin-top: 8px !important;
    margin-bottom: 4px !important;
    line-height: 1.3 !important;
}

.stMarkdown h1 { font-size: 22px !important; }
.stMarkdown h2 { font-size: 18px !important; border: none !important; padding: 0 !important; }
.stMarkdown h3 { font-size: 15px !important; }

.stMarkdown p {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #1D1D1F !important;
    margin-bottom: 8px !important;
}

.stMarkdown a {
    color: #0071E3 !important;
    text-decoration: none !important;
}

.stMarkdown a:hover { text-decoration: underline !important; }

.stMarkdown ul li, .stMarkdown ol li {
    line-height: 1.6 !important;
    margin-bottom: 4px !important;
    font-size: 14px !important;
    color: #1D1D1F !important;
}

/* Inline code */
.stMarkdown code, code {
    font-family: "SF Mono", "Fira Code", monospace !important;
    font-size: 12px !important;
    background: #F2F2F7 !important;
    color: #1D1D1F !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
    border: none !important;
}

/* Code blocks */
.stMarkdown pre {
    background: #F2F2F7 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    border: 1px solid #E5E5EA !important;
    overflow-x: auto !important;
}

.stMarkdown pre code {
    background: transparent !important;
    padding: 0 !important;
    font-size: 12px !important;
    color: #1D1D1F !important;
}

/* Captions & secondary text */
.stCaption, small, .stMarkdown small {
    font-size: 11px !important;
    color: #AEAEB2 !important;
    line-height: 1.5 !important;
}

/* ── Markdown tables ────────────────────────────────────────────────────── */
.stMarkdown table, table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
    margin-bottom: 12px !important;
}

.stMarkdown th, th {
    background: #F2F2F7 !important;
    color: #1D1D1F !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 8px 12px !important;
    border-bottom: 2px solid #E5E5EA !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
}

.stMarkdown td, td {
    padding: 8px 12px !important;
    border-bottom: 1px solid #F2F2F7 !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    color: #1D1D1F !important;
    vertical-align: top !important;
}

.stMarkdown tr:last-child td, tr:last-child td {
    border-bottom: none !important;
}

/* ── Horizontal rules ───────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E5E5EA !important;
    margin: 12px 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] {
    background: #F2F2F7 !important;
    border-radius: 4px !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stProgress"] > div > div {
    background: #0071E3 !important;
    border-radius: 4px !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   SOURCES SECTION  (rendered as stMarkdown below chat messages)
   ═══════════════════════════════════════════════════════════════════════ */

/* "Sources" label — matched via h6 used in source rendering */
.stMarkdown h6 {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #AEAEB2 !important;
    margin: 12px 0 8px !important;
    border: none !important;
    padding: 0 !important;
}

/* Source link chips */
.stMarkdown h6 + p a, .stMarkdown h6 ~ p a {
    color: #0071E3 !important;
    font-size: 13px !important;
    text-decoration: none !important;
}

.stMarkdown h6 + p a:hover, .stMarkdown h6 ~ p a:hover {
    text-decoration: underline !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   INTENT BADGE SPANS  (injected via st.html / intent_badge())
   — These are raw HTML spans so Streamlit doesn't interfere.
   Styles here reinforce the inline style with cascade specificity.
   ═══════════════════════════════════════════════════════════════════════ */
span[style*="border-radius"] {
    border-radius: 20px !important;
    padding: 3px 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    display: inline-block !important;
    margin-bottom: 4px !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   MISCELLANEOUS STREAMLIT WIDGETS
   ═══════════════════════════════════════════════════════════════════════ */

/* Remove stray input/select widget borders */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    border: 1px solid #E5E5EA !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    color: #1D1D1F !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.12) !important;
}

/* Spinner text */
.stSpinner p {
    font-size: 13px !important;
    color: #6E6E73 !important;
}

/* Alert / info / warning / error boxes */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}

/* Columns — no extra padding bleed */
[data-testid="column"] {
    padding: 0 8px !important;
}

[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
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
                _pdf_bytes = _generate_pdf(st.session_state.messages, session_id)
                st.download_button(
                    "📄",
                    data=_pdf_bytes,
                    file_name=f"conversation_{session_id}.pdf",
                    mime="application/pdf",
                    key="export_download",
                    help="Download conversation as PDF",
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
                    "register_user", "upgrade_plan", "list_accounts",
                    "out_of_scope", "prompt_injection",
                }

                # Skip streaming entirely when any multi-turn flow is active —
                # those messages must always reach process_message() directly.
                _any_flow_active = (
                    is_ticket_flow_active(session_id)
                    or is_register_flow_active(session_id)
                )
                if not _any_flow_active:
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
