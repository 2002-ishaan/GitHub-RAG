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

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from configs.settings import load_settings, load_prompts, setup_logging
from generation.rag_chain import RAGChain
from agent.intent_router import IntentRouter
from agent.session_state import SessionState
from agent.actions import (
    handle_create_ticket,
    handle_check_ticket,
    handle_check_billing,
    is_ticket_flow_active,
)
from agent.guardrails import get_guardrail_response, handle_insufficient_evidence


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Docs Assistant",
    page_icon="🐙",
    layout="wide",
)

# ── Intent label styling ───────────────────────────────────────────────────────
INTENT_LABELS = {
    "rag_query":        ("🔍 Knowledge Query",  "#0969DA", "#DDF4FF"),
    "create_ticket":    ("🎫 Create Ticket",     "#1A7F37", "#DAFBE1"),
    "check_ticket":     ("📋 Check Ticket",      "#6E40C9", "#F3EEFF"),
    "check_billing":    ("💳 Billing Check",     "#9A6700", "#FFF8C5"),
    "out_of_scope":     ("🚫 Out of Scope",      "#CF222E", "#FFEBE9"),
    "prompt_injection": ("⚠️ Blocked",           "#CF222E", "#FFEBE9"),
    "action_in_progress": ("🎫 Ticket Flow",     "#1A7F37", "#DAFBE1"),
    "close_tickets": ("🔒 Close Tickets", "#CF222E", "#FFEBE9"),
}


def intent_badge(intent: str) -> str:
    label, color, bg = INTENT_LABELS.get(intent, ("❓ Unknown", "#666", "#eee"))
    return (
        f'<span style="background:{bg};color:{color};padding:2px 8px;'
        f'border-radius:12px;font-size:0.75em;font-weight:600;'
        f'border:1px solid {color};">{label}</span>'
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

    elif intent == "check_ticket":
        response = handle_check_ticket(user_input, session_state)

    elif intent == "check_billing":
        response = handle_check_billing(user_input)
    
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


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    init_session()

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
        st.markdown("### 🐙 GitHub Docs Agent")
        st.caption(f"Session: `{session_id}`")
        st.divider()

        # Ticket panel
        st.markdown("#### 🎫 Your Tickets")
        tickets = session_state.list_tickets()
        if tickets:
            for t in tickets[:5]:
                status_icon = "📂" if t["status"] == "Open" else "✅"
                pri_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(t["priority"], "⚪")
                st.markdown(
                    f"**{t['ticket_id']}** {status_icon}  \n"
                    f"{t['category']} {pri_icon}  \n"
                    f"*{t['created_at'][:10]}*"
                )
                st.divider()
        else:
            st.caption("No tickets yet. Say *\"Create a support ticket\"* to start.")

        st.divider()

        # New session button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()

        st.divider()
        st.markdown("**Try these:**")
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
        st.markdown("## 🐙 GitHub Documentation Assistant")
        st.caption(
            "Ask anything about GitHub — repositories, billing, authentication, "
            "security, organizations. I can also create and check support tickets."
        )

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("intent"):
                    st.markdown(
                        intent_badge(msg["intent"]),
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Ask a GitHub question or request an action..."):
            # Show user message immediately
            st.session_state.messages.append({
                "role": "user", "content": user_input, "intent": None
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, intent = process_message(
                        user_input, settings, prompts,
                        rag_chain, intent_router, session_state,
                    )

                st.markdown(intent_badge(intent), unsafe_allow_html=True)
                st.markdown("")
                st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant", "content": response, "intent": intent
            })
            st.rerun()


if __name__ == "__main__":
    main()
