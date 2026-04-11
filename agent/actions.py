"""
agent/actions.py
────────────────────────────────────────────────────────────────
All 3 mock actions for the GitHub Documentation Assistant.

ACTION 1 — create_ticket (MULTI-TURN)
    Collects: category → description → priority
    Saves to SQLite, returns ticket ID

ACTION 2 — check_ticket (SINGLE-TURN)
    Input: ticket_id (e.g. TKT-001)
    Reads from SQLite, returns full ticket details

ACTION 3 — check_billing (SINGLE-TURN)
    Input: username (mock)
    Reads from mock JSON database, returns plan details
────────────────────────────────────────────────────────────────
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

# ── Mock billing database ──────────────────────────────────────────────────────

MOCK_ACCOUNTS = {
    "alice":   {"plan": "Pro",        "price": "$4/month",  "seats": 1,   "actions_minutes": 3000,  "storage_gb": 2},
    "bob":     {"plan": "Team",       "price": "$4/user/mo","seats": 5,   "actions_minutes": 50000, "storage_gb": 2},
    "carol":   {"plan": "Enterprise", "price": "Custom",    "seats": 100, "actions_minutes": 50000, "storage_gb": 50},
    "dave":    {"plan": "Free",       "price": "$0/month",  "seats": 1,   "actions_minutes": 2000,  "storage_gb": 0.5},
    "default": {"plan": "Free",       "price": "$0/month",  "seats": 1,   "actions_minutes": 2000,  "storage_gb": 0.5},
}

VALID_CATEGORIES = {
    "1": "Account & Authentication",
    "2": "Billing & Payments",
    "3": "Repository Issues",
    "4": "Organizations & Teams",
    "5": "Security & 2FA",
    "6": "Other",
    "account": "Account & Authentication",
    "authentication": "Account & Authentication",
    "billing": "Billing & Payments",
    "payments": "Billing & Payments",
    "repository": "Repository Issues",
    "repo": "Repository Issues",
    "organizations": "Organizations & Teams",
    "teams": "Organizations & Teams",
    "security": "Security & 2FA",
    "2fa": "Security & 2FA",
    "other": "Other",
}

VALID_PRIORITIES = {
    "high":   "High",
    "medium": "Medium",
    "med":    "Medium",
    "low":    "Low",
    "1":      "High",
    "2":      "Medium",
    "3":      "Low",
    "blocking": "High",
    "urgent":   "High",
    "minor":    "Low",
}


# ── Multi-turn ticket state ────────────────────────────────────────────────────

@dataclass
class TicketState:
    """Tracks progress of the multi-turn ticket creation flow."""
    step:        str            = "category"   # category → description → priority → done
    category:    Optional[str] = None
    description: Optional[str] = None
    priority:    Optional[str] = None

    def is_complete(self) -> bool:
        return (
            self.category is not None and
            self.description is not None and
            self.priority is not None
        )


# ── In-memory store for active ticket flows ───────────────────────────────────
# Key: session_id, Value: TicketState
_active_ticket_flows: dict[str, TicketState] = {}

# Stores info about the ticket most recently confirmed, so app.py can render a card.
_last_created_ticket: dict = {}

CANCEL_WORDS = {"cancel", "quit", "exit", "stop", "abort", "nevermind", "never mind"}
CONFIRM_WORDS = {"yes", "confirm", "submit", "ok", "looks good", "correct",
                 "proceed", "go ahead", "yep", "yeah", "y", "sure", "done"}
EDIT_WORDS    = {"edit", "no", "change", "redo", "restart", "modify", "update", "back"}


def get_ticket_flow_state(session_id: str) -> Optional[TicketState]:
    """Return the current TicketState for a session, or None if no flow is active."""
    return _active_ticket_flows.get(session_id)


def get_last_created_ticket() -> dict:
    """Return info about the most recently created ticket, then clear it."""
    result = dict(_last_created_ticket)
    _last_created_ticket.clear()
    return result


# ── Action 1: Create Support Ticket (Multi-turn) ───────────────────────────────

def handle_create_ticket(
    session_id: str,
    user_message: str,
    session_state,       # SessionState instance
    prompts: dict,
) -> str:
    """
    Multi-turn ticket creation.
    Steps: category → description → priority → review → confirm → save
    Cancel is accepted at any step.
    """

    # Get or create the ticket flow for this session
    if session_id not in _active_ticket_flows:
        _active_ticket_flows[session_id] = TicketState()

    state = _active_ticket_flows[session_id]
    msg   = user_message.strip().lower()

    # ── Escape hatch — cancel at any point ────────────────────────────────
    if msg in CANCEL_WORDS or any(w in msg.split() for w in CANCEL_WORDS):
        del _active_ticket_flows[session_id]
        return (
            "❌ **Ticket creation cancelled.**\n\n"
            "No problem — let me know if you need anything else, "
            "or say *\"Create a support ticket\"* whenever you're ready."
        )

    # ── Step 1: Prompt for category ────────────────────────────────────────
    if state.step == "category":
        state.step = "awaiting_category"
        return prompts["ticket_prompt"]["collecting_category"]

    if state.step == "awaiting_category":
        category = VALID_CATEGORIES.get(msg)
        if not category:
            for key, val in VALID_CATEGORIES.items():
                if key in msg:
                    category = val
                    break

        if not category:
            return (
                "I didn't recognise that category. Please choose one:\n\n"
                "1. Account & Authentication\n"
                "2. Billing & Payments\n"
                "3. Repository Issues\n"
                "4. Organizations & Teams\n"
                "5. Security & 2FA\n"
                "6. Other"
            )

        state.category = category
        state.step     = "awaiting_description"
        return prompts["ticket_prompt"]["collecting_description"].format(
            category=category
        )

    # ── Step 2: Collect description ────────────────────────────────────────
    if state.step == "awaiting_description":
        if len(user_message.strip()) < 10:
            return "Please provide a more detailed description (at least a sentence)."

        state.description = user_message.strip()
        state.step        = "awaiting_priority"
        return prompts["ticket_prompt"]["collecting_priority"]

    # ── Step 3: Collect priority ───────────────────────────────────────────
    if state.step == "awaiting_priority":
        priority = VALID_PRIORITIES.get(msg)
        if not priority:
            for key, val in VALID_PRIORITIES.items():
                if key in msg:
                    priority = val
                    break

        if not priority:
            return (
                "Please choose a priority:\n"
                "- **High** — blocking my work completely\n"
                "- **Medium** — causing problems but workable\n"
                "- **Low** — minor inconvenience"
            )

        state.priority = priority
        state.step     = "review"   # ← review before saving

        pri_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚪")
        return (
            f"**Here's your ticket — review before submitting:**\n\n"
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Category | {state.category} |\n"
            f"| Priority | {pri_emoji} {priority} |\n\n"
            f"**Description:** {state.description}\n\n"
            f"Confirm to submit, Edit to start over, or Cancel to exit."
        )

    # ── Review step: confirm or edit ───────────────────────────────────────
    if state.step == "review":
        if msg in CONFIRM_WORDS or any(w in msg.split() for w in CONFIRM_WORDS):
            ticket_id = session_state.create_ticket(
                session_id=session_id,
                category=state.category,
                description=state.description,
                priority=state.priority,
            )
            # Store for app.py to pick up and render a card
            _last_created_ticket.update({
                "ticket_id":   ticket_id,
                "category":    state.category,
                "description": state.description,
                "priority":    state.priority,
            })
            del _active_ticket_flows[session_id]
            return prompts["ticket_prompt"]["confirmation"].format(
                ticket_id=ticket_id,
                category=state.category,
                priority=state.priority,
            )

        elif msg in EDIT_WORDS or any(w in msg.split() for w in EDIT_WORDS):
            state.step        = "awaiting_category"
            state.category    = None
            state.description = None
            state.priority    = None
            return (
                "No problem — let's start over.\n\n"
                + prompts["ticket_prompt"]["collecting_category"]
            )

        else:
            return (
                "Please respond with:\n"
                "- **Yes** to submit the ticket\n"
                "- **Edit** to start over\n"
                "- **Cancel** to exit"
            )

    # Fallback — restart the flow
    _active_ticket_flows[session_id] = TicketState()
    return prompts["ticket_prompt"]["collecting_category"]


def is_ticket_flow_active(session_id: str) -> bool:
    """Check if a multi-turn ticket flow is in progress."""
    return session_id in _active_ticket_flows


# ── Action 2: Check Ticket Status (Single-turn) ────────────────────────────────

#--------------------Version 1 without RAG response ----------------------
# def handle_check_ticket(user_message: str, session_state) -> str:
#     """
#     Look up a ticket by ID.
#     Extracts ticket ID from the message (e.g. 'TKT-001', 'tkt001').
#     """
#     # Try to extract ticket ID from message
#     ticket_id = None

#     patterns = [
#         r"\b(TKT[-\s]?\d+)\b",   # TKT-001 or TKT 001
#         r"\b(tkt[-\s]?\d+)\b",   # lowercase
#         r"\bticket\s+#?(\d+)\b", # "ticket 1" or "ticket #1"
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, user_message, re.IGNORECASE)
#         if match:
#             raw = match.group(1).upper().replace(" ", "-")
#             # Normalise: TKT001 → TKT-001
#             ticket_id = re.sub(r"TKT(\d+)", r"TKT-\1", raw)
#             break

#     if not ticket_id:
#         return (
#             "Please provide a ticket ID. For example:\n"
#             "*\"Check ticket TKT-001\"*"
#         )

#     ticket = session_state.get_ticket(ticket_id)

#     if not ticket:
#         return (
#             f"❌ Ticket **{ticket_id}** was not found.\n\n"
#             f"Double-check the ticket ID or create a new ticket by saying "
#             f"*\"Create a support ticket\"*."
#         )

#     # Format the response nicely
#     priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(
#         ticket["priority"], "⚪"
#     )
#     status_emoji = {"Open": "📂", "Resolved": "✅", "Closed": "🔒"}.get(
#         ticket["status"], "📂"
#     )

#     return (
#         f"## Ticket {ticket['ticket_id']}\n\n"
#         f"| Field | Value |\n"
#         f"|---|---|\n"
#         f"| Status | {status_emoji} {ticket['status']} |\n"
#         f"| Category | {ticket['category']} |\n"
#         f"| Priority | {priority_emoji} {ticket['priority']} |\n"
#         f"| Created | {ticket['created_at'][:10]} |\n\n"
#         f"**Description:**\n{ticket['description']}"
#     )

#--------------------Version 2 with RAG response ----------------------
def handle_check_ticket(user_message: str, session_state, rag_chain=None) -> str:
    ticket_id = None

    patterns = [
        r"\b(TKT[-\s]?\d+)\b",
        r"\b(tkt[-\s]?\d+)\b",
        r"\bticket\s+#?(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            raw = match.group(1).upper().replace(" ", "-")
            ticket_id = re.sub(r"TKT(\d+)", r"TKT-\1", raw)
            break

    if not ticket_id:
        return (
            "Please provide a ticket ID. For example:\n"
            "*\"Check ticket TKT-001\"*"
        )

    ticket = session_state.get_ticket(ticket_id)

    if not ticket:
        return (
            f"❌ Ticket **{ticket_id}** was not found.\n\n"
            f"Double-check the ticket ID or create a new ticket by saying "
            f"*\"Create a support ticket\"*."
        )

    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(ticket["priority"], "⚪")
    status_emoji = {"Open": "📂", "Resolved": "✅", "Closed": "🔒"}.get(ticket["status"], "📂")

    # ── Save to variable instead of returning immediately ──
    ticket_response = (
        f"## Ticket {ticket['ticket_id']}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Status | {status_emoji} {ticket['status']} |\n"
        f"| Category | {ticket['category']} |\n"
        f"| Priority | {priority_emoji} {ticket['priority']} |\n"
        f"| Created | {ticket['created_at'][:10]} |\n\n"
        f"**Description:**\n{ticket['description']}"
    )

    # ── Append RAG guidance if available ──────────────────
    if rag_chain and ticket:
        rag_response = rag_chain.ask(question=ticket["description"])
        if rag_response.is_supported:
            ticket_response += (
                "\n\n---\n\n"
                "💡 **Suggested guidance based on your issue:**\n\n"
                + rag_response.formatted_answer()
            )

    return ticket_response  # ← single return at the end

# ── Action 3: Check Billing Plan (Single-turn) ─────────────────────────────────

def handle_check_billing(user_message: str) -> str:
    """
    Look up a mock account's billing plan.
    Tries to extract a username from the message.
    """
    # Try to extract username from message
    # Patterns: "check alice's plan", "what plan is bob on", "billing for carol"
    username = None

    for word in user_message.lower().split():
        clean = re.sub(r"[^a-z0-9]", "", word)
        if clean in MOCK_ACCOUNTS and clean != "default":
            username = clean
            break

    if not username:
        username = "default"
        note = (
            "\n\n> *No username found in your message. "
            "Showing the default Free plan. "
            "Try: \"Check billing for alice\" to see specific accounts.*"
        )
    else:
        note = ""

    account = MOCK_ACCOUNTS[username]
    plan    = account["plan"]

    plan_emoji = {
        "Free":       "🆓",
        "Pro":        "⭐",
        "Team":       "👥",
        "Enterprise": "🏢",
    }.get(plan, "📋")

    return (
        f"## {plan_emoji} GitHub {plan} Plan\n\n"
        f"| Feature | Value |\n"
        f"|---|---|\n"
        f"| Plan | **{plan}** |\n"
        f"| Price | {account['price']} |\n"
        f"| Seats | {account['seats']} |\n"
        f"| Actions minutes/month | {account['actions_minutes']:,} |\n"
        f"| Storage | {account['storage_gb']} GB |\n"
        f"{note}\n\n"
        f"For more details, visit: https://github.com/pricing"
    )

# close tickets by id
def handle_close_ticket_by_id(user_message: str, session_state) -> str:
    """Close a single ticket by ID extracted from the message."""
    ticket_id = None
    patterns = [
        r"\b(TKT[-\s]?\d+)\b",
        r"\b(tkt[-\s]?\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            raw = match.group(1).upper().replace(" ", "-")
            ticket_id = re.sub(r"TKT(\d+)", r"TKT-\1", raw)
            break

    if not ticket_id:
        return (
            "Please specify a ticket ID. For example:\n"
            "*\"Close ticket TKT-001\"*"
        )

    result = session_state.close_ticket_by_id(ticket_id)
    return result["message"]

def handle_close_tickets(user_message: str, session_state) -> str:
    """Close all open tickets."""
    count = session_state.close_all_tickets()
    if count == 0:
        return "✅ No open tickets to close."
    return (
        f"✅ **{count} ticket{'s' if count > 1 else ''} closed successfully.**\n\n"
        f"All your tickets have been marked as Closed."
    )