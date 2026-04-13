"""
agent/actions.py
────────────────────────────────────────────────────────────────
Actions for the GitHub Documentation Assistant.

ACTION 1 — create_ticket    (MULTI-TURN)
ACTION 2 — check_ticket     (SINGLE-TURN)
ACTION 3 — check_billing    (SINGLE-TURN) — reads from SQLite users table
ACTION 4 — register_user    (MULTI-TURN)  — collects username + plan, saves to SQLite
ACTION 5 — upgrade_plan     (SINGLE-TURN) — updates existing user's plan in SQLite
────────────────────────────────────────────────────────────────
"""

import re
from dataclasses import dataclass
from typing import Optional

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

def handle_check_billing(user_message: str, session_state, session_id: Optional[str] = None) -> str:
    """
    Look up an account's billing plan from SQLite.
    Tries to extract a username from the message.
    """
    # Try to extract username from message.
    # Patterns: "check alice's plan", "what plan is bob on", "billing for carol"
    # We track any candidate name we tried so we can give a "not found" error.
    SKIP_BILLING_WORDS = {
        "check", "view", "see", "billing", "plan", "subscription",
        "what", "is", "for", "my", "the", "on", "using", "account",
        "github", "show", "get", "details",
        # First-person pronouns — "what plan am I on?" must not extract "am" or "i"
        "am", "i", "a",
        # Informational words — must never be extracted as usernames
        "how", "where", "do", "does", "can", "cancel", "cancellation", "refund",
        "request", "cost", "price", "pricing", "fee", "fees", "charge",
        "downgrade", "upgrade", "change", "switch", "much", "if",
        "will", "lose", "have", "need", "unsubscribe", "happens",
        "invoice", "invoices", "receipt", "receipts", "find", "access",
        "download", "history", "see", "locate", "payment",
    }
    username      = None
    candidate_name = None   # the word we tried that wasn't in DB
    lower_msg = user_message.lower()

    # If this is clearly a billing-support issue, do not attempt username extraction.
    # Route users to ticket flow guidance instead of fabricated "user not found" names.
    support_issue_re = re.compile(
        r"\b(refund|dispute|disputed|charged|charge\s+twice|double\s+charge|receipt|invoice|payment\s+issue|billing\s+problem|correction)\b",
        re.IGNORECASE,
    )
    if support_issue_re.search(lower_msg):
        return (
            "I can help with this billing issue. For account-specific disputes, refunds, or receipt corrections, "
            "please create a **Billing & Payments** support ticket so it can be reviewed safely.\n\n"
            "Say: *\"Create a support ticket\"*"
        )

    # Session pronoun resolution — "my plan", "am I", "what plan am I on"
    if session_id and re.search(r"\b(my|me|mine|current_user|am\s+i)\b", lower_msg):
        resolved = session_state.get_current_user(session_id)
        if resolved:
            username = resolved

    if not username:
        for word in lower_msg.split():
            clean = re.sub(r"[^a-z0-9]", "", word)
            if not clean or len(clean) < 2 or clean in SKIP_BILLING_WORDS:
                continue
            candidate_name = clean
            user = session_state.get_user(clean)
            if user:
                username = clean
                break

    if not username:
        # Build live user list from SQLite (includes newly registered users)
        all_users = ", ".join(u["username"] for u in session_state.list_users()) or "none yet"
        if candidate_name:
            return (
                f"❌ **User '{candidate_name}' not found.**\n\n"
                f"No account exists for **{candidate_name}** in the system.\n"
                f"Registered users: {all_users}.\n\n"
                f"To create one, say: *\"Register a new account for {candidate_name}\"*"
            )
        return (
            f"Please specify a username. For example:\n"
            f"*\"Check billing for alice\"*\n\n"
            f"Registered users: {all_users}."
        )

    account = session_state.get_user(username)
    plan    = account["plan"]

    if session_id:
        session_state.set_current_user(session_id, username)

    plan_emoji = {"Free": "🆓", "Pro": "⭐", "Team": "👥", "Enterprise": "🏢"}.get(plan, "📋")

    return (
        f"## {plan_emoji} GitHub {plan} Plan — {username.title()}\n\n"
        f"| Feature | Value |\n"
        f"|---|---|\n"
        f"| Username | {username} |\n"
        f"| Plan | **{plan}** |\n"
        f"| Price | {account['price']} |\n"
        f"| Seats | {account['seats']} |\n"
        f"| Actions minutes/month | {account['actions_minutes']:,} |\n"
        f"| Storage | {account['storage_gb']} GB |\n"
        f"| Member since | {account['joined_date']} |\n\n"
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

def handle_close_tickets(session_state) -> str:
    """Close all open tickets."""
    count = session_state.close_all_tickets()
    if count == 0:
        return "✅ No open tickets to close."
    return (
        f"✅ **{count} ticket{'s' if count > 1 else ''} closed successfully.**\n\n"
        f"All your tickets have been marked as Closed."
    )


# ── Register flow state ────────────────────────────────────────────────────────

@dataclass
class RegisterState:
    """Tracks progress of the multi-turn user registration flow."""
    step:     str           = "awaiting_username"   # awaiting_username → awaiting_plan → review
    username: Optional[str] = None
    plan:     Optional[str] = None


_active_register_flows: dict[str, RegisterState] = {}

VALID_PLANS = {
    "free":       "Free",
    "pro":        "Pro",
    "team":       "Team",
    "enterprise": "Enterprise",
    "1":          "Free",
    "2":          "Pro",
    "3":          "Team",
    "4":          "Enterprise",
}


def is_register_flow_active(session_id: str) -> bool:
    return session_id in _active_register_flows


def get_register_flow_state(session_id: str) -> Optional[RegisterState]:
    return _active_register_flows.get(session_id)


_REGISTER_SKIP_WORDS = {
    "register", "add", "create", "sign", "up", "new", "user", "account",
    "for", "a", "an", "the", "please", "onboard", "me", "my",
}


def _extract_username_from_trigger(message: str) -> Optional[str]:
    """
    Try to pull a username out of the intent-trigger message.
    e.g. "Register a new account for michael" → "michael"
         "Add user john-doe" → "john-doe"
    Returns None if no clean candidate found.
    """
    # Keep alphanumeric + hyphens only, split on spaces
    words = re.sub(r"[^a-z0-9\-\s]", "", message.lower()).split()
    candidates = [w for w in words if w not in _REGISTER_SKIP_WORDS and len(w) >= 2]
    if candidates:
        return candidates[-1]   # username usually comes last ("for michael")
    return None


# ── Action 4: Register User (Multi-turn) ──────────────────────────────────────

def handle_register_user(
    session_id: str,
    user_message: str,
    session_state,
    prompts: dict,
) -> str:
    """
    Multi-turn user registration.
    Steps: (extract username from trigger OR ask) → ask plan → review → confirm → save.
    """
    fresh_start = session_id not in _active_register_flows

    if fresh_start:
        _active_register_flows[session_id] = RegisterState()
        state = _active_register_flows[session_id]

        # Try to pull username out of the trigger message
        # e.g. "Register a new account for michael" → username = "michael"
        extracted = _extract_username_from_trigger(user_message)
        if extracted:
            if session_state.get_user(extracted):
                del _active_register_flows[session_id]
                return (
                    f"❌ Username **{extracted}** is already registered.\n\n"
                    f"Say *\"Check billing for {extracted}\"* to view their plan, "
                    f"or *\"Upgrade {extracted} to <plan>\"* to change it."
                )
            state.username = extracted
            state.step     = "awaiting_plan"
            return prompts["register_prompt"]["collecting_plan"].format(username=extracted)
        else:
            # No username in trigger — ask for it
            state.step = "awaiting_username"
            return (
                "What username would you like to register?\n\n"
                "*(Letters, numbers, and hyphens only — 2 to 39 characters)*"
            )

    state = _active_register_flows[session_id]
    msg   = user_message.strip().lower()

    # Escape hatch
    if msg in CANCEL_WORDS or any(w in msg.split() for w in CANCEL_WORDS):
        del _active_register_flows[session_id]
        return (
            "❌ **Registration cancelled.**\n\n"
            "Let me know if you'd like to register later."
        )

    # ── Step 1: collect username (only reached when trigger had no username) ──
    if state.step == "awaiting_username":
        raw_username = user_message.strip()
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-]{0,37}[a-zA-Z0-9]$|^[a-zA-Z0-9]{2}$', raw_username):
            return (
                "That doesn't look like a valid GitHub username. "
                "Usernames can only contain letters, numbers, and hyphens, "
                "and must be 2–39 characters.\n\nPlease try again:"
            )
        if session_state.get_user(raw_username.lower()):
            return (
                f"❌ Username **{raw_username}** is already registered.\n\n"
                f"Try a different username."
            )
        state.username = raw_username.lower()
        state.step     = "awaiting_plan"
        return prompts["register_prompt"]["collecting_plan"].format(username=state.username)

    # ── Step 2: collect plan ───────────────────────────────────────────────
    if state.step == "awaiting_plan":
        plan = VALID_PLANS.get(msg)
        if not plan:
            for key, val in VALID_PLANS.items():
                if key in msg:
                    plan = val
                    break

        if not plan:
            return (
                "Please choose a plan:\n\n"
                "1. **Free** — $0/month · 2,000 Actions min · 0.5 GB storage\n"
                "2. **Pro** — $4/month · 3,000 Actions min · 2 GB storage\n"
                "3. **Team** — $4/user/mo · 50,000 Actions min · 2 GB storage\n"
                "4. **Enterprise** — Custom pricing · 50,000 Actions min · 50 GB storage"
            )

        state.plan = plan
        state.step = "review"

        defaults = session_state.PLAN_PRICING[plan]
        return (
            f"Here's your account summary:\n\n"
            f"- **Username:** {state.username}\n"
            f"- **Plan:** {plan} ({defaults['price']})\n"
            f"- **Storage:** {defaults['storage_gb']} GB\n"
            f"- **Actions:** {defaults['actions_minutes']:,} min/month\n\n"
            f"Type **confirm** to register, **edit** to change plan, or **cancel** to exit."
        )

    # ── Review step ────────────────────────────────────────────────────────
    if state.step == "review":
        if msg in CONFIRM_WORDS or any(w in msg.split() for w in CONFIRM_WORDS):
            try:
                user = session_state.create_user(state.username, state.plan)
            except ValueError as e:
                del _active_register_flows[session_id]
                return f"❌ {e}"

            session_state.set_current_user(session_id, user["username"])

            del _active_register_flows[session_id]
            return prompts["register_prompt"]["confirmation"].format(
                username=user["username"],
                plan=user["plan"],
                price=user["price"],
                actions_minutes=f"{user['actions_minutes']:,}",
                storage_gb=user["storage_gb"],
                joined_date=user["joined_date"],
            )

        elif msg in EDIT_WORDS or any(w in msg.split() for w in EDIT_WORDS):
            state.step = "awaiting_plan"
            state.plan = None
            return (
                "No problem — which plan would you like?\n\n"
                "1. **Free**  2. **Pro**  3. **Team**  4. **Enterprise**"
            )

        else:
            return (
                "Please respond with:\n"
                "- **confirm** to register\n"
                "- **edit** to choose a different plan\n"
                "- **cancel** to exit"
            )

    # Fallback
    _active_register_flows[session_id] = RegisterState()
    return prompts["register_prompt"]["collecting_plan"].format(username="your account")


# ── Action 5: Upgrade Plan (Single-turn) ──────────────────────────────────────

def handle_upgrade_plan(user_message: str, session_state, session_id: Optional[str] = None) -> str:
    """
    Parse "upgrade <username> to <plan>" and update the user's plan in SQLite.
    Returns a before/after comparison card.
    """
    msg = user_message.lower()

    # Extract plan from message
    new_plan = None
    for key, val in VALID_PLANS.items():
        if re.search(rf'\b{re.escape(key)}\b', msg):
            new_plan = val
            break

    # Extract username — word that isn't a plan keyword or common verb
    SKIP_WORDS = {
        "upgrade", "downgrade", "change", "switch", "update", "plan",
        "to", "from", "for", "the", "my", "their", "a", "an",
        "free", "pro", "team", "enterprise",
        "1", "2", "3", "4",
    }
    username = None
    for word in re.sub(r"[^a-z0-9\s]", "", msg).split():
        if word not in SKIP_WORDS and len(word) >= 2:
            if session_state.get_user(word):
                username = word
                break

    if not username:
        return (
            "Please specify a username. For example:\n"
            "*\"Upgrade alice to Enterprise\"*"
        )

    if not new_plan:
        return (
            "Please specify a plan (Free, Pro, Team, or Enterprise). For example:\n"
            "*\"Upgrade alice to Enterprise\"*"
        )

    old_user = session_state.get_user(username)
    old_plan = old_user["plan"]

    if old_plan == new_plan:
        return f"ℹ️ **{username}** is already on the **{new_plan}** plan. No changes made."

    try:
        updated = session_state.update_user_plan(username, new_plan)
    except ValueError as e:
        return f"❌ {e}"

    if session_id:
        session_state.set_current_user(session_id, username)

    direction = "⬆️ Upgraded" if (
        ["Free", "Pro", "Team", "Enterprise"].index(new_plan) >
        ["Free", "Pro", "Team", "Enterprise"].index(old_plan)
    ) else "⬇️ Downgraded"

    plan_emoji = {"Free": "🆓", "Pro": "⭐", "Team": "👥", "Enterprise": "🏢"}

    return (
        f"## {direction}: {username.title()}\n\n"
        f"| | Before | After |\n"
        f"|---|---|---|\n"
        f"| Plan | {plan_emoji.get(old_plan,'📋')} {old_plan} | {plan_emoji.get(new_plan,'📋')} **{new_plan}** |\n"
        f"| Price | {old_user['price']} | {updated['price']} |\n"
        f"| Seats | {old_user['seats']} | {updated['seats']} |\n"
        f"| Actions min/month | {old_user['actions_minutes']:,} | {updated['actions_minutes']:,} |\n"
        f"| Storage | {old_user['storage_gb']} GB | {updated['storage_gb']} GB |\n\n"
        f"Changes are effective immediately."
    )


# ── Action 6: List All Accounts (Single-turn) ─────────────────────────────────

def handle_list_accounts(session_state) -> str:
    """Return a formatted list of all registered users and their plans."""
    users = session_state.list_users()
    if not users:
        return "No accounts registered yet."

    plan_emoji = {"Free": "🆓", "Pro": "⭐", "Team": "👥", "Enterprise": "🏢"}

    lines = [f"## Registered Accounts ({len(users)} total)\n"]
    for u in users:
        emoji = plan_emoji.get(u["plan"], "📋")
        lines.append(
            f"- **{u['username']}** — {emoji} {u['plan']} ({u['price']}) "
            f"· joined {u['joined_date']}"
        )

    lines.append(
        "\nSay *\"Check billing for [username]\"* for full details, "
        "or *\"Upgrade [username] to [plan]\"* to change a plan."
    )
    return "\n".join(lines)