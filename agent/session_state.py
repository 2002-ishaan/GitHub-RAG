"""
agent/session_state.py
────────────────────────────────────────────────────────────────
SQLite-backed persistence for tickets and conversation sessions.

WHY SQLITE:
    - Zero setup, file-based, works everywhere
    - Survives app restarts (proves cross-session persistence)
    - Easy to inspect during demo

TABLES:
    tickets  — support tickets created by users
    sessions — conversation history per session
────────────────────────────────────────────────────────────────
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from loguru import logger


class SessionState:
    """Manages all persistent state via SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"SessionState ready | db={db_path}")

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id   TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority    TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'Open',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    history_json TEXT NOT NULL DEFAULT '[]',
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                );
            """)
        logger.debug("Database tables initialised")

    # ── Ticket methods ─────────────────────────────────────────────────────

    def create_ticket(
        self,
        session_id: str,
        category: str,
        description: str,
        priority: str,
    ) -> str:
        """Create a new support ticket. Returns ticket_id like TKT-001."""
        # Generate sequential-style ID
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            ticket_id = f"TKT-{count + 1:03d}"
            now = datetime.utcnow().isoformat()

            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_id, session_id, category, description, priority, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'Open', ?, ?)
                """,
                (ticket_id, session_id, category, description, priority, now, now),
            )

        logger.info(f"Ticket created: {ticket_id} | category={category} | priority={priority}")
        return ticket_id

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        """Look up a ticket by ID. Returns None if not found."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?",
                (ticket_id.upper(),)
            ).fetchone()

        if not row:
            return None

        return {
            "ticket_id":   row["ticket_id"],
            "session_id":  row["session_id"],
            "category":    row["category"],
            "description": row["description"],
            "priority":    row["priority"],
            "status":      row["status"],
            "created_at":  row["created_at"],
            "updated_at":  row["updated_at"],
        }

    def list_tickets(self, session_id: Optional[str] = None) -> List[dict]:
        """List tickets, optionally filtered by session."""
        with self._get_conn() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM tickets WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tickets ORDER BY created_at DESC"
                ).fetchall()

        return [dict(row) for row in rows]

    # ── Session / conversation history methods ─────────────────────────────

    def get_history(self, session_id: str) -> List[dict]:
        """Get conversation history for a session."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT history_json FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

        if not row:
            return []

        return json.loads(row["history_json"])

    def append_to_history(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        now = datetime.utcnow().isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, history_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    history_json = excluded.history_json,
                    updated_at   = excluded.updated_at
                """,
                (session_id, json.dumps(history), now, now),
            )

    def format_history_for_prompt(self, session_id: str, max_turns: int = 6) -> str:
        """Format recent history as a string for the RAG prompt."""
        history = self.get_history(session_id)
        # Take last N turns
        recent = history[-(max_turns * 2):]

        if not recent:
            return "No previous conversation."

        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def clear_history(self, session_id: str):
        """Clear conversation history for a session (new conversation)."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )

    def close_all_tickets(self, session_id: Optional[str] = None) -> int:
        """Close all open tickets. Returns count of tickets closed."""
        with self._get_conn() as conn:
            if session_id:
                result = conn.execute(
                    "UPDATE tickets SET status='Closed', updated_at=? WHERE status='Open' AND session_id=?",
                    (datetime.utcnow().isoformat(), session_id)
                )
            else:
                result = conn.execute(
                    "UPDATE tickets SET status='Closed', updated_at=? WHERE status='Open'",
                    (datetime.utcnow().isoformat(),)
                )
            return result.rowcount
    
    # Added: Close a single ticket by ID
    def close_ticket_by_id(self, ticket_id: str) -> dict:
        """Close a single ticket by ID. Returns a result dict."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT status FROM tickets WHERE ticket_id = ?",
                (ticket_id.upper(),)
            ).fetchone()

            if not row:
                return {"success": False, "message": f"❌ Ticket **{ticket_id}** was not found."}
            if row["status"] == "Closed":
                return {"success": False, "message": f"ℹ️ Ticket **{ticket_id}** is already closed."}

            conn.execute(
                "UPDATE tickets SET status='Closed', updated_at=? WHERE ticket_id=?",
                (datetime.utcnow().isoformat(), ticket_id.upper())
            )

        logger.info(f"Ticket closed: {ticket_id}")
        return {"success": True, "message": f"✅ Ticket **{ticket_id}** has been closed successfully."}

    # Added: For sidebar UI display all open tickets
    def get_open_tickets(self) -> List[dict]:
        """Return all tickets with status 'Open'."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status='Open' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]