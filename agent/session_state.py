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

                CREATE TABLE IF NOT EXISTS query_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    intent     TEXT    NOT NULL,
                    confidence REAL    NOT NULL DEFAULT 0.0,
                    question   TEXT    NOT NULL DEFAULT '',
                    is_gap     INTEGER NOT NULL DEFAULT 0
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

    # ── Analytics / query log ──────────────────────────────────────────────────

    def log_query(
        self,
        session_id: str,
        intent: str,
        confidence: float,
        question: str,
        is_gap: bool = False,
    ) -> None:
        """Record one query turn for analytics."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO query_log
                   (session_id, timestamp, intent, confidence, question, is_gap)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    intent,
                    round(confidence, 4),
                    question[:120],
                    1 if is_gap else 0,
                ),
            )

    def get_analytics_data(self) -> dict:
        """Return aggregated telemetry for the analytics dashboard."""
        with self._get_conn() as conn:
            intents = conn.execute(
                "SELECT intent, COUNT(*) AS count FROM query_log "
                "GROUP BY intent ORDER BY count DESC"
            ).fetchall()

            confidence_series = conn.execute(
                "SELECT timestamp, confidence, intent FROM query_log "
                "WHERE confidence > 0 ORDER BY timestamp ASC LIMIT 80"
            ).fetchall()

            top_questions = conn.execute(
                "SELECT question, COUNT(*) AS count FROM query_log "
                "WHERE intent='rag_query' AND is_gap=0 "
                "GROUP BY question ORDER BY count DESC LIMIT 5"
            ).fetchall()

            gaps = conn.execute(
                "SELECT question, timestamp FROM query_log "
                "WHERE is_gap=1 ORDER BY timestamp DESC LIMIT 15"
            ).fetchall()

            total = conn.execute(
                "SELECT COUNT(*) FROM query_log"
            ).fetchone()[0]

        return {
            "total_queries":     total,
            "intents":           [dict(r) for r in intents],
            "confidence_series": [dict(r) for r in confidence_series],
            "top_questions":     [dict(r) for r in top_questions],
            "gaps":              [dict(r) for r in gaps],
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's conversation history. Returns True if a row was removed."""
        with self._get_conn() as conn:
            result = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Session deleted: {session_id}")
        return deleted

    def list_sessions(self, limit: int = 10) -> List[dict]:
        """Return recent sessions with session_id, first user message, and timestamp."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, history_json, created_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        result = []
        for row in rows:
            history = json.loads(row["history_json"])
            first_user_msg = next(
                (msg["content"] for msg in history if msg["role"] == "user"), ""
            )
            result.append({
                "session_id":    row["session_id"],
                "first_message": first_user_msg,
                "created_at":    row["created_at"],
                "history":       history,
            })
        return result

    # Added: For sidebar UI display all open tickets
    def get_open_tickets(self) -> List[dict]:
        """Return all tickets with status 'Open'."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status='Open' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]