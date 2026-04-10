"""SQLite-based conversation persistence — save/load chat sessions."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from codator.domain.models import Message, Role

DEFAULT_DB_DIR = Path.home() / ".config" / "codator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "conversations.db"


class ConversationStore:
    """Persist and retrieve conversation histories in SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                model      TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                token_count     INTEGER DEFAULT 0,
                timestamp       REAL NOT NULL,
                metadata_json   TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv
                ON messages(conversation_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        conversation_id: str,
        messages: list[Message],
        title: str = "",
        model: str = "",
    ) -> None:
        """Save (or overwrite) a conversation."""
        now = time.time()
        if not title:
            # Auto-title from first user message
            for m in messages:
                if m.role == Role.USER:
                    title = m.content[:80]
                    break
            else:
                title = f"conversation-{conversation_id[:8]}"

        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute(
                """INSERT OR REPLACE INTO conversations
                   (id, title, model, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, title, model, now, now),
            )
            # Clear old messages for this conversation
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            # Insert all messages
            for msg in messages:
                meta = json.dumps(msg.metadata) if msg.metadata else "{}"
                cursor.execute(
                    """INSERT INTO messages
                       (conversation_id, role, content, token_count, timestamp,
                        metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        msg.role.value,
                        msg.content,
                        msg.token_count,
                        msg.timestamp,
                        meta,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, conversation_id: str) -> list[Message] | None:
        """Load messages for a conversation. Returns None if not found."""
        rows = self._conn.execute(
            """SELECT role, content, token_count, timestamp, metadata_json
               FROM messages
               WHERE conversation_id = ?
               ORDER BY id""",
            (conversation_id,),
        ).fetchall()
        if not rows:
            return None

        messages: list[Message] = []
        for row in rows:
            meta: dict[str, Any] = {}
            try:
                meta = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
            messages.append(Message(
                role=Role(row["role"]),
                content=row["content"],
                token_count=row["token_count"],
                timestamp=row["timestamp"],
                metadata=meta,
            ))
        return messages

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_conversations(
        self, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent conversations."""
        rows = self._conn.execute(
            """SELECT id, title, model, created_at, updated_at
               FROM conversations
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        self._conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        cursor = self._conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()
