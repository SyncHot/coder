"""Memory Tool — persistent knowledge store across sessions.

Stores facts, conventions, and learnings in a SQLite database so the AI
assistant can recall project-specific knowledge across conversations.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class MemoryTool(Tool):
    """Store and recall persistent facts/knowledge across sessions.

    Uses a local SQLite database to persist:
    - Project conventions and patterns
    - User preferences
    - Discovered facts about the codebase
    - Build/test commands that work
    """

    def __init__(self, db_path: str | Path = ""):
        if not db_path:
            db_path = Path.home() / ".codator" / "memory.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                source TEXT DEFAULT '',
                project TEXT DEFAULT '',
                created_at REAL NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(fact, subject, source, content=memories, content_rowid=id)
        """)
        conn.commit()
        conn.close()

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Store and recall persistent knowledge across sessions. "
            "Actions: store (save a fact), recall (search memories), "
            "list (show recent), forget (remove a memory)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "recall", "list", "forget"],
                },
                "fact": {
                    "type": "string",
                    "description": "The fact/knowledge to store (for 'store' action).",
                },
                "subject": {
                    "type": "string",
                    "description": "Category/topic (e.g. 'testing', 'auth', 'deployment').",
                },
                "source": {
                    "type": "string",
                    "description": "Where this fact was learned (file, user input, etc.).",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'recall' action).",
                },
                "memory_id": {
                    "type": "integer",
                    "description": "Memory ID (for 'forget' action).",
                },
                "project": {
                    "type": "string",
                    "description": "Project name to scope memories to.",
                },
            },
            "required": ["action"],
        }

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "list")
        project = kwargs.get("project", "")

        if action == "store":
            return self._store(
                fact=kwargs.get("fact", ""),
                subject=kwargs.get("subject", "general"),
                source=kwargs.get("source", ""),
                project=project,
            )
        elif action == "recall":
            return self._recall(
                query=kwargs.get("query", ""),
                project=project,
            )
        elif action == "list":
            return self._list(project=project)
        elif action == "forget":
            return self._forget(memory_id=kwargs.get("memory_id", 0))
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    def _store(self, fact: str, subject: str, source: str, project: str) -> ToolResult:
        if not fact:
            return ToolResult(success=False, error="'fact' is required for store action.")

        conn = self._get_conn()
        try:
            # Check for duplicate
            existing = conn.execute(
                "SELECT id FROM memories WHERE fact = ? AND project = ?",
                (fact, project),
            ).fetchone()
            if existing:
                return ToolResult(
                    success=True,
                    output=f"Memory already exists (id={existing['id']}). Skipped.",
                )

            now = time.time()
            cursor = conn.execute(
                "INSERT INTO memories (subject, fact, source, project, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (subject, fact, source, project, now),
            )
            mem_id = cursor.lastrowid
            # Update FTS
            conn.execute(
                "INSERT INTO memories_fts (rowid, fact, subject, source) VALUES (?, ?, ?, ?)",
                (mem_id, fact, subject, source),
            )
            conn.commit()
            return ToolResult(
                success=True,
                output=f"Stored memory #{mem_id}: [{subject}] {fact[:80]}",
                artifacts={"memory_id": mem_id},
            )
        finally:
            conn.close()

    def _recall(self, query: str, project: str) -> ToolResult:
        if not query:
            return self._list(project=project)

        conn = self._get_conn()
        try:
            # Full-text search
            rows = conn.execute(
                "SELECT m.id, m.subject, m.fact, m.source, m.created_at "
                "FROM memories_fts fts "
                "JOIN memories m ON fts.rowid = m.id "
                "WHERE memories_fts MATCH ? "
                "AND (m.project = ? OR m.project = '') "
                "ORDER BY rank LIMIT 10",
                (query, project),
            ).fetchall()

            if not rows:
                # Fallback: LIKE search
                rows = conn.execute(
                    "SELECT id, subject, fact, source, created_at FROM memories "
                    "WHERE (fact LIKE ? OR subject LIKE ?) "
                    "AND (project = ? OR project = '') "
                    "ORDER BY relevance_score DESC, created_at DESC LIMIT 10",
                    (f"%{query}%", f"%{query}%", project),
                ).fetchall()

            if not rows:
                return ToolResult(success=True, output=f"No memories found for: {query}")

            # Update access counts
            ids = [r["id"] for r in rows]
            conn.executemany(
                "UPDATE memories SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                [(time.time(), mid) for mid in ids],
            )
            conn.commit()

            lines = []
            for r in rows:
                lines.append(f"  #{r['id']} [{r['subject']}] {r['fact']}")
                if r["source"]:
                    lines.append(f"      source: {r['source']}")
            output = f"{len(rows)} memories found:\n" + "\n".join(lines)
            return ToolResult(success=True, output=output)
        finally:
            conn.close()

    def _list(self, project: str) -> ToolResult:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, subject, fact, source, created_at FROM memories "
                "WHERE (project = ? OR project = '') "
                "ORDER BY created_at DESC LIMIT 20",
                (project,),
            ).fetchall()

            if not rows:
                return ToolResult(success=True, output="No memories stored yet.")

            lines = []
            for r in rows:
                lines.append(f"  #{r['id']} [{r['subject']}] {r['fact'][:100]}")
            output = f"{len(rows)} memories:\n" + "\n".join(lines)
            return ToolResult(success=True, output=output)
        finally:
            conn.close()

    def _forget(self, memory_id: int) -> ToolResult:
        if not memory_id:
            return ToolResult(success=False, error="memory_id required for forget.")

        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            conn.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))
            conn.commit()
            return ToolResult(success=True, output=f"Memory #{memory_id} forgotten.")
        finally:
            conn.close()

    def get_context_memories(self, project: str = "", limit: int = 15) -> str:
        """Retrieve recent/relevant memories for injection into system prompt."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT subject, fact FROM memories "
                "WHERE (project = ? OR project = '') "
                "ORDER BY relevance_score DESC, last_accessed DESC NULLS LAST, "
                "created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
            if not rows:
                return ""
            lines = ["<project_memories>"]
            for r in rows:
                lines.append(f"- [{r['subject']}] {r['fact']}")
            lines.append("</project_memories>")
            return "\n".join(lines)
        finally:
            conn.close()
