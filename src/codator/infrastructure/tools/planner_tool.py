"""Planner Tool — structured task decomposition with dependencies.

Provides the AI with a persistent task tracking system within sessions,
similar to Claude's SQL-based todo tracking.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class PlannerTool(Tool):
    """Task planning and tracking with dependencies."""

    def __init__(self, db_path: str | Path = ""):
        if not db_path:
            db_path = Path.home() / ".codator" / "planner.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at REAL NOT NULL,
                status TEXT DEFAULT 'active'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                completed_at REAL,
                FOREIGN KEY (plan_id) REFERENCES plans(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_deps (
                task_id TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                PRIMARY KEY (task_id, depends_on),
                FOREIGN KEY (task_id) REFERENCES tasks(id),
                FOREIGN KEY (depends_on) REFERENCES tasks(id)
            )
        """)
        conn.commit()
        conn.close()

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return (
            "Task planning and tracking. Actions: create_plan, add_task, "
            "update_task, list_tasks, get_ready (tasks with no pending deps), "
            "complete_task, remove_task. Persists across conversation turns."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_plan", "add_task", "update_task", "list_tasks",
                        "get_ready", "complete_task", "remove_task", "show_plan",
                    ],
                },
                "plan_id": {"type": "string", "description": "Plan identifier."},
                "task_id": {"type": "string", "description": "Task identifier."},
                "title": {"type": "string", "description": "Plan/task title."},
                "description": {"type": "string", "description": "Detailed description."},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "blocked", "skipped"],
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs this task depends on.",
                },
                "priority": {"type": "integer", "description": "Priority (higher = more important)."},
            },
            "required": ["action"],
        }

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "list_tasks")
        dispatch = {
            "create_plan": self._create_plan,
            "add_task": self._add_task,
            "update_task": self._update_task,
            "list_tasks": self._list_tasks,
            "get_ready": self._get_ready,
            "complete_task": self._complete_task,
            "remove_task": self._remove_task,
            "show_plan": self._show_plan,
        }
        handler = dispatch.get(action)
        if not handler:
            return ToolResult(success=False, error=f"Unknown action: {action}")
        return handler(**kwargs)

    def _create_plan(self, **kwargs) -> ToolResult:
        plan_id = kwargs.get("plan_id", "")
        title = kwargs.get("title", "")
        if not plan_id or not title:
            return ToolResult(success=False, error="'plan_id' and 'title' required.")

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO plans (id, title, description, created_at) VALUES (?, ?, ?, ?)",
                (plan_id, title, kwargs.get("description", ""), time.time()),
            )
            conn.commit()
            return ToolResult(success=True, output=f"Plan created: {plan_id} - {title}")
        finally:
            conn.close()

    def _add_task(self, **kwargs) -> ToolResult:
        plan_id = kwargs.get("plan_id", "default")
        task_id = kwargs.get("task_id", "")
        title = kwargs.get("title", "")
        if not task_id or not title:
            return ToolResult(success=False, error="'task_id' and 'title' required.")

        conn = self._get_conn()
        try:
            # Auto-create plan if needed
            existing = conn.execute("SELECT id FROM plans WHERE id=?", (plan_id,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO plans (id, title, created_at) VALUES (?, ?, ?)",
                    (plan_id, plan_id, time.time()),
                )

            conn.execute(
                "INSERT OR REPLACE INTO tasks (id, plan_id, title, description, priority, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, plan_id, title, kwargs.get("description", ""),
                 kwargs.get("priority", 0), time.time()),
            )

            # Add dependencies
            depends_on = kwargs.get("depends_on", [])
            if depends_on:
                for dep in depends_on:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_deps (task_id, depends_on) VALUES (?, ?)",
                        (task_id, dep),
                    )

            conn.commit()
            dep_str = f" (depends on: {', '.join(depends_on)})" if depends_on else ""
            return ToolResult(success=True, output=f"Task added: {task_id} - {title}{dep_str}")
        finally:
            conn.close()

    def _update_task(self, **kwargs) -> ToolResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolResult(success=False, error="'task_id' required.")

        conn = self._get_conn()
        try:
            updates = []
            params = []
            for field in ("title", "description", "status", "priority"):
                if field in kwargs and kwargs[field] is not None:
                    updates.append(f"{field}=?")
                    params.append(kwargs[field])

            if kwargs.get("status") == "done":
                updates.append("completed_at=?")
                params.append(time.time())

            if not updates:
                return ToolResult(success=False, error="No fields to update.")

            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", params)
            conn.commit()
            return ToolResult(success=True, output=f"Task updated: {task_id}")
        finally:
            conn.close()

    def _complete_task(self, **kwargs) -> ToolResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolResult(success=False, error="'task_id' required.")

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=? WHERE id=?",
                (time.time(), task_id),
            )
            conn.commit()
            return ToolResult(success=True, output=f"Task completed: {task_id}")
        finally:
            conn.close()

    def _remove_task(self, **kwargs) -> ToolResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolResult(success=False, error="'task_id' required.")

        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM task_deps WHERE task_id=? OR depends_on=?", (task_id, task_id))
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
            return ToolResult(success=True, output=f"Task removed: {task_id}")
        finally:
            conn.close()

    def _list_tasks(self, **kwargs) -> ToolResult:
        plan_id = kwargs.get("plan_id", "")
        conn = self._get_conn()
        try:
            if plan_id:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE plan_id=? ORDER BY priority DESC, created_at",
                    (plan_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY plan_id, priority DESC, created_at"
                ).fetchall()

            if not rows:
                return ToolResult(success=True, output="No tasks found.")

            status_icons = {
                "pending": "○", "in_progress": "◐",
                "done": "●", "blocked": "✗", "skipped": "—",
            }
            lines = []
            for r in rows:
                icon = status_icons.get(r["status"], "?")
                lines.append(f"  {icon} [{r['status']:12s}] {r['id']}: {r['title']}")
            return ToolResult(
                success=True,
                output=f"Tasks ({len(rows)}):\n" + "\n".join(lines),
            )
        finally:
            conn.close()

    def _get_ready(self, **kwargs) -> ToolResult:
        """Get tasks with no pending dependencies (ready to execute)."""
        plan_id = kwargs.get("plan_id", "")
        conn = self._get_conn()
        try:
            query = """
                SELECT t.* FROM tasks t
                WHERE t.status = 'pending'
                AND NOT EXISTS (
                    SELECT 1 FROM task_deps td
                    JOIN tasks dep ON td.depends_on = dep.id
                    WHERE td.task_id = t.id AND dep.status NOT IN ('done', 'skipped')
                )
            """
            params = []
            if plan_id:
                query += " AND t.plan_id = ?"
                params.append(plan_id)
            query += " ORDER BY t.priority DESC"

            rows = conn.execute(query, params).fetchall()
            if not rows:
                return ToolResult(success=True, output="No ready tasks (all done or blocked).")

            lines = []
            for r in rows:
                lines.append(f"  → {r['id']}: {r['title']}")
                if r["description"]:
                    lines.append(f"    {r['description'][:100]}")
            return ToolResult(
                success=True,
                output=f"Ready tasks ({len(rows)}):\n" + "\n".join(lines),
            )
        finally:
            conn.close()

    def _show_plan(self, **kwargs) -> ToolResult:
        plan_id = kwargs.get("plan_id", "")
        if not plan_id:
            # Show all plans
            conn = self._get_conn()
            try:
                plans = conn.execute("SELECT * FROM plans ORDER BY created_at DESC").fetchall()
                if not plans:
                    return ToolResult(success=True, output="No plans.")
                lines = []
                for p in plans:
                    task_count = conn.execute(
                        "SELECT COUNT(*) as c FROM tasks WHERE plan_id=?", (p["id"],)
                    ).fetchone()["c"]
                    done_count = conn.execute(
                        "SELECT COUNT(*) as c FROM tasks WHERE plan_id=? AND status='done'", (p["id"],)
                    ).fetchone()["c"]
                    lines.append(f"  {p['id']}: {p['title']} [{done_count}/{task_count} done]")
                return ToolResult(success=True, output=f"Plans ({len(plans)}):\n" + "\n".join(lines))
            finally:
                conn.close()

        return self._list_tasks(plan_id=plan_id)
