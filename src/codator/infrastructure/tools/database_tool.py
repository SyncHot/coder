"""Database Tool — execute queries against SQLite, PostgreSQL, MySQL.

Provides structured query results, schema inspection, and safe query execution
with parameter binding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class DatabaseTool(Tool):
    """Connect to databases and execute queries with structured results."""

    def __init__(self):
        self._connections: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "database"

    @property
    def description(self) -> str:
        return (
            "Execute SQL queries against databases. "
            "Actions: query (SELECT), execute (INSERT/UPDATE/DELETE), "
            "schema (show tables/columns), connect (open connection). "
            "Supports SQLite (built-in), PostgreSQL (psycopg2), MySQL (pymysql)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query", "execute", "schema", "connect", "disconnect"],
                },
                "connection_string": {
                    "type": "string",
                    "description": "DB connection: 'sqlite:///path.db', 'postgresql://user:pass@host/db', 'mysql://user:pass@host/db'.",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute.",
                },
                "params": {
                    "type": "array",
                    "description": "Query parameters for safe binding.",
                },
                "alias": {
                    "type": "string",
                    "description": "Connection alias name (default: 'default').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 100).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "query")
        alias = kwargs.get("alias", "default")

        if action == "connect":
            return self._connect(kwargs.get("connection_string", ""), alias)
        elif action == "disconnect":
            return self._disconnect(alias)
        elif action == "schema":
            return await self._schema(alias, kwargs.get("sql", ""))
        elif action == "query":
            return await self._query(alias, kwargs.get("sql", ""), kwargs.get("params"), kwargs.get("limit", 100))
        elif action == "execute":
            return await self._execute_sql(alias, kwargs.get("sql", ""), kwargs.get("params"))
        return ToolResult(success=False, error=f"Unknown action: {action}")

    def _connect(self, conn_str: str, alias: str) -> ToolResult:
        if not conn_str:
            return ToolResult(success=False, error="'connection_string' required.")

        try:
            if conn_str.startswith("sqlite"):
                # sqlite:///path/to/db.db or sqlite:///:memory:
                db_path = conn_str.replace("sqlite:///", "").replace("sqlite://", "")
                if db_path == ":memory:" or not db_path:
                    conn = sqlite3.connect(":memory:")
                else:
                    conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                self._connections[alias] = {"type": "sqlite", "conn": conn}

            elif conn_str.startswith("postgresql") or conn_str.startswith("postgres"):
                try:
                    import psycopg2
                    import psycopg2.extras
                except ImportError:
                    return ToolResult(success=False, error="psycopg2 not installed. Run: pip install psycopg2-binary")
                conn = psycopg2.connect(conn_str)
                self._connections[alias] = {"type": "postgresql", "conn": conn}

            elif conn_str.startswith("mysql"):
                try:
                    import pymysql
                except ImportError:
                    return ToolResult(success=False, error="pymysql not installed. Run: pip install pymysql")
                from urllib.parse import urlparse
                parsed = urlparse(conn_str)
                conn = pymysql.connect(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 3306,
                    user=parsed.username or "root",
                    password=parsed.password or "",
                    database=parsed.path.lstrip("/"),
                    cursorclass=pymysql.cursors.DictCursor,
                )
                self._connections[alias] = {"type": "mysql", "conn": conn}
            else:
                return ToolResult(success=False, error=f"Unsupported DB type. Use sqlite://, postgresql://, mysql://")

            return ToolResult(success=True, output=f"Connected to {alias} ({conn_str[:50]}...)")
        except Exception as e:
            return ToolResult(success=False, error=f"Connection failed: {type(e).__name__}: {e}")

    def _disconnect(self, alias: str) -> ToolResult:
        if alias in self._connections:
            try:
                self._connections[alias]["conn"].close()
            except Exception:
                pass
            del self._connections[alias]
            return ToolResult(success=True, output=f"Disconnected: {alias}")
        return ToolResult(success=False, error=f"No connection named '{alias}'.")

    def _get_conn(self, alias: str) -> tuple[str, Any] | None:
        entry = self._connections.get(alias)
        if not entry:
            return None
        return entry["type"], entry["conn"]

    async def _query(self, alias: str, sql: str, params: Any, limit: int) -> ToolResult:
        if not sql:
            return ToolResult(success=False, error="'sql' required.")

        conn_info = self._get_conn(alias)
        if not conn_info:
            return ToolResult(success=False, error=f"No connection '{alias}'. Use action='connect' first.")

        db_type, conn = conn_info
        try:
            if db_type == "sqlite":
                cursor = conn.execute(sql, params or [])
                columns = [d[0] for d in cursor.description] if cursor.description else []
                rows = [dict(r) for r in cursor.fetchmany(limit)]
            elif db_type == "postgresql":
                import psycopg2.extras
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(sql, params or None)
                columns = [d.name for d in cursor.description] if cursor.description else []
                rows = [dict(r) for r in cursor.fetchmany(limit)]
                cursor.close()
            elif db_type == "mysql":
                cursor = conn.cursor()
                cursor.execute(sql, params or None)
                columns = [d[0] for d in cursor.description] if cursor.description else []
                rows = list(cursor.fetchmany(limit))
                cursor.close()
            else:
                return ToolResult(success=False, error="Unknown db type.")

            # Format as table
            output_parts = [f"Columns: {', '.join(columns)}", f"Rows: {len(rows)}"]
            if rows:
                # Simple table format
                for i, row in enumerate(rows[:50]):
                    output_parts.append(f"  {i+1}. {row}")

            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                artifacts={"columns": columns, "rows": rows, "row_count": len(rows)},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Query failed: {type(e).__name__}: {e}")

    async def _execute_sql(self, alias: str, sql: str, params: Any) -> ToolResult:
        if not sql:
            return ToolResult(success=False, error="'sql' required.")

        conn_info = self._get_conn(alias)
        if not conn_info:
            return ToolResult(success=False, error=f"No connection '{alias}'. Use action='connect' first.")

        db_type, conn = conn_info
        try:
            if db_type == "sqlite":
                cursor = conn.execute(sql, params or [])
                conn.commit()
                affected = cursor.rowcount
            elif db_type == "postgresql":
                cursor = conn.cursor()
                cursor.execute(sql, params or None)
                affected = cursor.rowcount
                conn.commit()
                cursor.close()
            elif db_type == "mysql":
                cursor = conn.cursor()
                cursor.execute(sql, params or None)
                affected = cursor.rowcount
                conn.commit()
                cursor.close()
            else:
                return ToolResult(success=False, error="Unknown db type.")

            return ToolResult(
                success=True,
                output=f"OK — {affected} row(s) affected.",
                artifacts={"rows_affected": affected},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Execute failed: {type(e).__name__}: {e}")

    async def _schema(self, alias: str, table_filter: str) -> ToolResult:
        conn_info = self._get_conn(alias)
        if not conn_info:
            return ToolResult(success=False, error=f"No connection '{alias}'. Use action='connect' first.")

        db_type, conn = conn_info
        try:
            if db_type == "sqlite":
                # List tables
                cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
                tables = cursor.fetchall()
                parts = [f"Tables ({len(tables)}):"]
                for t in tables:
                    tname = t["name"] if isinstance(t, dict) else t[0]
                    parts.append(f"  • {tname}")
                    if table_filter and table_filter.lower() in tname.lower():
                        # Show columns for matching tables
                        cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
                        for c in cols:
                            cn = c["name"] if isinstance(c, dict) else c[1]
                            ct = c["type"] if isinstance(c, dict) else c[2]
                            parts.append(f"      {cn} ({ct})")

            elif db_type == "postgresql":
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' ORDER BY table_name"
                )
                tables = cursor.fetchall()
                parts = [f"Tables ({len(tables)}):"]
                for t in tables:
                    tname = t[0]
                    parts.append(f"  • {tname}")
                cursor.close()

            elif db_type == "mysql":
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                parts = [f"Tables ({len(tables)}):"]
                for t in tables:
                    val = list(t.values())[0] if isinstance(t, dict) else t[0]
                    parts.append(f"  • {val}")
                cursor.close()
            else:
                parts = ["Unknown DB type"]

            return ToolResult(success=True, output="\n".join(parts))
        except Exception as e:
            return ToolResult(success=False, error=f"Schema query failed: {e}")
