"""Ethos Ticket Tool — create and manage tickets on Ethos OS NAS."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class EthosClient:
    """HTTP client for Ethos OS NAS API with auth handling."""

    def __init__(
        self,
        base_url: str = "",
        username: str = "",
        password: str = "",
        verify_ssl: bool = False,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._token: str | None = None
        self._csrf_token: str | None = None
        self._user: dict | None = None

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def login(self) -> dict:
        """Authenticate and store token."""
        async with httpx.AsyncClient(
            verify=self._verify_ssl, timeout=self._timeout
        ) as client:
            resp = await client.post(
                f"{self._base_url}/api/auth/login",
                json={"username": self._username, "password": self._password},
            )
            resp.raise_for_status()
            data = resp.json()

        if "token" not in data:
            raise RuntimeError(f"Login failed: {data.get('error', 'no token')}")

        self._token = data["token"]
        self._csrf_token = data.get("csrf_token")
        self._user = data.get("user")
        logger.info("Ethos login OK — user=%s", self._user)
        return data

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if self._csrf_token:
            h["X-CSRFToken"] = self._csrf_token
        return h

    async def api(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Call an Ethos API endpoint. Auto-logs in if needed."""
        if not self._token:
            await self.login()

        url = f"{self._base_url}/api{path}"
        kwargs: dict[str, Any] = {"headers": self._headers()}
        if body is not None:
            kwargs["json"] = body

        async with httpx.AsyncClient(
            verify=self._verify_ssl, timeout=timeout or self._timeout
        ) as client:
            resp = await client.request(method, url, **kwargs)

            # Re-auth on 401
            if resp.status_code == 401:
                await self.login()
                kwargs["headers"] = self._headers()
                resp = await client.request(method, url, **kwargs)

            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" in ct:
                return resp.json()
            return {"_text": resp.text, "_status": resp.status_code}

    async def api_safe(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[dict | None, str | None]:
        """Like api() but returns (data, error) instead of raising."""
        try:
            data = await self.api(path, method, body, timeout)
            return data, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


class EthosTicketTool(Tool):
    """Create and manage tickets in Ethos OS NAS Tickets app."""

    def __init__(self, client: EthosClient | None = None, **client_kwargs):
        if client:
            self._client = client
        else:
            self._client = EthosClient(
                base_url=client_kwargs.get("base_url")
                or os.environ.get("ETHOS_URL", ""),
                username=client_kwargs.get("username")
                or os.environ.get("ETHOS_USERNAME", ""),
                password=client_kwargs.get("password")
                or os.environ.get("ETHOS_PASSWORD", ""),
                verify_ssl=client_kwargs.get("verify_ssl", False),
            )
        self._project_cache: dict[str, str] = {}  # name → id

    @property
    def name(self) -> str:
        return "ethos_ticket"

    @property
    def description(self) -> str:
        return (
            "Create, list, and manage tickets in Ethos OS NAS Tickets app. "
            "Actions: create, list_projects, list_tickets, update, close. "
            "Tickets have: title, description, type (bug/task/feature), "
            "priority (critical/high/medium/low), column, labels, assignee."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "list_projects",
                        "list_tickets",
                        "update",
                        "close",
                    ],
                    "description": "Action to perform.",
                },
                "project": {
                    "type": "string",
                    "description": "Project name or ID.",
                },
                "title": {
                    "type": "string",
                    "description": "Ticket title (required for create).",
                },
                "description": {
                    "type": "string",
                    "description": "Ticket description / body (Markdown).",
                },
                "type": {
                    "type": "string",
                    "enum": ["bug", "task", "feature", "epic"],
                    "description": "Ticket type. Default: bug.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"],
                    "description": "Priority level. Default: medium.",
                },
                "complexity": {
                    "type": "string",
                    "enum": ["trivial", "simple", "medium", "complex", "epic"],
                    "description": "Complexity level. Default: medium.",
                },
                "column": {
                    "type": "string",
                    "description": "Board column. Default: Backlog.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply (e.g. ['qa-automated', 'backend']).",
                },
                "assignee": {
                    "type": "string",
                    "description": "Assign to user.",
                },
                "ticket_id": {
                    "type": "string",
                    "description": "Ticket ID (for update/close).",
                },
            },
            "required": ["action"],
        }

    async def _resolve_project_id(self, name_or_id: str) -> str:
        """Resolve a project name to its ID. Caches results."""
        if not name_or_id:
            raise ValueError("Project name or ID required.")

        # Already an ID?
        if len(name_or_id) > 20 or name_or_id.startswith("proj_"):
            return name_or_id

        if name_or_id.lower() in self._project_cache:
            return self._project_cache[name_or_id.lower()]

        data = await self._client.api("/tickets/projects")
        for p in data.get("projects", []):
            self._project_cache[p["name"].lower()] = p["id"]
            if p["name"].lower() == name_or_id.lower():
                return p["id"]

        raise ValueError(f"Project '{name_or_id}' not found.")

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "list_projects")

        try:
            if action == "list_projects":
                return await self._list_projects()
            elif action == "list_tickets":
                return await self._list_tickets(kwargs.get("project", ""))
            elif action == "create":
                return await self._create_ticket(kwargs)
            elif action == "update":
                return await self._update_ticket(kwargs)
            elif action == "close":
                return await self._close_ticket(kwargs)
            else:
                return ToolResult(
                    success=False, error=f"Unknown action: {action}"
                )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _list_projects(self) -> ToolResult:
        data = await self._client.api("/tickets/projects")
        projects = data.get("projects", [])
        if not projects:
            return ToolResult(success=True, output="No projects found.")
        lines = [f"📋 {len(projects)} project(s):"]
        for p in projects:
            lines.append(
                f"  • {p['name']} (id={p['id']}, "
                f"cols={','.join(p.get('columns', []))})"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            artifacts={"projects": projects},
        )

    async def _list_tickets(self, project: str) -> ToolResult:
        project_id = await self._resolve_project_id(project)
        data = await self._client.api(f"/tickets/projects/{project_id}")
        tickets = data.get("tickets", [])
        if not tickets:
            return ToolResult(
                success=True, output=f"No tickets in project."
            )
        lines = [f"🎫 {len(tickets)} ticket(s):"]
        for t in tickets[:30]:
            prio = t.get("priority", "?")
            lines.append(
                f"  [{prio.upper()}] {t.get('title', '?')} "
                f"(#{t['id'][:8]}, col={t.get('column', '?')})"
            )
        if len(tickets) > 30:
            lines.append(f"  ... and {len(tickets) - 30} more")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            artifacts={"tickets": tickets, "count": len(tickets)},
        )

    async def _create_ticket(self, kwargs: dict) -> ToolResult:
        title = kwargs.get("title", "")
        if not title:
            return ToolResult(
                success=False, error="Title is required."
            )

        project = kwargs.get("project", "")
        project_id = await self._resolve_project_id(project)

        payload = {
            "project_id": project_id,
            "title": title,
            "description": kwargs.get("description", ""),
            "type": kwargs.get("type", "bug"),
            "priority": kwargs.get("priority", "medium"),
            "complexity": kwargs.get("complexity", "medium"),
            "column": kwargs.get("column", "Backlog"),
            "labels": kwargs.get("labels", ["qa-automated"]),
            "assignee": kwargs.get("assignee", ""),
        }

        data = await self._client.api(
            "/tickets/tickets", method="POST", body=payload
        )
        if data.get("error"):
            return ToolResult(
                success=False, error=data["error"]
            )

        ticket_id = data.get("id", "?")
        return ToolResult(
            success=True,
            output=f"✅ Ticket created: [{kwargs.get('priority', 'medium').upper()}] {title} (#{ticket_id[:8]})",
            artifacts={"ticket_id": ticket_id, "data": data},
        )

    async def _update_ticket(self, kwargs: dict) -> ToolResult:
        ticket_id = kwargs.get("ticket_id", "")
        if not ticket_id:
            return ToolResult(
                success=False, error="ticket_id required."
            )

        payload = {}
        for field in (
            "title",
            "description",
            "type",
            "priority",
            "complexity",
            "column",
            "labels",
            "assignee",
        ):
            if field in kwargs:
                payload[field] = kwargs[field]

        data = await self._client.api(
            f"/tickets/tickets/{ticket_id}", method="PUT", body=payload
        )
        return ToolResult(
            success=True,
            output=f"Ticket #{ticket_id[:8]} updated.",
            artifacts={"data": data},
        )

    async def _close_ticket(self, kwargs: dict) -> ToolResult:
        ticket_id = kwargs.get("ticket_id", "")
        if not ticket_id:
            return ToolResult(
                success=False, error="ticket_id required."
            )

        data = await self._client.api(
            f"/tickets/tickets/{ticket_id}",
            method="PUT",
            body={"column": "Gotowe"},
        )
        return ToolResult(
            success=True,
            output=f"Ticket #{ticket_id[:8]} moved to Gotowe (closed).",
            artifacts={"data": data},
        )
