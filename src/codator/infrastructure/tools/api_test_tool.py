"""API Test Tool — structured HTTP client with assertions.

Make HTTP requests and validate responses with JSON path extraction,
status code checks, header validation, and timing measurements.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class APITestTool(Tool):
    """HTTP client for testing APIs with structured assertions."""

    def __init__(self, timeout: float = 30.0, verify_ssl: bool = True):
        self._timeout = timeout
        self._verify_ssl = verify_ssl

    @property
    def name(self) -> str:
        return "api_test"

    @property
    def description(self) -> str:
        return (
            "Make HTTP requests and validate responses. "
            "Supports GET/POST/PUT/DELETE/PATCH with headers, auth, body. "
            "Can assert status codes, extract JSON paths, measure timing."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                    "description": "HTTP method.",
                },
                "url": {
                    "type": "string",
                    "description": "Full URL to request.",
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers as key-value pairs.",
                },
                "body": {
                    "type": "object",
                    "description": "JSON request body.",
                },
                "form_data": {
                    "type": "object",
                    "description": "Form-encoded request body.",
                },
                "auth": {
                    "type": "object",
                    "description": "Auth: {type: 'basic'|'bearer'|'token', username, password, token}.",
                },
                "expect_status": {
                    "type": "integer",
                    "description": "Expected HTTP status code (assertion).",
                },
                "expect_json": {
                    "type": "object",
                    "description": "Expected JSON path values: {'path.to.key': 'expected_value'}.",
                },
                "expect_headers": {
                    "type": "object",
                    "description": "Expected response headers (case-insensitive).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds.",
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow redirects (default: true).",
                },
            },
            "required": ["method", "url"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url", "")
        if not url:
            return ToolResult(success=False, error="'url' is required.")

        headers = kwargs.get("headers", {})
        body = kwargs.get("body")
        form_data = kwargs.get("form_data")
        auth_conf = kwargs.get("auth", {})
        timeout = kwargs.get("timeout", self._timeout)
        follow = kwargs.get("follow_redirects", True)

        # Setup auth
        auth = None
        if auth_conf:
            auth_type = auth_conf.get("type", "bearer")
            if auth_type == "basic":
                auth = (auth_conf.get("username", ""), auth_conf.get("password", ""))
            elif auth_type in ("bearer", "token"):
                token = auth_conf.get("token", "")
                headers["Authorization"] = f"Bearer {token}"

        try:
            start_time = time.time()
            async with httpx.AsyncClient(
                verify=self._verify_ssl,
                follow_redirects=follow,
                timeout=timeout,
            ) as client:
                request_kwargs: dict[str, Any] = {"headers": headers}
                if auth and isinstance(auth, tuple):
                    request_kwargs["auth"] = auth
                if body is not None:
                    request_kwargs["json"] = body
                elif form_data:
                    request_kwargs["data"] = form_data

                response = await client.request(method, url, **request_kwargs)
            elapsed = time.time() - start_time
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s")
        except httpx.ConnectError as e:
            return ToolResult(success=False, error=f"Connection failed: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Request error: {type(e).__name__}: {e}")

        # Parse response body
        resp_body = ""
        resp_json = None
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                resp_json = response.json()
                resp_body = json.dumps(resp_json, indent=2, ensure_ascii=False)
            except Exception:
                resp_body = response.text
        else:
            resp_body = response.text

        # Truncate large bodies
        if len(resp_body) > 5000:
            resp_body = resp_body[:5000] + "\n... [truncated]"

        # Run assertions
        assertions_passed = True
        assertion_results = []

        expect_status = kwargs.get("expect_status")
        if expect_status is not None:
            passed = response.status_code == expect_status
            assertion_results.append(
                f"{'✓' if passed else '✗'} Status: {response.status_code} "
                f"(expected {expect_status})"
            )
            if not passed:
                assertions_passed = False

        expect_json = kwargs.get("expect_json", {})
        if expect_json and resp_json is not None:
            for path, expected in expect_json.items():
                actual = self._extract_json_path(resp_json, path)
                passed = actual == expected
                assertion_results.append(
                    f"{'✓' if passed else '✗'} JSON {path}: "
                    f"{actual!r} {'==' if passed else '!='} {expected!r}"
                )
                if not passed:
                    assertions_passed = False

        expect_hdrs = kwargs.get("expect_headers", {})
        if expect_hdrs:
            for key, val in expect_hdrs.items():
                actual = response.headers.get(key, "")
                passed = val.lower() in actual.lower()
                assertion_results.append(
                    f"{'✓' if passed else '✗'} Header {key}: {actual[:100]}"
                )
                if not passed:
                    assertions_passed = False

        # Build output
        parts = [
            f"{method} {url} → {response.status_code} ({elapsed:.2f}s)",
            f"Content-Type: {content_type}",
        ]
        if assertion_results:
            parts.append("\nAssertions:")
            parts.extend(f"  {r}" for r in assertion_results)
        parts.append(f"\nBody:\n{resp_body}")

        return ToolResult(
            success=assertions_passed,
            output="\n".join(parts),
            artifacts={
                "status_code": response.status_code,
                "elapsed_seconds": round(elapsed, 3),
                "headers": dict(response.headers),
                "json": resp_json,
            },
        )

    @staticmethod
    def _extract_json_path(data: Any, path: str) -> Any:
        """Simple dot-notation JSON path extraction.
        Supports: 'key.nested.array[0].field'
        """
        import re
        parts = re.split(r"\.(?![^\[]*\])", path)
        current = data
        for part in parts:
            # Handle array index: key[0]
            match = re.match(r"^(.+?)\[(\d+)\]$", part)
            if match:
                key, idx = match.group(1), int(match.group(2))
                if isinstance(current, dict):
                    current = current.get(key, None)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part, None)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current
