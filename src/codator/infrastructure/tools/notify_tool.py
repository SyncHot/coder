"""Notify Tool — send notifications via webhooks, email, or system.

Supports Slack webhooks, Discord webhooks, email (SMTP), and generic webhooks.
"""

from __future__ import annotations

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class NotifyTool(Tool):
    """Send notifications via Slack, Discord, email, or webhooks."""

    def __init__(
        self,
        slack_webhook: str = "",
        discord_webhook: str = "",
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        smtp_from: str = "",
    ):
        self._slack_webhook = slack_webhook
        self._discord_webhook = discord_webhook
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._smtp_from = smtp_from

    @property
    def name(self) -> str:
        return "notify"

    @property
    def description(self) -> str:
        return (
            "Send notifications. Channels: slack (webhook), discord (webhook), "
            "email (SMTP), webhook (generic POST). Useful for alerting on task "
            "completion, errors, or QA findings."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["slack", "discord", "email", "webhook"],
                    "description": "Notification channel.",
                },
                "message": {
                    "type": "string",
                    "description": "Message content (text or markdown).",
                },
                "title": {
                    "type": "string",
                    "description": "Message title/subject.",
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Override webhook URL.",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email (for 'email' channel).",
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "success"],
                    "description": "Message severity (affects formatting).",
                },
            },
            "required": ["channel", "message"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        channel = kwargs.get("channel", "")
        message = kwargs.get("message", "")

        if not message:
            return ToolResult(success=False, error="'message' required.")

        if channel == "slack":
            return await self._send_slack(message, kwargs)
        elif channel == "discord":
            return await self._send_discord(message, kwargs)
        elif channel == "email":
            return self._send_email(message, kwargs)
        elif channel == "webhook":
            return await self._send_webhook(message, kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown channel: {channel}")

    async def _send_slack(self, message: str, kwargs: dict) -> ToolResult:
        url = kwargs.get("webhook_url", self._slack_webhook)
        if not url:
            return ToolResult(success=False, error="No Slack webhook URL configured.")

        severity = kwargs.get("severity", "info")
        color_map = {
            "info": "#36a64f", "warning": "#ff9800",
            "error": "#f44336", "success": "#4caf50",
        }
        title = kwargs.get("title", "Codator Notification")

        payload = {
            "attachments": [{
                "color": color_map.get(severity, "#36a64f"),
                "title": title,
                "text": message,
                "footer": "codator",
            }]
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return ToolResult(success=True, output=f"Slack notification sent: {title}")
            return ToolResult(success=False, error=f"Slack returned {resp.status_code}: {resp.text}")
        except Exception as e:
            return ToolResult(success=False, error=f"Slack send failed: {e}")

    async def _send_discord(self, message: str, kwargs: dict) -> ToolResult:
        url = kwargs.get("webhook_url", self._discord_webhook)
        if not url:
            return ToolResult(success=False, error="No Discord webhook URL configured.")

        severity = kwargs.get("severity", "info")
        color_map = {
            "info": 0x36A64F, "warning": 0xFF9800,
            "error": 0xF44336, "success": 0x4CAF50,
        }
        title = kwargs.get("title", "Codator Notification")

        payload = {
            "embeds": [{
                "title": title,
                "description": message[:2000],
                "color": color_map.get(severity, 0x36A64F),
            }]
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code in (200, 204):
                return ToolResult(success=True, output=f"Discord notification sent: {title}")
            return ToolResult(success=False, error=f"Discord returned {resp.status_code}: {resp.text}")
        except Exception as e:
            return ToolResult(success=False, error=f"Discord send failed: {e}")

    def _send_email(self, message: str, kwargs: dict) -> ToolResult:
        if not self._smtp_host:
            return ToolResult(success=False, error="SMTP not configured.")

        to_addr = kwargs.get("to", "")
        if not to_addr:
            return ToolResult(success=False, error="'to' email address required.")

        subject = kwargs.get("title", "Codator Notification")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._smtp_from or self._smtp_user
        msg["To"] = to_addr
        msg.attach(MIMEText(message, "plain"))

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                if self._smtp_user:
                    server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)
            return ToolResult(success=True, output=f"Email sent to {to_addr}: {subject}")
        except Exception as e:
            return ToolResult(success=False, error=f"Email send failed: {e}")

    async def _send_webhook(self, message: str, kwargs: dict) -> ToolResult:
        url = kwargs.get("webhook_url", "")
        if not url:
            return ToolResult(success=False, error="'webhook_url' required.")

        title = kwargs.get("title", "Codator Notification")
        severity = kwargs.get("severity", "info")

        payload = {
            "title": title,
            "message": message,
            "severity": severity,
            "source": "codator",
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
            return ToolResult(
                success=resp.status_code < 400,
                output=f"Webhook {resp.status_code}: {resp.text[:200]}",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Webhook failed: {e}")
