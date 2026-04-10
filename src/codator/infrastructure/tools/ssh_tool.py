"""SSH Tool — remote command execution and file transfer via paramiko."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class SSHTool(Tool):
    """Execute commands and transfer files on remote hosts via SSH."""

    def __init__(self, host: str = "", port: int = 22, username: str = "",
                 password: str = "", key_path: str = "", timeout: int = 30):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._key_path = key_path
        self._timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        return "ssh"

    @property
    def description(self) -> str:
        return (
            "Execute commands on a remote host via SSH. "
            "Supports: connect, exec, upload, download, close."
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_paramiko(self):
        try:
            import paramiko  # noqa: F811
            return paramiko
        except ImportError:
            raise ImportError(
                "paramiko is required for SSH tool. "
                "Install with: pip install codator[tools]"
            )

    async def connect(self, host: str = "", port: int = 0,
                      username: str = "", password: str = "",
                      key_path: str = "") -> ToolResult:
        """Establish SSH connection."""
        h = host or self._host
        p = port or self._port
        u = username or self._username
        pw = password or self._password
        kp = key_path or self._key_path

        if not h or not u:
            return ToolResult(success=False, error="Host and username are required.")

        paramiko = self._ensure_paramiko()

        loop = asyncio.get_running_loop()

        def _connect():
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            kwargs: dict[str, Any] = {
                "hostname": h, "port": p, "username": u,
                "timeout": self._timeout,
            }
            if kp:
                kwargs["key_filename"] = kp
            elif pw:
                kwargs["password"] = pw
            client.connect(**kwargs)
            return client

        try:
            self._client = await loop.run_in_executor(None, _connect)
            return ToolResult(
                success=True,
                output=f"Connected to {u}@{h}:{p}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"SSH connect failed: {exc}")

    async def exec_command(self, command: str,
                           timeout: int = 0) -> ToolResult:
        """Execute a command on the remote host."""
        if self._client is None:
            return ToolResult(success=False, error="Not connected. Call connect() first.")

        t = timeout or self._timeout
        loop = asyncio.get_running_loop()

        def _exec():
            _, stdout, stderr = self._client.exec_command(command, timeout=t)
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            return out, err, exit_code

        try:
            out, err, code = await loop.run_in_executor(None, _exec)
            return ToolResult(
                success=(code == 0),
                output=out,
                error=err,
                exit_code=code,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"SSH exec failed: {exc}")

    async def upload_file(self, local_path: str,
                          remote_path: str) -> ToolResult:
        """Upload a file to the remote host via SFTP."""
        if self._client is None:
            return ToolResult(success=False, error="Not connected.")

        loop = asyncio.get_running_loop()

        def _upload():
            sftp = self._client.open_sftp()
            try:
                sftp.put(local_path, remote_path)
            finally:
                sftp.close()

        try:
            await loop.run_in_executor(None, _upload)
            return ToolResult(
                success=True,
                output=f"Uploaded {local_path} → {remote_path}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"SFTP upload failed: {exc}")

    async def download_file(self, remote_path: str,
                            local_path: str) -> ToolResult:
        """Download a file from the remote host via SFTP."""
        if self._client is None:
            return ToolResult(success=False, error="Not connected.")

        loop = asyncio.get_running_loop()

        def _download():
            sftp = self._client.open_sftp()
            try:
                sftp.get(remote_path, local_path)
            finally:
                sftp.close()

        try:
            await loop.run_in_executor(None, _download)
            return ToolResult(
                success=True,
                output=f"Downloaded {remote_path} → {local_path}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"SFTP download failed: {exc}")

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        """Dispatch to the appropriate sub-command.

        Parameters
        ----------
        action : str
            One of: connect, exec, upload, download, close.
        Plus action-specific kwargs (command, local_path, remote_path, etc.)
        """
        action = kwargs.pop("action", "exec")

        match action:
            case "connect":
                return await self.connect(**kwargs)
            case "exec":
                return await self.exec_command(
                    command=kwargs.get("command", ""),
                    timeout=kwargs.get("timeout", 0),
                )
            case "upload":
                return await self.upload_file(
                    local_path=kwargs.get("local_path", ""),
                    remote_path=kwargs.get("remote_path", ""),
                )
            case "download":
                return await self.download_file(
                    remote_path=kwargs.get("remote_path", ""),
                    local_path=kwargs.get("local_path", ""),
                )
            case "close":
                await self.close()
                return ToolResult(success=True, output="SSH connection closed.")
            case _:
                return ToolResult(
                    success=False,
                    error=f"Unknown SSH action: {action}. "
                    f"Use connect|exec|upload|download|close.",
                )

    async def close(self) -> None:
        """Close the SSH connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
