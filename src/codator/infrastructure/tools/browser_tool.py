"""Browser Tool — headless browser automation via Playwright."""

from __future__ import annotations

import base64
import logging
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class BrowserTool(Tool):
    """Automate a headless browser for E2E testing and web interaction."""

    def __init__(self, headless: bool = True, timeout: int = 30_000,
                 viewport_width: int = 1280, viewport_height: int = 720):
        self._headless = headless
        self._timeout = timeout
        self._viewport = {"width": viewport_width, "height": viewport_height}
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Control a headless browser for web testing and QA. "
            "Supports: launch, navigate, click, fill, screenshot, get_text, "
            "wait_for, evaluate, get_elements, get_console_errors, get_network_errors, close."
        )

    def _ensure_playwright(self):
        try:
            from playwright.async_api import async_playwright  # noqa: F811
            return async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for browser tool. "
                "Install with: pip install codator[tools] && playwright install chromium"
            )

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    async def launch(self) -> ToolResult:
        """Launch a headless Chromium instance."""
        async_playwright = self._ensure_playwright()

        if self._browser is not None:
            return ToolResult(success=True, output="Browser already running.")

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
            )
            self._page = await self._browser.new_page(viewport=self._viewport)
            self._page.set_default_timeout(self._timeout)
            # Set up event listeners for QA
            self._console_errors = []
            self._network_errors = []
            self._page.on("console", lambda msg: (
                self._console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "warning") else None
            ))
            self._page.on("response", lambda resp: (
                self._network_errors.append(f"{resp.status} {resp.url[:200]}")
                if resp.status >= 400 else None
            ))
            return ToolResult(success=True, output="Browser launched (Chromium, headless).")
        except Exception as exc:
            return ToolResult(success=False, error=f"Browser launch failed: {exc}")

    async def navigate(self, url: str) -> ToolResult:
        """Navigate to a URL."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            resp = await self._page.goto(url, wait_until="domcontentloaded")
            status = resp.status if resp else "unknown"
            title = await self._page.title()
            return ToolResult(
                success=True,
                output=f"Navigated to {url} (status={status}, title='{title}')",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Navigation failed: {exc}")

    async def click(self, selector: str) -> ToolResult:
        """Click an element by CSS selector."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            await self._page.click(selector)
            return ToolResult(success=True, output=f"Clicked: {selector}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Click failed: {exc}")

    async def fill(self, selector: str, value: str) -> ToolResult:
        """Fill an input element."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            await self._page.fill(selector, value)
            return ToolResult(success=True, output=f"Filled {selector} with value.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Fill failed: {exc}")

    async def screenshot(self, path: str = "") -> ToolResult:
        """Capture a screenshot. Returns base64 in artifacts['screenshot']."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            raw = await self._page.screenshot(
                path=path if path else None,
                full_page=True,
            )
            b64 = base64.b64encode(raw).decode("ascii")
            msg = f"Screenshot captured ({len(raw):,} bytes)"
            if path:
                msg += f", saved to {path}"
            return ToolResult(
                success=True,
                output=msg,
                artifacts={"screenshot": b64, "mime_type": "image/png"},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Screenshot failed: {exc}")

    async def get_text(self, selector: str = "body") -> ToolResult:
        """Extract visible text from a page or element."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            text = await self._page.inner_text(selector)
            # Truncate to avoid overwhelming the model
            if len(text) > 10_000:
                text = text[:10_000] + "\n... [truncated]"
            return ToolResult(success=True, output=text)
        except Exception as exc:
            return ToolResult(success=False, error=f"get_text failed: {exc}")


    async def wait_for(self, selector: str, timeout: int = 0) -> ToolResult:
        """Wait for an element to appear."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            t = timeout or self._timeout
            await self._page.wait_for_selector(selector, timeout=t)
            return ToolResult(success=True, output=f"Element found: {selector}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Timeout waiting for {selector}: {exc}")

    async def get_console_errors(self) -> ToolResult:
        """Return any JavaScript console errors captured since navigation."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        # Console messages are collected via event listener (set on launch)
        errors = getattr(self, "_console_errors", [])
        if not errors:
            return ToolResult(success=True, output="No console errors detected.")
        output = f"{len(errors)} console error(s):\n" + "\n".join(errors[-50:])
        return ToolResult(success=True, output=output)

    async def evaluate(self, expression: str) -> ToolResult:
        """Execute JavaScript in the page and return result."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            result = await self._page.evaluate(expression)
            output = str(result) if result is not None else "undefined"
            if len(output) > 10_000:
                output = output[:10_000] + "\n... [truncated]"
            return ToolResult(success=True, output=output)
        except Exception as exc:
            return ToolResult(success=False, error=f"JS evaluation failed: {exc}")

    async def get_elements(self, selector: str) -> ToolResult:
        """Count elements matching a selector and return basic info."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        try:
            elements = await self._page.query_selector_all(selector)
            count = len(elements)
            # Get text of first few elements
            infos = []
            for el in elements[:10]:
                text = (await el.inner_text() or "").strip()[:100]
                tag = await el.evaluate("el => el.tagName.toLowerCase()")
                infos.append(f"<{tag}> {text}")
            output = f"{count} element(s) matching '{selector}'"
            if infos:
                output += ":\n" + "\n".join(infos)
                if count > 10:
                    output += f"\n... and {count - 10} more"
            return ToolResult(success=True, output=output, artifacts={"count": count})
        except Exception as exc:
            return ToolResult(success=False, error=f"get_elements failed: {exc}")

    async def get_network_errors(self) -> ToolResult:
        """Return failed network requests (4xx/5xx) captured during browsing."""
        if self._page is None:
            return ToolResult(success=False, error="Browser not launched.")

        errors = getattr(self, "_network_errors", [])
        if not errors:
            return ToolResult(success=True, output="No network errors detected.")
        output = f"{len(errors)} failed request(s):\n" + "\n".join(errors[-50:])
        return ToolResult(success=True, output=output)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        """Dispatch to a browser sub-command.

        Parameters
        ----------
        action : str
            One of: launch, navigate, click, fill, screenshot, get_text, close.
        """
        action = kwargs.pop("action", "navigate")

        match action:
            case "launch":
                return await self.launch()
            case "navigate":
                return await self.navigate(url=kwargs.get("url", ""))
            case "click":
                return await self.click(selector=kwargs.get("selector", ""))
            case "fill":
                return await self.fill(
                    selector=kwargs.get("selector", ""),
                    value=kwargs.get("value", ""),
                )
            case "screenshot":
                return await self.screenshot(path=kwargs.get("path", ""))
            case "get_text":
                return await self.get_text(
                    selector=kwargs.get("selector", "body"),
                )
            case "close":
                await self.close()
                return ToolResult(success=True, output="Browser closed.")
            case "wait_for":
                return await self.wait_for(
                    selector=kwargs.get("selector", ""),
                    timeout=kwargs.get("timeout", 0),
                )
            case "evaluate":
                return await self.evaluate(expression=kwargs.get("expression", ""))
            case "get_elements":
                return await self.get_elements(selector=kwargs.get("selector", ""))
            case "get_console_errors":
                return await self.get_console_errors()
            case "get_network_errors":
                return await self.get_network_errors()
            case _:
                return ToolResult(
                    success=False,
                    error=f"Unknown browser action: {action}. "
                    f"Use launch|navigate|click|fill|screenshot|get_text|close.",
                )

    async def close(self) -> None:
        """Shut down the browser and playwright."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            self._page = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
