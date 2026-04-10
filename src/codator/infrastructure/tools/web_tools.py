"""Web tools — lightweight HTTP fetch and web search for the AI assistant."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# ── Shared helpers ──────────────────────────────────────────────────────

_TIMEOUT = 15.0
_MAX_DEFAULT = 5000

_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg"}


def _html_to_text(html: str, max_length: int = _MAX_DEFAULT) -> str:
    """Convert HTML to clean text using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: crude regex strip
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_length]

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_length]


# ── WebFetchTool ────────────────────────────────────────────────────────


class WebFetchTool(Tool):
    """Fetch a URL and return the page content as clean text."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and return its content as clean text. "
            "Use for reading documentation, APIs, Stack Overflow, etc."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Max characters to return (default 5000).",
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        max_length = int(kwargs.get("max_length", _MAX_DEFAULT))
        if not url:
            return ToolResult(success=False, error="No URL provided.")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "codator/1.0 (dev-assistant)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                text = _html_to_text(resp.text, max_length)
            else:
                text = resp.text[:max_length]

            return ToolResult(
                success=True,
                output=f"Fetched {url} ({len(text)} chars):\n\n{text}",
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                error=f"HTTP {exc.response.status_code} for {url}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Fetch failed: {exc}")


# ── WebSearchTool ───────────────────────────────────────────────────────


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo and return results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo. Returns top results with "
            "title, URL, and snippet. No API key needed."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        num = min(int(kwargs.get("num_results", 5)), 10)
        if not query:
            return ToolResult(success=False, error="No query provided.")

        # Try DuckDuckGo Instant Answers API first
        results = await self._ddg_instant(query)
        if not results:
            results = await self._ddg_html(query, num)

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for: {query}",
            )

        formatted = [f"Search results for: {query}\n"]
        for i, r in enumerate(results[:num], 1):
            formatted.append(f"{i}. **{r['title']}**\n   {r['url']}\n   {r['snippet']}\n")

        return ToolResult(success=True, output="\n".join(formatted))

    async def _ddg_instant(self, query: str) -> list[dict]:
        """Try DuckDuckGo Instant Answers API."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1"},
                )
                resp.raise_for_status()

            data = resp.json()
            results = []

            # Abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["Abstract"][:300],
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:8]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:200],
                    })

            return results
        except Exception as exc:
            logger.debug("DDG instant API failed: %s", exc)
            return []

    async def _ddg_html(self, query: str, num: int) -> list[dict]:
        """Scrape DuckDuckGo HTML search results as fallback."""
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "codator/1.0 (dev-assistant)"},
            ) as client:
                resp = await client.get(
                    f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                )
                resp.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            results = []
            for item in soup.select(".result__body")[:num]:
                title_el = item.select_one(".result__a")
                snippet_el = item.select_one(".result__snippet")
                if title_el:
                    title = title_el.get_text(strip=True)
                    href = title_el.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet[:200],
                    })

            return results
        except Exception as exc:
            logger.debug("DDG HTML scrape failed: %s", exc)
            return []
