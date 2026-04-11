"""Vision Tool — analyze screenshots via multimodal LLM (Ollama or Claude)."""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# Default vision models (in order of preference)
_OLLAMA_VISION_MODELS = ["minicpm-v", "llava:13b", "llava"]


class VisionTool(Tool):
    """Analyze images/screenshots using a multimodal LLM.

    Supports two backends:
    - Ollama with a vision model (minicpm-v, llava)
    - Claude API (claude-sonnet with vision)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vision_model: str = "",
        claude_api_key: str = "",
        claude_model: str = "claude-sonnet-4-20250514",
    ):
        self._ollama_url = ollama_url.rstrip("/")
        self._vision_model = vision_model
        self._claude_api_key = claude_api_key
        self._claude_model = claude_model
        self._resolved_model: str | None = None

    @property
    def name(self) -> str:
        return "vision"

    @property
    def description(self) -> str:
        return (
            "Analyze a screenshot or image using a vision-capable AI model. "
            "Actions: analyze (describe/find bugs in UI), compare (two images), "
            "extract_text (OCR-like text extraction from image)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["analyze", "extract_text"],
                    "description": "What to do with the image.",
                },
                "image_base64": {
                    "type": "string",
                    "description": "Base64-encoded image (PNG/JPEG).",
                },
                "prompt": {
                    "type": "string",
                    "description": "What to look for or analyze in the image.",
                },
            },
            "required": ["action", "image_base64"],
        }

    async def _resolve_ollama_model(self) -> str | None:
        """Find first available vision model on Ollama."""
        if self._resolved_model:
            return self._resolved_model
        if self._vision_model:
            self._resolved_model = self._vision_model
            return self._vision_model

        try:
            async with httpx.AsyncClient(base_url=self._ollama_url, timeout=10) as c:
                resp = await c.get("/api/tags")
                resp.raise_for_status()
                available = {m["name"] for m in resp.json().get("models", [])}
        except Exception:
            return None

        for candidate in _OLLAMA_VISION_MODELS:
            if candidate in available:
                self._resolved_model = candidate
                logger.info("Vision: using Ollama model %s", candidate)
                return candidate
            # Check without tag
            base = candidate.split(":")[0]
            matches = [m for m in available if m.startswith(base)]
            if matches:
                self._resolved_model = matches[0]
                logger.info("Vision: using Ollama model %s", matches[0])
                return matches[0]
        return None

    async def _analyze_ollama(self, image_b64: str, prompt: str) -> str:
        """Send image to Ollama vision model."""
        model = await self._resolve_ollama_model()
        if not model:
            raise RuntimeError(
                "No vision model available. Install one: ollama pull minicpm-v"
            )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {"num_ctx": 4096},
        }

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(
            base_url=self._ollama_url, timeout=timeout
        ) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]

    async def _analyze_claude(self, image_b64: str, prompt: str) -> str:
        """Send image to Claude API with vision."""
        if not self._claude_api_key:
            raise RuntimeError("No Claude API key configured for vision fallback.")

        headers = {
            "x-api-key": self._claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._claude_model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["content"][0]["text"]

    async def execute(self, **kwargs) -> ToolResult:
        """Analyze an image with vision AI.

        Parameters
        ----------
        action : str
            "analyze" or "extract_text"
        image_base64 : str
            Base64-encoded PNG/JPEG image data.
        prompt : str
            What to look for (default: general UI analysis).
        """
        action = kwargs.get("action", "analyze")
        image_b64 = kwargs.get("image_base64", "")
        prompt = kwargs.get("prompt", "")

        if not image_b64:
            return ToolResult(success=False, error="No image provided (image_base64 required).")

        if action == "extract_text":
            prompt = prompt or "Extract all visible text from this screenshot. Return it structured by UI sections."
        elif action == "analyze":
            prompt = prompt or (
                "Analyze this web application screenshot as a QA engineer. "
                "Report: 1) Visual bugs (misalignment, overflow, broken layout) "
                "2) UX issues (unclear labels, missing feedback) "
                "3) Functional concerns (broken elements, missing data) "
                "4) Accessibility issues. Be specific about element locations."
            )

        try:
            # Try Ollama first, fall back to Claude
            ollama_model = await self._resolve_ollama_model()
            if ollama_model:
                result = await self._analyze_ollama(image_b64, prompt)
            elif self._claude_api_key:
                result = await self._analyze_claude(image_b64, prompt)
            else:
                return ToolResult(
                    success=False,
                    error="No vision backend available. Either install a vision model "
                    "(ollama pull minicpm-v) or configure Claude API key.",
                )
            return ToolResult(success=True, output=result)
        except Exception as exc:
            return ToolResult(success=False, error=f"Vision analysis failed: {exc}")
