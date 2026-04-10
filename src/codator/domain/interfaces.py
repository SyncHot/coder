"""Abstract interfaces (protocols) for dependency inversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codator.domain.models import (
        GenerationResult,
        HardwareInfo,
        Message,
        ModelRecommendation,
        ProjectMap,
        ToolResult,
    )


class InferenceBackend(ABC):
    """Interface for both local (llama-cpp) and cloud (API) inference."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> GenerationResult:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class HardwareProbe(ABC):
    """Interface for hardware detection."""

    @abstractmethod
    def check(self) -> HardwareInfo:
        ...

    @abstractmethod
    def recommend_models(self, hw: HardwareInfo) -> list[ModelRecommendation]:
        ...


class ProjectIndexer(ABC):
    """Interface for code project scanning."""

    @abstractmethod
    async def index(self, root: str) -> ProjectMap:
        ...

    @abstractmethod
    async def update(self, root: str, changed_files: list[str]) -> ProjectMap:
        ...


class ContextManager(ABC):
    """Interface for conversation context management."""

    @abstractmethod
    def add_message(self, msg: Message) -> None:
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        ...

    @abstractmethod
    def total_tokens(self) -> int:
        ...

    @abstractmethod
    async def maybe_compact(self) -> bool:
        """Check if compaction is needed and perform it. Returns True if compacted."""
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class Tool(ABC):
    """Interface for agentic tools (SSH, browser, terminal, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'ssh', 'browser', 'terminal'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the model."""
        ...

    @property
    def parameters_schema(self) -> dict:
        """JSON Schema for the tool's parameters (OpenAI function calling format)."""
        return {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters."""
        ...

    async def close(self) -> None:
        """Release resources (connections, browsers, etc.)."""
