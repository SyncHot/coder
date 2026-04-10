"""Domain models — value objects and entities used across all layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------

class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    SUMMARY = "summary"  # synthetic role for compacted snapshots


@dataclass
class Message:
    role: Role
    content: str
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_dict(self) -> dict[str, str]:
        """Format suitable for llama-cpp or API calls."""
        # Map SUMMARY back to system for LLM consumption
        role = "system" if self.role == Role.SUMMARY else self.role.value
        return {"role": role, "content": self.content}


# ---------------------------------------------------------------------------
# Hardware / GPU info
# ---------------------------------------------------------------------------

class InferenceMode(StrEnum):
    FULL_GPU = "full_gpu"
    HYBRID = "hybrid"
    CPU_ONLY = "cpu_only"


@dataclass
class HardwareInfo:
    gpu_name: str = "unknown"
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    cpu_cores: int = 0
    rocm_version: str = ""

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def recommended_mode(self) -> InferenceMode:
        if self.vram_total_mb >= 12_000:
            return InferenceMode.FULL_GPU
        elif self.vram_total_mb >= 4_000:
            return InferenceMode.HYBRID
        return InferenceMode.CPU_ONLY


@dataclass
class ModelRecommendation:
    name: str
    params: str  # e.g. "7B", "34B"
    quant: str  # e.g. "Q8_0", "Q4_K_M"
    mode: InferenceMode
    n_gpu_layers: int  # -1 means all layers on GPU
    estimated_vram_mb: int
    description: str


# ---------------------------------------------------------------------------
# Project indexing
# ---------------------------------------------------------------------------

class SymbolKind(StrEnum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    VARIABLE = "variable"
    IMPORT = "import"


@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    file_path: str
    line: int
    end_line: int | None = None
    signature: str = ""


@dataclass
class FileInfo:
    path: str
    language: str
    size_bytes: int
    symbols: list[Symbol] = field(default_factory=list)
    last_modified: float = 0.0


@dataclass
class ProjectMap:
    root: str
    files: dict[str, FileInfo] = field(default_factory=dict)
    total_files: int = 0
    total_symbols: int = 0
    indexed_at: float = field(default_factory=time.time)

    def summary(self, max_files: int = 50) -> str:
        """Compact text representation for injecting into LLM context."""
        lines = [
            f"Project: {self.root}",
            f"Files: {self.total_files}, Symbols: {self.total_symbols}",
            "",
        ]
        for i, (path, finfo) in enumerate(sorted(self.files.items())):
            if i >= max_files:
                lines.append(f"  ... and {self.total_files - max_files} more files")
                break
            syms = ", ".join(f"{s.name}({s.kind.value})" for s in finfo.symbols[:10])
            trunc = "..." if len(finfo.symbols) > 10 else ""
            lines.append(f"  {path} [{finfo.language}]: {syms}{trunc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context state
# ---------------------------------------------------------------------------

@dataclass
class ContextSnapshot:
    """Result of a compaction operation."""
    summary_text: str
    original_token_count: int
    compacted_token_count: int
    messages_removed: int
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    time_seconds: float = 0.0
    model_name: str = ""
    stopped_by: str = ""  # "eos", "limit", "error"
