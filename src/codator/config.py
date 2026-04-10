"""Application configuration loaded from TOML + environment variables."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Pydantic configuration models
# ---------------------------------------------------------------------------

class InferenceConfig(BaseModel):
    model_path: str = ""
    context_size: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.3
    n_gpu_layers: int = -1
    n_threads: int = 8


class ContextConfig(BaseModel):
    compaction_threshold: float = 0.80
    keep_last_messages: int = 2
    summary_model: str = "fast"


class APIConfig(BaseModel):
    provider: str = "none"
    claude_api_key: str = ""
    openai_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""


class ProjectConfig(BaseModel):
    index_extensions: list[str] = Field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
        ".c", ".cpp", ".h", ".java", ".rb", ".php", ".sh",
        ".toml", ".yaml", ".yml", ".json", ".md", ".txt",
    ])
    exclude_dirs: list[str] = Field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "target", "build", "dist", ".tox", ".mypy_cache",
    ])
    max_file_size: int = 524_288


class SSHConfig(BaseModel):
    host: str = ""
    port: int = 22
    username: str = ""
    password: str = ""
    key_path: str = ""
    timeout: int = 30


class BrowserConfig(BaseModel):
    headless: bool = True
    timeout: int = 30_000
    viewport_width: int = 1280
    viewport_height: int = 720


class TerminalConfig(BaseModel):
    working_dir: str = "."
    timeout: int = 60
    require_confirm: bool = True
    dangerous_patterns: list[str] = Field(default_factory=lambda: [
        r"rm\s+-rf\s+/",
        r"sudo\s+rm",
        r"mkfs\.",
        r"dd\s+if=",
        r"chmod\s+777",
        r":\(\)\s*\{",
        r">\s*/dev/sd",
        r"shutdown",
        r"reboot",
        r"init\s+0",
    ])


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:14b-instruct-q6_K"


class AppSettings(BaseSettings):
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    terminal: TerminalConfig = Field(default_factory=TerminalConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _config_search_paths() -> list[Path]:
    """Return config file search paths (CWD, package dir, home)."""
    pkg_root = Path(__file__).resolve().parent.parent.parent  # src/../..
    return [
        Path("codator.toml"),
        Path("config/default.toml"),
        pkg_root / "config" / "default.toml",
        Path.home() / ".config" / "codator" / "config.toml",
    ]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> AppSettings:
    """Load configuration from TOML file, with env-var overrides for API keys."""
    raw: dict[str, Any] = {}

    if path and path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        for candidate in _config_search_paths():
            if candidate.exists():
                raw = tomllib.loads(candidate.read_text(encoding="utf-8"))
                break

    # Strip top-level [app] section (metadata only)
    raw.pop("app", None)

    settings = AppSettings(**raw)

    # Env-var overrides for sensitive keys
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        settings.api.claude_api_key = key
    if key := os.environ.get("OPENAI_API_KEY"):
        settings.api.openai_api_key = key

    return settings


# Module-level singleton (lazy)
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def reload_settings(path: Path | None = None) -> AppSettings:
    global _settings
    _settings = load_config(path)
    return _settings
