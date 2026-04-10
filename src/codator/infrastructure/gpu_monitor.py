"""Real-time GPU monitoring (ROCm) and Ollama runtime stats."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_ROCM_TIMEOUT = 10
_OLLAMA_TIMEOUT = 5
_OLLAMA_BASE = "http://localhost:11434"

# Only consider discrete GPUs (skip integrated with ≤512 MB VRAM).
_MIN_VRAM_BYTES = 1_000_000_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunningModel:
    name: str
    size_bytes: int
    processor: str
    num_ctx: int


@dataclass
class OllamaStats:
    running_models: list[RunningModel] = field(default_factory=list)
    total_models: int = 0


@dataclass
class GPUStats:
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: float = 0.0
    gpu_name: str = "unknown"
    temperature_c: int = 0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Async monitor combining ROCm GPU metrics and Ollama runtime info."""

    # -- ROCm helpers -------------------------------------------------------

    @staticmethod
    async def _run_rocm_smi(*args: str) -> str | None:
        if not shutil.which("rocm-smi"):
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "rocm-smi", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_ROCM_TIMEOUT)
            if proc.returncode == 0:
                return stdout.decode()
        except (asyncio.TimeoutError, OSError) as exc:
            logger.debug("rocm-smi %s failed: %s", args, exc)
        return None

    @staticmethod
    def _pick_gpu0_value(output: str, key_pattern: str) -> str | None:
        """Return the first GPU[0] line matching *key_pattern*."""
        for line in output.splitlines():
            if line.startswith("GPU[0]") and re.search(key_pattern, line):
                # Value is everything after the last ':'
                return line.rsplit(":", 1)[-1].strip()
        return None

    async def _parse_vram(self) -> tuple[int, int, int]:
        """Return (total_mb, used_mb, free_mb) for the discrete GPU."""
        output = await self._run_rocm_smi("--showmeminfo", "vram")
        if not output:
            return 0, 0, 0

        total_bytes = 0
        used_bytes = 0

        for line in output.splitlines():
            if not line.startswith("GPU[0]"):
                continue
            if "Total Memory" in line:
                val = line.rsplit(":", 1)[-1].strip()
                total_bytes = int(val)
            elif "Total Used" in line:
                val = line.rsplit(":", 1)[-1].strip()
                used_bytes = int(val)

        if total_bytes < _MIN_VRAM_BYTES:
            return 0, 0, 0

        total_mb = total_bytes // _BYTES_PER_MB
        used_mb = used_bytes // _BYTES_PER_MB
        return total_mb, used_mb, total_mb - used_mb

    async def _parse_temperature(self) -> int:
        output = await self._run_rocm_smi("--showtemp")
        if not output:
            return 0
        val = self._pick_gpu0_value(output, r"[Tt]emperature")
        if val:
            # Strip units like "°C" or "c" and grab the number
            m = re.search(r"[\d.]+", val)
            if m:
                return int(float(m.group()))
        return 0

    async def _parse_utilization(self) -> float:
        output = await self._run_rocm_smi("--showuse")
        if not output:
            return 0.0
        val = self._pick_gpu0_value(output, r"[Uu]se|[Uu]tilization|[Aa]ctivity")
        if val:
            m = re.search(r"[\d.]+", val)
            if m:
                return float(m.group())
        return 0.0

    async def _parse_gpu_name(self) -> str:
        output = await self._run_rocm_smi("--showproductname")
        if not output:
            return "unknown"
        for line in output.splitlines():
            if not line.startswith("GPU[0]"):
                continue
            lower = line.lower()
            if "card series" in lower or "marketing" in lower:
                return line.rsplit(":", 1)[-1].strip()
            if "radeon" in lower or "rx" in lower:
                return line.rsplit(":", 1)[-1].strip()
        return "unknown"

    # -- Public API ---------------------------------------------------------

    async def get_gpu_stats(self) -> GPUStats:
        """Gather GPU metrics from rocm-smi (all invocations run in parallel)."""
        vram_task = self._parse_vram()
        temp_task = self._parse_temperature()
        util_task = self._parse_utilization()
        name_task = self._parse_gpu_name()

        (total, used, free), temp, util, name = await asyncio.gather(
            vram_task, temp_task, util_task, name_task,
        )

        return GPUStats(
            vram_total_mb=total,
            vram_used_mb=used,
            vram_free_mb=free,
            gpu_utilization_pct=util,
            gpu_name=name,
            temperature_c=temp,
        )

    async def get_ollama_stats(self) -> OllamaStats:
        """Query Ollama /api/ps for currently loaded models."""
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                resp = await client.get(f"{_OLLAMA_BASE}/api/ps")
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Ollama /api/ps unavailable: %s", exc)
            return OllamaStats()

        models: list[RunningModel] = []
        for entry in data.get("models", []):
            details = entry.get("details", {})
            models.append(RunningModel(
                name=entry.get("name", ""),
                size_bytes=entry.get("size", 0),
                processor=details.get("processor", entry.get("processor", "")),
                num_ctx=entry.get("num_ctx", 0),
            ))

        return OllamaStats(running_models=models, total_models=len(models))

    async def get_full_status(self) -> dict:
        """Combined GPU + Ollama snapshot suitable for API/web responses."""
        gpu_stats, ollama_stats = await asyncio.gather(
            self.get_gpu_stats(),
            self.get_ollama_stats(),
        )

        return {
            "gpu": {
                "name": gpu_stats.gpu_name,
                "vram_total_mb": gpu_stats.vram_total_mb,
                "vram_used_mb": gpu_stats.vram_used_mb,
                "vram_free_mb": gpu_stats.vram_free_mb,
                "utilization_pct": gpu_stats.gpu_utilization_pct,
                "temperature_c": gpu_stats.temperature_c,
            },
            "ollama": {
                "running_models": [
                    {
                        "name": m.name,
                        "size_bytes": m.size_bytes,
                        "processor": m.processor,
                        "num_ctx": m.num_ctx,
                    }
                    for m in ollama_stats.running_models
                ],
                "total_models": ollama_stats.total_models,
            },
        }
