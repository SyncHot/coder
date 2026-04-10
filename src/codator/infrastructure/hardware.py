"""Hardware detection — VRAM, RAM, ROCm status, model recommendations."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import psutil

from codator.domain.interfaces import HardwareProbe
from codator.domain.models import HardwareInfo, InferenceMode, ModelRecommendation


class AMDHardwareProbe(HardwareProbe):
    """Detect AMD GPU hardware via rocm-smi and system utilities."""

    def check(self) -> HardwareInfo:
        hw = HardwareInfo(
            ram_total_mb=psutil.virtual_memory().total // (1024 * 1024),
            ram_free_mb=psutil.virtual_memory().available // (1024 * 1024),
            cpu_cores=psutil.cpu_count(logical=False) or 1,
        )

        # ROCm version
        rocm_ver_file = Path("/opt/rocm/.info/version")
        if rocm_ver_file.exists():
            hw.rocm_version = rocm_ver_file.read_text().strip()

        # GPU info via rocm-smi
        if shutil.which("rocm-smi"):
            hw = self._parse_rocm_smi(hw)
        elif shutil.which("rocminfo"):
            hw = self._parse_rocminfo(hw)

        return hw

    def _parse_rocm_smi(self, hw: HardwareInfo) -> HardwareInfo:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                # rocm-smi JSON format varies by version; try common keys
                for card_key, card_data in data.items():
                    if isinstance(card_data, dict):
                        total = card_data.get("VRAM Total Memory (B)", 0)
                        used = card_data.get("VRAM Total Used Memory (B)", 0)
                        if total:
                            hw.vram_total_mb = int(total) // (1024 * 1024)
                            hw.vram_used_mb = int(used) // (1024 * 1024)
                            hw.vram_free_mb = hw.vram_total_mb - hw.vram_used_mb
                        break
        except (subprocess.TimeoutExpired, Exception):
            pass

        # GPU name
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "card series" in line.lower() or "marketing" in line.lower():
                    hw.gpu_name = line.split(":")[-1].strip()
                    break
                # Fallback: any line with "Radeon" or "RX"
                if "radeon" in line.lower() or "rx" in line.lower():
                    hw.gpu_name = line.strip()
                    break
        except (subprocess.TimeoutExpired, Exception):
            pass

        return hw

    def _parse_rocminfo(self, hw: HardwareInfo) -> HardwareInfo:
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "Marketing Name" in line:
                    hw.gpu_name = line.split(":")[-1].strip()
                # Pool sizes in KB
                m = re.search(r"Size:\s+(\d+)\s*\(0x", line)
                if m and hw.vram_total_mb == 0:
                    size_kb = int(m.group(1))
                    if size_kb > 1_000_000:  # likely VRAM (>1 GB in KB)
                        hw.vram_total_mb = size_kb // 1024
                        hw.vram_free_mb = hw.vram_total_mb
        except (subprocess.TimeoutExpired, Exception):
            pass
        return hw

    def recommend_models(self, hw: HardwareInfo) -> list[ModelRecommendation]:
        recs: list[ModelRecommendation] = []
        vram = hw.vram_total_mb

        # Always recommend a small summary model
        recs.append(ModelRecommendation(
            name="Phi-3-mini-3.8B / Qwen2.5-3B",
            params="3B",
            quant="Q4_K_M",
            mode=InferenceMode.FULL_GPU,
            n_gpu_layers=-1,
            estimated_vram_mb=2_200,
            description="Fast summary/compaction model (~2 GB VRAM)",
        ))

        if vram >= 12_000:
            # Full GPU: 7B-14B models
            recs.append(ModelRecommendation(
                name="DeepSeek-Coder-V2-Lite-Instruct / Llama-3-8B-Instruct",
                params="7B-8B",
                quant="Q8_0",
                mode=InferenceMode.FULL_GPU,
                n_gpu_layers=-1,
                estimated_vram_mb=9_500,
                description="Primary coding model, full GPU, high quality Q8_0",
            ))
            recs.append(ModelRecommendation(
                name="Qwen2.5-Coder-14B-Instruct",
                params="14B",
                quant="Q5_K_M",
                mode=InferenceMode.FULL_GPU,
                n_gpu_layers=-1,
                estimated_vram_mb=12_500,
                description="Larger coding model, full GPU, balanced quant",
            ))

        if vram >= 8_000 and hw.ram_free_mb >= 16_000:
            # Hybrid: 30B-34B models with layer splitting
            gpu_layers = max(20, int(vram / 350))  # rough estimate: ~350 MB per layer
            recs.append(ModelRecommendation(
                name="CodeLlama-34B-Instruct / DeepSeek-Coder-33B",
                params="33B-34B",
                quant="Q4_K_M",
                mode=InferenceMode.HYBRID,
                n_gpu_layers=gpu_layers,
                estimated_vram_mb=vram - 1_000,  # leave 1 GB headroom
                description=f"Hybrid mode: {gpu_layers} layers on GPU, rest on DDR5",
            ))

        return recs


def check_hardware() -> tuple[HardwareInfo, list[ModelRecommendation]]:
    """Convenience function — detect hardware and get model recommendations."""
    probe = AMDHardwareProbe()
    hw = probe.check()
    recs = probe.recommend_models(hw)
    return hw, recs
