"""Tests for hardware detection."""

from codator.domain.models import HardwareInfo, InferenceMode
from codator.infrastructure.hardware import AMDHardwareProbe


class TestHardwareInfo:
    def test_vram_total_gb(self):
        hw = HardwareInfo(vram_total_mb=16384)
        assert hw.vram_total_gb == 16.0

    def test_recommended_mode_full_gpu(self):
        hw = HardwareInfo(vram_total_mb=16384)
        assert hw.recommended_mode == InferenceMode.FULL_GPU

    def test_recommended_mode_hybrid(self):
        hw = HardwareInfo(vram_total_mb=8000)
        assert hw.recommended_mode == InferenceMode.HYBRID

    def test_recommended_mode_cpu(self):
        hw = HardwareInfo(vram_total_mb=2000)
        assert hw.recommended_mode == InferenceMode.CPU_ONLY


class TestAMDHardwareProbe:
    def test_recommend_models_16gb(self):
        probe = AMDHardwareProbe()
        hw = HardwareInfo(vram_total_mb=16384, ram_free_mb=24000)
        recs = probe.recommend_models(hw)
        assert len(recs) >= 3  # summary + full GPU + hybrid
        modes = {r.mode for r in recs}
        assert InferenceMode.FULL_GPU in modes
        assert InferenceMode.HYBRID in modes

    def test_recommend_models_low_vram(self):
        probe = AMDHardwareProbe()
        hw = HardwareInfo(vram_total_mb=4000, ram_free_mb=8000)
        recs = probe.recommend_models(hw)
        # Should still get the summary model at minimum
        assert len(recs) >= 1
