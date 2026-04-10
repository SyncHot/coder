"""Tests for ModelSelector — complexity classification and model selection."""

from __future__ import annotations

import pytest

from codator.core.model_selector import ModelChoice, ModelSelector


@pytest.fixture()
def selector() -> ModelSelector:
    """ModelSelector with typical Ollama model list."""
    models = [
        {"name": "qwen2.5-coder:1.5b"},
        {"name": "qwen2.5-coder:7b"},
        {"name": "qwen2.5-coder:14b-instruct-q6_K"},
    ]
    return ModelSelector(available_models=models, vram_total_mb=16_304)


class TestModelSelector:
    def test_simple_query_picks_fast(self, selector: ModelSelector):
        choice = selector.select_model("hello")
        assert isinstance(choice, ModelChoice)
        assert "1.5b" in choice.model_name or "deepseek" in choice.model_name

    def test_complex_query_picks_heavy(self, selector: ModelSelector):
        choice = selector.select_model(
            "refactor the entire authentication module to use async/await "
            "with proper error handling and retry logic"
        )
        assert isinstance(choice, ModelChoice)
        # Should pick 14b or 7b (not 1.5b)
        assert "1.5b" not in choice.model_name

    def test_num_ctx_within_bounds(self, selector: ModelSelector):
        choice = selector.select_model("explain what this function does")
        assert choice.num_ctx > 0
        assert choice.num_ctx <= 32_768

    def test_reason_is_set(self, selector: ModelSelector):
        choice = selector.select_model("summarize this file")
        assert choice.reason

    def test_estimate_vram(self):
        assert ModelSelector.estimate_vram_mb("qwen2.5-coder:7b") > 0
        # Unknown model returns default
        assert ModelSelector.estimate_vram_mb("unknown-model:latest") > 0

    def test_optimal_num_ctx(self, selector: ModelSelector):
        ctx = selector.optimal_num_ctx(1000, "medium")
        assert ctx >= 4_096

    def test_no_available_models_fallback(self):
        selector = ModelSelector(available_models=[], vram_total_mb=16_304)
        choice = selector.select_model("hello")
        assert isinstance(choice, ModelChoice)
        # Should still return something reasonable

    def test_medium_complexity(self, selector: ModelSelector):
        choice = selector.select_model(
            "add a new endpoint for user profile"
        )
        assert isinstance(choice, ModelChoice)
        assert choice.num_ctx > 0
