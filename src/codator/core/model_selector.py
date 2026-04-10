"""Hardware-aware model selector — picks the optimal Ollama model per query.

Analyses query complexity via keyword heuristics and message length, then
selects the best model that fits in available VRAM with an appropriate
context window size.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------


@dataclass
class ModelChoice:
    """Result of model selection for a given query."""

    model_name: str
    num_ctx: int
    reason: str


# ---------------------------------------------------------------------------
# Keyword patterns for complexity classification
# ---------------------------------------------------------------------------

_FAST_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bexplain\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\blist\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\bplan\b",
        r"\bthanks\b",
        r"\bthank you\b",
        r"\bdescribe briefly\b",
        r"\bname the\b",
        r"\bdefine\b",
    ]
]

_MEDIUM_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\breview\b",
        r"\brefactor\b",
        r"\bfix\b",
        r"\bbug\b",
        r"\bdocument\b",
        r"\bdocstring\b",
        r"\btype hints?\b",
        r"\btests?\b",
        r"\bunit test\b",
        r"\bsingle file\b",
        r"\brename\b",
        r"\bclean ?up\b",
        r"\boptimize\b",
        r"\bimprove\b",
        r"\bmigrate\b",
    ]
]

_COMPLEX_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdebug\b",
        r"\barchitect",
        r"\bdesign pattern\b",
        r"\bmulti[- ]?file\b",
        r"\brefactor entire\b",
        r"\brewrite\b",
        r"\bfrom scratch\b",
        r"\balgorithm\b",
        r"\bconcurrency\b",
        r"\basync\b",
        r"\bperformance\b",
        r"\bsecurity\b",
        r"\bvulnerabilit",
        r"\bsystem design\b",
        r"\bCI/?CD\b",
        r"\bpipeline\b",
        r"\bdatabase schema\b",
        r"\bscalab",
        r"\bintegrat(e|ion)\b",
        r"\bfull.+implementation\b",
    ]
]

# Length thresholds (character count) used as a secondary signal
_FAST_MAX_LENGTH = 80
_COMPLEX_MIN_LENGTH = 500


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


class ModelSelector:
    """Pick the best Ollama model for each query based on complexity and VRAM.

    Parameters
    ----------
    available_models:
        List of dicts describing installed models, each with at least a
        ``"name"`` key (e.g. ``"qwen2.5-coder:7b"``).  An optional
        ``"size"`` key (bytes) is used as a fallback for VRAM estimation.
    vram_total_mb:
        Total GPU VRAM in MiB.  Defaults to 16 304 MiB (AMD RX 9070 XT).
    """

    # ---- tier configuration (edit these to change model assignments) -------

    FAST_MODELS: list[str] = [
        "qwen2.5-coder:1.5b",
        "deepseek-coder:latest",
    ]

    MEDIUM_MODELS: list[str] = [
        "qwen2.5-coder:7b",
    ]

    COMPLEX_MODELS: list[str] = [
        "qwen2.5-coder:14b-instruct-q6_K",
        "qwen2.5-coder:32b-instruct-q3_K_M",
        "deepseek-r1:32b",
    ]

    # Known VRAM requirements (MiB) — measured/estimated with Ollama
    _VRAM_MAP: dict[str, int] = {
        "qwen2.5-coder:1.5b": 1_200,
        "qwen2.5-coder:7b": 5_500,
        "qwen2.5-coder:14b-instruct-q6_K": 12_500,
        "qwen2.5-coder:32b-instruct-q3_K_M": 15_800,
        "deepseek-r1:32b": 19_000,
        "deepseek-coder:latest": 1_000,
    }

    # Maximum context each model can handle (tokens)
    _MAX_CTX: dict[str, int] = {
        "qwen2.5-coder:1.5b": 32_768,
        "qwen2.5-coder:7b": 32_768,
        "qwen2.5-coder:14b-instruct-q6_K": 32_768,
        "qwen2.5-coder:32b-instruct-q3_K_M": 32_768,
        "deepseek-r1:32b": 131_072,
        "deepseek-coder:latest": 16_384,
    }

    # Context-window ranges per complexity tier
    _CTX_RANGES: dict[str, tuple[int, int]] = {
        "fast": (2_048, 4_096),
        "medium": (4_096, 16_384),
        "complex": (8_192, 32_768),
    }

    # -----------------------------------------------------------------------

    def __init__(
        self,
        available_models: list[dict],
        vram_total_mb: int = 16_304,
    ) -> None:
        self._available: set[str] = {m["name"] for m in available_models}
        self._models_raw = available_models
        self._vram_total = vram_total_mb

    # ----- Public API ------------------------------------------------------

    def select_model(
        self,
        query: str,
        current_context_tokens: int = 0,
    ) -> ModelChoice:
        """Analyse *query* and return the optimal model + context size.

        Parameters
        ----------
        query:
            The user's message / prompt.
        current_context_tokens:
            Tokens already consumed by conversation history.  Used to
            calculate an appropriate ``num_ctx`` value.
        """
        complexity = self._classify(query)
        tier = self._tier_for(complexity)

        model = self._pick_from_tier(tier, complexity)
        num_ctx = self.optimal_num_ctx(current_context_tokens, complexity)
        num_ctx = min(num_ctx, self._MAX_CTX.get(model, num_ctx))

        reason = (
            f"complexity={complexity}, "
            f"vram_required={self.estimate_vram_mb(model)} MiB / "
            f"{self._vram_total} MiB available"
        )
        logger.info("Model selected: %s (num_ctx=%d, %s)", model, num_ctx, reason)
        return ModelChoice(model_name=model, num_ctx=num_ctx, reason=reason)

    @staticmethod
    def estimate_vram_mb(model_name: str) -> int:
        """Return estimated VRAM usage in MiB for *model_name*.

        Falls back to a conservative heuristic when the model is unknown.
        """
        if model_name in ModelSelector._VRAM_MAP:
            return ModelSelector._VRAM_MAP[model_name]
        # Heuristic: assume ~1 GB per billion parameters (rough upper bound)
        logger.warning("No VRAM estimate for %r — using 2 048 MiB fallback", model_name)
        return 2_048

    def optimal_num_ctx(
        self,
        current_tokens: int,
        task_complexity: str,
    ) -> int:
        """Calculate an appropriate ``num_ctx`` for the current state.

        The result is clamped to the range defined for *task_complexity*
        and always leaves headroom above *current_tokens* for generation.
        """
        ctx_min, ctx_max = self._CTX_RANGES.get(
            task_complexity, self._CTX_RANGES["medium"],
        )

        # Start from the tier minimum, but grow if the conversation already
        # occupies more tokens than that.
        ideal = max(ctx_min, current_tokens + 1_024)
        return max(ctx_min, min(ideal, ctx_max))

    # ----- Internals -------------------------------------------------------

    def _classify(self, query: str) -> str:
        """Return ``'fast'``, ``'medium'``, or ``'complex'``."""
        complex_score = sum(1 for p in _COMPLEX_KEYWORDS if p.search(query))
        medium_score = sum(1 for p in _MEDIUM_KEYWORDS if p.search(query))
        fast_score = sum(1 for p in _FAST_KEYWORDS if p.search(query))

        length = len(query)

        # Strong complex signals
        if complex_score >= 2 or (complex_score >= 1 and length >= _COMPLEX_MIN_LENGTH):
            return "complex"

        # Length alone can push to complex
        if length >= _COMPLEX_MIN_LENGTH and medium_score == 0 and fast_score == 0:
            return "complex"

        if medium_score >= 1 and fast_score == 0:
            return "medium"
        if medium_score >= 1 and complex_score >= 1:
            return "complex"

        # Short + fast keywords → fast
        if fast_score >= 1 or length <= _FAST_MAX_LENGTH:
            return "fast"

        # Default to medium for moderate-length queries with no clear signal
        return "medium"

    def _tier_for(self, complexity: str) -> list[str]:
        """Return the ordered candidate list for *complexity*."""
        if complexity == "fast":
            return self.FAST_MODELS
        if complexity == "medium":
            return self.MEDIUM_MODELS
        return self.COMPLEX_MODELS

    def _pick_from_tier(self, tier: list[str], complexity: str) -> str:
        """Select the best available model from *tier*, respecting VRAM.

        Falls back to progressively smaller tiers when nothing fits.
        """
        # Try each model in the tier (ordered best → smallest)
        for model in tier:
            if model in self._available and self.estimate_vram_mb(model) <= self._vram_total:
                return model

        # Fallback chain: complex → medium → fast
        fallback_order: list[list[str]] = {
            "complex": [self.MEDIUM_MODELS, self.FAST_MODELS],
            "medium": [self.FAST_MODELS],
            "fast": [],
        }.get(complexity, [self.FAST_MODELS])

        for fallback_tier in fallback_order:
            for model in fallback_tier:
                if model in self._available and self.estimate_vram_mb(model) <= self._vram_total:
                    logger.warning(
                        "Preferred tier unavailable for %s — falling back to %s",
                        complexity,
                        model,
                    )
                    return model

        # Last resort: return whatever is available and fits
        for model_name in self._available:
            if self.estimate_vram_mb(model_name) <= self._vram_total:
                logger.warning("All tiers exhausted — using %s as last resort", model_name)
                return model_name

        # Nothing fits — return the smallest known model anyway and let
        # Ollama deal with the consequences.
        if self.FAST_MODELS:
            smallest = self.FAST_MODELS[0]
        elif self._available:
            smallest = next(iter(self._available))
        else:
            smallest = "qwen2.5-coder:1.5b"  # hardcoded fallback
            logger.error("No models available at all — using default %s", smallest)
        logger.error(
            "No model fits in %d MiB VRAM — selecting %s anyway", self._vram_total, smallest,
        )
        return smallest
