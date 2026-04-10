"""Curated model catalog — recommendations for coding, agent, and analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    """A single model in the catalog."""

    name: str
    ollama_tag: str
    params: str
    quant: str
    size_gb: float
    description: str
    runtime: str  # "gpu", "hybrid", "cpu"
    vram_mb: int  # 0 for cpu-only
    ram_mb: int  # estimated RAM needed for cpu mode
    context_max: int
    scores: dict[str, int] = field(default_factory=dict)  # coding/agent/analysis 1-5
    tags: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Hardware thresholds (based on user's AMD 0x7550 16 GB VRAM, 31 GB RAM)
# ---------------------------------------------------------------------------
VRAM_BUDGET_MB = 16_304
RAM_BUDGET_MB = 31_237


def _compat(entry: ModelEntry) -> str:
    """Return compatibility label for the current hardware."""
    if entry.runtime == "gpu" and entry.vram_mb <= VRAM_BUDGET_MB:
        return "fits_gpu"
    if entry.runtime == "hybrid":
        return "hybrid"
    if entry.runtime == "cpu" and entry.ram_mb <= RAM_BUDGET_MB:
        return "fits_ram"
    if entry.runtime == "cpu" and entry.ram_mb > RAM_BUDGET_MB:
        return "tight_ram"
    return "unknown"


# ---------------------------------------------------------------------------
# Curated catalog
# ---------------------------------------------------------------------------

CATALOG: list[ModelEntry] = [
    # ---- GPU — coding specialists ----
    ModelEntry(
        name="Qwen 2.5 Coder 1.5B",
        ollama_tag="qwen2.5-coder:1.5b",
        params="1.5B", quant="Q4_K_M", size_gb=1.0,
        description="Ultra-fast coding assistant for autocomplete, quick edits, and summaries. Low VRAM.",
        runtime="gpu", vram_mb=1_200, ram_mb=0, context_max=32_768,
        scores={"coding": 2, "agent": 1, "analysis": 1},
        tags=["coding", "fast", "lightweight"],
        notes="Best for: quick tasks, summaries, context compaction.",
    ),
    ModelEntry(
        name="Qwen 2.5 Coder 7B",
        ollama_tag="qwen2.5-coder:7b",
        params="7B", quant="Q4_K_M", size_gb=4.7,
        description="Solid general-purpose coding model. Good balance of speed and quality.",
        runtime="gpu", vram_mb=5_500, ram_mb=0, context_max=32_768,
        scores={"coding": 3, "agent": 2, "analysis": 3},
        tags=["coding", "balanced"],
        notes="Good for: everyday coding, explanations, moderate edits.",
    ),
    ModelEntry(
        name="Qwen 2.5 Coder 14B Instruct",
        ollama_tag="qwen2.5-coder:14b-instruct-q6_K",
        params="14B", quant="Q6_K", size_gb=11.4,
        description="High-quality coding model with strong instruction following. Recommended default.",
        runtime="gpu", vram_mb=12_500, ram_mb=0, context_max=32_768,
        scores={"coding": 4, "agent": 3, "analysis": 4},
        tags=["coding", "agent", "recommended"],
        notes="⭐ Recommended default. Good for: agent tasks, code review, refactoring.",
    ),
    ModelEntry(
        name="DeepSeek Coder V2 16B",
        ollama_tag="deepseek-coder-v2:16b",
        params="16B", quant="Q4_K_M", size_gb=8.9,
        description="DeepSeek's coding specialist. Strong at multi-file understanding and generation.",
        runtime="gpu", vram_mb=10_500, ram_mb=0, context_max=65_536,
        scores={"coding": 4, "agent": 3, "analysis": 4},
        tags=["coding", "large-context"],
        notes="Alternative to Qwen 14B with larger context window.",
    ),
    ModelEntry(
        name="CodeGemma 7B",
        ollama_tag="codegemma:7b",
        params="7B", quant="Q4_K_M", size_gb=5.0,
        description="Google's coding model. Good at code completion and infill tasks.",
        runtime="gpu", vram_mb=5_800, ram_mb=0, context_max=8_192,
        scores={"coding": 3, "agent": 2, "analysis": 2},
        tags=["coding", "google"],
        notes="Strong completion model, weaker on long multi-step tasks.",
    ),
    ModelEntry(
        name="StarCoder2 15B",
        ollama_tag="starcoder2:15b",
        params="15B", quant="Q4_K_M", size_gb=9.0,
        description="BigCode's open coding model. Trained on The Stack v2. Great code generation.",
        runtime="gpu", vram_mb=10_000, ram_mb=0, context_max=16_384,
        scores={"coding": 4, "agent": 2, "analysis": 3},
        tags=["coding", "open-source"],
        notes="Strong raw code generation but weaker at multi-step reasoning.",
    ),

    # ---- GPU — reasoning / agent specialists ----
    ModelEntry(
        name="DeepSeek R1 14B",
        ollama_tag="deepseek-r1:14b",
        params="14B", quant="Q4_K_M", size_gb=9.0,
        description="Chain-of-thought reasoning model. Thinks step-by-step before answering.",
        runtime="gpu", vram_mb=10_200, ram_mb=0, context_max=65_536,
        scores={"coding": 3, "agent": 4, "analysis": 4},
        tags=["reasoning", "agent", "analysis"],
        notes="Good for: complex analysis, debugging, architecture decisions.",
    ),
    ModelEntry(
        name="Phi-4 14B",
        ollama_tag="phi4:14b",
        params="14B", quant="Q4_K_M", size_gb=9.1,
        description="Microsoft's efficient reasoning model. Strong logical and analytical capabilities.",
        runtime="gpu", vram_mb=10_500, ram_mb=0, context_max=16_384,
        scores={"coding": 3, "agent": 3, "analysis": 4},
        tags=["reasoning", "microsoft", "analysis"],
        notes="Great reasoning-to-size ratio. Good for analysis tasks.",
    ),

    # ---- Hybrid (GPU+CPU offload) — 32B class ----
    ModelEntry(
        name="Qwen 2.5 Coder 32B Instruct",
        ollama_tag="qwen2.5-coder:32b-instruct-q3_K_M",
        params="32B", quant="Q3_K_M", size_gb=15.1,
        description="Top-tier coding model. Near GPT-4 quality for code tasks. Tight VRAM fit.",
        runtime="hybrid", vram_mb=15_800, ram_mb=18_000, context_max=32_768,
        scores={"coding": 5, "agent": 4, "analysis": 4},
        tags=["coding", "agent", "premium", "recommended"],
        notes="⭐ Best quality for coding. Tight fit in 16GB VRAM, may need partial CPU offload.",
    ),
    ModelEntry(
        name="DeepSeek R1 32B",
        ollama_tag="deepseek-r1:32b",
        params="32B", quant="Q4_K_M", size_gb=19.0,
        description="Powerful reasoning model. Chain-of-thought for complex multi-step problems.",
        runtime="hybrid", vram_mb=19_000, ram_mb=22_000, context_max=131_072,
        scores={"coding": 4, "agent": 5, "analysis": 5},
        tags=["reasoning", "agent", "analysis", "premium"],
        notes="⭐ Best for agent tasks. Needs GPU+CPU offload. Slower but highest quality reasoning.",
    ),
    ModelEntry(
        name="Codestral 22B",
        ollama_tag="codestral:22b",
        params="22B", quant="Q4_K_M", size_gb=12.9,
        description="Mistral's dedicated coding model. Strong at code generation and fill-in-the-middle.",
        runtime="gpu", vram_mb=14_500, ram_mb=0, context_max=32_768,
        scores={"coding": 4, "agent": 3, "analysis": 3},
        tags=["coding", "mistral"],
        notes="Fits in VRAM. Good alternative to Qwen 32B with faster inference.",
    ),
    ModelEntry(
        name="QwQ 32B",
        ollama_tag="qwq:32b",
        params="32B", quant="Q4_K_M", size_gb=19.9,
        description="Qwen's dedicated reasoning model. Extended thinking for complex problems.",
        runtime="hybrid", vram_mb=20_000, ram_mb=23_000, context_max=32_768,
        scores={"coding": 3, "agent": 5, "analysis": 5},
        tags=["reasoning", "agent", "analysis", "premium"],
        notes="Dedicated reasoning model. Great for complex agent workflows.",
    ),
    ModelEntry(
        name="Devstral Small 24B",
        ollama_tag="devstral",
        params="24B", quant="Q4_K_M", size_gb=14.5,
        description="Mistral's agent-focused coding model. Built for agentic workflows and tool use.",
        runtime="gpu", vram_mb=15_000, ram_mb=0, context_max=131_072,
        scores={"coding": 4, "agent": 5, "analysis": 4},
        tags=["coding", "agent", "mistral", "recommended"],
        notes="⭐ Purpose-built for coding agents. 128K context. Tight GPU fit.",
    ),

    # ---- CPU+RAM — 70B+ class (slow but powerful) ----
    ModelEntry(
        name="Qwen 2.5 Coder 72B (Q2_K)",
        ollama_tag="qwen2.5-coder:72b-instruct-q2_K",
        params="72B", quant="Q2_K", size_gb=27.0,
        description="Largest open coding model. GPT-4 class quality. CPU-only, slow but excellent.",
        runtime="cpu", vram_mb=0, ram_mb=29_000, context_max=32_768,
        scores={"coding": 5, "agent": 5, "analysis": 5},
        tags=["coding", "agent", "analysis", "premium", "cpu-only", "72b"],
        notes="⚠ CPU-only (~2-5 tok/s). Needs ~29GB RAM. Tight fit with 31GB. Best quality.",
    ),
    ModelEntry(
        name="Qwen 2.5 72B (Q2_K)",
        ollama_tag="qwen2.5:72b-instruct-q2_K",
        params="72B", quant="Q2_K", size_gb=27.0,
        description="Largest Qwen general model. Excellent reasoning and instruction following.",
        runtime="cpu", vram_mb=0, ram_mb=29_000, context_max=32_768,
        scores={"coding": 4, "agent": 5, "analysis": 5},
        tags=["general", "reasoning", "agent", "cpu-only", "72b"],
        notes="⚠ CPU-only (~2-5 tok/s). More general-purpose than coder variant.",
    ),
    ModelEntry(
        name="DeepSeek R1 70B (Q2_K)",
        ollama_tag="deepseek-r1:70b-q2_K",
        params="70B", quant="Q2_K", size_gb=26.0,
        description="Full-size DeepSeek reasoning model. Top-tier chain-of-thought for complex problems.",
        runtime="cpu", vram_mb=0, ram_mb=28_000, context_max=131_072,
        scores={"coding": 4, "agent": 5, "analysis": 5},
        tags=["reasoning", "agent", "analysis", "cpu-only", "72b"],
        notes="⚠ CPU-only (~1-3 tok/s). Extremely strong reasoning but very slow.",
    ),
    ModelEntry(
        name="Llama 3.1 70B (Q2_K)",
        ollama_tag="llama3.1:70b-instruct-q2_K",
        params="70B", quant="Q2_K", size_gb=26.0,
        description="Meta's flagship model. Strong general capabilities including coding and reasoning.",
        runtime="cpu", vram_mb=0, ram_mb=28_000, context_max=131_072,
        scores={"coding": 4, "agent": 4, "analysis": 5},
        tags=["general", "coding", "agent", "cpu-only", "72b"],
        notes="⚠ CPU-only (~2-4 tok/s). Well-rounded 70B with 128K context.",
    ),
    ModelEntry(
        name="Qwen 3 32B (Q4_K_M)",
        ollama_tag="qwen3:32b",
        params="32B", quant="Q4_K_M", size_gb=20.0,
        description="Latest Qwen 3 with hybrid thinking. Toggles between fast and deep reasoning.",
        runtime="hybrid", vram_mb=20_000, ram_mb=23_000, context_max=32_768,
        scores={"coding": 4, "agent": 5, "analysis": 5},
        tags=["reasoning", "agent", "analysis", "hybrid-thinking"],
        notes="Supports /think and /no_think modes. Great for agent workflows.",
    ),
]


def get_catalog() -> list[dict]:
    """Return the full catalog as dicts with compatibility info."""
    results = []
    for entry in CATALOG:
        compat = _compat(entry)
        results.append({
            "name": entry.name,
            "ollama_tag": entry.ollama_tag,
            "params": entry.params,
            "quant": entry.quant,
            "size_gb": entry.size_gb,
            "description": entry.description,
            "runtime": entry.runtime,
            "vram_mb": entry.vram_mb,
            "ram_mb": entry.ram_mb,
            "context_max": entry.context_max,
            "scores": entry.scores,
            "tags": entry.tags,
            "notes": entry.notes,
            "compat": compat,
        })
    return results


def search_catalog(
    query: str = "",
    tag: str = "",
    runtime: str = "",
    min_coding: int = 0,
    min_agent: int = 0,
) -> list[dict]:
    """Filter catalog by query, tag, runtime, or minimum scores."""
    results = get_catalog()
    if query:
        q = query.lower()
        results = [
            r for r in results
            if q in r["name"].lower()
            or q in r["description"].lower()
            or q in r["ollama_tag"].lower()
            or any(q in t for t in r["tags"])
        ]
    if tag:
        results = [r for r in results if tag.lower() in r["tags"]]
    if runtime:
        results = [r for r in results if r["runtime"] == runtime]
    if min_coding:
        results = [r for r in results if r["scores"].get("coding", 0) >= min_coding]
    if min_agent:
        results = [r for r in results if r["scores"].get("agent", 0) >= min_agent]
    return results


def get_recommendations(use_case: str = "agent") -> list[dict]:
    """Return top recommendations sorted by relevance for a use case."""
    catalog = get_catalog()
    score_key = {
        "coding": "coding",
        "agent": "agent",
        "analysis": "analysis",
    }.get(use_case, "agent")

    # Sort by: score desc, then prefer gpu over hybrid over cpu
    runtime_order = {"gpu": 0, "hybrid": 1, "cpu": 2}
    catalog.sort(key=lambda r: (
        -r["scores"].get(score_key, 0),
        runtime_order.get(r["runtime"], 3),
        r["size_gb"],
    ))
    return catalog
