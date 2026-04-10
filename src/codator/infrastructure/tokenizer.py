"""Token counting utilities using tiktoken (for API) and llama-cpp tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Lazy-loaded tiktoken encoder
_tiktoken_enc = None


def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base — works for GPT-4/Claude estimates)."""
    return len(_get_tiktoken().encode(text))


def count_tokens_llama(text: str, model) -> int:
    """Count tokens using the loaded llama-cpp model's native tokenizer."""
    if model is not None and hasattr(model, "tokenize"):
        return len(model.tokenize(text.encode("utf-8")))
    return count_tokens_tiktoken(text)
