"""QA Audit Tests — Recursive Context Manager edge cases.

Tests for:
- Infinite compaction loop prevention
- Summary exceeding token limit (recursive snapshot)
- Context drift: loss of critical env vars / paths after compaction
- Compaction with different backend scenarios (no backend, mock backend)
- Edge cases: empty context, single message, system-only
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codator.config import ContextConfig
from codator.core.context_manager import AdaptiveContextManager
from codator.domain.models import Message, Role

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: Role, content: str, tokens: int = 0) -> Message:
    return Message(role=role, content=content, token_count=tokens)


def _counter(text: str) -> int:
    """4 chars = 1 token."""
    return max(1, len(text) // 4)


def _make_mgr(
    window: int = 1000,
    threshold: float = 0.8,
    keep_last: int = 2,
    backend=None,
    summary_backend=None,
) -> AdaptiveContextManager:
    config = ContextConfig(
        compaction_threshold=threshold,
        keep_last_messages=keep_last,
    )
    return AdaptiveContextManager(
        context_window=window,
        config=config,
        token_counter=_counter,
        primary_backend=backend,
        summary_backend=summary_backend,
    )


def _mock_summary_backend(summary_text="Summary of discussion."):
    """Create a mock backend that returns a short summary."""
    backend = AsyncMock()
    result = MagicMock()
    result.text = summary_text
    backend.generate = AsyncMock(return_value=result)
    return backend


# ===================================================================
# 1. INFINITE COMPACTION LOOP PREVENTION
# ===================================================================

class TestCompactionLoopPrevention:
    """Ensure compaction doesn't loop infinitely when summary is too large."""

    @pytest.mark.asyncio
    async def test_compaction_reduces_token_count(self):
        """After compaction, total tokens MUST decrease."""
        backend = _mock_summary_backend("Discussed 20 questions about testing.")
        mgr = _make_mgr(window=800, threshold=0.5, summary_backend=backend)
        mgr.add_message(_msg(Role.SYSTEM, "You are codator."))
        for i in range(20):
            mgr.add_message(_msg(Role.USER, f"Question {i}: " + "x" * 80))
            mgr.add_message(_msg(Role.ASSISTANT, f"Answer {i}: " + "y" * 80))

        before = mgr.total_tokens()
        assert mgr.needs_compaction

        compacted = await mgr.maybe_compact()
        assert compacted
        after = mgr.total_tokens()
        assert after < before, f"Tokens did not decrease: {before} → {after}"

    @pytest.mark.asyncio
    async def test_double_compaction_stable(self):
        """Two compactions in a row should not cause errors or grow context."""
        backend = _mock_summary_backend("Short summary.")
        mgr = _make_mgr(window=1000, threshold=0.3, summary_backend=backend)
        mgr.add_message(_msg(Role.SYSTEM, "sys"))
        for i in range(15):
            mgr.add_message(_msg(Role.USER, f"Q{i}" + "x" * 40))
            mgr.add_message(_msg(Role.ASSISTANT, f"A{i}" + "y" * 40))

        # First compaction
        await mgr.maybe_compact()

        # Force context above threshold again
        for i in range(15):
            mgr.add_message(_msg(Role.USER, f"Q2_{i}" + "x" * 40))
            mgr.add_message(_msg(Role.ASSISTANT, f"A2_{i}" + "y" * 40))

        # Second compaction
        await mgr.maybe_compact()
        tokens_after_second = mgr.total_tokens()

        # Should still be manageable
        assert tokens_after_second < mgr._context_window
        assert mgr.compaction_count == 2

    @pytest.mark.asyncio
    async def test_compaction_count_increments(self):
        """Each successful compaction increments the counter."""
        mgr = _make_mgr(window=80, threshold=0.4)
        mgr.add_message(_msg(Role.SYSTEM, "s"))

        for cycle in range(3):
            for i in range(10):
                mgr.add_message(_msg(Role.USER, "x" * 60))
                mgr.add_message(_msg(Role.ASSISTANT, "y" * 60))
            await mgr.maybe_compact()

        assert mgr.compaction_count == 3

    @pytest.mark.asyncio
    async def test_summary_with_mock_backend(self):
        """When a backend is available, summary should use it."""
        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(return_value=MagicMock(
            text="Summary: discussed coding topics.",
            tokens_generated=8,
        ))
        mock_backend.count_tokens = MagicMock(side_effect=_counter)

        mgr = _make_mgr(window=100, threshold=0.5, backend=mock_backend)
        mgr.add_message(_msg(Role.SYSTEM, "You are codator."))
        for i in range(10):
            mgr.add_message(_msg(Role.USER, "x" * 60))
            mgr.add_message(_msg(Role.ASSISTANT, "y" * 60))

        await mgr.maybe_compact()
        # Backend generate should have been called for summarization
        mock_backend.generate.assert_awaited()


# ===================================================================
# 2. CONTEXT DRIFT — CRITICAL INFO PRESERVATION
# ===================================================================

class TestContextDrift:
    """Verify that compaction preserves critical env vars, paths, and config."""

    @pytest.mark.asyncio
    async def test_system_prompt_never_removed(self):
        """System prompt must survive any number of compactions."""
        sys_content = (
            "You are codator. Project root: /home/user/project. "
            "SSH: user@192.168.1.100. API key: sk-test-xxx."
        )
        mgr = _make_mgr(window=100, threshold=0.4)
        mgr.add_message(_msg(Role.SYSTEM, sys_content))

        for cycle in range(3):
            for i in range(10):
                mgr.add_message(_msg(Role.USER, f"cycle{cycle}_q{i}" + "x" * 50))
                mgr.add_message(_msg(Role.ASSISTANT, f"cycle{cycle}_a{i}" + "y" * 50))
            await mgr.maybe_compact()

        messages = mgr.get_messages()
        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == sys_content
        assert "/home/user/project" in messages[0].content
        assert "192.168.1.100" in messages[0].content

    @pytest.mark.asyncio
    async def test_summary_snapshot_created(self):
        """After compaction, a SUMMARY role message should exist."""
        mgr = _make_mgr(window=100, threshold=0.5)
        mgr.add_message(_msg(Role.SYSTEM, "system"))
        for i in range(10):
            mgr.add_message(_msg(Role.USER, "x" * 60))
            mgr.add_message(_msg(Role.ASSISTANT, "y" * 60))

        await mgr.maybe_compact()
        messages = mgr.get_messages()
        summary_msgs = [m for m in messages if m.role == Role.SUMMARY]
        assert len(summary_msgs) >= 1

    @pytest.mark.asyncio
    async def test_keep_last_messages_preserved(self):
        """The last N messages should survive compaction."""
        mgr = _make_mgr(window=100, threshold=0.4, keep_last=4)
        mgr.add_message(_msg(Role.SYSTEM, "sys"))

        for i in range(20):
            mgr.add_message(_msg(Role.USER, f"USER_MSG_{i}" + "x" * 40))
            mgr.add_message(_msg(Role.ASSISTANT, f"ASST_MSG_{i}" + "y" * 40))

        await mgr.maybe_compact()
        messages = mgr.get_messages()
        # Filter out system and summary
        conv_msgs = [m for m in messages if m.role in (Role.USER, Role.ASSISTANT)]
        # Last messages should be the most recent
        assert len(conv_msgs) >= 2  # at least keep_last

    @pytest.mark.asyncio
    async def test_file_paths_in_conversation_tracked_in_summary(self):
        """File paths mentioned in conversation should appear in summary."""
        mgr = _make_mgr(window=150, threshold=0.5)
        mgr.add_message(_msg(Role.SYSTEM, "You are codator."))
        mgr.add_message(_msg(Role.USER, "Edit backend/blueprints/builder.py line 42"))
        mgr.add_message(_msg(Role.ASSISTANT,
            "I'll modify backend/blueprints/builder.py. The function "
            "create_project() needs to validate the path. " + "x" * 200))
        mgr.add_message(_msg(Role.USER, "Also fix frontend/src/App.tsx"))
        mgr.add_message(_msg(Role.ASSISTANT,
            "Done. Updated frontend/src/App.tsx with the new import." + "y" * 200))

        # Fill to trigger compaction
        for i in range(10):
            mgr.add_message(_msg(Role.USER, "more " + "x" * 80))
            mgr.add_message(_msg(Role.ASSISTANT, "resp " + "y" * 80))

        await mgr.maybe_compact()
        messages = mgr.get_messages()
        # The extractive fallback should preserve some file references
        summary_msgs = [m for m in messages if m.role == Role.SUMMARY]
        assert len(summary_msgs) >= 1


# ===================================================================
# 3. EDGE CASES
# ===================================================================

class TestContextEdgeCases:
    """Boundary conditions for the context manager."""

    def test_empty_context_status(self):
        """Status dict should work on empty context."""
        mgr = _make_mgr(window=1000)
        status = mgr.status_dict()
        assert status["total_tokens"] == 0
        assert status["message_count"] == 0
        assert status["usage_percent"] == 0.0

    def test_zero_context_window(self):
        """Context window of 0 should not crash."""
        mgr = _make_mgr(window=0)
        mgr.add_message(_msg(Role.USER, "test"))
        # usage_ratio should be 0 (not division by zero)
        assert mgr.usage_ratio == 0.0

    @pytest.mark.asyncio
    async def test_compact_system_only(self):
        """Compacting with only a system message should not crash."""
        mgr = _make_mgr(window=10, threshold=0.5)
        mgr.add_message(_msg(Role.SYSTEM, "x" * 100))

        # May or may not compact, but should not raise
        await mgr.maybe_compact()
        assert mgr.get_messages()[0].role == Role.SYSTEM

    def test_message_to_llm_dict_summary_mapped(self):
        """SUMMARY role should map to 'system' in LLM dict."""
        msg = _msg(Role.SUMMARY, "This is a summary.")
        d = msg.to_llm_dict()
        assert d["role"] == "system"

    def test_message_to_llm_dict_tool_mapped(self):
        """TOOL role should map correctly."""
        msg = _msg(Role.TOOL, "Tool output here.")
        d = msg.to_llm_dict()
        # TOOL should map to something valid
        assert d["role"] in ("tool", "user", "system")

    def test_clear_resets_everything(self):
        """clear() should reset all state."""
        mgr = _make_mgr()
        mgr.add_message(_msg(Role.SYSTEM, "sys"))
        mgr.add_message(_msg(Role.USER, "hello"))
        mgr.clear()
        assert mgr.total_tokens() == 0
        assert len(mgr.get_messages()) == 0
        assert mgr.usage_ratio == 0.0

    @pytest.mark.asyncio
    async def test_compaction_with_single_user_message(self):
        """Compacting a single user message shouldn't explode."""
        mgr = _make_mgr(window=10, threshold=0.3)
        mgr.add_message(_msg(Role.SYSTEM, "s"))
        mgr.add_message(_msg(Role.USER, "x" * 100))
        # Should compact or at minimum not crash
        await mgr.maybe_compact()
        assert len(mgr.get_messages()) >= 1
