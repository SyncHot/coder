"""Tests for AdaptiveContextManager."""

from __future__ import annotations

import pytest

from codator.core.context_manager import AdaptiveContextManager
from codator.domain.models import Message, Role


def _make_msg(role: Role, content: str, tokens: int = 0) -> Message:
    msg = Message(role=role, content=content, token_count=tokens)
    return msg


def _simple_counter(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


class TestAdaptiveContextManager:
    def _make_manager(self, window: int = 1000, threshold: float = 0.8) -> AdaptiveContextManager:
        from codator.config import ContextConfig
        config = ContextConfig(compaction_threshold=threshold, keep_last_messages=2)
        return AdaptiveContextManager(
            context_window=window,
            config=config,
            token_counter=_simple_counter,
        )

    def test_add_message_and_count(self):
        mgr = self._make_manager()
        mgr.add_message(_make_msg(Role.USER, "Hello world"))
        assert mgr.total_tokens() > 0
        assert len(mgr.get_messages()) == 1

    def test_token_count_auto_filled(self):
        mgr = self._make_manager()
        msg = _make_msg(Role.USER, "Hello world", tokens=0)
        mgr.add_message(msg)
        assert msg.token_count > 0

    def test_usage_ratio(self):
        mgr = self._make_manager(window=100)
        # "x" * 200 → ~50 tokens at 4 chars/token
        mgr.add_message(_make_msg(Role.USER, "x" * 200))
        assert mgr.usage_ratio == pytest.approx(0.5, abs=0.1)

    def test_needs_compaction_false_when_below_threshold(self):
        mgr = self._make_manager(window=1000, threshold=0.8)
        mgr.add_message(_make_msg(Role.USER, "short message"))
        assert not mgr.needs_compaction

    def test_needs_compaction_true_when_above_threshold(self):
        mgr = self._make_manager(window=100, threshold=0.8)
        # Fill to >80%: need ~80 tokens → ~320 chars
        mgr.add_message(_make_msg(Role.USER, "x" * 400))
        assert mgr.needs_compaction

    @pytest.mark.asyncio
    async def test_maybe_compact_below_threshold(self):
        mgr = self._make_manager(window=10000)
        mgr.add_message(_make_msg(Role.USER, "small"))
        compacted = await mgr.maybe_compact()
        assert not compacted
        assert mgr.compaction_count == 0

    @pytest.mark.asyncio
    async def test_maybe_compact_above_threshold(self):
        mgr = self._make_manager(window=100, threshold=0.5)
        mgr.add_message(_make_msg(Role.SYSTEM, "You are a helpful assistant."))
        # Add enough messages to exceed threshold
        for i in range(10):
            mgr.add_message(_make_msg(Role.USER, f"Question {i}: " + "x" * 50))
            mgr.add_message(_make_msg(Role.ASSISTANT, f"Answer {i}: " + "y" * 50))

        compacted = await mgr.maybe_compact()
        assert compacted
        assert mgr.compaction_count == 1

        # After compaction: system + summary + last 2 messages
        messages = mgr.get_messages()
        assert messages[0].role == Role.SYSTEM
        assert any(m.role == Role.SUMMARY for m in messages)
        # Total messages should be much fewer
        assert len(messages) <= 5

    @pytest.mark.asyncio
    async def test_snapshot_preserves_system_prompt(self):
        mgr = self._make_manager(window=80, threshold=0.5)
        sys_content = "System: you are codator."
        mgr.add_message(_make_msg(Role.SYSTEM, sys_content))

        for i in range(5):
            mgr.add_message(_make_msg(Role.USER, "x" * 60))
            mgr.add_message(_make_msg(Role.ASSISTANT, "y" * 60))

        await mgr.maybe_compact()
        messages = mgr.get_messages()
        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == sys_content

    def test_clear(self):
        mgr = self._make_manager()
        mgr.add_message(_make_msg(Role.USER, "hello"))
        mgr.clear()
        assert mgr.total_tokens() == 0
        assert len(mgr.get_messages()) == 0

    def test_status_dict(self):
        mgr = self._make_manager(window=1000)
        mgr.add_message(_make_msg(Role.USER, "test"))
        status = mgr.status_dict()
        assert "total_tokens" in status
        assert "context_window" in status
        assert "usage_percent" in status
        assert status["context_window"] == 1000
