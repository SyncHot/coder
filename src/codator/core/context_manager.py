"""AdaptiveContextManager — smart context compaction with Summary Snapshots.

Policy:
  1. Context is NEVER silently truncated. The user is always notified.
  2. When token count exceeds `compaction_threshold` (default 80%) of the context window,
     the manager generates a "Summary Snapshot" using a small fast model (3B).
  3. After compaction: [system prompt] + [summary snapshot] + [last N user messages].
  4. The snapshot is tagged with role=SUMMARY so the user can inspect it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codator.domain.interfaces import ContextManager, InferenceBackend
from codator.domain.models import ContextSnapshot, Message, Role

if TYPE_CHECKING:
    from codator.config import ContextConfig

logger = logging.getLogger(__name__)

# Number of recent TOOL messages to preserve across compaction
KEEP_TOOL_MESSAGES = 4

# Template for the summary generation prompt
SUMMARY_PROMPT = """\
You are a technical context summarizer. Summarize the following conversation \
between a developer and an AI coding assistant. Preserve:
- All code snippets, file paths, function/class names mentioned
- Key decisions made and their rationale
- Current state of the task (what's done, what's pending)
- Any errors or bugs discussed

Be concise but precise. Use bullet points. Do NOT omit technical details.

=== CONVERSATION ===
{conversation}
=== END CONVERSATION ===

Produce a structured summary:"""


class AdaptiveContextManager(ContextManager):
    """Manages conversation history with intelligent compaction.

    Architecture:
        messages[0]     = system prompt (always preserved)
        messages[1]     = summary snapshot (if compacted, role=SUMMARY)
        messages[2..]   = conversation messages (user/assistant pairs)
    """

    def __init__(
        self,
        *,
        context_window: int = 8192,
        config: ContextConfig | None = None,
        primary_backend: InferenceBackend | None = None,
        summary_backend: InferenceBackend | None = None,
        token_counter: callable | None = None,
    ):
        from codator.config import ContextConfig
        self._config = config or ContextConfig()
        self._context_window = context_window
        self._primary = primary_backend
        self._summary = summary_backend  # small 3B model for compaction
        self._messages: list[Message] = []
        self._snapshots: list[ContextSnapshot] = []
        self._compaction_count = 0

        # Token counter — use backend's if available, else tiktoken
        if token_counter:
            self._count = token_counter
        elif primary_backend:
            self._count = primary_backend.count_tokens
        else:
            from codator.infrastructure.tokenizer import count_tokens_tiktoken
            self._count = count_tokens_tiktoken

    # ----- Public interface -----

    def add_message(self, msg: Message) -> None:
        if msg.token_count == 0:
            msg.token_count = self._count(msg.content)
        self._messages.append(msg)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def total_tokens(self) -> int:
        return sum(m.token_count for m in self._messages)

    def remaining_tokens(self) -> int:
        """Tokens still available before hitting the context window limit."""
        return max(0, self._context_window - self.total_tokens())

    @property
    def usage_ratio(self) -> float:
        """Current context usage as a fraction (0.0 – 1.0)."""
        if self._context_window <= 0:
            return 0.0
        return self.total_tokens() / self._context_window

    @property
    def needs_compaction(self) -> bool:
        return self.usage_ratio >= self._config.compaction_threshold

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def snapshots(self) -> list[ContextSnapshot]:
        return list(self._snapshots)

    async def maybe_compact(self) -> str | None:
        """Check usage and compact if threshold exceeded. Returns notification string or None."""
        if not self.needs_compaction:
            return None

        logger.info(
            "Context compaction triggered: %d/%d tokens (%.0f%%)",
            self.total_tokens(), self._context_window,
            self.usage_ratio * 100,
        )

        snapshot = await self._generate_snapshot()
        notification = self._apply_compaction(snapshot)
        self._compaction_count += 1
        return notification

    def clear(self) -> None:
        self._messages.clear()
        self._snapshots.clear()
        self._compaction_count = 0

    # ----- Context inspection (for UI) -----

    def status_dict(self) -> dict:
        """Return current state for CLI/web display."""
        return {
            "total_tokens": self.total_tokens(),
            "context_window": self._context_window,
            "usage_percent": round(self.usage_ratio * 100, 1),
            "message_count": len(self._messages),
            "compaction_count": self._compaction_count,
            "threshold_percent": round(self._config.compaction_threshold * 100, 1),
        }

    # ----- Internal -----

    async def _generate_snapshot(self) -> ContextSnapshot:
        """Generate a Summary Snapshot of the current conversation."""
        # Separate system prompt from conversation
        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        conv_msgs = [m for m in self._messages if m.role not in (Role.SYSTEM, Role.SUMMARY)]

        # Format conversation for summarization
        conversation_text = self._format_conversation(conv_msgs)
        original_tokens = self.total_tokens()

        # Use summary backend if available, otherwise fall back to primary
        backend = self._summary or self._primary
        if backend is None:
            # No backend — produce a simple extractive summary
            summary_text = self._extractive_fallback(conv_msgs)
        else:
            summary_prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
            summary_msg = Message(role=Role.USER, content=summary_prompt)

            # Use only system + summary request (don't recurse the full context)
            gen_messages = system_msgs + [summary_msg]
            result = await backend.generate(
                gen_messages, max_tokens=1024, temperature=0.1,
            )
            summary_text = result.text

        summary_tokens = self._count(summary_text)

        snapshot = ContextSnapshot(
            summary_text=summary_text,
            original_token_count=original_tokens,
            compacted_token_count=summary_tokens,
            messages_removed=len(conv_msgs),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def _apply_compaction(self, snapshot: ContextSnapshot) -> str:
        """Replace conversation with [system] + [snapshot] + [last N messages]."""
        keep_n = self._config.keep_last_messages

        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        conv_msgs = [m for m in self._messages if m.role not in (Role.SYSTEM, Role.SUMMARY)]

        # Keep last N messages (pairs ideally, but at minimum the raw count)
        kept_messages = conv_msgs[-keep_n:] if keep_n > 0 else []

        # Also keep recent tool results that aren't already in kept_messages
        kept_ids = set(id(m) for m in kept_messages)
        recent_tool_msgs = [
            m for m in conv_msgs if m.role == Role.TOOL and id(m) not in kept_ids
        ][-KEEP_TOOL_MESSAGES:]

        # Build summary message
        summary_msg = Message(
            role=Role.SUMMARY,
            content=(
                f"[CONTEXT SNAPSHOT #{self._compaction_count + 1}]\n"
                f"(Compacted {snapshot.messages_removed} messages → summary)\n\n"
                f"{snapshot.summary_text}"
            ),
            token_count=self._count(snapshot.summary_text) + 20,  # overhead
        )

        # Reassemble: system + summary + recent_tools + kept_messages
        self._messages = system_msgs + [summary_msg] + recent_tool_msgs + kept_messages

        logger.info(
            "Compaction complete: %d → %d tokens, kept %d messages + %d tool msgs + snapshot",
            snapshot.original_token_count, self.total_tokens(),
            len(kept_messages), len(recent_tool_msgs),
        )
        return (
            f"⚠️ Context compacted: {snapshot.original_token_count:,} → "
            f"{self.total_tokens():,} tokens "
            f"({snapshot.messages_removed} messages summarized)"
        )

    def _format_conversation(self, messages: list[Message]) -> str:
        parts = []
        for m in messages:
            label = m.role.value.upper()
            # Truncate very long messages in the summary input
            content = m.content
            if len(content) > 3000:
                logger.info("Truncating message from %d to 2800 chars for summary", len(content))
                content = content[:2800] + "\n... [truncated for summary]"
            parts.append(f"[{label}]: {content}")
        return "\n\n".join(parts)

    def _extractive_fallback(self, messages: list[Message]) -> str:
        """Simple extractive summary when no model is available."""
        lines = ["Summary of conversation so far:"]
        for m in messages:
            # Take first 2 lines of each message
            first_lines = m.content.strip().split("\n")[:2]
            prefix = "User" if m.role == Role.USER else "Assistant"
            lines.append(f"- {prefix}: {' '.join(first_lines)[:200]}")
        return "\n".join(lines)
