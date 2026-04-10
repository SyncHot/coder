"""Tests for ConversationStore — SQLite conversation persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from codator.core.conversation_store import ConversationStore
from codator.domain.models import Message, Role


@pytest.fixture()
def store(tmp_path: Path) -> ConversationStore:
    db = tmp_path / "test.db"
    s = ConversationStore(db_path=db)
    yield s
    s.close()


def _make_messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are a helper."),
        Message(role=Role.USER, content="Hello!"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ]


class TestConversationStore:
    def test_save_and_load(self, store: ConversationStore):
        msgs = _make_messages()
        store.save("conv1", msgs, title="Test conv")

        loaded = store.load("conv1")
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0].role == Role.SYSTEM
        assert loaded[1].content == "Hello!"
        assert loaded[2].role == Role.ASSISTANT

    def test_load_nonexistent(self, store: ConversationStore):
        assert store.load("nope") is None

    def test_overwrite(self, store: ConversationStore):
        msgs1 = _make_messages()
        store.save("conv1", msgs1)

        msgs2 = [Message(role=Role.USER, content="New msg")]
        store.save("conv1", msgs2)

        loaded = store.load("conv1")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].content == "New msg"

    def test_list_conversations(self, store: ConversationStore):
        store.save("a", _make_messages(), title="First")
        store.save("b", _make_messages(), title="Second")

        convs = store.list_conversations()
        assert len(convs) == 2
        assert convs[0]["title"] in ("First", "Second")

    def test_delete(self, store: ConversationStore):
        store.save("conv1", _make_messages())
        assert store.delete("conv1") is True
        assert store.load("conv1") is None
        assert store.delete("conv1") is False

    def test_auto_title(self, store: ConversationStore):
        msgs = [Message(role=Role.USER, content="My first question")]
        store.save("auto", msgs)
        convs = store.list_conversations()
        assert convs[0]["title"] == "My first question"

    def test_metadata_round_trip(self, store: ConversationStore):
        msgs = [
            Message(
                role=Role.ASSISTANT,
                content="result",
                metadata={"tool_call_id": "tc123"},
            ),
        ]
        store.save("meta", msgs)
        loaded = store.load("meta")
        assert loaded is not None
        assert loaded[0].metadata["tool_call_id"] == "tc123"
