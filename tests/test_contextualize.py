"""Tests for the Contextual Retrieval helpers.

We do not hit the live API — a stubbed client demonstrates the call shape,
and the `apply_context` helper is pure.
"""

from __future__ import annotations

from dataclasses import dataclass

from personal_rag.contextualize import (
    CONTEXT_SYSTEM_PROMPT,
    apply_context,
    contextualize_chunk,
)


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Response:
    content: list


class _StubClient:
    """Minimal stub of anthropic.Anthropic that records the last request and replies."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[dict] = []
        self.messages = self  # type: ignore[assignment]

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return _Response(content=[_Block(type="text", text=self._reply)])


def test_apply_context_prepends_context():
    out = apply_context("From doc X, section Y.", "body line")
    assert out.startswith("From doc X, section Y.")
    assert out.endswith("body line")


def test_apply_context_passthrough_on_empty():
    assert apply_context("", "body") == "body"
    assert apply_context("   ", "body") == "body"


def test_contextualize_chunk_sends_cached_document():
    client = _StubClient("This is the section on RAG chunking.")
    got = contextualize_chunk(
        chunk="chunk text",
        full_doc="full document text",
        client=client,  # type: ignore[arg-type]
        model="claude-haiku-4-5",
    )
    assert got == "This is the section on RAG chunking."

    # Exactly one call, with system prompt split and document cached.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    system = call["system"]
    assert isinstance(system, list) and len(system) == 2
    assert system[0]["text"] == CONTEXT_SYSTEM_PROMPT
    assert "full document text" in system[1]["text"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    # And the user message names the chunk
    assert "chunk text" in call["messages"][0]["content"]


def test_contextualize_chunk_empty_returns_empty():
    client = _StubClient("should not be called")
    assert contextualize_chunk("", "full doc", client=client) == ""  # type: ignore[arg-type]
    assert client.calls == []
