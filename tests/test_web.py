"""Tests for the /ui HTML endpoints. FastAPI TestClient drives the app
without binding a port; we don't mock retrieve/generate — instead we test
the cases where they fail cleanly (no API key set)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from personal_rag.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_ui_home_renders_form(client):
    resp = client.get("/ui")
    assert resp.status_code == 200
    body = resp.text
    assert "personal-rag" in body
    assert '<form method="post" action="/ui/ask">' in body
    assert "Ask something about your notes" in body


def test_ui_ask_without_api_key_shows_error(client, monkeypatch):
    # Scrub the API key so generate() raises on purpose
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("personal_rag.web.retrieve", lambda q, s, **kw: [])  # empty retrieval path
    resp = client.post("/ui/ask", data={"question": "anything"})
    # The empty-passages branch in generate() still returns an Answer without
    # hitting the API, so the page renders an answer block (not an error).
    assert resp.status_code == 200
    assert "No relevant passages" in resp.text


def test_ui_ask_escapes_user_input(client, monkeypatch):
    """User-supplied question text must be HTML-escaped in the echoed form."""
    monkeypatch.setattr("personal_rag.web.retrieve", lambda q, s, **kw: [])
    nasty = "<script>alert('xss')</script>"
    resp = client.post("/ui/ask", data={"question": nasty})
    assert resp.status_code == 200
    assert "<script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def test_ui_ask_renders_cited_marker_for_cited_passages(client, monkeypatch):
    """When a passage index appears in the answer text as [N], the table
    row should mark it cited."""
    from personal_rag.generate import Answer

    fake_passages = [
        {"id": "a:0", "source": "note1.md", "chunk_index": 0, "text": "first content"},
        {"id": "b:0", "source": "note2.md", "chunk_index": 0, "text": "second content"},
    ]

    def fake_retrieve(q, s, **kw):
        return fake_passages

    def fake_generate(q, passages, s, **kw):
        return Answer(
            text="The first note says something important [1].",
            used_passages=passages,
            cited_indices=[1],
        )

    monkeypatch.setattr("personal_rag.web.retrieve", fake_retrieve)
    monkeypatch.setattr("personal_rag.web.generate", fake_generate)

    resp = client.post("/ui/ask", data={"question": "what does note1 say?"})
    assert resp.status_code == 200
    body = resp.text
    # Cited passage row carries the green checkmark class; uncited the dim dot.
    assert '<span class="cited">✓</span>' in body
    assert '<span class="uncited">·</span>' in body
    assert "first content" in body
