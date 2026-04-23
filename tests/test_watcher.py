"""Tests for the watch-mode debouncer. File watching itself isn't exercised
here — that would depend on OS inotify/FSEvents behavior. We test the
debouncing + dispatch logic directly."""

from __future__ import annotations

import time
from pathlib import Path

from personal_rag.watcher import _DebounceQueue, _PendingEvent


def test_debounce_coalesces_rapid_events(tmp_path):
    calls: list[_PendingEvent] = []
    q = _DebounceQueue(delay=0.2, on_ready=calls.append)
    q.start()
    try:
        p = tmp_path / "a.md"
        p.write_text("x", encoding="utf-8")
        # Three rapid events for the same file
        q.push(p, "upsert")
        q.push(p, "upsert")
        q.push(p, "upsert")
        time.sleep(0.5)  # past the debounce window
    finally:
        q.stop()

    # Only ONE call, even though we pushed three events
    assert len(calls) == 1
    assert calls[0].path == p
    assert calls[0].kind == "upsert"


def test_debounce_rejects_unsupported_suffixes(tmp_path):
    calls: list[_PendingEvent] = []
    q = _DebounceQueue(delay=0.2, on_ready=calls.append)
    q.start()
    try:
        q.push(tmp_path / "ignored.jpg", "upsert")
        q.push(tmp_path / "also_ignored.zip", "upsert")
        time.sleep(0.4)
    finally:
        q.stop()
    assert calls == []


def test_debounce_handles_multiple_paths_independently(tmp_path):
    calls: list[_PendingEvent] = []
    q = _DebounceQueue(delay=0.2, on_ready=calls.append)
    q.start()
    try:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("x", encoding="utf-8")
        b.write_text("y", encoding="utf-8")
        q.push(a, "upsert")
        q.push(b, "upsert")
        time.sleep(0.5)
    finally:
        q.stop()

    paths = {evt.path for evt in calls}
    assert paths == {a, b}


def test_delete_wins_latest_wins(tmp_path):
    calls: list[_PendingEvent] = []
    q = _DebounceQueue(delay=0.2, on_ready=calls.append)
    q.start()
    try:
        p = tmp_path / "c.md"
        q.push(p, "upsert")
        # A delete lands while we're still within the debounce window for the
        # upsert — the latest kind wins.
        q.push(p, "delete")
        time.sleep(0.4)
    finally:
        q.stop()
    assert len(calls) == 1
    assert calls[0].kind == "delete"


def test_apply_event_skips_vanished_file(tmp_path, monkeypatch):
    """If a file is gone by the time the debounce fires, don't crash."""
    from personal_rag.config import Settings
    from personal_rag.watcher import _apply_event, _PendingEvent

    ingested = []

    def fake_ingest(path, settings):
        ingested.append(path)
        return {"chunks_total": 0}

    monkeypatch.setattr("personal_rag.watcher._ingest_path", fake_ingest)

    logs: list[str] = []
    evt = _PendingEvent(path=tmp_path / "gone.md", kind="upsert", at=0.0)
    _apply_event(evt, Settings(_env_file=None), logs.append)
    # No ingest call since the file doesn't exist
    assert ingested == []
