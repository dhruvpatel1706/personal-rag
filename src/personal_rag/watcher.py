"""Filesystem watcher that keeps the index in sync with a directory of notes.

We debounce events so that a `save` storm (editors touch the file multiple
times in quick succession, especially with swap files and auto-backup plugins)
doesn't fan out into N re-ingestions. The debounce is per-path; unrelated
files don't block each other.

Delete events remove every row with that source. Unsupported file types
(anything not in `ingest.SUPPORTED_SUFFIXES`) are silently ignored.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from personal_rag.config import Settings
from personal_rag.embed import embedding_dim
from personal_rag.index import Index
from personal_rag.ingest import SUPPORTED_SUFFIXES
from personal_rag.ingest import ingest as _ingest_path

log = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_S = 1.5


@dataclass
class _PendingEvent:
    path: Path
    kind: str  # "upsert" or "delete"
    at: float  # monotonic time of the last observed event for this path


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: "_DebounceQueue"):
        super().__init__()
        self.queue = queue

    def on_created(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.queue.push(Path(event.src_path), "upsert")

    def on_modified(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.queue.push(Path(event.src_path), "upsert")

    def on_moved(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            # Treat as delete-then-upsert: the old path is gone, the new one
            # needs re-ingesting.
            self.queue.push(Path(event.src_path), "delete")
            self.queue.push(Path(event.dest_path), "upsert")

    def on_deleted(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self.queue.push(Path(event.src_path), "delete")


class _DebounceQueue:
    """Coalesces rapid events on the same path into a single action."""

    def __init__(self, *, delay: float, on_ready: Callable[[_PendingEvent], None]):
        self.delay = delay
        self.on_ready = on_ready
        self._pending: dict[Path, _PendingEvent] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def push(self, path: Path, kind: str) -> None:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return
        with self._lock:
            self._pending[path] = _PendingEvent(path=path, kind=kind, at=time.monotonic())

    def _drain(self) -> None:
        while not self._stop.is_set():
            time.sleep(min(0.25, self.delay / 4))
            now = time.monotonic()
            ready: list[_PendingEvent] = []
            with self._lock:
                for path, evt in list(self._pending.items()):
                    if now - evt.at >= self.delay:
                        ready.append(evt)
                        del self._pending[path]
            for evt in ready:
                try:
                    self.on_ready(evt)
                except Exception:  # noqa: BLE001
                    log.exception("watcher callback failed for %s", evt.path)


def _apply_event(
    evt: _PendingEvent, settings: Settings, console_log: Callable[[str], None]
) -> None:
    dim = embedding_dim(settings.embedding_model)
    index = Index(settings.db_path, settings.table_name, dim)
    if evt.kind == "delete":
        removed = index.remove_source(str(evt.path.resolve()))
        if removed:
            console_log(f"removed {removed} chunk(s) for {evt.path}")
        return
    if not evt.path.exists():
        # File briefly existed then vanished before debounce fired.
        return
    try:
        result = _ingest_path(evt.path, settings)
    except Exception as exc:  # noqa: BLE001
        console_log(f"[error] {evt.path}: {exc}")
        return
    n = result.get("chunks_total", 0)
    if n:
        console_log(f"indexed {n} chunk(s) from {evt.path}")


def watch(
    target: Path,
    settings: Settings,
    *,
    debounce_s: float = DEFAULT_DEBOUNCE_S,
    console_log: Callable[[str], None] = print,
) -> None:
    """Block on a filesystem watch of `target`, reindexing as files change.

    Handles SIGINT cleanly. Use Ctrl-C to stop.
    """
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Watch target must be an existing directory: {target}")

    queue = _DebounceQueue(
        delay=debounce_s,
        on_ready=lambda evt: _apply_event(evt, settings, console_log),
    )
    handler = _Handler(queue)

    observer = Observer()
    observer.schedule(handler, str(target), recursive=True)
    observer.start()
    queue.start()

    console_log(f"watching {target} (debounce {debounce_s:.1f}s). Ctrl-C to stop.")
    try:
        while observer.is_alive():
            observer.join(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        queue.stop()
        observer.stop()
        observer.join(timeout=2.0)
        console_log("stopped.")
