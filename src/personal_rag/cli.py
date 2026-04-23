"""Typer CLI: ingest and ask."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from personal_rag import __version__
from personal_rag.config import get_settings
from personal_rag.generate import generate
from personal_rag.ingest import IngestError, ingest
from personal_rag.retrieve import retrieve

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local retrieval-augmented QA over your own documents.",
)
console = Console()
err = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"personal-rag {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Entry point callback — hosts global flags."""
    return


@app.command(name="ingest")
def ingest_cmd(
    path: Path = typer.Argument(..., exists=True, help="File or directory to index."),
    contextual: bool = typer.Option(
        False,
        "--contextual/--no-contextual",
        help=(
            "Use Anthropic's Contextual Retrieval: prefix each chunk with a 1-2 sentence "
            "context from a small model (default: claude-haiku-4-5). Significantly improves "
            "retrieval recall but costs one extra (cached) API call per chunk."
        ),
    ),
) -> None:
    """Read files at PATH, chunk, embed, and upsert into the index."""
    settings = get_settings()
    if contextual:
        settings = settings.model_copy(update={"contextual": True})

    with console.status(f"[cyan]Ingesting {path}...", spinner="dots"):
        try:
            result = ingest(path, settings)
        except IngestError as exc:
            err.print(f"[red]Ingest failed:[/red] {exc}")
            raise typer.Exit(1)

    title_suffix = " (contextual)" if result.get("contextual") else ""
    console.print(
        Panel.fit(
            f"Indexed [bold]{result['files_ingested']}[/bold] files "
            f"into [bold]{result['chunks_total']}[/bold] chunks{title_suffix}.",
            title="Ingest complete",
            border_style="green",
        )
    )
    if result["per_file"]:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("file")
        table.add_column("chunks", justify="right")
        for f, n in sorted(result["per_file"].items()):
            table.add_row(f, str(n))
        console.print(table)


@app.command(name="ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Question to answer."),
    k: int = typer.Option(5, "--top-k", "-k", help="Passages to retrieve.", min=1),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Print retrieved passages."
    ),
    hybrid: bool = typer.Option(
        None,
        "--hybrid/--dense",
        help="Hybrid BM25+dense retrieval (v0.3) vs dense-only. Default: settings.hybrid.",
    ),
) -> None:
    """Ask a question — retrieves relevant chunks and answers with Claude."""
    settings = get_settings()
    if hybrid is not None:
        settings = settings.model_copy(update={"hybrid": hybrid})
    with console.status("[cyan]Retrieving...", spinner="dots"):
        passages = retrieve(question, settings, k=k)

    with console.status(f"[cyan]Generating answer with {settings.model}...", spinner="dots"):
        try:
            answer = generate(question, passages, settings)
        except Exception as exc:
            err.print(f"[red]Generation failed:[/red] {exc}")
            raise typer.Exit(1)

    console.print(Panel(answer.text, title="Answer", border_style="cyan"))

    if show_sources and passages:
        table = Table(show_header=True, header_style="bold cyan", title="Retrieved passages")
        table.add_column("#", justify="right")
        table.add_column("cited?", justify="center")
        table.add_column("source")
        table.add_column("chunk", justify="right")
        table.add_column("preview")
        cited = set(answer.cited_indices)
        for i, p in enumerate(passages, 1):
            preview = p["text"][:100].replace("\n", " ") + ("..." if len(p["text"]) > 100 else "")
            marker = "[green]✓[/green]" if i in cited else "[dim]·[/dim]"
            table.add_row(str(i), marker, p["source"], str(p["chunk_index"]), preview)
        console.print(table)
        if cited and len(cited) < len(passages):
            console.print(
                f"[dim]{len(cited)}/{len(passages)} passages were cited inline. "
                "Uncited passages were retrieved but not used by the model.[/dim]"
            )


@app.command(name="serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, help="Reload on code change (dev only)."),
) -> None:
    """Start the FastAPI server — exposes /query for programmatic access."""
    import uvicorn

    uvicorn.run("personal_rag.server:app", host=host, port=port, reload=reload)


@app.command(name="watch")
def watch_cmd(
    path: Path = typer.Argument(..., exists=True, help="Directory to watch."),
    debounce_s: float = typer.Option(
        1.5,
        "--debounce",
        help="Seconds of quiet required after the last change before re-ingesting.",
    ),
) -> None:
    """Watch a directory and re-ingest files on any change. Ctrl-C to stop."""
    from personal_rag.watcher import watch

    settings = get_settings()
    if not path.is_dir():
        err.print(f"[red]Not a directory:[/red] {path}")
        raise typer.Exit(1)

    def _log(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    try:
        watch(path.resolve(), settings, debounce_s=debounce_s, console_log=_log)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command(name="similar")
def similar_cmd(
    target: str = typer.Argument(
        ...,
        help="Either a chunk id (e.g. 'path/to/note.md:3') or a source path.",
    ),
    k: int = typer.Option(5, "--k", min=1, help="How many results to return."),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="'chunk', 'source', or 'auto' (chunk if target contains ':', else source).",
    ),
) -> None:
    """Find chunks/notes similar to the given chunk id or source path.

    Two shapes: chunk-level ('what else have I written that's close to this
    paragraph?') or source-level ('which of my notes is related to this whole
    note?'). Source mode aggregates per-chunk search results across the
    whole source and ranks other sources by reciprocal rank.
    """
    from personal_rag.similar import similar_to_chunk, similar_to_source

    settings = get_settings()

    if mode == "auto":
        resolved_mode = "chunk" if ":" in target else "source"
    else:
        resolved_mode = mode

    try:
        if resolved_mode == "chunk":
            rows = similar_to_chunk(settings, target, k=k)
            if not rows:
                console.print(f"[dim]No similar chunks for {target!r}.[/dim]")
                return
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("source")
            table.add_column("chunk")
            table.add_column("preview")
            for row in rows:
                preview = row["text"].strip().replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                table.add_row(row["source"], str(row["chunk_index"]), preview)
            console.print(table)
        elif resolved_mode == "source":
            ranked = similar_to_source(settings, target, k=k)
            if not ranked:
                console.print(f"[dim]No similar sources for {target!r}.[/dim]")
                return
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("source")
            table.add_column("score", justify="right")
            for src, score in ranked:
                table.add_row(src, f"{score:.4f}")
            console.print(table)
        else:
            err.print(f"[red]Unknown mode {mode!r}. Use 'chunk', 'source', or 'auto'.[/red]")
            raise typer.Exit(1)
    except KeyError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
