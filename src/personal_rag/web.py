"""Tiny HTML UI mounted on the FastAPI app.

Deliberately minimal: no React, no htmx, no template engine. Server-rendered
HTML on GET / POST. This gets you a usable "chat with my notes" surface in
about 150 lines, no frontend toolchain, no build step.

For anything more interactive we'd reach for htmx or a proper SPA, but at
that point the backend is the hard part anyway — the point of this file
is that personal-rag is usable from a browser without touching any of that.
"""

from __future__ import annotations

import html

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from personal_rag.config import get_settings
from personal_rag.generate import generate
from personal_rag.retrieve import retrieve

router = APIRouter()

PAGE_CSS = """
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 780px;
         margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; margin: 0 0 1rem; }
  form { display: flex; gap: .5rem; margin-bottom: 1.5rem; }
  input[type=text] { flex: 1; padding: .5rem .7rem; border: 1px solid #999;
                     border-radius: 6px; font: inherit; }
  button { padding: .5rem 1rem; border: 1px solid #999; border-radius: 6px;
           background: #f3f3f3; cursor: pointer; font: inherit; }
  .answer { background: #f9f9f9; padding: 1rem 1.25rem; border-left: 3px solid #3b82f6;
            border-radius: 4px; white-space: pre-wrap; }
  .passages { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 13px; }
  .passages th, .passages td { text-align: left; padding: .4rem .5rem;
                               border-bottom: 1px solid #e4e4e4; vertical-align: top; }
  .passages th { background: #f3f3f3; font-weight: 600; }
  .cited { color: #16a34a; font-weight: 600; }
  .uncited { color: #999; }
  .footer { margin-top: 2rem; font-size: 12px; color: #666; }
  .error { color: #b91c1c; background: #fee2e2; padding: .75rem 1rem; border-radius: 4px; }
  @media (prefers-color-scheme: dark) {
    body { background: #111; color: #eee; }
    input[type=text] { background: #1a1a1a; color: #eee; border-color: #444; }
    button { background: #1a1a1a; color: #eee; border-color: #444; }
    .answer { background: #1a1a1a; border-left-color: #60a5fa; }
    .passages th { background: #1a1a1a; }
    .passages th, .passages td { border-bottom-color: #333; }
    .footer { color: #888; }
    .error { background: #4a1414; color: #fee; }
  }
"""

INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>personal-rag</title>
<style>{css}</style></head>
<body>
<h1>personal-rag</h1>
<form method="post" action="/ui/ask">
  <input type="text" name="question" placeholder="Ask something about your notes..."
         value="{question}" autofocus required>
  <button type="submit">Ask</button>
</form>
{body}
<p class="footer">
  {model} · top-k {top_k} · embedding {embed}
</p>
</body>
</html>
"""


def _render(body: str, *, question: str = "") -> HTMLResponse:
    settings = get_settings()
    return HTMLResponse(
        INDEX_TEMPLATE.format(
            css=PAGE_CSS,
            question=html.escape(question, quote=True),
            body=body,
            model=html.escape(settings.model),
            top_k=settings.top_k,
            embed=html.escape(settings.embedding_model),
        )
    )


@router.get("/ui", response_class=HTMLResponse)
async def ui_home(request: Request) -> HTMLResponse:
    return _render("")


@router.post("/ui/ask", response_class=HTMLResponse)
async def ui_ask(question: str = Form(...)) -> HTMLResponse:
    settings = get_settings()
    try:
        passages = retrieve(question, settings)
        answer = generate(question, passages, settings)
    except Exception as exc:  # noqa: BLE001
        return _render(
            f'<div class="error">{html.escape(str(exc))}</div>',
            question=question,
        )

    cited = set(answer.cited_indices)

    rows_html = []
    for i, p in enumerate(passages, 1):
        source = html.escape(p.get("source", ""))
        preview = html.escape(p.get("text", "")[:220]).replace("\n", " ")
        if i in cited:
            mark = '<span class="cited">✓</span>'
        else:
            mark = '<span class="uncited">·</span>'
        rows_html.append(
            f"<tr><td>{i}</td><td>{mark}</td><td>{source}</td>"
            f"<td>{p.get('chunk_index', '')}</td><td>{preview}</td></tr>"
        )

    passages_html = ""
    if passages:
        passages_html = (
            '<table class="passages">'
            "<thead><tr><th>#</th><th>cited?</th><th>source</th>"
            "<th>chunk</th><th>preview</th></tr></thead>"
            f'<tbody>{"".join(rows_html)}</tbody>'
            "</table>"
        )

    body = f'<div class="answer">{html.escape(answer.text)}</div>{passages_html}'
    return _render(body, question=question)
