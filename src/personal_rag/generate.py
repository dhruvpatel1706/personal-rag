"""Call Claude with retrieved context and return an answer + inline citations.

v0.4 switches to inline `[N]` citations embedded in the answer text, rather
than a trailing "Sources:" list. This makes it obvious which sentence
came from which passage and lets the UI surface passages on-demand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anthropic

from personal_rag.config import Settings

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the provided context passages. If the answer isn't in the context, say so \
plainly — do not invent facts.

Cite inline using [N] where N is the passage number as listed above the question. \
Every factual claim that comes from a passage must carry the bracketed citation at \
the end of its sentence or clause. Multiple passages supporting one point are fine: \
[1][3]. Don't cite passages you didn't actually rely on.

Keep the answer to 1-4 sentences unless the user asks for more detail. Do NOT add a \
trailing 'Sources:' list — the client renders the passage map separately."""

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    used_passages: list[dict]
    cited_indices: list[int] = field(default_factory=list)
    """1-based passage numbers actually cited inline. Excludes anything the model
    was shown but didn't use."""


def extract_citations(text: str, *, max_index: int) -> list[int]:
    """Parse `[N]` citations from `text`. Returns a sorted unique list of indices.

    Drops indices > max_index (the model hallucinated a passage number).
    """
    seen: set[int] = set()
    for m in _CITE_RE.finditer(text):
        try:
            n = int(m.group(1))
        except ValueError:  # pragma: no cover — regex guarantees digits
            continue
        if 1 <= n <= max_index:
            seen.add(n)
    return sorted(seen)


def generate(
    question: str,
    passages: list[dict],
    settings: Settings,
    *,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    """Synthesize an answer from `question` and retrieved `passages`."""
    if not question.strip():
        raise ValueError("Empty question.")
    if not passages:
        return Answer(
            text="No relevant passages were retrieved. Try ingesting more documents "
            "or rephrasing the question.",
            used_passages=[],
            cited_indices=[],
        )

    if client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    context_block = "\n\n".join(
        f"[Passage {i + 1} — {p['source']} chunk {p['chunk_index']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    user_prompt = (
        f"Context passages:\n\n{context_block}\n\n"
        f"---\nQuestion: {question}\n\n"
        "Answer using only the passages above, citing inline with [N]."
    )

    response = client.messages.create(
        model=settings.model,
        max_tokens=1500,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return Answer(
        text=text,
        used_passages=passages,
        cited_indices=extract_citations(text, max_index=len(passages)),
    )
