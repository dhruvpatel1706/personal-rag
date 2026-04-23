"""Call Claude with retrieved context and return an answer + citations."""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from personal_rag.config import Settings

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the provided context passages. If the answer isn't in the context, say so \
plainly — do not invent facts.

Format:
1. A concise answer (1-4 sentences).
2. A "Sources" list that cites which passages (by number) you used.

Never cite a passage you didn't actually use."""


@dataclass
class Answer:
    text: str
    used_passages: list[dict]


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
        "Answer using only the passages above."
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
    return Answer(text=text, used_passages=passages)
