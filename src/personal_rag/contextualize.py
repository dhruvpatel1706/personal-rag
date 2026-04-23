"""Contextual retrieval: prefix each chunk with a short situating context.

Based on Anthropic's September 2024 post "Introducing Contextual Retrieval". The
idea is that a chunk taken out of context ("It supports streaming and tool use")
is hard for an embedding model to situate, but a prefix like "From the Anthropic
Python SDK README, section on the Messages API..." makes the same chunk much
more retrievable.

We generate the context once per (document, chunk) pair with a cheap + fast model
and *prompt-cache the document*, so the full doc is sent only on the first chunk
of each file — every subsequent chunk in that doc reuses the cache.
"""

from __future__ import annotations

import anthropic

CONTEXT_SYSTEM_PROMPT = (
    "You write concise situating-context strings for chunks of a larger document "
    "so they can be found by semantic search. Your output is prepended verbatim to the chunk "
    "before embedding, so keep it short (one or two sentences) and make it specific — name the "
    "document topic, the section, and why this chunk matters relative to the rest."
)


def _build_user_prompt(chunk: str) -> str:
    return (
        "Here is the chunk we want to situate:\n\n"
        f"<chunk>\n{chunk}\n</chunk>\n\n"
        "Give a short succinct context (1–2 sentences) that names the document topic and "
        "this chunk's place in it, so a search over embeddings can retrieve this chunk when "
        "it's relevant. Answer with ONLY the context string — no preamble, no quotes."
    )


def contextualize_chunk(
    chunk: str,
    full_doc: str,
    *,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 150,
) -> str:
    """Generate a situating-context string for `chunk` relative to `full_doc`.

    The document is sent as a cached prefix so repeated calls with the same
    `full_doc` (one per chunk) pay the cache-read cost, not the full doc cost.
    """
    if not chunk.strip():
        return ""

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": CONTEXT_SYSTEM_PROMPT,
            },
            {
                "type": "text",
                "text": f"<document>\n{full_doc}\n</document>",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": _build_user_prompt(chunk)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return text


def apply_context(context: str, chunk: str) -> str:
    """Prepend a one-line context to the chunk. Empty context = chunk unchanged."""
    context = (context or "").strip()
    if not context:
        return chunk
    return f"{context}\n\n{chunk}"
