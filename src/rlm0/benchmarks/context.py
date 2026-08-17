"""Turn one long benchmark context into documents the harness can grade against.

Public long-context benchmarks hand over a single undivided string. The
harness here grades evidence as well as answers, and evidence is a set of
document identifiers, so an adapter that passed the string through whole would
leave every solver citing the one document it was given. The evidence axis
would still compute, still print, and mean nothing. That is the failure this
project spends most of its energy avoiding elsewhere, so it is not acceptable
here either.

Chunking is therefore a deliberate deviation from the official prompt, and it
is recorded as one in every manifest an adapter writes. What it buys is that
"which parts of the context did you actually read" becomes a question with an
answer, which on an aggregation benchmark is the whole point: an OOLONG
counting question is only answerable by reading all of the window, so the
required evidence set is all of it, and a solver that answers correctly having
cited eight chunks out of two hundred has been caught.

Identifiers are derived from a digest of the sample key and the chunk index,
never from the chunk text. A citation is only evidence of reading if the
identifier could not have been guessed, and an identifier derived from content
can be reconstructed by a solver that saw the content in a summary.
"""

from __future__ import annotations

import hashlib

from rlm0.harness.corpus import Document

__all__ = ["DEFAULT_CHUNK_CHARS", "chunk_context", "locate"]

DEFAULT_CHUNK_CHARS = 2000
"""Roughly five hundred tokens, which keeps a 128K window near 250 documents.

Small enough that citing a chunk is a claim about a specific passage, large
enough that the identifier headers stay a rounding error on the context.
"""

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _doc_id(key: str, index: int) -> str:
    """An opaque, stable identifier for chunk `index` of context `key`."""
    digest = hashlib.sha256(f"{key}:{index}".encode()).digest()
    token = "".join(_ALPHABET[byte % len(_ALPHABET)] for byte in digest[:8])
    return f"DOC-{token}"


def _split_lines(text: str, target_chars: int) -> list[str]:
    """Accumulate whole lines up to the target, never splitting one.

    Line integrity matters more than even chunk sizes here. OOLONG-synth
    contexts are numbered lists of independent short texts, so a chunk boundary
    inside a line would produce a fragment that supports no label at all, and
    the evidence set would then be a claim about a passage that says nothing.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        addition = len(line) + 1
        if current and size + addition > target_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += addition
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def chunk_context(
    text: str, key: str, *, target_chars: int = DEFAULT_CHUNK_CHARS
) -> tuple[Document, ...]:
    """Split a benchmark context into identified documents.

    `key` should be the sample identifier, so that two samples sharing a
    context window still get distinct document identifiers. Sharing them would
    let a solver carry a citation across samples, and the evidence score would
    then be partly a memory test.
    """
    if target_chars < 1:
        raise ValueError("target_chars must be at least 1")
    return tuple(
        Document(doc_id=_doc_id(key, index), lines=tuple(chunk.split("\n")))
        for index, chunk in enumerate(_split_lines(text, target_chars))
    )


def locate(documents: tuple[Document, ...], needle: str) -> frozenset[str]:
    """Which documents contain `needle`, compared case-insensitively.

    Used by the needle adapters to derive the required evidence set from the
    text rather than from a field the dataset happens to provide. An empty
    result is a signal the caller must treat as fatal: a needle benchmark whose
    needle is not in its own haystack has no gradeable evidence, and scoring it
    anyway is how a grader becomes decoration.
    """
    lowered = needle.lower()
    return frozenset(
        doc.doc_id
        for doc in documents
        if lowered in "\n".join(doc.lines).lower()
    )
