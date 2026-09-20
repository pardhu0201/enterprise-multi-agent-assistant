"""Heading-aware chunking.

Plain fixed-window splitting destroys the structure that policy documents rely
on ("15 working days" means nothing without the *Notice period* heading above
it). So documents are first cut on Markdown headings, each section is then
split with LangChain's recursive splitter, and every chunk is stored with a
breadcrumb prefix (``Document > Section > Subsection``).

That breadcrumb is part of the embedded text, which measurably improves both
retrieval and the readability of citations in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class ParsedDocument:
    title: str
    body: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class TextChunk:
    ordinal: int
    heading: str
    content: str
    token_estimate: int


def parse_frontmatter(raw: str, fallback_title: str) -> ParsedDocument:
    """Read the small ``key: value`` YAML-ish header the seed corpus uses."""
    metadata: dict[str, str] = {}
    body = raw
    match = _FRONTMATTER_RE.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                metadata[key.strip().lower()] = value.strip()
        body = raw[match.end() :]

    title = metadata.get("title", "").strip()
    if not title:
        for line in body.splitlines():
            head = _HEADING_RE.match(line.strip())
            if head:
                title = head.group(2).strip()
                break
    return ParsedDocument(title=title or fallback_title, body=body.strip(), metadata=metadata)


def _sections(body: str) -> list[tuple[str, str]]:
    """Split a Markdown body into ``(breadcrumb, text)`` sections."""
    sections: list[tuple[str, str]] = []
    trail: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((" > ".join(trail), text))
        buffer.clear()

    for line in body.splitlines():
        head = _HEADING_RE.match(line.strip())
        if head:
            flush()
            level = len(head.group(1))
            trail[:] = trail[: level - 1]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(head.group(2).strip())
        else:
            buffer.append(line)
    flush()
    return sections or [("", body.strip())]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token) for budgeting only."""
    return max(1, len(text) // 4)


def chunk_document(
    title: str,
    body: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[TextChunk] = []
    ordinal = 0
    for breadcrumb, text in _sections(body):
        parts: list[str] = []
        for part in [title, *breadcrumb.split(" > ")]:
            if part and part not in parts:
                parts.append(part)
        crumb = " > ".join(parts)
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 40:  # drop stubs left behind by heading splits
                continue
            content = f"[{crumb}]\n{piece}" if crumb else piece
            chunks.append(
                TextChunk(
                    ordinal=ordinal,
                    heading=crumb,
                    content=content,
                    token_estimate=estimate_tokens(content),
                )
            )
            ordinal += 1
    return chunks
