"""Reasoning agent - turns retrieved evidence into a cited answer.

With Claude available this is a constrained synthesis call: only the numbered
passages may be used, and every claim must carry a citation marker.

Without a key it degrades to *extractive* summarisation rather than pretending
to reason: the highest-scoring sentences from the evidence are selected with an
IDF-weighted overlap against the question and returned verbatim with their
citation numbers. That keeps the free demo honest - it never produces a
sentence that is not in the corpus.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.prompts import REASONING_SYSTEM
from app.agents.state import AgentState, RunContext, trace_event
from app.logging_config import get_logger
from app.rag.embeddings import tokenize

log = get_logger(__name__)

# Source documents are hard-wrapped Markdown, so a naive split on newlines cuts
# sentences in half. Unwrap continuation lines first, then split on real
# sentence boundaries and list-item starts.
_UNWRAP_RE = re.compile(r"(?<![.!?:;|])\n(?![\n\s]|[-*|#])")
# The negative lookahead keeps a trailing citation marker attached to the
# sentence it belongs to - "...EUR 12). [1]" is one unit, not two.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?!\[\d)|\n{2,}|\n(?=\s*[-*])")
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*]\s+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "is",
    "are",
    "be",
    "with",
    "that",
    "this",
    "it",
    "as",
    "at",
    "by",
    "from",
    "can",
    "i",
    "my",
    "me",
    "do",
    "does",
    "how",
    "what",
    "when",
    "where",
    "who",
    "why",
    "we",
    "you",
    "your",
    "our",
    "if",
    "any",
    "there",
    "will",
    "would",
    "should",
}


class ReasonerOutput(BaseModel):
    answer: str = Field(description="Markdown answer with [n] citation markers.")
    used_citations: list[int] = Field(
        default_factory=list, description="Passage numbers actually cited."
    )
    insufficient_evidence: bool = Field(
        default=False, description="True when the passages cannot answer the question."
    )
    follow_up_question: str = Field(
        default="", description="One clarifying question, or empty string."
    )


def split_sentences(text: str) -> list[str]:
    """Sentence-ish units suitable for extraction and groundedness checks."""
    body = text.split("]\n", 1)[-1] if text.startswith("[") else text
    body = _UNWRAP_RE.sub(" ", body)
    parts = []
    for raw in _SENTENCE_RE.split(body):
        cleaned = " ".join(_BULLET_PREFIX_RE.sub("", raw).split())
        if len(cleaned) > 25 and not cleaned.startswith(("#", "|", "---")):
            parts.append(cleaned)
    return parts


def extractive_answer(query: str, retrieved: list[dict], limit: int = 4) -> ReasonerOutput:
    """Deterministic fallback: select, never generate."""
    if not retrieved:
        return ReasonerOutput(
            answer=(
                "I could not find anything in the company knowledge base that covers this. "
                "Try rephrasing, or ask HR to publish the relevant policy."
            ),
            used_citations=[],
            insufficient_evidence=True,
            follow_up_question="Which policy area does this fall under - HR, finance or IT?",
        )

    # IDF over the retrieved set so shared boilerplate counts for little.
    doc_tokens = [set(tokenize(item["content"])) for item in retrieved]
    n_docs = len(doc_tokens) or 1
    df = Counter()
    for tokens in doc_tokens:
        df.update(tokens)

    q_tokens = [t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 2]
    if not q_tokens:
        q_tokens = tokenize(query)
    q_set = set(q_tokens)

    scored: list[tuple[float, int, str]] = []
    for item in retrieved:
        index = item["index"]
        rank_boost = 1.0 / math.sqrt(index)
        # A sentence inherits the subject of its section. "Requests of 3 or more
        # working days require 10 working days of notice" never says "annual
        # leave" - its heading does, and a reader carries that context down.
        # Scoring the sentence alone systematically loses to off-topic
        # sentences that happen to repeat the question's words.
        heading_terms = set(tokenize(item.get("heading", ""))) & q_set

        for sentence in split_sentences(item["content"]):
            s_tokens = set(tokenize(sentence))
            direct = q_set & s_tokens
            inherited = heading_terms - direct
            if not direct:
                continue
            idf = {t: math.log(1 + n_docs / (1 + df[t])) for t in direct | inherited}
            weight = sum(idf[t] for t in direct) + 0.6 * sum(idf[t] for t in inherited)
            # Reward sentences that answer more of the question at once.
            breadth = len(direct | inherited) / len(q_set)
            length_penalty = 1.0 / (1.0 + abs(len(sentence) - 160) / 320)
            scored.append((weight * (0.5 + breadth) * rank_boost * length_penalty, index, sentence))

    scored.sort(key=lambda row: row[0], reverse=True)

    chosen: list[tuple[int, str]] = []
    seen_sentences: set[str] = set()
    per_passage: Counter = Counter()
    for _, index, sentence in scored:
        key = sentence[:80].lower()
        if key in seen_sentences or per_passage[index] >= 2:
            continue
        seen_sentences.add(key)
        per_passage[index] += 1
        chosen.append((index, sentence))
        if len(chosen) >= limit:
            break

    if not chosen:
        top = retrieved[0]
        return ReasonerOutput(
            answer=(
                f"The closest match in the knowledge base is **{top['document_title']}**"
                f" ({top['heading'] or 'general'}) [{top['index']}], but it does not directly "
                "answer your question."
            ),
            used_citations=[top["index"]],
            insufficient_evidence=True,
            follow_up_question="Can you add a bit more detail about what you need?",
        )

    chosen.sort(key=lambda row: row[0])
    titles = {item["index"]: item["document_title"] for item in retrieved}
    lead_sources = sorted({titles[i] for i, _ in chosen})
    bullets = "\n".join(f"- {sentence} [{index}]" for index, sentence in chosen)
    answer = (
        f"Here is what {', '.join(lead_sources)} says:\n\n{bullets}\n\n"
        "_Extractive summary (demo mode): sentences are quoted directly from the "
        "cited policy documents._"
    )
    return ReasonerOutput(
        answer=answer,
        used_citations=sorted({index for index, _ in chosen}),
        insufficient_evidence=False,
        follow_up_question="",
    )


def _smalltalk_answer() -> ReasonerOutput:
    return ReasonerOutput(
        answer=(
            "Hi - I'm the Northwind internal assistant. I can answer questions from "
            "company policy documents and prepare HR, finance and IT requests for you. "
            'Try: *"How much notice do I need to book annual leave, and can you request '
            '3 days off for me starting next Monday?"*'
        ),
        used_citations=[],
        insufficient_evidence=False,
        follow_up_question="",
    )


def reasoning_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()

    query = state["query"]
    retrieved = state.get("retrieved") or []

    if state.get("intent") == "smalltalk":
        output = _smalltalk_answer()
        return {
            "answer": output.answer,
            "used_citations": [],
            "insufficient_evidence": False,
            "follow_up_question": "",
            "trace": [
                trace_event(
                    ctx, "reasoning", "Conversational reply (no evidence needed)", started=started
                )
            ],
        }

    history = state.get("history") or []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) or "(none)"

    user = (
        f"Employee question:\n{query}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Numbered context passages:\n\n{state.get('context_block') or '(no passages found)'}"
    )

    result = ctx.llm.structured(
        agent="reasoning",
        system=REASONING_SYSTEM,
        user=user,
        schema=ReasonerOutput,
        fallback=lambda: extractive_answer(query, retrieved),
    )
    ctx.usage.add("reasoning", result)
    output: ReasonerOutput = result.value  # type: ignore[assignment]

    valid_indices = {item["index"] for item in retrieved}
    used = [c for c in output.used_citations if c in valid_indices]

    return {
        "answer": output.answer.strip(),
        "used_citations": used,
        "insufficient_evidence": output.insufficient_evidence or not retrieved,
        "follow_up_question": output.follow_up_question,
        "trace": [
            trace_event(
                ctx,
                "reasoning",
                (
                    f"Drafted answer citing {len(used)} passage(s)"
                    if used
                    else "Drafted answer with no usable citations"
                ),
                status="warning" if output.insufficient_evidence else "ok",
                payload={
                    "used_citations": used,
                    "insufficient_evidence": output.insufficient_evidence,
                    "mode": result.mode,
                    "characters": len(output.answer),
                },
                started=started,
            )
        ],
    }
