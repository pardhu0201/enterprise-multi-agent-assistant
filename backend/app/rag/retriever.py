"""Hybrid retrieval: BM25 + dense vectors, fused by score and rank together.

Why hybrid: policy questions mix rare exact tokens ("Form HR-14", "10 working
days") where lexical matching wins, with paraphrases ("time off for a new
baby" -> "parental leave") where dense vectors win. Neither leg is reliable
alone, and their scores are on incomparable scales, so `_fuse` combines a
min-max normalised score with a normalised reciprocal rank from each leg.

Two corrections sit on top of the fusion: a heading boost, because a section
whose own title answers the question is almost always the right passage; and
near-duplicate suppression, which removes the overlapping windows chunking
produces without demoting genuinely distinct sections.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.logging_config import get_logger
from app.rag.embeddings import get_embedder, tokenize
from app.rag.store import ScoredChunk, corpus_snapshot, corpus_version, dense_search

log = get_logger(__name__)

RRF_K = 12
HEADING_BOOST = 0.35
_QUERY_STOPWORDS = {
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
    "do",
    "does",
    "did",
    "how",
    "what",
    "when",
    "where",
    "who",
    "why",
    "can",
    "i",
    "my",
    "me",
    "we",
    "you",
    "your",
    "need",
    "get",
    "much",
    "many",
    "long",
    "if",
    "any",
    "there",
    "will",
    "would",
    "should",
    "have",
    "has",
    "was",
    "were",
}


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
@dataclass
class BM25Index:
    chunks: list[ScoredChunk]
    doc_tokens: list[list[str]]
    doc_freq: Counter
    avg_len: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunks: list[ScoredChunk]) -> BM25Index:
        doc_tokens = [tokenize(c.content) for c in chunks]
        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))
        avg_len = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(chunks=chunks, doc_tokens=doc_tokens, doc_freq=doc_freq, avg_len=avg_len or 1.0)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []
        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float32)
        for term in set(q_tokens):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                norm = 1.0 - self.b + self.b * (len(tokens) / self.avg_len)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        order = np.argsort(-scores)[:limit]
        return [(int(i), float(scores[int(i)])) for i in order if scores[int(i)] > 0]


_index_cache: dict[str, BM25Index] = {}


def _get_index(db: Session) -> BM25Index:
    version = corpus_version(db)
    cached = _index_cache.get(version)
    if cached is None:
        _index_cache.clear()
        cached = BM25Index.build(corpus_snapshot(db))
        _index_cache[version] = cached
        log.info("Built BM25 index over %d chunks (version=%s)", len(cached.chunks), version)
    return cached


def invalidate_index() -> None:
    _index_cache.clear()


# ---------------------------------------------------------------------------
# Fusion + diversification
# ---------------------------------------------------------------------------
NEAR_DUPLICATE_JACCARD = 0.6


def _mmr(candidates: list[ScoredChunk], top_k: int, penalty: float = 0.5) -> list[ScoredChunk]:
    """Near-duplicate suppression in the style of Maximal Marginal Relevance.

    Classic MMR applies a smooth redundancy penalty, which on a corpus of
    tightly-related policy sections demotes genuinely distinct passages (the
    *Requesting annual leave* section looks similar to *Annual leave
    entitlement*, but only one of them contains the notice periods). Here the
    penalty only engages above a near-duplicate threshold, so it removes the
    overlapping windows that chunking produces without reordering real content.
    """
    if len(candidates) <= top_k:
        return candidates
    token_sets = [set(tokenize(c.content)) for c in candidates]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_value = None, -1e9
        for i in remaining:
            redundancy = 0.0
            for j in selected:
                union = token_sets[i] | token_sets[j]
                if union:
                    redundancy = max(redundancy, len(token_sets[i] & token_sets[j]) / len(union))
            excess = max(0.0, redundancy - NEAR_DUPLICATE_JACCARD) / (1.0 - NEAR_DUPLICATE_JACCARD)
            value = candidates[i].score - penalty * excess
            if value > best_value:
                best_idx, best_value = i, value
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)  # type: ignore[arg-type]
    return [candidates[i] for i in selected]


def _fuse(legs: list[tuple[list[tuple[str, float]], float]]) -> dict[str, float]:
    """Combine ranked result lists from different scorers.

    Pure Reciprocal Rank Fusion throws away score magnitude, which on a corpus
    of this size flattens a decisive BM25 win (9.5 vs 5.7) into a near-tie and
    lets the weaker dense leg decide the order. Pure score fusion has the
    opposite problem: the two scorers are on incomparable scales and one
    outlier dominates.

    So each leg contributes both - a min-max normalised score and a normalised
    reciprocal rank, averaged - and the legs are then summed with their
    weights.
    """
    fused: dict[str, float] = {}
    for hits, weight in legs:
        if not hits:
            continue
        scores = [s for _, s in hits]
        low, high = min(scores), max(scores)
        span = (high - low) or 1.0
        for rank, (chunk_id, score) in enumerate(hits):
            normalised = (score - low) / span
            reciprocal = (RRF_K + 1) / (RRF_K + rank + 1)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * (
                0.6 * normalised + 0.4 * reciprocal
            )
    return fused


def expand_query(query: str) -> list[str]:
    """Add a couple of cheap domain-aware paraphrases to widen lexical recall."""
    base = query.strip()
    synonyms = {
        "time off": "leave vacation holiday",
        "vacation": "annual leave",
        "pto": "annual leave entitlement",
        "sick": "sick leave medical certificate",
        "wfh": "remote work from home hybrid",
        "work from home": "remote work hybrid policy",
        "reimburse": "expense claim reimbursement receipt",
        "expense": "claim reimbursement receipt finance portal",
        "laptop": "hardware equipment it service desk",
        "password": "credentials multi-factor authentication access",
        "mfa": "multi-factor authentication identity account",
        "maternity": "parental leave",
        "paternity": "parental leave",
        "promotion": "career framework calibration level",
        "bonus": "compensation payout target",
    }
    lowered = base.lower()
    extra = [v for k, v in synonyms.items() if k in lowered]
    variants = [base]
    if extra:
        variants.append(f"{base} {' '.join(extra)}")
    return variants


def retrieve(
    db: Session,
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
) -> list[ScoredChunk]:
    top_k = top_k or settings.retrieval_top_k
    candidates = candidates or settings.retrieval_candidates

    index = _get_index(db)
    if not index.chunks:
        return []

    embedder = get_embedder()
    # The hashing embedder is a weaker signal than a real sentence encoder, so
    # it gets a smaller vote in the fusion. With sentence-transformers enabled
    # the two legs are weighted equally.
    dense_weight = 0.55 if getattr(embedder, "name", "") == "hashing" else 1.0

    by_id: dict[str, ScoredChunk] = {}
    legs: list[tuple[list[tuple[str, float]], float]] = []

    for position, variant in enumerate(expand_query(query)):
        # Later variants are paraphrases - they should not outvote the original.
        variant_weight = 1.0 if position == 0 else 0.7

        # Dense leg
        dense_hits: list[tuple[str, float]] = []
        for hit in dense_search(db, embedder.embed_query(variant), candidates):
            existing = by_id.setdefault(hit.chunk_id, hit)
            existing.dense_score = max(existing.dense_score, hit.dense_score)
            dense_hits.append((hit.chunk_id, hit.dense_score))
        legs.append((dense_hits, dense_weight * variant_weight))

        # Lexical leg
        lexical_hits: list[tuple[str, float]] = []
        for idx, score in index.search(variant, candidates):
            chunk = index.chunks[idx]
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                existing = ScoredChunk(**{**chunk.__dict__})
                by_id[chunk.chunk_id] = existing
            existing.lexical_score = max(existing.lexical_score, score)
            lexical_hits.append((chunk.chunk_id, score))
        legs.append((lexical_hits, variant_weight))

    fused = _fuse(legs)

    # Heading boost: a section whose own title answers the question is almost
    # always the right passage, and pure term frequency under-weights it
    # because headings are short.
    content_terms = {t for t in tokenize(query) if t not in _QUERY_STOPWORDS and len(t) > 2}
    if content_terms:
        for chunk_id, chunk in by_id.items():
            heading_terms = set(tokenize(chunk.heading))
            overlap = len(content_terms & heading_terms) / len(content_terms)
            fused[chunk_id] = fused.get(chunk_id, 0.0) * (1.0 + HEADING_BOOST * overlap)

    ranked = sorted(by_id.values(), key=lambda c: fused.get(c.chunk_id, 0.0), reverse=True)
    best = max(fused.values()) if fused else 1.0
    for chunk in ranked:
        chunk.score = fused.get(chunk.chunk_id, 0.0) / (best or 1.0)

    diversified = _mmr(ranked[: max(top_k * 3, top_k)], top_k)
    for position, chunk in enumerate(diversified, start=1):
        chunk.rank = position
    return diversified


def build_context_block(chunks: list[ScoredChunk], token_budget: int = 3200) -> str:
    """Render retrieved chunks as a numbered, citable context block.

    Numbering here is the contract the reasoning agent cites against and the
    verification agent checks - so it is built once, in one place.
    """
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        body = chunk.content
        cost = max(1, len(body) // 4)
        if used + cost > token_budget and parts:
            break
        used += cost
        parts.append(
            f"[{i}] source={chunk.document_title} | section={chunk.heading or 'n/a'} "
            f"| department={chunk.department}\n{body}"
        )
    return "\n\n---\n\n".join(parts)
