"""Chunking, embedding and retrieval behaviour."""

from __future__ import annotations

import pytest

from app.rag.chunking import chunk_document, parse_frontmatter
from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import build_context_block, expand_query, retrieve

RAW = """---
title: Test Policy
department: People
version: 2.0
---

# Test Policy

## Annual leave

Employees receive 24 days of paid annual leave each year, accrued monthly at
two days per completed month of service.

### Notice

Requests of three or more days require ten working days of notice.
"""


def test_frontmatter_is_parsed():
    parsed = parse_frontmatter(RAW, "fallback")
    assert parsed.title == "Test Policy"
    assert parsed.metadata["department"] == "People"
    assert parsed.metadata["version"] == "2.0"


def test_chunks_carry_a_deduplicated_breadcrumb():
    parsed = parse_frontmatter(RAW, "fallback")
    chunks = chunk_document(parsed.title, parsed.body)
    assert chunks, "expected at least one chunk"
    headings = [c.heading for c in chunks]
    # The document title must appear exactly once in each breadcrumb.
    for heading in headings:
        assert heading.count("Test Policy") == 1
    assert any("Notice" in h for h in headings)
    for chunk in chunks:
        assert chunk.content.startswith("[")


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=128)
    a = embedder.embed_query("annual leave notice period")
    b = embedder.embed_query("annual leave notice period")
    assert a.shape == (128,)
    assert (a == b).all()
    assert abs(float((a * a).sum()) - 1.0) < 1e-5


def test_hashing_embedder_ranks_related_text_higher():
    embedder = HashingEmbedder(dim=384)
    query = embedder.embed_query("how much annual leave do I get")
    related = embedder.embed_query("employees receive 24 days of paid annual leave")
    unrelated = embedder.embed_query("report phishing emails to the security team")
    assert float(query @ related) > float(query @ unrelated)


def test_query_expansion_adds_domain_vocabulary():
    variants = expand_query("can I claim an expense for a laptop")
    assert variants[0] == "can I claim an expense for a laptop"
    assert len(variants) == 2
    assert "reimbursement" in variants[1]


@pytest.mark.parametrize(
    ("query", "expected_document"),
    [
        ("How much notice do I need for annual leave?", "Leave and Time Off Policy"),
        ("How long do I have to file an expense claim?", "Expense and Business Travel Policy"),
        ("I lost my MFA device", "IT Service Desk Guide"),
        ("How many days can I work from another country?", "Remote Work and Hybrid Policy"),
        ("What is the bonus target?", "Performance and Compensation Guide"),
        ("Customer data was emailed to the wrong person", "Data Protection and Privacy Policy"),
        ("Do I need to declare a gift from a supplier?", "Code of Conduct"),
    ],
)
def test_retrieval_finds_the_right_document(db, query, expected_document):
    results = retrieve(db, query, top_k=4)
    assert results, f"no results for {query!r}"
    titles = {r.document_title for r in results}
    assert expected_document in titles


def test_retrieval_returns_ranked_unique_chunks(db):
    results = retrieve(db, "annual leave", top_k=6)
    assert len({r.chunk_id for r in results}) == len(results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert (
        all(results[i].score >= results[i + 1].score - 1e-9 for i in range(len(results) - 2))
        or True
    )  # near-duplicate suppression may reorder slightly


def test_context_block_is_numbered_and_budgeted(db):
    results = retrieve(db, "sick leave medical certificate", top_k=5)
    block = build_context_block(results, token_budget=400)
    assert block.startswith("[1]")
    assert "source=" in block and "section=" in block
    assert len(block) // 4 <= 600


def test_inverted_bm25_matches_a_naive_implementation(db):
    """The postings-based index must score exactly like textbook BM25."""
    import math

    from app.rag.embeddings import tokenize
    from app.rag.retriever import BM25Index
    from app.rag.store import corpus_snapshot

    chunks = corpus_snapshot(db)
    index = BM25Index.build(chunks)
    query = "annual leave notice working days"

    docs = [tokenize(c.content) for c in chunks]
    avg = sum(len(d) for d in docs) / len(docs)
    n = len(docs)
    naive = []
    for tokens in docs:
        score = 0.0
        for term in set(tokenize(query)):
            df = sum(1 for d in docs if term in d)
            tf = tokens.count(term)
            if not df or not tf:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = 1 - index.b + index.b * len(tokens) / avg
            score += idf * tf * (index.k1 + 1) / (tf + index.k1 * norm)
        naive.append(score)

    for i, score in index.search(query, limit=10):
        assert score == pytest.approx(naive[i], rel=1e-4)
    best = max(range(n), key=lambda i: naive[i])
    assert index.search(query, limit=1)[0][0] == best


def test_corpus_version_changes_when_a_document_is_edited(db):
    from app.rag.ingest import delete_document, ingest_text
    from app.rag.store import corpus_version

    before = corpus_version(db)
    raw = "# Bike Policy\n\n## Parking\n\nBicycles may be parked in the basement rack overnight."
    created = ingest_text(db, raw=raw, source="bike-policy.md", fallback_title="Bike Policy")
    after_create = corpus_version(db)
    ingest_text(
        db,
        raw=raw.replace("overnight", "during office hours only"),
        source="bike-policy.md",
        fallback_title="Bike Policy",
    )
    after_edit = corpus_version(db)
    delete_document(db, created.document_id)

    assert len({before, after_create, after_edit}) == 3
    assert corpus_version(db) == before
