"""Pluggable embedding backends.

Two implementations ship with the project:

``hashing``
    A dependency-free hashed bag-of-features encoder (words + character
    n-grams, sub-linear term frequency, L2 normalised). It costs nothing to
    run, needs no model download, and is deterministic - which is what keeps
    the public demo free and CI reproducible. Paired with BM25 in the hybrid
    retriever it is a genuinely usable lexical/semantic-ish signal.

``sentence-transformers``
    Real dense embeddings (default ``all-MiniLM-L6-v2``) for when quality
    matters more than install size. Enabled with
    ``EMBEDDING_PROVIDER=sentence-transformers``.

Both satisfy the same interface, so the vector store, retriever and ingestion
pipeline are unaware of which one is active.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Light suffix normalisation so inflections match.

    Policy prose and employee questions rarely agree on inflection - the
    document says "Expenses must be claimed within 30 days", the employee asks
    "how long do I have to file an expense claim". Without this, those share no
    tokens at all and the correct sentence loses to worse ones.

    The trailing-``e`` strip after ``-ed``/``-ing`` removal is what keeps the
    forms consistent: "requires" -> "require" -> "requir" matches
    "required" -> "requir". Collisions like care/car are acceptable in a
    lexical index; missed matches are not.
    """
    if len(word) <= 3 or word.isdigit():
        return word

    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    if word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]

    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, stemmed word tokenizer shared by the embedder and BM25."""
    return [stem(w) for w in _WORD_RE.findall(text.lower())]


def raw_tokens(text: str) -> list[str]:
    """Unstemmed tokens, for checks that must compare surface forms."""
    return _WORD_RE.findall(text.lower())


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing over words, bigrams and character 4-grams."""

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    # -- features ----------------------------------------------------------
    # Relative importance of each feature family. Character n-grams are far
    # more numerous than words, so without down-weighting they swamp word
    # identity and every document ends up looking alike.
    FEATURE_WEIGHTS = {"w": 1.0, "b": 0.7, "c": 0.3}

    @staticmethod
    def _features(text: str) -> Counter[str]:
        words = tokenize(text)
        feats: Counter[str] = Counter()
        for w in words:
            feats[f"w:{w}"] += 1
            # character 4-grams give partial robustness to morphology
            padded = f"^{w}$"
            for i in range(len(padded) - 3):
                feats[f"c:{padded[i : i + 4]}"] += 1
        for a, b in zip(words, words[1:], strict=False):
            feats[f"b:{a}_{b}"] += 1
        return feats

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % dim, sign

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, count in self._features(text).items():
            idx, sign = self._bucket(feature, self.dim)
            weight = self.FEATURE_WEIGHTS.get(feature[0], 1.0)
            # sub-linear tf damping, like BM25/TF-IDF
            vec[idx] += sign * weight * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_one(text)


class SentenceTransformerEmbedder:
    """Dense transformer embeddings - opt-in, requires `sentence-transformers`."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy, heavy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"sentence-transformers", "st", "minilm"}:
        try:
            embedder = SentenceTransformerEmbedder()
            log.info("Embeddings: sentence-transformers (dim=%d)", embedder.dim)
            return embedder
        except Exception as exc:
            log.warning(
                "sentence-transformers unavailable (%s); falling back to hashing embedder", exc
            )
    embedder = HashingEmbedder(dim=settings.embedding_dim)
    log.info("Embeddings: hashing (dim=%d)", embedder.dim)
    return embedder


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a stack of row vectors."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    denom[denom == 0] = 1.0
    return (matrix @ query) / denom
