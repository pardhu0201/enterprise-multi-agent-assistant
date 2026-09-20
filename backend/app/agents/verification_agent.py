"""Verification agent - the quality gate between the draft and the employee.

This node deliberately does **not** trust an LLM to grade an LLM. It runs
deterministic checks first and uses the model as a second opinion on top:

Programmatic checks (always run, in both modes)
    * citation validity - every ``[n]`` marker points at a real passage
    * citation coverage - what share of factual sentences are cited at all
    * lexical support   - IDF-weighted overlap between each cited sentence and
      the passage it cites
    * numeric grounding - every number in the answer must appear in the
      evidence; a fabricated threshold is the highest-signal failure mode in
      policy RAG
    * action safety     - preflight blockers and warnings from the tool

Model check (when a key is configured)
    Claude reviews the same material adversarially and returns unsupported
    claims plus a calibrated confidence.

The two are blended into a single confidence score, which - together with the
tool's risk level - decides whether the turn is answered, sent back for a
broader search, escalated, or queued for human approval.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agents.prompts import VERIFICATION_SYSTEM
from app.agents.reasoning_agent import split_sentences
from app.agents.state import AgentState, RunContext, trace_event
from app.config import settings
from app.logging_config import get_logger
from app.rag.embeddings import tokenize

log = get_logger(__name__)

# BM25 score a confidently on-topic passage reaches on this corpus; used to
# turn an unbounded lexical score into a 0-1 relevance signal.
STRONG_MATCH_BM25 = 12.0
# Even a perfectly written answer cannot score above this fraction of its
# quality when the evidence does not address the question.
RELEVANCE_FLOOR = 0.15

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_NUMBER_RE = re.compile(r"(?<![\w\[])(\d+(?:[.,]\d+)?)\s*(%|percent|days?|hours?|weeks?)?")
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
    "you",
    "your",
    "will",
    "must",
    "may",
    "can",
    "not",
    "if",
    "any",
    "all",
    "has",
    "have",
    # interrogative scaffolding - present in every question, informative in none
    "how",
    "what",
    "when",
    "where",
    "who",
    "why",
    "do",
    "does",
    "did",
    "need",
    "much",
    "many",
    "long",
    "i",
    "my",
    "me",
    "we",
    "our",
    "there",
    "about",
    "next",
    "per",
    "each",
    "get",
    "am",
    "was",
    "were",
    "should",
    "would",
}


class VerificationOutput(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    action_risk_notes: list[str] = Field(default_factory=list)
    llm_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    recommendation: str = Field(default="answer", description="answer | clarify | escalate")


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------
def _normalise_number(raw: str) -> str:
    value = raw.replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_numbers(text: str) -> set[str]:
    stripped = _CITATION_RE.sub(" ", text)
    return {_normalise_number(m.group(1)) for m in _NUMBER_RE.finditer(stripped)}


def _is_claim_sentence(sentence: str) -> bool:
    """Sentences that assert policy, as opposed to framing or questions.

    Lead-ins ("Here is what the policy says:"), headings and italic meta notes
    carry no factual claim, so holding them to a citation requirement would
    understate coverage on every well-formed answer.
    """
    cleaned = sentence.strip()
    if len(cleaned) < 30 or cleaned.endswith(("?", ":")):
        return False
    if cleaned.startswith(("_", "#", "|", ">")):
        return False
    words = [w for w in tokenize(cleaned) if w not in _STOPWORDS]
    return len(words) >= 5


def lexical_support(sentence: str, passage: str, idf: dict[str, float]) -> float:
    """IDF-weighted share of a sentence's content words present in a passage."""
    s_tokens = [t for t in tokenize(sentence) if t not in _STOPWORDS and len(t) > 2]
    if not s_tokens:
        return 1.0
    p_tokens = set(tokenize(passage))
    total = sum(idf.get(t, 1.0) for t in s_tokens)
    matched = sum(idf.get(t, 1.0) for t in s_tokens if t in p_tokens)
    return matched / total if total else 0.0


def query_coverage(query: str, evidence_text: str) -> float:
    """Share of the question's content words that appear in the evidence.

    This is the signal that catches an out-of-scope question. Retrieval always
    returns *something*, and an extractive or well-behaved generative answer
    will always cite what it was given - so citation coverage and lexical
    support stay high even when the corpus cannot answer the question at all.
    Only comparing the question against the evidence exposes that gap.
    """
    terms = {t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 2}
    if not terms:
        return 1.0
    evidence_terms = set(tokenize(evidence_text))
    return len(terms & evidence_terms) / len(terms)


def run_programmatic_checks(
    query: str,
    answer: str,
    retrieved: list[dict],
    proposed_action: dict | None,
    intent: str = "question",
) -> dict:
    passages = {item["index"]: item["content"] for item in retrieved}
    evidence_text = "\n".join(passages.values())

    # IDF across the retrieved evidence
    df: Counter = Counter()
    for content in passages.values():
        df.update(set(tokenize(content)))
    n_docs = max(1, len(passages))
    idf = {term: math.log(1 + n_docs / (1 + count)) for term, count in df.items()}

    citations = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
    invalid = sorted({c for c in citations if c not in passages})

    sentences = [s for s in split_sentences(answer) if _is_claim_sentence(s)]
    cited_sentences = [s for s in sentences if _CITATION_RE.search(s)]
    coverage = (len(cited_sentences) / len(sentences)) if sentences else 1.0

    support_scores: list[float] = []
    weak_sentences: list[str] = []
    for sentence in cited_sentences:
        refs = [int(m.group(1)) for m in _CITATION_RE.finditer(sentence)]
        texts = [passages[r] for r in refs if r in passages]
        if not texts:
            support_scores.append(0.0)
            weak_sentences.append(sentence)
            continue
        best = max(lexical_support(sentence, text, idf) for text in texts)
        support_scores.append(best)
        if best < 0.45:
            weak_sentences.append(sentence)
    support = sum(support_scores) / len(support_scores) if support_scores else 0.0

    evidence_numbers = extract_numbers(evidence_text)
    answer_numbers = extract_numbers(answer)
    ungrounded_numbers = sorted(
        n for n in answer_numbers - evidence_numbers if len(n) > 1 or int(float(n)) > 3
    )

    blockers = list((proposed_action or {}).get("blockers") or [])
    warnings = list((proposed_action or {}).get("warnings") or [])

    # Two independent relevance signals, combined pessimistically. Term
    # coverage catches a question whose vocabulary is absent from the corpus;
    # absolute BM25 catches one whose terms are common but whose subject is
    # not covered. Taking the minimum means either can veto.
    term_coverage = query_coverage(query, evidence_text) if retrieved else 0.0
    lexical_scores = [i.get("lexical_score") for i in retrieved]
    if any(score is not None for score in lexical_scores):
        top_lexical = max(float(score or 0.0) for score in lexical_scores)
        retrieval_strength = min(1.0, top_lexical / STRONG_MATCH_BM25)
        relevance = min(term_coverage, retrieval_strength)
    else:
        retrieval_strength = term_coverage
        relevance = term_coverage

    # A number that appears nowhere in the evidence means a threshold, deadline
    # or entitlement was invented - the most damaging failure mode this system
    # has, and the one a fluent answer hides best. It outweighs every other
    # signal, so the penalty has to be large enough to override an otherwise
    # well-formed, fully cited answer.
    penalties = 0.0
    penalties += 0.25 if invalid else 0.0
    penalties += 0.40 if ungrounded_numbers else 0.0
    penalties += 0.15 if blockers else 0.0
    penalties += 0.05 * min(2, len(warnings))

    # --- answer confidence ------------------------------------------------
    # How well-formed the answer is, gated by whether the evidence addresses
    # the question at all. A perfectly cited answer to a question the corpus
    # does not cover must not score highly.
    quality = 0.30 * coverage + 0.45 * support + 0.25 * (1.0 if retrieved else 0.0)
    answer_confidence = quality * (RELEVANCE_FLOOR + (1.0 - RELEVANCE_FLOOR) * relevance)
    answer_confidence = max(0.0, min(1.0, answer_confidence - penalties))

    # --- action confidence -------------------------------------------------
    # "Book 3 days from the 16th" asks the corpus nothing, so grading it on
    # evidence coverage is the wrong question entirely. What matters is
    # whether the prepared arguments are complete, valid and policy-clean.
    action_confidence: float | None = None
    if proposed_action is not None:
        action_confidence = 1.0
        if not proposed_action.get("valid", False):
            action_confidence -= 0.20
        if blockers:
            action_confidence -= 0.45
        if proposed_action.get("missing_fields"):
            action_confidence -= 0.25
        elif proposed_action.get("confirmation_needed"):
            action_confidence -= 0.10
        action_confidence -= 0.08 * min(2, len(warnings))
        action_confidence = max(0.0, min(1.0, action_confidence))

    if action_confidence is None:
        score = answer_confidence
    elif intent == "action":
        # Nothing was asked, so the action is the whole turn.
        score = action_confidence
    else:  # "mixed" - the employee asked *and* requested; both must hold
        score = (
            min(answer_confidence, action_confidence) * 0.5
            + ((answer_confidence + action_confidence) / 2) * 0.5
        )

    return {
        "citation_count": len(citations),
        "invalid_citations": invalid,
        "claim_sentences": len(sentences),
        "cited_sentences": len(cited_sentences),
        "citation_coverage": round(coverage, 3),
        "lexical_support": round(support, 3),
        "weak_sentences": weak_sentences[:5],
        "ungrounded_numbers": ungrounded_numbers,
        "query_coverage": round(relevance, 3),
        "retrieval_strength": round(retrieval_strength, 3),
        "action_blockers": blockers,
        "action_warnings": warnings,
        "answer_confidence": round(answer_confidence, 3),
        "action_confidence": (
            round(action_confidence, 3) if action_confidence is not None else None
        ),
        "programmatic_confidence": round(max(0.0, min(1.0, score)), 3),
    }


def _flags_from(checks: dict, state: AgentState) -> list[str]:
    flags: list[str] = []

    # A pure action turn asks the corpus nothing, so answer-quality flags would
    # be noise on it - only the action's own flags mean anything there.
    grade_answer = state.get("intent") != "action"

    if grade_answer:
        if not state.get("retrieved"):
            flags.append("no_evidence_retrieved")
        if state.get("insufficient_evidence"):
            flags.append("insufficient_evidence")
        if checks["claim_sentences"] and checks["citation_coverage"] < 0.6:
            flags.append("low_citation_coverage")
        if checks["cited_sentences"] and checks["lexical_support"] < 0.45:
            flags.append("weak_evidence_support")
        if checks["query_coverage"] < 0.4:
            flags.append("question_not_covered_by_corpus")

    # These are defects regardless of intent.
    if checks["invalid_citations"]:
        flags.append("invalid_citation")
    if checks["ungrounded_numbers"]:
        flags.append("ungrounded_numbers")
    if checks["action_blockers"]:
        flags.append("action_blocked")
    if checks["action_warnings"]:
        flags.append("policy_warning")
    if checks.get("action_confidence") is not None and checks["action_confidence"] < 0.6:
        flags.append("incomplete_action")
    return flags


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def verification_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()

    answer = state.get("answer", "")
    retrieved = state.get("retrieved") or []
    proposed_action = state.get("proposed_action")

    checks = run_programmatic_checks(
        state["query"], answer, retrieved, proposed_action, state.get("intent", "question")
    )

    def deterministic_review() -> VerificationOutput:
        recommendation = "answer"
        if checks["action_blockers"] or checks["invalid_citations"]:
            recommendation = "escalate"
        elif not retrieved or state.get("insufficient_evidence"):
            recommendation = "clarify"
        return VerificationOutput(
            unsupported_claims=checks["weak_sentences"],
            citation_issues=(
                [f"Answer cites passage(s) {checks['invalid_citations']} that do not exist"]
                if checks["invalid_citations"]
                else []
            ),
            action_risk_notes=[*checks["action_blockers"], *checks["action_warnings"]],
            llm_confidence=checks["programmatic_confidence"],
            recommendation=recommendation,
        )

    if state.get("intent") == "smalltalk":
        review = VerificationOutput(llm_confidence=1.0, recommendation="answer")
        result_mode = "skipped"
        confidence = 1.0
    else:
        action_text = "(none)"
        if proposed_action:
            action_text = (
                f"tool={proposed_action['tool_name']}\n"
                f"arguments={proposed_action['arguments']}\n"
                f"preview={proposed_action.get('preview')}\n"
                f"blockers={proposed_action.get('blockers')}\n"
                f"warnings={proposed_action.get('warnings')}"
            )
        user = (
            f"Employee request:\n{state['query']}\n\n"
            f"Numbered evidence passages:\n{state.get('context_block') or '(none)'}\n\n"
            f"Draft answer:\n{answer or '(empty)'}\n\n"
            f"Prepared action:\n{action_text}\n\n"
            f"Automated checks already computed:\n"
            f"- citation coverage: {checks['citation_coverage']}\n"
            f"- lexical support: {checks['lexical_support']}\n"
            f"- invalid citations: {checks['invalid_citations']}\n"
            f"- numbers not found in evidence: {checks['ungrounded_numbers']}"
        )
        result = ctx.llm.structured(
            agent="verification",
            system=VERIFICATION_SYSTEM,
            user=user,
            schema=VerificationOutput,
            fallback=deterministic_review,
            max_tokens=2500,
        )
        ctx.usage.add("verification", result)
        review: VerificationOutput = result.value  # type: ignore[assignment]
        result_mode = result.mode
        if result_mode == "claude":
            # Blend: neither signal is trusted alone, and the lower one dominates.
            blended = 0.5 * checks["programmatic_confidence"] + 0.5 * review.llm_confidence
            confidence = min(blended, max(checks["programmatic_confidence"], review.llm_confidence))
        else:
            confidence = checks["programmatic_confidence"]

    flags = _flags_from(checks, state)
    if review.unsupported_claims and "weak_evidence_support" not in flags:
        flags.append("model_flagged_claims")
    if review.recommendation == "escalate":
        flags.append("verifier_escalated")

    confidence = round(max(0.0, min(1.0, confidence)), 3)

    # --- routing decision -------------------------------------------------
    needs_action = proposed_action is not None
    tool_requires_approval = bool(proposed_action and proposed_action.get("requires_approval"))
    low_confidence = confidence < settings.confidence_threshold

    can_retry = (
        state.get("retrieval_attempt", 1) <= settings.max_retrieval_retries
        and state.get("intent") != "smalltalk"
        and (low_confidence or not retrieved)
        and not needs_action
    )

    if can_retry:
        decision = "retry"
    elif needs_action and (tool_requires_approval or low_confidence or checks["action_blockers"]):
        decision = "approval"
    elif review.recommendation == "escalate" or (low_confidence and retrieved):
        decision = "escalate"
    elif review.recommendation == "clarify" or state.get("insufficient_evidence"):
        decision = "clarify"
    else:
        decision = "answer"

    verification = {
        **checks,
        "llm_confidence": round(review.llm_confidence, 3),
        "confidence": confidence,
        "recommendation": review.recommendation,
        "decision": decision,
        "unsupported_claims": review.unsupported_claims,
        "citation_issues": review.citation_issues,
        "action_risk_notes": review.action_risk_notes,
        "mode": result_mode,
        "threshold": settings.confidence_threshold,
    }

    status = "ok"
    if flags:
        status = "warning"
    if decision in {"escalate"} or checks["action_blockers"]:
        status = "error"

    summary = {
        "retry": "Evidence too thin - sending back for a broader search",
        "approval": "Action prepared - human approval required",
        "escalate": "Low confidence - escalating to a human reviewer",
        "clarify": "Evidence incomplete - asking the employee to clarify",
        "answer": "Answer verified against the cited evidence",
    }[decision]

    return {
        "verification": verification,
        "confidence": confidence,
        "flags": flags,
        "requires_approval": decision == "approval",
        "status": {
            "retry": "running",
            "approval": "awaiting_approval",
            "escalate": "escalated",
            "clarify": "needs_clarification",
            "answer": "completed",
        }[decision],
        "trace": [
            trace_event(
                ctx,
                "verification",
                f"{summary} (confidence {confidence:.2f})",
                status=status,
                payload=verification,
                started=started,
            )
        ],
    }
