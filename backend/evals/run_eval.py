"""Offline evaluation harness.

Runs the golden set through the real graph and reports the numbers that
actually matter for a policy assistant:

* **Routing accuracy**   - did the planner pick the right intent and tool?
* **Retrieval recall@k** - was the document that contains the answer retrieved?
* **Retrieval MRR**      - how high up was it?
* **Answer coverage**    - does the answer contain the facts it should, and
                           none of the near-miss numbers it should not?
* **Citation validity**  - every ``[n]`` marker points at a real passage.
* **Groundedness**       - IDF-weighted support of each cited sentence.
* **Action accuracy**    - are the prepared tool arguments correct?
* **Safety**             - was every sensitive action stopped at the gate?

Run it against either mode:

    python -m evals.run_eval              # demo mode (free, deterministic)
    ANTHROPIC_API_KEY=sk-... python -m evals.run_eval

Add ``--json report.json`` to write machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.runner import run_turn  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.db.init_db import initialise  # noqa: E402
from app.llm.client import get_llm  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
RECALL_K = 5


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("**", "").replace("*", "").split())


def evaluate_case(db, case: dict) -> dict:
    started = time.perf_counter()

    # --- retrieval measured independently of the answer ---
    ranked = retrieve(db, case["query"], top_k=RECALL_K)
    titles = [chunk.document_title for chunk in ranked]
    expected_document = case.get("document")
    if expected_document:
        recall = expected_document in titles
        position = titles.index(expected_document) + 1 if recall else 0
        mrr = 1.0 / position if position else 0.0
    else:
        recall, mrr = None, None

    # --- full graph ---
    result = run_turn(db, query=case["query"])
    answer = _normalise(result["answer"])

    contains = [_normalise(s) in answer for s in case.get("must_contain", [])]
    avoids = [_normalise(s) not in answer for s in case.get("must_not_contain", [])]

    verification = result.get("verification") or {}
    action = result.get("proposed_action")

    argument_hits: list[bool] = []
    for key, expected in (case.get("expected_arguments") or {}).items():
        actual = (action or {}).get("arguments", {}).get(key)
        argument_hits.append(str(actual) == str(expected))

    intent_ok = result["intent"] == case["intent"]
    tool_ok = ((action or {}).get("tool_name") if action else None) == case.get("tool")

    sensitive = bool(action and action.get("requires_approval"))
    gated = (not sensitive) or result["status"] == "awaiting_approval"

    return {
        "id": case["id"],
        "query": case["query"],
        "intent_expected": case["intent"],
        "intent_actual": result["intent"],
        "intent_ok": intent_ok,
        "tool_expected": case.get("tool"),
        "tool_actual": (action or {}).get("tool_name"),
        "tool_ok": tool_ok,
        "recall_at_k": recall,
        "reciprocal_rank": mrr,
        "retrieved": titles,
        "coverage": (sum(contains) / len(contains)) if contains else None,
        "avoided_distractors": all(avoids) if avoids else None,
        "citation_valid": not verification.get("invalid_citations"),
        "citation_coverage": verification.get("citation_coverage"),
        "groundedness": verification.get("lexical_support"),
        "ungrounded_numbers": verification.get("ungrounded_numbers", []),
        "confidence": result["confidence"],
        "low_confidence_expected": case.get("expect_low_confidence", False),
        "argument_accuracy": (sum(argument_hits) / len(argument_hits) if argument_hits else None),
        "action_gated": gated,
        "status": result["status"],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _mean(values: list) -> float:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 3) if numbers else 0.0


def _rate(values: list) -> float:
    flags = [v for v in values if isinstance(v, bool)]
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def summarise(rows: list[dict]) -> dict:
    low_conf_rows = [r for r in rows if r["low_confidence_expected"]]
    return {
        "cases": len(rows),
        "mode": get_llm().mode,
        "routing": {
            "intent_accuracy": _rate([r["intent_ok"] for r in rows]),
            "tool_accuracy": _rate([r["tool_ok"] for r in rows]),
        },
        "retrieval": {
            f"recall_at_{RECALL_K}": _rate([r["recall_at_k"] for r in rows]),
            "mrr": _mean([r["reciprocal_rank"] for r in rows]),
        },
        "answer": {
            "fact_coverage": _mean([r["coverage"] for r in rows]),
            "distractor_avoidance": _rate([r["avoided_distractors"] for r in rows]),
            "citation_validity": _rate([r["citation_valid"] for r in rows]),
            "citation_coverage": _mean([r["citation_coverage"] for r in rows]),
            "groundedness": _mean([r["groundedness"] for r in rows]),
            "hallucinated_number_rate": _rate([bool(r["ungrounded_numbers"]) for r in rows]),
        },
        "actions": {
            "argument_accuracy": _mean([r["argument_accuracy"] for r in rows]),
            "human_gate_enforced": _rate([r["action_gated"] for r in rows]),
        },
        "calibration": {
            "mean_confidence": _mean([r["confidence"] for r in rows]),
            "out_of_scope_flagged": _rate([r["confidence"] < 0.55 for r in low_conf_rows])
            if low_conf_rows
            else None,
        },
        "performance": {"median_latency_ms": _median([r["latency_ms"] for r in rows])},
    }


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the assistant on the golden set")
    parser.add_argument("--json", type=Path, help="Write the full report to this path")
    parser.add_argument("--case", help="Run only the case with this id")
    args = parser.parse_args()

    configure_logging("WARNING")
    initialise(seed=True)

    payload = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}")
            return 2

    rows: list[dict] = []
    with session_scope() as db:
        for case in cases:
            row = evaluate_case(db, case)
            rows.append(row)
            marks = "".join(
                [
                    "R" if row["recall_at_k"] else ("-" if row["recall_at_k"] is None else "x"),
                    "I" if row["intent_ok"] else "x",
                    "T" if row["tool_ok"] else "x",
                    "C" if row["citation_valid"] else "x",
                ]
            )
            print(f"  [{marks}] conf={row['confidence']:.2f} {row['latency_ms']:>5}ms  {row['id']}")

    report = {"summary": summarise(rows), "results": rows}
    print("\n" + json.dumps(report["summary"], indent=2))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nFull report written to {args.json}")

    summary = report["summary"]
    ok = (
        summary["retrieval"][f"recall_at_{RECALL_K}"] >= 0.75
        and summary["routing"]["intent_accuracy"] >= 0.75
        and summary["answer"]["citation_validity"] >= 0.95
        and summary["actions"]["human_gate_enforced"] == 1.0
    )
    print("\nRESULT:", "PASS" if ok else "BELOW THRESHOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
