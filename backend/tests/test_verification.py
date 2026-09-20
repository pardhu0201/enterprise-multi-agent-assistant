"""The verification agent's deterministic groundedness checks.

These run identically with and without an API key, so they are the safety net
that does not depend on a model behaving itself.
"""

from __future__ import annotations

from app.agents.verification_agent import (
    extract_numbers,
    lexical_support,
    run_programmatic_checks,
)

QUERY = "How many days of annual leave do I get and how much notice do I need?"

PASSAGES = [
    {
        "index": 1,
        "content": (
            "[Leave Policy > Annual leave]\n"
            "Every full-time employee receives 24 days of paid annual leave per calendar "
            "year, accrued at 2 days per completed month of service."
        ),
        "score": 1.0,
        "lexical_score": 14.2,
    },
    {
        "index": 2,
        "content": (
            "[Leave Policy > Requesting annual leave]\n"
            "Requests of 3 or more consecutive working days require 10 working days of "
            "notice before the first day of leave."
        ),
        "score": 0.8,
        "lexical_score": 11.6,
    },
]


def test_extract_numbers_ignores_citation_markers():
    numbers = extract_numbers("You get 24 days [1] and need 10 days notice [2].")
    assert "24" in numbers and "10" in numbers
    assert "1" not in numbers and "2" not in numbers


def test_lexical_support_rewards_overlap():
    idf = {"annual": 1.0, "leave": 1.0, "days": 1.0, "notice": 1.0, "working": 1.0}
    high = lexical_support(
        "Requests require 10 working days of notice", PASSAGES[1]["content"], idf
    )
    low = lexical_support(
        "Expenses must be filed through the finance portal", PASSAGES[1]["content"], idf
    )
    assert high > 0.8
    assert low < high


def test_grounded_answer_scores_well():
    answer = (
        "Every full-time employee receives 24 days of paid annual leave per calendar year [1]. "
        "Requests of 3 or more consecutive working days require 10 working days of notice [2]."
    )
    checks = run_programmatic_checks(QUERY, answer, PASSAGES, None)
    assert checks["invalid_citations"] == []
    assert checks["ungrounded_numbers"] == []
    assert checks["citation_coverage"] == 1.0
    assert checks["lexical_support"] > 0.8
    assert checks["programmatic_confidence"] > 0.75


def test_fabricated_number_is_caught():
    answer = "Employees receive 45 days of paid annual leave per calendar year [1]."
    checks = run_programmatic_checks(QUERY, answer, PASSAGES, None)
    assert "45" in checks["ungrounded_numbers"]
    assert checks["programmatic_confidence"] < 0.75


def test_invalid_citation_is_caught():
    answer = "Employees receive 24 days of annual leave [7]."
    checks = run_programmatic_checks(QUERY, answer, PASSAGES, None)
    assert checks["invalid_citations"] == [7]


def test_uncited_claims_reduce_coverage():
    answer = (
        "Every full-time employee receives 24 days of paid annual leave per calendar year [1]. "
        "Managers must always approve requests within one hour of submission."
    )
    checks = run_programmatic_checks(QUERY, answer, PASSAGES, None)
    assert checks["citation_coverage"] < 1.0


def test_action_blockers_are_surfaced():
    action = {
        "tool_name": "submit_leave_request",
        "blockers": ["Requested 30 days but only 5.0 annual days remain in the balance."],
        "warnings": ["Only 1 working days' notice."],
    }
    checks = run_programmatic_checks(QUERY, "Prepared your request [1].", PASSAGES, action)
    assert checks["action_blockers"]
    assert checks["action_warnings"]
    assert checks["programmatic_confidence"] < 0.8
