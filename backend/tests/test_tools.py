"""Business tool validation, preflight guards and date parsing."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.agents.dateparse import add_working_days, find_duration_days, infer_range
from app.tools.business_tools import working_days_between, working_days_notice
from app.tools.registry import TOOLS, get_tool
from app.tools.schemas import LeaveRequestArgs


# --- date handling ---------------------------------------------------------
def test_working_days_skips_weekends():
    # Mon 2026-10-12 .. Fri 2026-10-16
    assert working_days_between(date(2026, 10, 12), date(2026, 10, 16)) == 5
    # Includes a weekend
    assert working_days_between(date(2026, 10, 12), date(2026, 10, 19)) == 6
    # Weekend only
    assert working_days_between(date(2026, 10, 17), date(2026, 10, 18)) == 0


def test_notice_is_counted_in_working_days():
    today = date(2026, 10, 1)  # Thursday
    assert working_days_notice(today, date(2026, 10, 1)) == 0
    assert working_days_notice(today, date(2026, 10, 15)) == 9


def test_add_working_days():
    assert add_working_days(date(2026, 10, 12), 1) == date(2026, 10, 12)
    assert add_working_days(date(2026, 10, 12), 3) == date(2026, 10, 14)
    assert add_working_days(date(2026, 10, 15), 3) == date(2026, 10, 19)  # spans a weekend


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("book leave from 2026-10-12 to 2026-10-16", (date(2026, 10, 12), date(2026, 10, 16))),
        ("3 days off starting 2026-10-12", (date(2026, 10, 12), date(2026, 10, 14))),
        ("leave on 14 November", (date(2026, 11, 14), date(2026, 11, 14))),
        ("off from Nov 2 to Nov 6", (date(2026, 11, 2), date(2026, 11, 6))),
    ],
)
def test_infer_range(text, expected):
    assert infer_range(text, today=date(2026, 9, 20)) == expected


def test_duration_words_and_digits():
    assert find_duration_days("take three days off") == 3
    assert find_duration_days("I need 5 working days") == 5
    assert find_duration_days("no duration here") is None


# --- schema validation -----------------------------------------------------
def test_leave_args_reject_reversed_range():
    with pytest.raises(ValidationError):
        LeaveRequestArgs(start_date=date(2026, 10, 16), end_date=date(2026, 10, 12))


def test_leave_args_reject_unknown_type():
    with pytest.raises(ValidationError):
        LeaveRequestArgs(leave_type="sabbatical", start_date="2026-10-12", end_date="2026-10-13")


# --- preflight guards ------------------------------------------------------
def test_preflight_blocks_past_dates(db):
    tool = get_tool("submit_leave_request")
    yesterday = date.today() - timedelta(days=1)
    args = tool.validate(
        {
            "leave_type": "annual",
            "start_date": yesterday.isoformat(),
            "end_date": yesterday.isoformat(),
        }
    )
    result = tool.preflight(db, "E-1001", args)
    assert not result.ok
    assert any("past" in b for b in result.blockers)


def test_preflight_blocks_insufficient_balance(db):
    tool = get_tool("submit_leave_request")
    start = date.today() + timedelta(days=30)
    args = tool.validate(
        {
            "leave_type": "annual",
            "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=60)).isoformat(),
        }
    )
    # E-1002 has 5 days of annual leave left in the seed data.
    result = tool.preflight(db, "E-1002", args)
    assert not result.ok
    assert any("remain" in b for b in result.blockers)


def test_preflight_warns_about_short_notice(db):
    tool = get_tool("submit_leave_request")
    start = date.today() + timedelta(days=1)
    args = tool.validate(
        {
            "leave_type": "annual",
            "start_date": start.isoformat(),
            "end_date": (start + timedelta(days=6)).isoformat(),
        }
    )
    result = tool.preflight(db, "E-1001", args)
    assert any("notice" in w for w in result.warnings)


def test_expense_preflight_enforces_claim_window(db):
    tool = get_tool("submit_expense_claim")
    old = date.today() - timedelta(days=45)
    args = tool.validate({"category": "meals", "amount": 900, "incurred_on": old.isoformat()})
    result = tool.preflight(db, "E-1001", args)
    assert not result.ok
    assert any("30 days" in b for b in result.blockers)


def test_expense_preflight_routes_by_amount(db):
    tool = get_tool("submit_expense_claim")
    args = tool.validate(
        {"category": "travel", "amount": 75000, "incurred_on": date.today().isoformat()}
    )
    result = tool.preflight(db, "E-1001", args)
    assert result.preview["approval_route"] == "Finance Controller"


def test_read_only_tool_needs_no_approval():
    assert TOOLS["check_leave_balance"].requires_approval is False
    assert all(
        spec.requires_approval for name, spec in TOOLS.items() if name != "check_leave_balance"
    )
    for spec in TOOLS.values():
        # Running a tool inside the graph is only allowed for pure lookups.
        if spec.read_only:
            assert spec.requires_approval is False


# --- hard guards stay in sync with the policy text ---------------------------
def test_preflight_constants_match_the_policy_corpus():
    """The preflight thresholds are hard-coded guards that mirror the policy.

    If HR edits the documents, this fails instead of the guard silently
    enforcing a stale rule while the assistant quotes the new one.
    """
    from app.config import CORPUS_DIR
    from app.tools import business_tools as bt

    leave = (CORPUS_DIR / "leave-and-time-off-policy.md").read_text(encoding="utf-8")
    expense = (CORPUS_DIR / "expense-and-travel-policy.md").read_text(encoding="utf-8")
    flat_leave = " ".join(leave.replace("*", "").split())
    flat_expense = " ".join(expense.replace("*", "").split())

    assert (
        f"{bt.LONG_LEAVE_THRESHOLD_DAYS} or more consecutive working days require "
        f"{bt.LONG_LEAVE_NOTICE_DAYS} working days" in flat_leave
    )
    assert f"require {bt.SHORT_LEAVE_NOTICE_DAYS} working days of notice" in flat_leave
    assert f"longer than {bt.MAX_CONSECUTIVE_DAYS} consecutive working days" in flat_leave
    assert f"above INR {bt.RECEIPT_THRESHOLD:,.0f}" in flat_expense
    assert f"Up to {bt.MANAGER_APPROVAL_THRESHOLD:,.0f}" in flat_expense
    assert f"Above {bt.FINANCE_APPROVAL_THRESHOLD:,.0f}" in flat_expense
    assert f"older than {bt.EXPENSE_WINDOW_DAYS} days" in flat_expense


# --- approvals service --------------------------------------------------------
def test_stale_session_cannot_execute_an_approval_twice(db):
    """Two deciders racing on one request: exactly one may execute it."""
    from app.agents.runner import run_turn
    from app.db.base import SessionLocal
    from app.db.models import Approval, LeaveRequest
    from app.services.approvals import ApprovalError, decide

    start = date.today() + timedelta(days=50)
    start += timedelta(days=(7 - start.weekday()) % 7)  # a Monday
    turn = run_turn(db, query=f"Book annual leave for 1 day on {start.isoformat()}")
    approval_id = turn["approval_id"]

    # Session A reads the request while it is still pending...
    stale = SessionLocal()
    try:
        assert stale.get(Approval, approval_id).status == "pending"

        # ...session B approves and executes it...
        first = decide(db, approval_id, decision="approve", decided_by="priya@northwind.example")
        assert first["status"] == "approved"

        # ...and A, still holding its stale 'pending' copy, must be refused.
        with pytest.raises(ApprovalError):
            decide(stale, approval_id, decision="approve", decided_by="marcus@northwind.example")
    finally:
        stale.close()

    executed = db.query(LeaveRequest).filter(LeaveRequest.approval_id == approval_id).count()
    assert executed == 1


def test_eval_date_placeholders_resolve_relative_to_today():
    from evals.run_eval import resolve_dates

    today = date(2026, 10, 7)  # a Wednesday
    resolved = resolve_dates(
        {
            "q": "book from {{monday+5}} for 3 days",
            "args": {"start": "{{monday+5}}", "end": "{{monday+5+2}}", "on": "{{today-5}}"},
            "n": 3,
        },
        today,
    )
    assert resolved["q"] == "book from 2026-10-12 for 3 days"  # Oct 7 + 5 = Mon Oct 12
    assert resolved["args"] == {"start": "2026-10-12", "end": "2026-10-14", "on": "2026-10-02"}
    assert resolved["n"] == 3
    assert resolve_dates("{{monday+0}}", date(2026, 10, 12)) == "2026-10-12"  # already Monday
