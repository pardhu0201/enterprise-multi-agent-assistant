"""End-to-end behaviour of the LangGraph multi-agent workflow."""

from __future__ import annotations

from datetime import date, timedelta

from app.agents.graph import graph_topology
from app.agents.planner import rule_based_plan
from app.agents.runner import iter_turn, run_turn


def test_planner_routes_questions_and_actions():
    assert rule_based_plan("What is the annual leave entitlement?").intent == "question"
    assert rule_based_plan("hi").intent == "smalltalk"

    action = rule_based_plan("Please submit a leave request for me next Monday")
    assert action.intent in {"action", "mixed"}
    assert action.candidate_tool == "submit_leave_request"

    expense = rule_based_plan("I want to claim an expense for a client dinner")
    assert expense.candidate_tool == "submit_expense_claim"


def test_question_turn_is_answered_with_citations(db):
    result = run_turn(db, query="How long do I have to file an expense claim?")
    assert result["status"] in {"completed", "needs_clarification", "escalated"}
    assert result["citations"], "expected retrieved citations"
    assert result["proposed_action"] is None
    assert result["approval_id"] is None
    assert "Expense" in " ".join(c["document_title"] for c in result["citations"])

    agents = [event["agent"] for event in result["trace"]]
    assert agents[:3] == ["planner", "retrieval", "reasoning"]
    assert "verification" in agents
    assert "workflow" not in agents


def test_action_turn_stops_at_the_approval_gate(db):
    start = date.today() + timedelta(days=21)
    result = run_turn(
        db,
        query=(
            "Please submit a leave request for 3 days starting "
            f"{start.isoformat()} for a family event"
        ),
    )
    assert result["status"] == "awaiting_approval"
    assert result["requires_approval"] is True
    assert result["approval_id"]

    action = result["proposed_action"]
    assert action["tool_name"] == "submit_leave_request"
    assert action["arguments"]["start_date"] == start.isoformat()
    assert action["preview"]["working_days"] == 3
    assert action["requires_approval"] is True

    agents = [event["agent"] for event in result["trace"]]
    assert "workflow" in agents
    assert agents[-1] == "approval_gate"

    # Nothing may have been written to the system of record yet.
    from app.db.models import LeaveRequest

    assert (
        db.query(LeaveRequest).filter(LeaveRequest.approval_id == result["approval_id"]).count()
        == 0
    )


def test_blocked_action_is_flagged_not_executed(db):
    yesterday = date.today() - timedelta(days=2)
    result = run_turn(
        db, query=f"Book me annual leave from {yesterday.isoformat()} to {yesterday.isoformat()}"
    )
    action = result["proposed_action"]
    assert action is not None
    assert action["blockers"], "a past-dated request must be blocked"
    assert "action_blocked" in result["flags"]


def test_smalltalk_skips_retrieval(db):
    result = run_turn(db, query="hello")
    assert result["intent"] == "smalltalk"
    assert result["citations"] == []
    assert result["confidence"] == 1.0


def test_streaming_emits_ordered_events(db):
    events = list(iter_turn(db, query="What is the sick leave entitlement?"))
    names = [e["event"] for e in events]
    assert names[0] == "run_started"
    assert names[-1] == "final"
    steps = [e for e in events if e["event"] == "agent_step"]
    assert [s["data"]["seq"] for s in steps] == sorted(s["data"]["seq"] for s in steps)


def test_conversation_history_is_persisted(db):
    first = run_turn(db, query="What is the annual leave entitlement?")
    second = run_turn(
        db, query="And how much notice do I need?", conversation_id=first["conversation_id"]
    )
    assert second["conversation_id"] == first["conversation_id"]

    from app.db.models import Message

    messages = db.query(Message).filter(Message.conversation_id == first["conversation_id"]).all()
    assert len(messages) >= 4
    assert {m.role for m in messages} == {"user", "assistant"}


def test_graph_topology_is_described():
    topology = graph_topology()
    node_ids = {n["id"] for n in topology["nodes"]}
    assert {
        "planner",
        "retrieval",
        "reasoning",
        "workflow",
        "verification",
        "approval_gate",
    } <= node_ids
    for edge in topology["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
