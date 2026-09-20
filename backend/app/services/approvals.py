"""Approval queue - the only code path that can execute a business action.

The agent graph can *propose*; only a human decision routed through here can
*perform*. Arguments are re-validated at execution time rather than trusting
what was stored when the proposal was made, and every decision is written to
the audit log.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Approval, AuditLog, Message, Run, utcnow
from app.logging_config import get_logger
from app.tools.registry import get_tool

log = get_logger(__name__)


class ApprovalError(Exception):
    """Raised when a decision cannot be applied."""


def list_approvals(db: Session, status: str | None = None, limit: int = 50) -> list[Approval]:
    stmt = select(Approval).order_by(Approval.created_at.desc()).limit(limit)
    if status and status != "all":
        stmt = stmt.where(Approval.status == status)
    return list(db.execute(stmt).scalars())


def get_approval(db: Session, approval_id: str) -> Approval | None:
    return db.get(Approval, approval_id)


def _summarise(tool_name: str, result: dict) -> str:
    reference = result.get("reference", "")
    if tool_name == "submit_leave_request":
        return (
            f"Submitted. Your {result.get('leave_type')} leave from "
            f"{result.get('start_date')} to {result.get('end_date')} "
            f"({result.get('working_days')} working days) is with "
            f"{result.get('routed_to')} for sign-off. Reference **{reference}**."
        )
    if tool_name == "submit_expense_claim":
        return (
            f"Submitted. Expense claim for {result.get('currency')} "
            f"{result.get('amount'):,.2f} ({result.get('category')}) is in the finance "
            f"queue. Reference **{reference}**."
        )
    if tool_name == "raise_it_ticket":
        return (
            f"Ticket raised with the IT service desk at {result.get('priority')} "
            f"priority. Reference **{reference}**."
        )
    return f"Action `{tool_name}` completed. Reference **{reference}**."


def decide(
    db: Session,
    approval_id: str,
    *,
    decision: str,
    decided_by: str = "manager@northwind.example",
    note: str = "",
) -> dict:
    """Apply an approve/reject decision, executing the tool on approval."""
    if decision not in {"approve", "reject"}:
        raise ApprovalError("decision must be 'approve' or 'reject'")

    approval = db.get(Approval, approval_id)
    if approval is None:
        raise ApprovalError(f"Approval {approval_id} not found")
    if approval.status != "pending":
        raise ApprovalError(f"Approval {approval_id} is already {approval.status}")

    approval.decided_by = decided_by
    approval.decision_note = note
    approval.decided_at = utcnow()

    if decision == "reject":
        approval.status = "rejected"
        assistant_message = f"The request was rejected by {decided_by}. Nothing was submitted." + (
            f"\n\nReason: {note}" if note else ""
        )
        approval.execution_result = None
        _finish(db, approval, assistant_message, "rejected")
        return _payload(approval, assistant_message)

    tool = get_tool(approval.tool_name)
    if tool is None:
        approval.status = "failed"
        message = f"Cannot execute unknown tool `{approval.tool_name}`."
        _finish(db, approval, message, "failed")
        raise ApprovalError(message)

    try:
        # Re-validate at execution time - never trust stored arguments blindly.
        validated = tool.validate(approval.arguments)
        preflight = tool.preflight(db, approval.requested_by, validated)
        if not preflight.ok:
            approval.status = "failed"
            message = "Execution blocked by policy:\n" + "\n".join(
                f"- {b}" for b in preflight.blockers
            )
            _finish(db, approval, message, "failed")
            return _payload(approval, message)

        result = tool.execute(db, approval.requested_by, validated, approval.id)
    except Exception as exc:
        log.exception("Tool execution failed for approval %s", approval_id)
        approval.status = "failed"
        message = f"The action could not be completed: {exc}"
        _finish(db, approval, message, "failed")
        return _payload(approval, message)

    approval.status = "approved"
    approval.execution_result = result
    message = _summarise(approval.tool_name, result)
    _finish(db, approval, message, "completed")
    return _payload(approval, message)


def _finish(db: Session, approval: Approval, message: str, run_status: str) -> None:
    db.add(
        AuditLog(
            actor=approval.decided_by or "system",
            action=f"approval.{approval.status}",
            entity="approval",
            entity_id=approval.id,
            payload={
                "tool": approval.tool_name,
                "arguments": approval.arguments,
                "note": approval.decision_note,
            },
        )
    )
    if approval.conversation_id:
        db.add(
            Message(
                conversation_id=approval.conversation_id,
                role="assistant",
                content=message,
                run_id=approval.run_id,
            )
        )
    run = db.get(Run, approval.run_id)
    if run is not None:
        run.status = run_status
        run.requires_approval = False
    db.commit()
    db.refresh(approval)


def _payload(approval: Approval, message: str) -> dict:
    return {
        "approval_id": approval.id,
        "status": approval.status,
        "tool_name": approval.tool_name,
        "message": message,
        "execution_result": approval.execution_result,
        "decided_by": approval.decided_by,
        "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
        "run_id": approval.run_id,
        "conversation_id": approval.conversation_id,
    }
