"""Deterministic implementations behind the tool registry.

These stand in for the enterprise systems a real deployment would call (HRIS,
finance, ITSM). They are intentionally boring, synchronous and fully
auditable - all the intelligence lives in the agents, all the *authority*
lives here behind the approval gate.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.db.models import (
    AuditLog,
    Employee,
    ExpenseClaim,
    ITTicket,
    LeaveRequest,
    utcnow,
)
from app.tools.schemas import ExpenseClaimArgs, ITTicketArgs, LeaveBalanceArgs, LeaveRequestArgs

# Policy constants mirrored from the seed corpus. Kept here as *hard* guards;
# the verification agent separately checks the answer against retrieved text.
LONG_LEAVE_THRESHOLD_DAYS = 3
LONG_LEAVE_NOTICE_DAYS = 10
SHORT_LEAVE_NOTICE_DAYS = 2
MAX_CONSECUTIVE_DAYS = 10
RECEIPT_THRESHOLD = 1000.0
MANAGER_APPROVAL_THRESHOLD = 10000.0
FINANCE_APPROVAL_THRESHOLD = 50000.0
EXPENSE_WINDOW_DAYS = 30


def working_days_between(start: date, end: date) -> float:
    """Inclusive count of Mon-Fri days."""
    days = 0
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor += timedelta(days=1)
    return float(days)


def working_days_notice(from_day: date, start: date) -> float:
    if start <= from_day:
        return 0.0
    return working_days_between(from_day + timedelta(days=1), start - timedelta(days=1))


def get_employee(db: Session, employee_id: str) -> Employee | None:
    return db.get(Employee, employee_id)


def _audit(db: Session, actor: str, action: str, entity: str, entity_id: str, payload: dict):
    db.add(
        AuditLog(actor=actor, action=action, entity=entity, entity_id=entity_id, payload=payload)
    )


# ---------------------------------------------------------------------------
# Leave
# ---------------------------------------------------------------------------
def preflight_leave_request(db: Session, employee_id: str, args: LeaveRequestArgs):
    from app.tools.registry import PreflightResult

    warnings: list[str] = []
    blockers: list[str] = []

    employee = get_employee(db, employee_id)
    days = working_days_between(args.start_date, args.end_date)
    today = date.today()
    notice = working_days_notice(today, args.start_date)

    if days <= 0:
        blockers.append("The selected dates contain no working days (weekend only).")

    if args.start_date < today:
        blockers.append(
            f"Start date {args.start_date.isoformat()} is in the past; "
            "retroactive leave must be filed by HR."
        )

    required_notice = (
        LONG_LEAVE_NOTICE_DAYS if days >= LONG_LEAVE_THRESHOLD_DAYS else SHORT_LEAVE_NOTICE_DAYS
    )
    if args.leave_type == "annual" and notice < required_notice:
        warnings.append(
            f"Only {notice:.0f} working days' notice - the annual leave policy asks for "
            f"{required_notice} for a {days:.0f}-day request. Manager may need to waive this."
        )

    if days > MAX_CONSECUTIVE_DAYS:
        warnings.append(
            f"{days:.0f} consecutive working days exceeds the {MAX_CONSECUTIVE_DAYS}-day "
            "limit and needs HR Business Partner sign-off."
        )

    balance = None
    if employee is not None:
        if args.leave_type == "annual":
            balance = employee.annual_leave_total - employee.annual_leave_used
        elif args.leave_type == "sick":
            balance = employee.sick_leave_total - employee.sick_leave_used
        if balance is not None and days > balance:
            blockers.append(
                f"Requested {days:.0f} days but only {balance:.1f} {args.leave_type} "
                "days remain in the balance."
            )

    if args.leave_type == "sick" and days > 2:
        warnings.append("Sick leave beyond 2 consecutive days requires a medical certificate.")

    preview = {
        "employee_id": employee_id,
        "employee_name": employee.name if employee else "Unknown",
        "manager": employee.manager if employee else "Unassigned",
        "leave_type": args.leave_type,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "working_days": days,
        "notice_working_days": notice,
        "balance_before": balance,
        "balance_after": (balance - days) if balance is not None else None,
        "reason": args.reason,
    }
    return PreflightResult(ok=not blockers, preview=preview, warnings=warnings, blockers=blockers)


def execute_leave_request(
    db: Session, employee_id: str, args: LeaveRequestArgs, approval_id: str | None
) -> dict:
    days = working_days_between(args.start_date, args.end_date)
    record = LeaveRequest(
        employee_id=employee_id,
        leave_type=args.leave_type,
        start_date=args.start_date,
        end_date=args.end_date,
        working_days=days,
        reason=args.reason,
        status="submitted",
        approval_id=approval_id,
    )
    db.add(record)

    employee = get_employee(db, employee_id)
    if employee is not None:
        if args.leave_type == "annual":
            employee.annual_leave_used += days
        elif args.leave_type == "sick":
            employee.sick_leave_used += days

    db.flush()
    _audit(
        db,
        actor=employee_id,
        action="leave_request.submitted",
        entity="leave_request",
        entity_id=record.id,
        payload={"days": days, "type": args.leave_type, "approval_id": approval_id},
    )
    db.commit()
    return {
        "reference": f"LR-{record.id[:8].upper()}",
        "status": record.status,
        "leave_type": record.leave_type,
        "start_date": record.start_date.isoformat(),
        "end_date": record.end_date.isoformat(),
        "working_days": days,
        "routed_to": employee.manager if employee else "HR Operations",
        "submitted_at": utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Expenses
# ---------------------------------------------------------------------------
def preflight_expense_claim(db: Session, employee_id: str, args: ExpenseClaimArgs):
    from app.tools.registry import PreflightResult

    warnings: list[str] = []
    blockers: list[str] = []

    if args.incurred_on:
        age = (date.today() - args.incurred_on).days
        if age > EXPENSE_WINDOW_DAYS:
            blockers.append(
                f"Expense is {age} days old; claims must be filed within "
                f"{EXPENSE_WINDOW_DAYS} days."
            )
    else:
        warnings.append("No expense date supplied - finance will ask for one.")

    if args.amount > RECEIPT_THRESHOLD:
        warnings.append(
            f"Amount exceeds {args.currency} {RECEIPT_THRESHOLD:,.0f}; an itemised receipt "
            "must be attached in the finance portal."
        )

    if args.amount > FINANCE_APPROVAL_THRESHOLD:
        route = "Finance Controller"
    elif args.amount > MANAGER_APPROVAL_THRESHOLD:
        route = "Reporting manager"
    else:
        route = "Auto-approval queue"

    employee = get_employee(db, employee_id)
    preview = {
        "employee_id": employee_id,
        "employee_name": employee.name if employee else "Unknown",
        "category": args.category,
        "amount": args.amount,
        "currency": args.currency,
        "description": args.description,
        "incurred_on": args.incurred_on.isoformat() if args.incurred_on else None,
        "approval_route": route,
    }
    return PreflightResult(ok=not blockers, preview=preview, warnings=warnings, blockers=blockers)


def execute_expense_claim(
    db: Session, employee_id: str, args: ExpenseClaimArgs, approval_id: str | None
) -> dict:
    record = ExpenseClaim(
        employee_id=employee_id,
        category=args.category,
        amount=args.amount,
        currency=args.currency,
        description=args.description,
        incurred_on=args.incurred_on,
        status="submitted",
        approval_id=approval_id,
    )
    db.add(record)
    db.flush()
    _audit(
        db,
        actor=employee_id,
        action="expense_claim.submitted",
        entity="expense_claim",
        entity_id=record.id,
        payload={"amount": args.amount, "currency": args.currency},
    )
    db.commit()
    return {
        "reference": f"EX-{record.id[:8].upper()}",
        "status": record.status,
        "amount": record.amount,
        "currency": record.currency,
        "category": record.category,
        "submitted_at": utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# IT service desk
# ---------------------------------------------------------------------------
def preflight_it_ticket(db: Session, employee_id: str, args: ITTicketArgs):
    from app.tools.registry import PreflightResult

    warnings: list[str] = []
    if args.category == "access":
        warnings.append("Access requests need manager confirmation before provisioning.")
    if args.priority == "critical":
        warnings.append("Critical tickets page the on-call engineer immediately.")

    employee = get_employee(db, employee_id)
    preview = {
        "employee_id": employee_id,
        "employee_name": employee.name if employee else "Unknown",
        "category": args.category,
        "priority": args.priority,
        "summary": args.summary,
        "details": args.details,
        "target_response": {
            "critical": "1 hour",
            "high": "4 business hours",
            "normal": "1 business day",
            "low": "3 business days",
        }[args.priority],
    }
    return PreflightResult(ok=True, preview=preview, warnings=warnings, blockers=[])


def execute_it_ticket(
    db: Session, employee_id: str, args: ITTicketArgs, approval_id: str | None
) -> dict:
    record = ITTicket(
        employee_id=employee_id,
        category=args.category,
        priority=args.priority,
        summary=args.summary,
        details=args.details,
        status="open",
        approval_id=approval_id,
    )
    db.add(record)
    db.flush()
    _audit(
        db,
        actor=employee_id,
        action="it_ticket.opened",
        entity="it_ticket",
        entity_id=record.id,
        payload={"category": args.category, "priority": args.priority},
    )
    db.commit()
    return {
        "reference": f"IT-{record.id[:8].upper()}",
        "status": record.status,
        "priority": record.priority,
        "summary": record.summary,
        "submitted_at": utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Read-only lookup
# ---------------------------------------------------------------------------
def preflight_leave_balance(db: Session, employee_id: str, args: LeaveBalanceArgs):
    from app.tools.registry import PreflightResult

    employee = get_employee(db, employee_id)
    if employee is None:
        return PreflightResult(
            ok=False,
            preview={},
            warnings=[],
            blockers=[f"No employee record for {employee_id}."],
        )
    return PreflightResult(
        ok=True, preview=_balance_payload(employee, args.leave_type), warnings=[], blockers=[]
    )


def execute_leave_balance(
    db: Session, employee_id: str, args: LeaveBalanceArgs, approval_id: str | None
) -> dict:
    employee = get_employee(db, employee_id)
    if employee is None:
        return {"error": f"No employee record for {employee_id}."}
    return _balance_payload(employee, args.leave_type)


def _balance_payload(employee: Employee, leave_type: str) -> dict:
    return {
        "employee_id": employee.id,
        "employee_name": employee.name,
        "annual": {
            "entitlement": employee.annual_leave_total,
            "used": employee.annual_leave_used,
            "remaining": employee.annual_leave_total - employee.annual_leave_used,
        },
        "sick": {
            "entitlement": employee.sick_leave_total,
            "used": employee.sick_leave_used,
            "remaining": employee.sick_leave_total - employee.sick_leave_used,
        },
        "requested_type": leave_type,
    }
