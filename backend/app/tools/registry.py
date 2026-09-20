"""Tool registry: the only place where the assistant can touch a business system.

Each tool separates three phases, which is what makes safe human-in-the-loop
possible:

``validate``
    Parse the model's proposed arguments with Pydantic. Structurally invalid
    actions never get further than here.
``preflight``
    Run cheap, deterministic business checks against live data (leave balance,
    notice period, receipt thresholds) and return human-readable warnings plus
    a rendered preview of what *would* happen.
``execute``
    Actually write to the system of record. For anything flagged
    ``requires_approval`` this is unreachable until a human approves the
    queued request - the agent graph itself cannot call it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.tools import business_tools as impl
from app.tools.schemas import (
    ExpenseClaimArgs,
    ITTicketArgs,
    LeaveBalanceArgs,
    LeaveRequestArgs,
)


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    preview: dict
    warnings: list[str]
    blockers: list[str]

    @property
    def issues(self) -> list[str]:
        return [*self.blockers, *self.warnings]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    risk: str  # low | medium | high
    requires_approval: bool
    preflight: Callable[[Session, str, BaseModel], PreflightResult]
    execute: Callable[[Session, str, BaseModel, str | None], dict]
    keywords: tuple[str, ...] = ()

    def json_schema(self) -> dict:
        return self.args_model.model_json_schema()

    def validate(self, arguments: dict) -> BaseModel:
        return self.args_model.model_validate(arguments)


TOOLS: dict[str, ToolSpec] = {
    "submit_leave_request": ToolSpec(
        name="submit_leave_request",
        description=(
            "Submit a leave request to the HR system for the current employee. "
            "Use when the employee asks to take, book, apply for or request time off."
        ),
        args_model=LeaveRequestArgs,
        risk="high",
        requires_approval=True,
        preflight=impl.preflight_leave_request,
        execute=impl.execute_leave_request,
        keywords=("leave", "time off", "vacation", "holiday", "pto", "sick", "day off", "absence"),
    ),
    "submit_expense_claim": ToolSpec(
        name="submit_expense_claim",
        description=(
            "File an expense reimbursement claim. Use when the employee wants to "
            "claim, reimburse or expense a cost they have already incurred."
        ),
        args_model=ExpenseClaimArgs,
        risk="high",
        requires_approval=True,
        preflight=impl.preflight_expense_claim,
        execute=impl.execute_expense_claim,
        keywords=("expense", "reimburse", "claim", "receipt", "invoice", "spent"),
    ),
    "raise_it_ticket": ToolSpec(
        name="raise_it_ticket",
        description=(
            "Open a ticket with the IT service desk for access, hardware, software "
            "or a security incident."
        ),
        args_model=ITTicketArgs,
        risk="medium",
        requires_approval=True,
        preflight=impl.preflight_it_ticket,
        execute=impl.execute_it_ticket,
        keywords=("laptop", "access", "vpn", "password", "software", "it support", "ticket"),
    ),
    "check_leave_balance": ToolSpec(
        name="check_leave_balance",
        description="Read the current employee's remaining leave balance. Read-only.",
        args_model=LeaveBalanceArgs,
        risk="low",
        requires_approval=False,
        preflight=impl.preflight_leave_balance,
        execute=impl.execute_leave_balance,
        keywords=("balance", "how many days", "remaining leave", "left"),
    ),
}


def get_tool(name: str) -> ToolSpec | None:
    return TOOLS.get(name)


def tool_catalogue() -> list[dict]:
    """Compact description of every tool, for prompts and the /api/tools route."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "risk": spec.risk,
            "requires_approval": spec.requires_approval,
            "parameters": spec.json_schema(),
        }
        for spec in TOOLS.values()
    ]


def anthropic_tool_definitions() -> list[dict]:
    """The same catalogue in Messages API tool-definition shape."""
    definitions = []
    for spec in TOOLS.values():
        schema = spec.json_schema()
        schema["additionalProperties"] = False
        definitions.append(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": schema,
            }
        )
    return definitions
