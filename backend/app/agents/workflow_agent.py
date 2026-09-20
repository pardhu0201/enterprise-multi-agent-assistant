"""Workflow agent - prepares a business action, never performs one.

Output contract: a *proposal* containing the tool name, Pydantic-validated
arguments, a preflight preview computed against live data, and any blockers or
warnings. Execution is physically impossible from this node - the tool's
`execute` callable is only reachable from the approvals API after a human
decision.
"""

from __future__ import annotations

import json
import time
from datetime import date

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from app.agents.dateparse import infer_range
from app.agents.prompts import WORKFLOW_SYSTEM
from app.agents.state import AgentState, RunContext, trace_event
from app.logging_config import get_logger
from app.tools.business_tools import get_employee
from app.tools.registry import ToolSpec, get_tool

log = get_logger(__name__)


class ActionExtraction(BaseModel):
    arguments_json: str = Field(
        description="A single JSON object literal matching the tool's argument schema."
    )
    missing_fields: list[str] = Field(
        default_factory=list, description="Schema fields that had to be guessed or were absent."
    )
    confirmation_needed: bool = Field(
        default=False, description="True when a guessed field could change the submission."
    )
    rationale: str = Field(default="", description="One sentence for the human approver.")


# ---------------------------------------------------------------------------
# Deterministic extraction (demo mode / fallback)
# ---------------------------------------------------------------------------
def _rule_based_arguments(tool: ToolSpec, query: str, today: date) -> ActionExtraction:
    lowered = query.lower()
    missing: list[str] = []

    if tool.name == "submit_leave_request":
        leave_type = "annual"
        for keyword, value in (
            ("sick", "sick"),
            ("medical", "sick"),
            ("maternity", "parental"),
            ("paternity", "parental"),
            ("parental", "parental"),
            ("unpaid", "unpaid"),
        ):
            if keyword in lowered:
                leave_type = value
                break

        start, end = infer_range(query, today)
        if start is None:
            missing.append("start_date")
            start = today
        if end is None:
            missing.append("end_date")
            end = start

        args = {
            "leave_type": leave_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "reason": query.strip()[:240],
        }
        return ActionExtraction(
            arguments_json=json.dumps(args),
            missing_fields=missing,
            confirmation_needed=bool(missing),
            rationale=(
                f"Prepared a leave request ({leave_type}) from "
                f"{args['start_date']} to {args['end_date']}."
            ),
        )

    if tool.name == "submit_expense_claim":
        import re

        amount_match = re.search(r"(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d{1,2})?)", lowered)
        amount = float(amount_match.group(1).replace(",", "")) if amount_match else 0.0
        if not amount:
            missing.append("amount")
            amount = 1.0
        category = "travel"
        for keyword, value in (
            ("meal", "meals"),
            ("food", "meals"),
            ("lunch", "meals"),
            ("dinner", "meals"),
            ("laptop", "equipment"),
            ("monitor", "equipment"),
            ("equipment", "equipment"),
            ("course", "training"),
            ("training", "training"),
            ("conference", "training"),
        ):
            if keyword in lowered:
                category = value
                break
        dates = infer_range(query, today)[0]
        args = {
            "category": category,
            "amount": amount,
            "currency": "USD" if "$" in query else "INR",
            "description": query.strip()[:240],
            "incurred_on": dates.isoformat() if dates else None,
        }
        return ActionExtraction(
            arguments_json=json.dumps(args),
            missing_fields=missing,
            confirmation_needed=bool(missing),
            rationale=(
                f"Prepared an expense claim ({category}) for {args['currency']} {amount:,.2f}."
            ),
        )

    if tool.name == "raise_it_ticket":
        category = "access"
        for keyword, value in (
            ("laptop", "hardware"),
            ("monitor", "hardware"),
            ("keyboard", "hardware"),
            ("install", "software"),
            ("licence", "software"),
            ("license", "software"),
            ("breach", "incident"),
            ("phishing", "incident"),
            ("incident", "incident"),
        ):
            if keyword in lowered:
                category = value
                break
        priority = "high" if any(w in lowered for w in ("urgent", "asap", "blocked")) else "normal"
        if category == "incident":
            priority = "critical"
        args = {
            "category": category,
            "priority": priority,
            "summary": query.strip()[:120],
            "details": query.strip()[:500],
        }
        return ActionExtraction(
            arguments_json=json.dumps(args),
            missing_fields=[],
            confirmation_needed=False,
            rationale=f"Prepared an IT ticket ({category}) at {priority} priority.",
        )

    return ActionExtraction(
        arguments_json=json.dumps({}),
        missing_fields=["all"],
        confirmation_needed=True,
        rationale="No deterministic parser for this tool.",
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def workflow_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    started = time.perf_counter()

    tool_name = state.get("candidate_tool") or ""
    tool = get_tool(tool_name)
    if tool is None:
        return {
            "proposed_action": None,
            "action_note": "No business action required.",
            "trace": [trace_event(ctx, "workflow", "No business action required", started=started)],
        }

    today = date.today()
    employee = get_employee(ctx.db, ctx.employee_id)
    profile = (
        f"id={ctx.employee_id}, name={employee.name}, department={employee.department}, "
        f"manager={employee.manager}, annual_leave_remaining="
        f"{employee.annual_leave_total - employee.annual_leave_used:.1f}, "
        f"sick_leave_remaining={employee.sick_leave_total - employee.sick_leave_used:.1f}"
        if employee
        else f"id={ctx.employee_id} (no HR record found)"
    )

    user = (
        f"Today is {today.isoformat()} ({today.strftime('%A')}).\n\n"
        f"Employee profile: {profile}\n\n"
        f"Tool: {tool.name}\n"
        f"Purpose: {tool.description}\n"
        f"Argument JSON Schema:\n{json.dumps(tool.json_schema(), indent=2)}\n\n"
        f"Relevant policy passages:\n{state.get('context_block') or '(none)'}\n\n"
        f"Employee request:\n{state['query']}"
    )

    result = ctx.llm.structured(
        agent="workflow",
        system=WORKFLOW_SYSTEM,
        user=user,
        schema=ActionExtraction,
        fallback=lambda: _rule_based_arguments(tool, state["query"], today),
        max_tokens=3000,
    )
    ctx.usage.add("workflow", result)
    extraction: ActionExtraction = result.value  # type: ignore[assignment]

    # --- validate ---------------------------------------------------------
    raw_arguments: dict = {}
    validation_error = ""
    try:
        parsed = json.loads(extraction.arguments_json)
        raw_arguments = parsed if isinstance(parsed, dict) else {}
        validated = tool.validate(raw_arguments)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        validation_error = str(exc)
        log.info("[workflow] model arguments rejected (%s); retrying with rule parser", exc)
        fallback = _rule_based_arguments(tool, state["query"], today)
        try:
            raw_arguments = json.loads(fallback.arguments_json)
            validated = tool.validate(raw_arguments)
            extraction = fallback
            validation_error = f"{validation_error} (recovered with deterministic parser)"
        except (json.JSONDecodeError, ValidationError, TypeError) as exc2:
            return {
                "proposed_action": {
                    "tool_name": tool.name,
                    "risk": tool.risk,
                    "requires_approval": tool.requires_approval,
                    "arguments": raw_arguments,
                    "valid": False,
                    "blockers": [f"Could not build a valid request: {exc2}"],
                    "warnings": [],
                    "preview": {},
                    "missing_fields": extraction.missing_fields,
                    "confirmation_needed": True,
                    "rationale": extraction.rationale,
                },
                "action_note": "Action could not be prepared.",
                "trace": [
                    trace_event(
                        ctx,
                        "workflow",
                        f"Could not prepare '{tool.name}' - invalid arguments",
                        status="error",
                        payload={"error": str(exc2), "raw": raw_arguments},
                        started=started,
                    )
                ],
            }

    # --- preflight against live data --------------------------------------
    preflight = tool.preflight(ctx.db, ctx.employee_id, validated)

    proposed = {
        "tool_name": tool.name,
        "description": tool.description,
        "risk": tool.risk,
        "requires_approval": tool.requires_approval,
        "arguments": json.loads(validated.model_dump_json()),
        "valid": preflight.ok,
        "blockers": preflight.blockers,
        "warnings": preflight.warnings,
        "preview": preflight.preview,
        "missing_fields": extraction.missing_fields,
        "confirmation_needed": extraction.confirmation_needed or bool(extraction.missing_fields),
        "rationale": extraction.rationale,
        "extraction_mode": result.mode,
    }
    if validation_error:
        proposed["validation_note"] = validation_error

    status = "error" if preflight.blockers else ("warning" if preflight.warnings else "ok")
    summary = (
        f"Prepared '{tool.name}' for approval"
        if preflight.ok
        else f"Prepared '{tool.name}' but it is blocked"
    )

    return {
        "proposed_action": proposed,
        "action_note": extraction.rationale,
        "trace": [
            trace_event(
                ctx,
                "workflow",
                summary,
                status=status,
                payload={
                    "tool": tool.name,
                    "arguments": proposed["arguments"],
                    "blockers": preflight.blockers,
                    "warnings": preflight.warnings,
                    "mode": result.mode,
                },
                started=started,
            )
        ],
    }
