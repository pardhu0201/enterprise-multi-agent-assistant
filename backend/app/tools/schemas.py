"""Argument schemas for the business tools the workflow agent can propose.

These double as the JSON Schemas handed to Claude, so a malformed action is
rejected by Pydantic *before* it ever reaches an approval queue.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class LeaveRequestArgs(BaseModel):
    leave_type: str = Field(
        default="annual",
        description="One of: annual, sick, parental, unpaid.",
    )
    start_date: date = Field(description="First day of leave, ISO format YYYY-MM-DD.")
    end_date: date = Field(description="Last day of leave inclusive, ISO format YYYY-MM-DD.")
    reason: str = Field(default="", description="Short reason supplied by the employee.")

    @field_validator("leave_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"annual", "sick", "parental", "unpaid"}:
            raise ValueError("leave_type must be annual, sick, parental or unpaid")
        return v

    @field_validator("end_date")
    @classmethod
    def _ordered(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date cannot be before start_date")
        return v


class ExpenseClaimArgs(BaseModel):
    category: str = Field(default="travel", description="travel, meals, equipment or training.")
    amount: float = Field(gt=0, description="Claim amount, positive.")
    currency: str = Field(default="INR", description="ISO currency code.")
    description: str = Field(default="", description="What the expense was for.")
    incurred_on: date | None = Field(default=None, description="Date the expense was incurred.")

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"travel", "meals", "equipment", "training", "other"}:
            raise ValueError("unsupported expense category")
        return v


class ITTicketArgs(BaseModel):
    category: str = Field(default="access", description="access, hardware, software or incident.")
    priority: str = Field(default="normal", description="low, normal, high or critical.")
    summary: str = Field(min_length=3, description="One-line summary of the request.")
    details: str = Field(default="", description="Any extra detail the employee gave.")

    @field_validator("priority")
    @classmethod
    def _known_priority(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"low", "normal", "high", "critical"}:
            raise ValueError("priority must be low, normal, high or critical")
        return v


class LeaveBalanceArgs(BaseModel):
    leave_type: str = Field(default="annual", description="annual or sick.")
