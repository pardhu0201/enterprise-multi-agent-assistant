"""Opt-in protection for state-changing endpoints.

The public demo is deliberately open so anyone can try the approval flow. Set
``ADMIN_TOKEN`` and every endpoint that changes shared state - approval
decisions, knowledge-base uploads/deletes/reindex, conversation deletes -
then requires an ``X-Admin-Token`` header with that value.

This is a single shared secret, not user authentication: a real deployment
would put SSO in front of the API and derive the decider's identity from the
session instead of the request body.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import settings


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = settings.admin_token.strip()
    if not expected:
        return  # open demo mode
    # Constant-time comparison so the token cannot be recovered by timing.
    if not x_admin_token or not hmac.compare_digest(x_admin_token.strip(), expected):
        raise HTTPException(status_code=401, detail="Admin token required")
