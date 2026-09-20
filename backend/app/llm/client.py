"""Thin, typed wrapper around the Anthropic Messages API.

Design notes
------------
* Every agent talks to Claude through :meth:`LLMClient.structured`, which uses
  the Messages API's structured-output mode (``output_config.format``) so the
  response is always schema-valid JSON - no regex scraping of prose.
* Each call carries a deterministic *fallback*. If no ``ANTHROPIC_API_KEY`` is
  configured, or the API errors/refuses, the fallback runs instead and the run
  continues in "demo" mode. The graph, retrieval, verification and approval
  logic are identical in both modes; only the language model is swapped out.
  That is what lets the public deployment run at zero cost.
* Server-side refusal fallbacks are enabled by default on Opus 5. If the beta
  is not enabled for the org, the client silently drops to the stable endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResult:
    """Outcome of one agent-to-model call."""

    value: BaseModel
    mode: str  # "claude" | "demo"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    note: str = ""


@dataclass
class UsageTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    def add(self, agent: str, result: LLMResult) -> None:
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.by_agent[agent] = self.by_agent.get(agent, 0) + result.output_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_agent": self.by_agent,
        }


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema, tightened for the API's structured-output mode."""
    schema = model.model_json_schema()

    def tighten(node: dict) -> None:
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
            node.setdefault("required", sorted(node.get("properties", {}).keys()))
        for key in ("properties", "$defs"):
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    tighten(child)
        items = node.get("items")
        if isinstance(items, dict):
            tighten(items)

    tighten(schema)
    return schema


class LLMClient:
    """Single entry point for all model traffic."""

    def __init__(self) -> None:
        self._client = None
        self._use_beta = True
        self.model = settings.anthropic_model
        if settings.llm_enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=settings.anthropic_api_key,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=2,
                )
                log.info("LLM: Anthropic %s (effort=%s)", self.model, settings.llm_effort)
            except Exception as exc:  # pragma: no cover - import/credential failure
                log.error("Could not initialise Anthropic client (%s); using demo mode", exc)
                self._client = None
        else:
            log.info("LLM: demo mode (no ANTHROPIC_API_KEY configured)")

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def mode(self) -> str:
        return "claude" if self.available else "demo"

    # -- core ---------------------------------------------------------------
    def structured(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[T],
        fallback: Callable[[], T],
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Ask Claude for a schema-valid object, or compute the fallback."""
        if not self.available:
            return LLMResult(value=fallback(), mode="demo", note="no api key")

        started = time.perf_counter()
        try:
            response = self._create(system=system, user=user, schema=schema, max_tokens=max_tokens)
        except Exception as exc:
            log.warning("[%s] model call failed (%s); using deterministic fallback", agent, exc)
            return LLMResult(value=fallback(), mode="demo", note=f"api error: {type(exc).__name__}")

        latency_ms = int((time.perf_counter() - started) * 1000)

        if getattr(response, "stop_reason", None) == "refusal":  # pragma: no cover
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            log.warning("[%s] model refused (category=%s); using fallback", agent, category)
            return LLMResult(value=fallback(), mode="demo", note=f"refusal:{category}")

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            value = schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("[%s] unparseable model output (%s); using fallback", agent, exc)
            return LLMResult(value=fallback(), mode="demo", note="schema mismatch")

        usage = getattr(response, "usage", None)
        return LLMResult(
            value=value,
            mode="claude",
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=latency_ms,
        )

    # -- transport ----------------------------------------------------------
    def _create(self, *, system: str, user: str, schema: type[BaseModel], max_tokens: int | None):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Adaptive thinking: the model decides how much reasoning each
            # agent's task needs, bounded by the configured effort level.
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": settings.llm_effort,
                "format": {"type": "json_schema", "schema": _strict_schema(schema)},
            },
        }

        if self._use_beta:
            try:
                return self._client.beta.messages.create(
                    **kwargs,
                    betas=[REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                )
            except Exception as exc:
                if "anthropic-beta" in str(exc) or "fallbacks" in str(exc):
                    log.info("Refusal fallbacks unavailable for this org; using stable endpoint")
                    self._use_beta = False
                else:
                    raise

        return self._client.messages.create(**kwargs)


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
