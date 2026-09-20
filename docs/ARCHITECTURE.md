# Architecture

## The agent graph

The graph is a compiled [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` (`backend/app/agents/graph.py`), stateless and built once at
import time. Per-request services (DB session, LLM client, usage tracker) are
injected through the LangGraph `config`, so one compiled graph safely serves
concurrent requests.

```
START → planner → retrieval → reasoning ─┬→ workflow ─┬→ verification
                       ↑                  └────────────┘        │
                       │                                        │
                       └──────────── retry (max 1) ──────────────┤
                                                                 │
                                        approval_gate ←───────────┘  (action proposed)
                                              │
                                             END
```

- **`planner`** classifies intent (`question` / `action` / `mixed` /
  `smalltalk`), picks a candidate tool from the registry (never invents one),
  and writes 1–3 retrieval queries.
- **`retrieval`** runs hybrid search (see below) and packs the highest-scoring
  chunks into a token-budgeted, numbered context block — the same numbering
  the reasoning and verification agents both key off.
- **`reasoning`** answers using *only* the numbered context. Every claim must
  carry a `[n]` marker. With no API key, this becomes **extractive**: it
  selects and quotes the highest-scoring sentences rather than generating
  text, so it cannot state a fact that isn't verbatim in a cited document.
- **`workflow`** (only entered for `action`/`mixed` intent) extracts tool
  arguments, validates them against a Pydantic schema, and runs a **preflight**
  against live data (leave balance, notice period, receipt thresholds) —
  producing blockers/warnings before anything is ever proposed to a human.
- **`verification`** is the quality gate — see below.
- **`approval_gate`** persists an `Approval` row and stops. It is the only
  node that touches the `Approval` table; nothing upstream of it can execute a
  tool.

Routing after `verification` is conditional: too little evidence sends the
turn back to `retrieval` once with a widened search; a proposed action with
elevated risk or low confidence routes to `approval_gate`; otherwise the turn
answers directly.

## Retrieval

`backend/app/rag/retriever.py`. Two independently-computed candidate lists are
fused:

1. **BM25** (`k1=1.5, b=0.75`) over a stemmed token index, rebuilt whenever the
   corpus checksum changes.
2. **Dense vectors** — either a dependency-free hashed bag-of-features
   embedder (default; word + bigram + char-4gram features, weighted so word
   identity dominates) or real `sentence-transformers` embeddings when
   `EMBEDDING_PROVIDER=sentence-transformers` is set. On Postgres the vector
   column is native `vector(N)` with pgvector cosine search; on SQLite it
   falls back to in-process numpy cosine — identical calling code either way.

Fusion (`_fuse`) blends each leg's **min-max normalised score** with its
**normalised reciprocal rank** (60/40), rather than pure RRF — on a corpus
this size, pure rank fusion throws away BM25's often-decisive score margin.
A **heading-relevance boost** rewards a chunk whose section title already
answers the question. A **near-duplicate suppression** pass (Jaccard over
token sets, engaged only above a similarity threshold) removes the
overlapping windows chunking produces without reordering genuinely distinct
sections — a targeted alternative to classic MMR, which over-penalises
policy documents whose sections are naturally similar.

Chunking (`chunking.py`) splits on Markdown headings first, then recursively
within each section, and prefixes every chunk with a de-duplicated breadcrumb
(`Document > Section > Subsection`) — retrieval and citation readability both
improve measurably from this alone.

## Verification — why it doesn't just ask the model "was that right?"

`backend/app/agents/verification_agent.py` runs deterministic checks in every
mode, and treats the model's self-reported confidence (when available) as a
second opinion that can only pull the score *down*, never up:

- **Citation validity** — every `[n]` marker must point at a real passage.
- **Citation coverage** — share of factual sentences that carry a citation at
  all (lead-ins, headings, and meta-notes are excluded from the denominator).
- **Evidence support** — IDF-weighted token overlap between each cited
  sentence and the passage it cites.
- **Numeric grounding** — every number in the answer must appear somewhere in
  the evidence; an invented threshold or deadline is the highest-signal
  failure mode in policy RAG, and is penalised harder than any other defect.
- **Question-vs-corpus coverage** — retrieval always returns *something*, so
  an out-of-scope question can still produce a fully-cited, well-supported
  *non-answer*. This check compares the question's own vocabulary against the
  evidence (plus an absolute BM25 floor) and gates the whole score by it — the
  reason the assistant flags "what's the share price forecast?" instead of
  confidently answering from unrelated policy text.
- **Action safety** — preflight blockers/warnings from the tool layer.

Answer-quality and action-quality are scored **separately** and combined by
intent: a pure `action` turn (no question asked) is graded entirely on
whether the prepared arguments are complete and policy-clean; a `mixed` turn
requires both scores to hold.

## Human-in-the-loop

`backend/app/services/approvals.py` is the only code path that calls a tool's
`execute()`. The agent graph builds a `ProposedAction` and, for anything
`requires_approval=True`, that's as far as it goes — an `Approval` row is
persisted and the graph ends. Approving re-validates the arguments (never
trusts what was stored), re-runs the preflight against current data, executes,
and writes to the audit log. Rejecting is a dead end — nothing runs.

## Data model

`backend/app/db/models.py` — `Document`/`Chunk` (knowledge base),
`Conversation`/`Message`/`Run`/`RunEvent` (chat + full per-agent trace),
`Approval`/`AuditLog` (governance), and a small mock HRIS
(`Employee`/`LeaveRequest`/`ExpenseClaim`/`ITTicket`) that the workflow agent
reads and the approval flow writes to.

## Evaluation

`backend/evals/golden_set.json` + `run_eval.py` scores the system against 27
labelled cases spanning every document and both question/action intents:
routing accuracy, retrieval recall@5/MRR, citation validity, groundedness,
hallucinated-number rate, action-argument accuracy, and — critically — whether
100% of sensitive actions were actually stopped at the human gate. CI runs
this on every push and fails if any threshold regresses.
