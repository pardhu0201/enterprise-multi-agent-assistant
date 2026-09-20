import { useEffect, useMemo, useRef, useState } from "react";
import { CornerDownLeft, Loader2, RotateCcw, Sparkles, User } from "lucide-react";

import ActionCard from "../components/ActionCard";
import AgentTimeline from "../components/AgentTimeline";
import AnswerBody from "../components/AnswerBody";
import SourcePanel from "../components/SourcePanel";
import { ConfidenceMeter, FlagList, PageHeader, StatusPill } from "../components/primitives";
import { streamChat, type ChatResponse, type TraceEntry } from "../lib/api";

interface Turn {
  id: string;
  question: string;
  response?: ChatResponse;
  error?: string;
}

const SUGGESTIONS = [
  "Summarise our leave policy and book me 3 days off starting next Monday",
  "How much notice do I need for annual leave, and how many days do I get?",
  "I paid 2,400 for a client dinner last week - please file the expense claim",
  "I lost my MFA device, what should I do?",
  "What is the share price forecast for next quarter?",
];

export default function ChatPage({ onApprovalChange }: { onApprovalChange?: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [running, setRunning] = useState(false);
  const [liveTrace, setLiveTrace] = useState<TraceEntry[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedTurn, setSelectedTurn] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [tab, setTab] = useState<"trace" | "sources">("trace");

  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, liveTrace.length, running]);

  const inspected = useMemo(() => {
    const explicit = turns.find((t) => t.id === selectedTurn);
    if (explicit?.response) return explicit;
    return [...turns].reverse().find((t) => t.response);
  }, [turns, selectedTurn]);

  const send = async (text: string) => {
    const question = text.trim();
    if (!question || running) return;

    const id = crypto.randomUUID();
    setTurns((prev) => [...prev, { id, question }]);
    setDraft("");
    setRunning(true);
    setLiveTrace([]);
    setSelectedTurn(id);
    setHighlighted(null);
    setTab("trace");

    await streamChat(question, conversationId, {
      onStart: ({ conversation_id }) => setConversationId(conversation_id),
      onStep: (step) => setLiveTrace((prev) => [...prev, step]),
      onFinal: (response) => {
        setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, response } : t)));
        if (response.requires_approval) onApprovalChange?.();
      },
      onError: (message) =>
        setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message } : t))),
    }).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : "The assistant is unreachable";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message } : t)));
    });

    setRunning(false);
    setLiveTrace([]);
    inputRef.current?.focus();
  };

  const reset = () => {
    setTurns([]);
    setConversationId(null);
    setLiveTrace([]);
    setSelectedTurn(null);
  };

  const jumpToSource = (index: number) => {
    setTab("sources");
    setHighlighted(index);
    window.setTimeout(() => {
      document.getElementById(`source-${index}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }, 60);
  };

  return (
    <div className="flex min-h-screen flex-col xl:h-screen">
      <PageHeader
        title="Assistant"
        subtitle="Ask about company policy or ask for a request to be raised. Sensitive actions stop for your approval."
        actions={
          turns.length > 0 ? (
            <button className="btn-ghost" onClick={reset}>
              <RotateCcw size={14} />
              New conversation
            </button>
          ) : undefined
        }
      />

      <div className="grid min-h-0 flex-1 gap-0 xl:grid-cols-[minmax(0,1fr)_400px]">
        {/* ---------------- conversation ---------------- */}
        <div className="flex min-w-0 flex-col">
          <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-5 py-6 sm:px-7">
            {turns.length === 0 && <Welcome onPick={send} />}

            {turns.map((turn) => (
              <div key={turn.id} className="space-y-4">
                <div className="flex justify-end">
                  <div className="flex max-w-[85%] items-start gap-2.5">
                    <p className="rounded-2xl rounded-br-sm bg-[var(--color-surface-3)] px-4 py-2.5 text-[14.5px] leading-relaxed">
                      {turn.question}
                    </p>
                    <span className="mt-1 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[var(--color-surface-2)] text-[var(--color-ink-3)]">
                      <User size={12} />
                    </span>
                  </div>
                </div>

                {turn.error && (
                  <div className="rounded-lg border border-[rgba(251,113,133,0.4)] bg-[rgba(251,113,133,0.08)] px-4 py-3 text-[13px] text-[var(--color-danger)]">
                    {turn.error}
                  </div>
                )}

                {turn.response && (
                  <button
                    type="button"
                    onClick={() => setSelectedTurn(turn.id)}
                    className={[
                      "block w-full rounded-xl border p-4 text-left transition-colors sm:p-5",
                      inspected?.id === turn.id
                        ? "border-[var(--color-line-strong)] bg-[var(--color-surface)]"
                        : "border-[var(--color-line)] bg-[rgba(14,19,34,0.55)] hover:border-[var(--color-line-strong)]",
                    ].join(" ")}
                  >
                    <div className="mb-3 flex flex-wrap items-center gap-2">
                      <span className="grid h-6 w-6 place-items-center rounded-full bg-gradient-to-br from-[var(--color-brand)] to-[var(--color-brand-2)] text-[#0a0d18]">
                        <Sparkles size={12} />
                      </span>
                      <span className="text-[13px] font-medium">Assistant</span>
                      <StatusPill status={turn.response.status} />
                      <span className="chip">
                        {turn.response.llm_mode === "claude" ? "Claude" : "demo mode"}
                      </span>
                      <span className="ml-auto font-mono text-[11px] text-[var(--color-ink-3)]">
                        {turn.response.latency_ms}ms
                      </span>
                    </div>

                    <AnswerBody
                      text={turn.response.answer}
                      citations={turn.response.citations}
                      onCitationClick={(index) => {
                        setSelectedTurn(turn.id);
                        jumpToSource(index);
                      }}
                    />

                    {turn.response.follow_up_question && (
                      <p className="mt-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface-2)] px-3 py-2 text-[13px] text-[var(--color-ink-2)]">
                        {turn.response.follow_up_question}
                      </p>
                    )}

                    {turn.response.flags.length > 0 && (
                      <div className="mt-3">
                        <FlagList flags={turn.response.flags} />
                      </div>
                    )}
                  </button>
                )}

                {turn.response?.proposed_action && (
                  <ActionCard
                    action={turn.response.proposed_action}
                    approvalId={turn.response.approval_id}
                    onDecided={() => onApprovalChange?.()}
                  />
                )}
              </div>
            ))}

            {running && (
              <div className="rounded-xl border border-[var(--color-line)] bg-[rgba(14,19,34,0.55)] p-4 sm:p-5">
                <div className="mb-2 flex items-center gap-2 text-[13px] text-[var(--color-ink-2)]">
                  <Loader2 size={13} className="animate-spin text-[var(--color-brand)]" />
                  Agents are working
                </div>
                <AgentTimeline trace={liveTrace} running />
              </div>
            )}

            <div ref={endRef} />
          </div>

          {/* ---------------- composer ---------------- */}
          <div className="border-t border-[var(--color-line)] bg-[rgba(8,11,20,0.85)] px-5 py-4 backdrop-blur sm:px-7">
            <div className="flex items-end gap-2.5">
              <textarea
                ref={inputRef}
                className="field max-h-40 min-h-[46px] resize-none"
                rows={1}
                placeholder="Ask about a policy, or ask for a request to be raised…"
                value={draft}
                disabled={running}
                onChange={(e) => {
                  setDraft(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send(draft);
                  }
                }}
              />
              <button
                className="btn-primary h-[46px] px-4"
                onClick={() => void send(draft)}
                disabled={running || !draft.trim()}
                aria-label="Send message"
              >
                {running ? (
                  <Loader2 size={15} className="animate-spin" />
                ) : (
                  <CornerDownLeft size={15} />
                )}
              </button>
            </div>
            <p className="mt-2 text-[11px] text-[var(--color-ink-3)]">
              Enter to send, Shift+Enter for a new line. Nothing is submitted to a business
              system without your explicit approval.
            </p>
          </div>
        </div>

        {/* ---------------- inspector ---------------- */}
        <aside className="min-w-0 border-t border-[var(--color-line)] bg-[rgba(14,19,34,0.4)] xl:min-h-0 xl:overflow-y-auto xl:border-t-0 xl:border-l">
          <div className="sticky top-0 z-10 flex gap-1 border-b border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-2">
            {(["trace", "sources"] as const).map((key) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={[
                  "rounded-md px-3 py-1.5 text-[12px] font-medium capitalize transition-colors",
                  tab === key
                    ? "bg-[var(--color-surface-3)] text-[var(--color-ink)]"
                    : "text-[var(--color-ink-3)] hover:text-[var(--color-ink-2)]",
                ].join(" ")}
              >
                {key === "trace" ? "Agent trace" : "Sources"}
                {key === "sources" && inspected?.response?.citations.length
                  ? ` (${inspected.response.citations.length})`
                  : ""}
              </button>
            ))}
          </div>

          <div className="space-y-5 p-4">
            {!inspected?.response ? (
              <p className="px-1 py-8 text-center text-[13px] text-[var(--color-ink-3)]">
                Ask something to see how the agents reached the answer.
              </p>
            ) : tab === "trace" ? (
              <>
                <ConfidenceMeter value={inspected.response.confidence} />
                <AgentTimeline trace={inspected.response.trace} running={false} />
                <VerificationDetails response={inspected.response} />
              </>
            ) : (
              <SourcePanel
                citations={inspected.response.citations}
                used={inspected.response.used_citations}
                highlighted={highlighted}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function VerificationDetails({ response }: { response: ChatResponse }) {
  const v = response.verification ?? {};
  const rows: [string, string][] = [];
  const pct = (x: unknown) => (typeof x === "number" ? `${Math.round(x * 100)}%` : "-");

  rows.push(["Citation coverage", pct(v.citation_coverage)]);
  rows.push(["Evidence support", pct(v.lexical_support)]);
  rows.push(["Question coverage", pct(v.query_coverage)]);
  if (typeof v.answer_confidence === "number") {
    rows.push(["Answer confidence", pct(v.answer_confidence)]);
  }
  if (typeof v.action_confidence === "number") {
    rows.push(["Action confidence", pct(v.action_confidence)]);
  }
  if (v.mode === "claude") rows.push(["Model confidence", pct(v.llm_confidence)]);
  rows.push(["Decision", String(v.decision ?? "-")]);

  const usage = response.token_usage as { input_tokens?: number; output_tokens?: number } | null;

  return (
    <div className="panel-2 p-3.5">
      <div className="label mb-2.5">Verification</div>
      <dl className="space-y-1.5">
        {rows.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-3 text-[12px]">
            <dt className="text-[var(--color-ink-3)]">{label}</dt>
            <dd className="font-mono text-[var(--color-ink-2)]">{value}</dd>
          </div>
        ))}
      </dl>

      {Array.isArray(v.unsupported_claims) && v.unsupported_claims.length > 0 && (
        <div className="mt-3 border-t border-[var(--color-line)] pt-2.5">
          <div className="label mb-1.5">Claims needing review</div>
          <ul className="ml-4 list-disc space-y-1 text-[12px] text-[var(--color-ink-2)]">
            {v.unsupported_claims.slice(0, 3).map((claim: string) => (
              <li key={claim}>{claim}</li>
            ))}
          </ul>
        </div>
      )}

      {usage && (usage.input_tokens ?? 0) > 0 && (
        <p className="mt-3 border-t border-[var(--color-line)] pt-2.5 font-mono text-[11px] text-[var(--color-ink-3)]">
          {usage.input_tokens} in / {usage.output_tokens} out tokens
        </p>
      )}
    </div>
  );
}

function Welcome({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto max-w-2xl py-8">
      <h2 className="text-xl font-semibold tracking-tight">
        Ask anything about company policy.
      </h2>
      <p className="mt-2 text-sm leading-relaxed text-[var(--color-ink-2)]">
        Four agents coordinate on every turn: a <strong>retrieval</strong> agent searches the
        policy corpus, a <strong>reasoning</strong> agent answers with citations, a{" "}
        <strong>workflow</strong> agent prepares any request you ask for, and a{" "}
        <strong>verification</strong> agent checks the result against the evidence before you
        see it. Anything that writes to a business system waits for your approval.
      </p>

      <div className="mt-6 space-y-2">
        <div className="label">Try one of these</div>
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => onPick(suggestion)}
            className="block w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-surface-2)] px-3.5 py-2.5 text-left text-[13.5px] text-[var(--color-ink-2)] transition-colors hover:border-[var(--color-line-strong)] hover:text-[var(--color-ink)]"
          >
            {suggestion}
          </button>
        ))}
      </div>
      <p className="mt-3 text-[11.5px] text-[var(--color-ink-3)]">
        The last one is deliberately outside the knowledge base - watch the verification agent
        flag it instead of inventing an answer.
      </p>
    </div>
  );
}
