import { useCallback, useEffect, useState } from "react";
import { Check, Inbox, Loader2, RefreshCw, X } from "lucide-react";

import { api, type Approval } from "../lib/api";
import {
  ConfidenceMeter,
  EmptyState,
  FlagList,
  KeyValue,
  PageHeader,
  RiskBadge,
  StatusPill,
} from "../components/primitives";

const FILTERS = ["pending", "approved", "rejected", "all"] as const;

export default function ApprovalsPage({ onApprovalChange }: { onApprovalChange?: () => void }) {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("pending");
  const [rows, setRows] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setRows(await api.approvals(filter));
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the approval queue");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    void load();
  }, [load]);

  const decide = async (id: string, decision: "approve" | "reject") => {
    setBusy(id);
    try {
      await api.decide(id, decision);
      await load();
      onApprovalChange?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not record the decision");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Approval queue"
        subtitle="Every action the assistant prepares lands here first. Approving is the only path that writes to a business system."
        actions={
          <button className="btn-ghost" onClick={() => void load()}>
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        }
      />

      <div className="px-5 py-5 sm:px-7">
        <div className="mb-5 flex flex-wrap gap-1.5">
          {FILTERS.map((option) => (
            <button
              key={option}
              onClick={() => setFilter(option)}
              className={[
                "rounded-lg px-3 py-1.5 text-[12.5px] font-medium capitalize transition-colors",
                filter === option
                  ? "bg-[var(--color-surface-3)] text-[var(--color-ink)]"
                  : "text-[var(--color-ink-3)] hover:bg-[var(--color-surface-2)]",
              ].join(" ")}
            >
              {option}
            </button>
          ))}
        </div>

        {error && (
          <p className="mb-4 rounded-lg border border-[rgba(251,113,133,0.4)] bg-[rgba(251,113,133,0.08)] px-3.5 py-2.5 text-[13px] text-[var(--color-danger)]">
            {error}
          </p>
        )}

        {loading && rows.length === 0 ? (
          <div className="flex justify-center py-16 text-[var(--color-ink-3)]">
            <Loader2 size={20} className="animate-spin" />
          </div>
        ) : rows.length === 0 ? (
          <EmptyState
            icon={<Inbox size={18} />}
            title={filter === "pending" ? "Nothing waiting on you" : "No requests here"}
            body="Ask the assistant to submit a leave request, file an expense or raise a ticket, and it will appear in this queue."
          />
        ) : (
          <div className="space-y-3">
            {rows.map((row) => (
              <article key={row.id} className="panel p-4 sm:p-5">
                <div className="flex flex-wrap items-center gap-2">
                  <code className="rounded bg-[var(--color-surface-2)] px-2 py-0.5 font-mono text-[12px] text-[var(--color-brand-2)]">
                    {row.tool_name}
                  </code>
                  <StatusPill status={row.status} />
                  <RiskBadge risk={row.risk} />
                  <span className="chip">{row.requested_by}</span>
                  <span className="ml-auto font-mono text-[11px] text-[var(--color-ink-3)]">
                    {new Date(row.created_at).toLocaleString()}
                  </span>
                </div>

                {row.rationale && (
                  <p className="mt-3 text-[13px] text-[var(--color-ink-2)]">{row.rationale}</p>
                )}

                <div className="mt-3.5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_200px]">
                  <div className="panel-2 p-3.5">
                    <div className="label mb-2">Arguments</div>
                    <KeyValue data={row.arguments} />
                  </div>
                  <div className="space-y-3">
                    <ConfidenceMeter value={row.confidence} />
                    <FlagList flags={row.flags} />
                  </div>
                </div>

                {row.execution_result && (
                  <div className="mt-3.5 rounded-lg border border-[rgba(52,211,153,0.35)] bg-[rgba(52,211,153,0.07)] p-3.5">
                    <div
                      className="label mb-2"
                      style={{ color: "var(--color-ok)" }}
                    >
                      Executed
                    </div>
                    <KeyValue data={row.execution_result} />
                  </div>
                )}

                {row.decided_by && (
                  <p className="mt-3 text-[12px] text-[var(--color-ink-3)]">
                    {row.status} by {row.decided_by}
                    {row.decision_note ? ` - "${row.decision_note}"` : ""}
                  </p>
                )}

                {row.status === "pending" && (
                  <div className="mt-4 flex gap-2 border-t border-[var(--color-line)] pt-3.5">
                    <button
                      className="btn-ok"
                      disabled={busy === row.id}
                      onClick={() => void decide(row.id, "approve")}
                    >
                      {busy === row.id ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Check size={14} />
                      )}
                      Approve
                    </button>
                    <button
                      className="btn-danger"
                      disabled={busy === row.id}
                      onClick={() => void decide(row.id, "reject")}
                    >
                      <X size={14} />
                      Reject
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
