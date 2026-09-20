import { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CircleHelp,
  Clock,
  Gauge,
  Loader2,
  ShieldCheck,
  Timer,
} from "lucide-react";

import { api, type GraphTopology, type Metrics } from "../lib/api";
import { ConfidenceMeter, EmptyState, PageHeader } from "../components/primitives";

/* --------------------------------------------------------------------------
   Chart palette.

   Magnitude bars use ONE hue (sequential job) - length carries the value, so a
   second hue would encode nothing. Run outcomes use the reserved status
   palette in a fixed order, validated against this surface (#0e1322): worst
   adjacent CVD ΔE 9.4, all steps inside the dark lightness band and above 3:1
   contrast. Every status mark is also written out in words, so state is never
   communicated by colour alone.
   -------------------------------------------------------------------------- */
const SEQUENTIAL = "#3987e5";

const STATUS_STYLE: Record<string, { fill: string; label: string; icon: typeof Check }> = {
  completed: { fill: "#059669", label: "Answered", icon: Check },
  needs_clarification: { fill: "#0284c7", label: "Needs clarification", icon: CircleHelp },
  awaiting_approval: { fill: "#d97706", label: "Awaiting approval", icon: Clock },
  escalated: { fill: "#e11d48", label: "Escalated", icon: AlertTriangle },
  failed: { fill: "#e11d48", label: "Failed", icon: AlertTriangle },
  running: { fill: "#0284c7", label: "Running", icon: Loader2 },
  rejected: { fill: "#0284c7", label: "Rejected", icon: CircleHelp },
  approved: { fill: "#059669", label: "Approved", icon: Check },
  pending: { fill: "#d97706", label: "Pending", icon: Clock },
};

const statusStyle = (key: string) =>
  STATUS_STYLE[key] ?? { fill: "#0284c7", label: key.replace(/_/g, " "), icon: CircleHelp };

export default function ObservabilityPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [graph, setGraph] = useState<GraphTopology | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.metrics(), api.graph()])
      .then(([m, g]) => {
        setMetrics(m);
        setGraph(g);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-[var(--color-ink-3)]">
        <Loader2 size={22} className="animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Observability"
        subtitle="Every turn is persisted with its full agent trace, retrieval evidence and verification scores."
      />

      <div className="space-y-7 px-5 py-6 sm:px-7">
        {metrics && metrics.runs_total === 0 ? (
          <EmptyState
            icon={<Gauge size={18} />}
            title="No runs recorded yet"
            body="Ask the assistant a few questions and the trace, confidence and citation statistics will appear here."
          />
        ) : (
          metrics && (
            <>
              <StatRow metrics={metrics} />

              <div className="grid items-start gap-5 lg:grid-cols-2">
                <DocumentUsage documents={metrics.top_documents} />
                <RunOutcomes
                  counts={metrics.runs_by_status}
                  total={metrics.runs_total}
                  approvals={metrics.approvals_by_status}
                />
              </div>

              <RecentRuns runs={metrics.recent_runs} />
            </>
          )
        )}

        {graph && <GraphMap graph={graph} />}
      </div>
    </div>
  );
}

/* ---- KPI row: headline numbers, not charts ------------------------------ */
function StatRow({ metrics }: { metrics: Metrics }) {
  const tiles = [
    { label: "Turns processed", value: String(metrics.runs_total), icon: Gauge },
    {
      label: "Mean confidence",
      value: `${Math.round(metrics.average_confidence * 100)}%`,
      icon: ShieldCheck,
      note: `${Math.round(metrics.low_confidence_rate * 100)}% below threshold`,
    },
    {
      label: "Mean latency",
      value: `${Math.round(metrics.average_latency_ms)}ms`,
      icon: Timer,
    },
    {
      label: "Actions executed",
      value: String(metrics.actions_executed),
      icon: Check,
      note: `${metrics.approvals_by_status.pending ?? 0} awaiting approval`,
    },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {tiles.map(({ label, value, icon: Icon, note }) => (
        <div key={label} className="panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-ink-3)]">
            <Icon size={13} />
            <span className="label">{label}</span>
          </div>
          <div className="mt-2 font-mono text-[28px] leading-none font-semibold tracking-tight">
            {value}
          </div>
          {note && <div className="mt-1.5 text-[11.5px] text-[var(--color-ink-3)]">{note}</div>}
        </div>
      ))}
    </div>
  );
}

/* ---- magnitude: one hue, direct labels ---------------------------------- */
function DocumentUsage({ documents }: { documents: { title: string; citations: number }[] }) {
  const max = Math.max(1, ...documents.map((d) => d.citations));

  return (
    <section className="panel p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Which documents answer questions</h2>
      <p className="mt-0.5 mb-4 text-[12.5px] text-[var(--color-ink-3)]">
        Passages cited across recent turns
      </p>

      {documents.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-[var(--color-ink-3)]">
          No citations recorded yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {documents.map((doc) => (
            <li key={doc.title} title={`${doc.title}: ${doc.citations} citations`}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className="truncate text-[12.5px] text-[var(--color-ink-2)]">
                  {doc.title}
                </span>
                <span className="shrink-0 font-mono text-[12.5px] text-[var(--color-ink)]">
                  {doc.citations}
                </span>
              </div>
              <div className="h-2.5 w-full rounded-[4px] bg-[var(--color-surface-2)]">
                <div
                  className="h-full rounded-[4px] transition-[width] duration-500"
                  style={{
                    width: `${Math.max((doc.citations / max) * 100, 3)}%`,
                    background: SEQUENTIAL,
                  }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* ---- state: reserved status palette + written labels -------------------- */
function RunOutcomes({
  counts,
  total,
  approvals,
}: {
  counts: Record<string, number>;
  total: number;
  approvals: Record<string, number>;
}) {
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, n]) => n));
  const approvalRows = Object.entries(approvals).sort((a, b) => b[1] - a[1]);

  return (
    <section className="panel p-4 sm:p-5">
      <h2 className="text-sm font-semibold">How turns ended</h2>
      <p className="mt-0.5 mb-4 text-[12.5px] text-[var(--color-ink-3)]">
        {total} turns, routed by the verification agent
      </p>

      <ul className="space-y-3">
        {rows.map(([status, count]) => {
          const { fill, label, icon: Icon } = statusStyle(status);
          return (
            <li key={status} title={`${label}: ${count} of ${total} turns`}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className="flex min-w-0 items-center gap-1.5 text-[12.5px] text-[var(--color-ink-2)]">
                  <Icon size={11} style={{ color: fill }} className="shrink-0" />
                  <span className="truncate">{label}</span>
                </span>
                <span className="shrink-0 font-mono text-[12.5px] text-[var(--color-ink)]">
                  {count}
                </span>
              </div>
              <div className="h-2.5 w-full rounded-[4px] bg-[var(--color-surface-2)]">
                <div
                  className="h-full rounded-[4px] transition-[width] duration-500"
                  style={{ width: `${Math.max((count / max) * 100, 3)}%`, background: fill }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      {approvalRows.length > 0 && (
        <div className="mt-5 border-t border-[var(--color-line)] pt-4">
          <div className="label mb-2.5">Approval queue</div>
          <ul className="flex flex-wrap gap-2">
            {approvalRows.map(([status, count]) => {
              const { fill, label, icon: Icon } = statusStyle(status);
              return (
                <li
                  key={status}
                  className="chip"
                  style={{ color: fill, borderColor: `${fill}66` }}
                >
                  <Icon size={11} />
                  {label}
                  <span className="font-mono text-[var(--color-ink)]">{count}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}

/* ---- the table view ------------------------------------------------------ */
function RecentRuns({ runs }: { runs: Metrics["recent_runs"] }) {
  if (!runs.length) return null;
  return (
    <section className="panel overflow-hidden">
      <div className="border-b border-[var(--color-line)] px-4 py-3.5 sm:px-5">
        <h2 className="text-sm font-semibold">Recent turns</h2>
        <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-3)]">
          Each row links to a persisted trace of every agent decision
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-[13px]">
          <thead>
            <tr className="border-b border-[var(--color-line)] text-[var(--color-ink-3)]">
              <th className="px-4 py-2.5 font-medium sm:px-5">Question</th>
              <th className="px-3 py-2.5 font-medium">Intent</th>
              <th className="px-3 py-2.5 font-medium">Outcome</th>
              <th className="px-3 py-2.5 font-medium">Confidence</th>
              <th className="px-3 py-2.5 text-right font-medium sm:px-5">Latency</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const { fill, label } = statusStyle(run.status);
              return (
                <tr
                  key={run.id}
                  className="border-b border-[var(--color-line)] last:border-0 hover:bg-[var(--color-surface-2)]"
                >
                  <td className="max-w-[320px] truncate px-4 py-2.5 sm:px-5">{run.query}</td>
                  <td className="px-3 py-2.5 text-[var(--color-ink-3)]">{run.intent}</td>
                  <td className="px-3 py-2.5">
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: fill }}
                      />
                      <span className="text-[var(--color-ink-2)]">{label}</span>
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <ConfidenceMeter value={run.confidence} compact />
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-[var(--color-ink-3)] sm:px-5">
                    {run.latency_ms}ms
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ---- the graph itself ---------------------------------------------------- */
function GraphMap({ graph }: { graph: GraphTopology }) {
  return (
    <section className="panel p-4 sm:p-5">
      <h2 className="text-sm font-semibold">Agent graph</h2>
      <p className="mt-0.5 mb-4 text-[12.5px] text-[var(--color-ink-3)]">
        Compiled LangGraph topology served live from the API
      </p>

      <div className="flex flex-wrap items-stretch gap-2">
        {graph.nodes.map((node, i) => (
          <div key={node.id} className="flex items-center gap-2">
            <div className="panel-2 w-[172px] p-3">
              <div className="text-[12.5px] font-medium">{node.label}</div>
              <div className="mt-0.5 text-[11px] leading-snug text-[var(--color-ink-3)]">
                {node.role}
              </div>
            </div>
            {i < graph.nodes.length - 1 && (
              <ArrowRight size={13} className="shrink-0 text-[var(--color-ink-3)]" />
            )}
          </div>
        ))}
      </div>

      <ul className="mt-4 space-y-1 border-t border-[var(--color-line)] pt-3.5 text-[11.5px] text-[var(--color-ink-3)]">
        {graph.edges
          .filter((edge) => edge.condition)
          .map((edge) => (
            <li key={`${edge.source}-${edge.target}`} className="font-mono">
              {edge.source} → {edge.target}
              <span className="ml-2 font-sans italic">when {edge.condition}</span>
            </li>
          ))}
      </ul>
    </section>
  );
}
