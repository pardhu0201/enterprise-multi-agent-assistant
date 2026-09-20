import { useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronRight,
  Compass,
  FileSearch,
  Lightbulb,
  Loader2,
  ShieldCheck,
  UserCheck,
  Wrench,
  X,
} from "lucide-react";

import type { TraceEntry } from "../lib/api";

const AGENT_ICONS: Record<string, typeof Compass> = {
  planner: Compass,
  retrieval: FileSearch,
  reasoning: Lightbulb,
  workflow: Wrench,
  verification: ShieldCheck,
  approval_gate: UserCheck,
};

const STATUS_COLOR: Record<string, string> = {
  ok: "var(--color-ok)",
  warning: "var(--color-warn)",
  error: "var(--color-danger)",
};

/**
 * Live view of the agent graph for one turn. Each node appends its entry as it
 * finishes, so the panel doubles as the explanation of *why* the assistant
 * answered the way it did - which is the point of the whole architecture.
 */
export default function AgentTimeline({
  trace,
  running,
}: {
  trace: TraceEntry[];
  running: boolean;
}) {
  return (
    <ol className="space-y-1">
      {trace.map((entry) => (
        <TimelineStep key={`${entry.seq}-${entry.agent}`} entry={entry} />
      ))}

      {running && (
        <li className="flex items-center gap-3 px-1 py-2 text-sm text-[var(--color-ink-3)]">
          <span className="running-dot grid h-6 w-6 place-items-center rounded-full border border-[var(--color-line-strong)] bg-[var(--color-surface-2)]">
            <Loader2 size={12} className="animate-spin text-[var(--color-brand)]" />
          </span>
          {trace.length ? "Next agent working…" : "Starting the agent graph…"}
        </li>
      )}
    </ol>
  );
}

function TimelineStep({ entry }: { entry: TraceEntry }) {
  const [open, setOpen] = useState(false);
  const Icon = AGENT_ICONS[entry.agent] ?? Compass;
  const color = STATUS_COLOR[entry.status] ?? "var(--color-ink-3)";
  const StatusIcon = entry.status === "error" ? X : entry.status === "warning" ? AlertTriangle : Check;
  const details = summarise(entry);

  return (
    <li className="rise">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 rounded-lg px-1 py-2 text-left transition-colors hover:bg-[var(--color-surface-2)]"
        aria-expanded={open}
      >
        <span
          className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border"
          style={{ borderColor: `${color}55`, background: `${color}14`, color }}
        >
          <Icon size={12} />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-baseline gap-2">
            <span className="text-[13px] font-medium text-[var(--color-ink)]">{entry.label}</span>
            <StatusIcon size={11} style={{ color }} className="shrink-0 self-center" />
            <span className="ml-auto shrink-0 font-mono text-[11px] text-[var(--color-ink-3)]">
              {entry.duration_ms}ms
            </span>
          </span>
          <span className="mt-0.5 block text-[13px] leading-snug text-[var(--color-ink-2)]">
            {entry.summary}
          </span>
        </span>

        {details.length > 0 && (
          <ChevronRight
            size={14}
            className={`mt-1 shrink-0 text-[var(--color-ink-3)] transition-transform ${open ? "rotate-90" : ""}`}
          />
        )}
      </button>

      {open && details.length > 0 && (
        <dl className="mb-1 ml-9 space-y-1.5 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface-2)] p-3">
          {details.map(([key, value]) => (
            <div key={key} className="flex gap-3 text-[12px]">
              <dt className="w-32 shrink-0 text-[var(--color-ink-3)]">{key}</dt>
              <dd className="min-w-0 flex-1 font-mono break-words text-[var(--color-ink-2)]">
                {value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </li>
  );
}

/** Pick the handful of payload fields worth showing per agent. */
function summarise(entry: TraceEntry): [string, string][] {
  const p = entry.payload ?? {};
  const rows: [string, string][] = [];
  const push = (label: string, value: unknown) => {
    if (value === undefined || value === null) return;
    if (Array.isArray(value)) {
      if (!value.length) return;
      rows.push([label, value.map(String).join(", ")]);
    } else if (typeof value === "object") {
      rows.push([label, JSON.stringify(value)]);
    } else if (String(value).length) {
      rows.push([label, String(value)]);
    }
  };

  switch (entry.agent) {
    case "planner":
      push("rationale", p.rationale);
      push("search queries", p.search_queries);
      push("tool", p.candidate_tool);
      break;
    case "retrieval":
      push("queries", p.queries);
      push("documents", p.documents);
      push("top score", p.top_score);
      break;
    case "reasoning":
      push("citations used", p.used_citations);
      push("answer length", p.characters ? `${p.characters} chars` : undefined);
      push("insufficient evidence", p.insufficient_evidence ? "yes" : undefined);
      break;
    case "workflow":
      push("tool", p.tool);
      push("arguments", p.arguments);
      push("blockers", p.blockers);
      push("warnings", p.warnings);
      break;
    case "verification":
      push("decision", p.decision);
      push("citation coverage", fmt(p.citation_coverage));
      push("evidence support", fmt(p.lexical_support));
      push("question coverage", fmt(p.query_coverage));
      push("invalid citations", p.invalid_citations);
      push("ungrounded numbers", p.ungrounded_numbers);
      push("unsupported claims", p.unsupported_claims);
      break;
    case "approval_gate":
      push("tool", p.tool_name);
      push("risk", p.risk);
      push("blockers", p.blockers);
      push("warnings", p.warnings);
      break;
  }

  if (p.mode) push("mode", p.mode === "claude" ? "Claude" : "deterministic");
  return rows;
}

function fmt(value: unknown) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : undefined;
}
