import { FileText } from "lucide-react";

import type { Citation } from "../lib/api";

/**
 * The evidence the answer was built from, in the same numbering the answer
 * cites. Each card shows both retrieval scores so the hybrid search is
 * inspectable rather than a black box.
 */
export default function SourcePanel({
  citations,
  used,
  highlighted,
}: {
  citations: Citation[];
  used: number[];
  highlighted?: number | null;
}) {
  if (!citations.length) return null;
  const usedSet = new Set(used);

  return (
    <div className="space-y-2">
      {citations.map((citation) => {
        const isUsed = usedSet.has(citation.index);
        const isActive = highlighted === citation.index;
        return (
          <article
            key={citation.chunk_id}
            id={`source-${citation.index}`}
            className={[
              "rounded-lg border p-3 transition-colors",
              isActive
                ? "border-[var(--color-brand)] bg-[rgba(124,140,255,0.08)]"
                : "border-[var(--color-line)] bg-[var(--color-surface-2)]",
              isUsed ? "" : "opacity-60",
            ].join(" ")}
          >
            <div className="flex items-start gap-2.5">
              <span
                className={[
                  "mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded font-mono text-[11px] font-semibold",
                  isUsed
                    ? "bg-[rgba(124,140,255,0.2)] text-[var(--color-brand)]"
                    : "bg-[var(--color-surface-3)] text-[var(--color-ink-3)]",
                ].join(" ")}
              >
                {citation.index}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 text-[13px] font-medium">
                  <FileText size={12} className="shrink-0 text-[var(--color-ink-3)]" />
                  <span className="truncate">{citation.document_title}</span>
                </div>
                <div className="truncate text-[11px] text-[var(--color-ink-3)]">
                  {citation.heading || citation.department}
                </div>
              </div>
            </div>

            <p className="mt-2 text-[12.5px] leading-relaxed text-[var(--color-ink-2)]">
              {citation.snippet}
            </p>

            <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[10.5px] text-[var(--color-ink-3)]">
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                fused {citation.score.toFixed(2)}
              </span>
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                bm25 {citation.lexical_score.toFixed(1)}
              </span>
              <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5 font-mono">
                vec {citation.dense_score.toFixed(2)}
              </span>
              {!isUsed && <span className="italic">retrieved, not cited</span>}
            </div>
          </article>
        );
      })}
    </div>
  );
}
