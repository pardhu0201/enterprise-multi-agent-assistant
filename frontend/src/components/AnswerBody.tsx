import { Fragment, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Citation } from "../lib/api";

/**
 * Renders a Markdown answer and turns every `[n]` marker into a clickable pill
 * that jumps to the matching source. Citations are the product here, not
 * decoration - if a claim cannot be traced back to a document, the reader
 * should be able to see that immediately.
 */
export default function AnswerBody({
  text,
  citations,
  onCitationClick,
}: {
  text: string;
  citations: Citation[];
  onCitationClick?: (index: number) => void;
}) {
  const valid = new Set(citations.map((c) => c.index));

  return (
    <div className="answer">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{withCitations(children, valid, citations, onCitationClick)}</p>,
          li: ({ children }) => <li>{withCitations(children, valid, citations, onCitationClick)}</li>,
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}

const CITATION_RE = /\[(\d{1,2})\]/g;

function withCitations(
  children: ReactNode,
  valid: Set<number>,
  citations: Citation[],
  onClick?: (index: number) => void,
): ReactNode {
  const transform = (node: ReactNode, key: string | number): ReactNode => {
    if (typeof node === "string") return renderString(node, valid, citations, onClick, key);
    if (Array.isArray(node)) return node.map((child, i) => transform(child, `${key}-${i}`));
    return node;
  };
  return transform(children, "c");
}

function renderString(
  text: string,
  valid: Set<number>,
  citations: Citation[],
  onClick: ((index: number) => void) | undefined,
  key: string | number,
): ReactNode {
  if (!CITATION_RE.test(text)) return text;
  CITATION_RE.lastIndex = 0;

  const parts: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const index = Number(match[1]);
    const source = citations.find((c) => c.index === index);

    parts.push(
      valid.has(index) ? (
        <button
          key={`${key}-${match.index}`}
          type="button"
          className="citation-pill"
          title={source ? `${source.document_title} - ${source.heading || "general"}` : undefined}
          onClick={() => onClick?.(index)}
        >
          {index}
        </button>
      ) : (
        // A marker pointing at a passage that does not exist is a defect the
        // verification agent flags; surface it rather than hiding it.
        <span
          key={`${key}-${match.index}`}
          className="citation-pill"
          style={{
            color: "var(--color-danger)",
            borderColor: "rgba(251,113,133,0.45)",
            background: "rgba(251,113,133,0.14)",
          }}
          title="This citation does not match any retrieved passage"
        >
          {index}
        </span>
      ),
    );
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) parts.push(text.slice(cursor));
  return <Fragment key={key}>{parts}</Fragment>;
}
