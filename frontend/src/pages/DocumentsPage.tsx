import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, Loader2, Search, Trash2, Upload } from "lucide-react";

import { api, type Citation, type DocumentSummary } from "../lib/api";
import { EmptyState, PageHeader } from "../components/primitives";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Citation[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [tookMs, setTookMs] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setDocuments(await api.documents());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const upload = async (file: File) => {
    setUploading(true);
    setStatus("");
    try {
      const result = await api.uploadDocument(file, "Uploaded");
      setStatus(`${result.status}: "${result.title}" indexed into ${result.chunks} chunks.`);
      await load();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const search = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const response = await api.search(query, 6);
      setResults(response.results);
      setTookMs(response.took_ms);
    } finally {
      setSearching(false);
    }
  };

  const remove = async (id: string, title: string) => {
    if (!window.confirm(`Remove "${title}" from the knowledge base?`)) return;
    await api.deleteDocument(id);
    setResults(null);
    await load();
  };

  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Knowledge base"
        subtitle={`${documents.length} documents, ${totalChunks} indexed chunks. Upload a policy to see it become answerable immediately.`}
        actions={
          <>
            <input
              ref={fileInput}
              type="file"
              accept=".md,.markdown,.txt,.pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void upload(file);
              }}
            />
            <button
              className="btn-primary"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
            >
              {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
              Upload document
            </button>
          </>
        }
      />

      <div className="grid gap-6 px-5 py-6 sm:px-7 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* ---- documents ---- */}
        <section>
          <h2 className="label mb-3">Indexed documents</h2>
          {status && (
            <p className="mb-3 rounded-lg border border-[var(--color-line-strong)] bg-[var(--color-surface-2)] px-3.5 py-2.5 text-[13px] text-[var(--color-ink-2)]">
              {status}
            </p>
          )}

          {loading ? (
            <div className="flex justify-center py-12 text-[var(--color-ink-3)]">
              <Loader2 size={20} className="animate-spin" />
            </div>
          ) : documents.length === 0 ? (
            <EmptyState
              icon={<FileText size={18} />}
              title="No documents indexed"
              body="Upload a Markdown, text or PDF policy document to populate the knowledge base."
            />
          ) : (
            <ul className="space-y-2">
              {documents.map((doc) => (
                <li key={doc.id} className="panel-2 flex items-start gap-3 p-3.5">
                  <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-[var(--color-surface-3)] text-[var(--color-ink-3)]">
                    <FileText size={13} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13.5px] font-medium">{doc.title}</div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-[var(--color-ink-3)]">
                      <span>{doc.department}</span>
                      <span>·</span>
                      <span>v{doc.version}</span>
                      <span>·</span>
                      <span className="font-mono">{doc.chunk_count} chunks</span>
                      {doc.effective_date && (
                        <>
                          <span>·</span>
                          <span>effective {doc.effective_date}</span>
                        </>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => void remove(doc.id, doc.title)}
                    className="shrink-0 rounded-md p-1.5 text-[var(--color-ink-3)] transition-colors hover:bg-[rgba(251,113,133,0.12)] hover:text-[var(--color-danger)]"
                    aria-label={`Remove ${doc.title}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* ---- retrieval preview ---- */}
        <section>
          <h2 className="label mb-3">Retrieval preview</h2>
          <p className="mb-3 text-[13px] text-[var(--color-ink-3)]">
            Runs the same hybrid search the retrieval agent uses - BM25 and vector scores
            fused, then de-duplicated.
          </p>

          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search
                size={14}
                className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-[var(--color-ink-3)]"
              />
              <input
                className="field pl-9"
                placeholder="e.g. medical certificate for sick leave"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void search()}
              />
            </div>
            <button className="btn-ghost" onClick={() => void search()} disabled={searching}>
              {searching ? <Loader2 size={14} className="animate-spin" /> : "Search"}
            </button>
          </div>

          {results && (
            <>
              <p className="mt-3 font-mono text-[11px] text-[var(--color-ink-3)]">
                {results.length} passages in {tookMs}ms
              </p>
              <ol className="mt-2 space-y-2">
                {results.map((hit) => (
                  <li key={hit.chunk_id} className="panel-2 p-3.5">
                    <div className="flex items-center gap-2 text-[13px] font-medium">
                      <span className="grid h-5 w-5 place-items-center rounded bg-[rgba(124,140,255,0.18)] font-mono text-[11px] text-[var(--color-brand)]">
                        {hit.index}
                      </span>
                      <span className="truncate">{hit.document_title}</span>
                    </div>
                    <div className="mt-0.5 truncate text-[11px] text-[var(--color-ink-3)]">
                      {hit.heading}
                    </div>
                    <p className="mt-2 text-[12.5px] leading-relaxed text-[var(--color-ink-2)]">
                      {hit.snippet}
                    </p>
                    <div className="mt-2 flex gap-1.5 font-mono text-[10.5px] text-[var(--color-ink-3)]">
                      <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5">
                        fused {hit.score.toFixed(2)}
                      </span>
                      <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5">
                        bm25 {hit.lexical_score.toFixed(1)}
                      </span>
                      <span className="rounded bg-[var(--color-surface-3)] px-1.5 py-0.5">
                        vec {hit.dense_score.toFixed(2)}
                      </span>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
