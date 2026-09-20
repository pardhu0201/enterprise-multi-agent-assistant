/**
 * Typed client for the assistant API.
 *
 * `streamChat` parses the server-sent-event stream by hand rather than using
 * EventSource, because the chat endpoint is a POST (EventSource is GET-only)
 * and we want the agent timeline to fill in live as each node finishes.
 */

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export const apiUrl = (path: string) => `${BASE}/api${path}`;

// --- domain types ----------------------------------------------------------
export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  department: string;
  source: string;
  heading: string;
  snippet: string;
  score: number;
  dense_score: number;
  lexical_score: number;
}

export interface TraceEntry {
  seq: number;
  agent: string;
  label: string;
  status: "ok" | "warning" | "error";
  summary: string;
  payload: Record<string, unknown>;
  duration_ms: number;
}

export interface ProposedAction {
  tool_name: string;
  description?: string;
  risk: string;
  requires_approval: boolean;
  arguments: Record<string, unknown>;
  valid: boolean;
  blockers: string[];
  warnings: string[];
  preview: Record<string, unknown>;
  missing_fields: string[];
  confirmation_needed: boolean;
  rationale: string;
  extraction_mode?: string;
}

export interface Verification {
  citation_coverage?: number;
  lexical_support?: number;
  query_coverage?: number;
  retrieval_strength?: number;
  invalid_citations?: number[];
  ungrounded_numbers?: string[];
  unsupported_claims?: string[];
  citation_issues?: string[];
  action_risk_notes?: string[];
  confidence?: number;
  llm_confidence?: number;
  answer_confidence?: number;
  action_confidence?: number | null;
  recommendation?: string;
  decision?: string;
  threshold?: number;
  mode?: string;
  [key: string]: unknown;
}

export interface ChatResponse {
  run_id: string;
  conversation_id: string;
  answer: string;
  status: string;
  intent: string;
  confidence: number;
  flags: string[];
  citations: Citation[];
  used_citations: number[];
  proposed_action: ProposedAction | null;
  approval_id: string | null;
  requires_approval: boolean;
  verification: Verification;
  follow_up_question: string;
  trace: TraceEntry[];
  llm_mode: string;
  latency_ms: number;
  token_usage: Record<string, unknown> | null;
}

export interface Approval {
  id: string;
  run_id: string;
  conversation_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  risk: string;
  rationale: string;
  flags: string[];
  confidence: number;
  status: string;
  requested_by: string;
  decided_by: string | null;
  decision_note: string;
  execution_result: Record<string, unknown> | null;
  created_at: string;
  decided_at: string | null;
}

export interface DocumentSummary {
  id: string;
  title: string;
  source: string;
  doc_type: string;
  department: string;
  version: string;
  effective_date: string | null;
  chunk_count: number;
  created_at: string;
}

export interface Health {
  status: string;
  version: string;
  llm_mode: string;
  llm_model: string;
  database: string;
  vector_backend: string;
  embedding_provider: string;
  documents: number;
  chunks: number;
}

export interface Metrics {
  runs_total: number;
  runs_by_status: Record<string, number>;
  approvals_by_status: Record<string, number>;
  average_confidence: number;
  average_latency_ms: number;
  low_confidence_rate: number;
  actions_executed: number;
  top_documents: { title: string; citations: number }[];
  recent_runs: {
    id: string;
    query: string;
    intent: string;
    status: string;
    confidence: number;
    latency_ms: number;
    llm_mode: string;
    created_at: string;
  }[];
}

export interface GraphTopology {
  nodes: { id: string; label: string; role: string }[];
  edges: { source: string; target: string; condition?: string }[];
}

export interface ToolSummary {
  name: string;
  description: string;
  risk: string;
  requires_approval: boolean;
  parameters: Record<string, unknown>;
}

export interface Employee {
  id: string;
  name: string;
  department: string;
  manager: string;
  location: string;
  annual_remaining: number;
  sick_remaining: number;
}

// --- transport -------------------------------------------------------------
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  graph: () => request<GraphTopology>("/graph"),
  tools: () => request<ToolSummary[]>("/tools"),
  employees: () => request<Employee[]>("/employees"),
  metrics: () => request<Metrics>("/metrics"),

  documents: () => request<DocumentSummary[]>("/documents"),
  deleteDocument: (id: string) => request<{ deleted: string }>(`/documents/${id}`, { method: "DELETE" }),
  reindex: () => request<unknown[]>("/documents/reindex", { method: "POST" }),
  uploadDocument: (file: File, department: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("department", department);
    return request<{ title: string; chunks: number; status: string }>("/documents/upload", {
      method: "POST",
      body: form,
    });
  },
  search: (query: string, topK = 6) =>
    request<{ query: string; results: Citation[]; took_ms: number }>("/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    }),

  approvals: (status = "all") => request<Approval[]>(`/approvals?status=${status}`),
  decide: (id: string, decision: "approve" | "reject", note = "") =>
    request<{ status: string; message: string; execution_result: Record<string, unknown> | null }>(
      `/approvals/${id}/decision`,
      { method: "POST", body: JSON.stringify({ decision, note }) },
    ),
  audit: () => request<Record<string, unknown>[]>("/audit"),
};

// --- streaming -------------------------------------------------------------
export interface StreamHandlers {
  onStart?: (data: { run_id: string; conversation_id: string; llm_mode: string }) => void;
  onStep?: (step: TraceEntry & { node: string }) => void;
  onFinal?: (data: ChatResponse) => void;
  onError?: (message: string) => void;
}

export async function streamChat(
  message: string,
  conversationId: string | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl("/chat/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId }),
    signal,
  });

  if (!response.ok || !response.body) {
    handlers.onError?.(`Request failed (${response.status})`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (raw: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    let payload: unknown;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }
    if (event === "run_started") handlers.onStart?.(payload as never);
    else if (event === "agent_step") handlers.onStep?.(payload as never);
    else if (event === "final") handlers.onFinal?.(payload as ChatResponse);
    else if (event === "error") handlers.onError?.((payload as { message: string }).message);
  };

  // SSE frames are separated by a blank line; a chunk boundary can land
  // anywhere, so hold the tail back until the next read completes it.
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) if (frame.trim()) dispatch(frame);
  }
  if (buffer.trim()) dispatch(buffer);
}
