"""HTTP surface, including the full human-in-the-loop approval round trip."""

from __future__ import annotations

import json
from datetime import date, timedelta


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["documents"] >= 8
    assert body["chunks"] > 40
    assert body["llm_mode"] == "demo"


def test_graph_and_tools_are_exposed(client):
    topology = client.get("/api/graph").json()
    assert len(topology["nodes"]) == 6

    tools = client.get("/api/tools").json()
    names = {t["name"] for t in tools}
    assert "submit_leave_request" in names
    for tool in tools:
        assert "parameters" in tool and tool["parameters"]["type"] == "object"


def test_documents_and_search(client):
    documents = client.get("/api/documents").json()
    assert len(documents) >= 8
    assert all(d["chunk_count"] > 0 for d in documents)

    response = client.post("/api/search", json={"query": "medical certificate", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 3
    assert body["results"][0]["index"] == 1
    assert "Leave" in body["results"][0]["document_title"]


def test_upload_rejects_unsupported_types(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("policy.exe", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_and_delete_document(client):
    content = (
        "---\ntitle: Pet Policy\ndepartment: Workplace\n---\n\n"
        "# Pet Policy\n\n## Office pets\n\n"
        "Employees may bring a dog to the Hyderabad office on the first Friday of "
        "each month, provided the dog is registered with reception in advance and "
        "remains on a lead in shared areas at all times.\n"
    )
    response = client.post(
        "/api/documents/upload",
        files={"file": ("pet-policy.md", content.encode(), "text/markdown")},
        data={"department": "Workplace"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "created"
    assert body["chunks"] >= 1

    found = client.post("/api/search", json={"query": "can I bring my dog to the office"}).json()
    assert any(r["document_title"] == "Pet Policy" for r in found["results"])

    assert client.delete(f"/api/documents/{body['document_id']}").status_code == 200
    after = client.post("/api/search", json={"query": "bring my dog to the office"}).json()
    assert all(r["document_title"] != "Pet Policy" for r in after["results"])


def test_chat_question_round_trip(client):
    response = client.post("/api/chat", json={"message": "What is the entitlement for sick leave?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert body["requires_approval"] is False
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["verification"]["decision"] in {"answer", "clarify", "escalate", "retry"}

    run = client.get(f"/api/runs/{body['run_id']}").json()
    assert run["query"] == "What is the entitlement for sick leave?"
    assert len(run["trace"]) >= 4


def test_chat_stream_emits_sse_events(client):
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "How do I report a phishing email?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())

    assert "event: run_started" in payload
    assert "event: agent_step" in payload
    assert "event: final" in payload

    final_line = [
        line for line in payload.splitlines() if line.startswith("data: ") and '"trace"' in line
    ][-1]
    final = json.loads(final_line[len("data: ") :])
    assert final["answer"]


def test_approval_round_trip_executes_the_action(client):
    start = date.today() + timedelta(days=25)
    chat = client.post(
        "/api/chat",
        json={
            "message": (
                f"Please book 2 days of annual leave from {start.isoformat()} "
                f"to {(start + timedelta(days=1)).isoformat()}"
            )
        },
    ).json()
    assert chat["status"] == "awaiting_approval"
    approval_id = chat["approval_id"]

    pending = client.get("/api/approvals?status=pending").json()
    assert any(a["id"] == approval_id for a in pending)

    decision = client.post(
        f"/api/approvals/{approval_id}/decision",
        json={"decision": "approve", "decided_by": "priya.raman@northwind.example"},
    )
    assert decision.status_code == 200
    body = decision.json()
    assert body["status"] == "approved"
    assert body["execution_result"]["reference"].startswith("LR-")
    assert body["execution_result"]["working_days"] == 2

    # A second decision on the same request must be refused.
    again = client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approve"})
    assert again.status_code == 409

    audit = client.get("/api/audit").json()
    assert any(entry["action"] == "approval.approved" for entry in audit)

    conversation = client.get(f"/api/conversations/{chat['conversation_id']}").json()
    assert any("LR-" in m["content"] for m in conversation["messages"])


def test_rejection_does_not_execute(client):
    start = date.today() + timedelta(days=40)
    chat = client.post(
        "/api/chat",
        json={"message": f"Book annual leave for 1 day on {start.isoformat()}"},
    ).json()
    approval_id = chat["approval_id"]
    assert approval_id

    body = client.post(
        f"/api/approvals/{approval_id}/decision",
        json={"decision": "reject", "note": "Sprint review that week"},
    ).json()
    assert body["status"] == "rejected"
    assert body["execution_result"] is None
    assert "rejected" in body["message"].lower()


def test_metrics_summarise_runs(client):
    body = client.get("/api/metrics").json()
    assert body["runs_total"] > 0
    assert body["approvals_by_status"]
    assert 0.0 <= body["average_confidence"] <= 1.0
    assert isinstance(body["top_documents"], list)


def test_employees_expose_leave_balances(client):
    employees = client.get("/api/employees").json()
    assert {e["id"] for e in employees} >= {"E-1001", "E-1002", "E-1003"}
    assert all(e["annual_remaining"] >= 0 for e in employees)
