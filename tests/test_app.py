"""Checks for app.py: the web routes and the summary the UI lights the graph from.

The agent is replaced by the rules diagnosis (diagnose.rule_answer), so these
tests need the local FalkorDB but no LLM key, and run in well under a second.
"""
import pytest
from fastapi.testclient import TestClient

import agent
import app
import diagnose
import verify
from db import get_graph
from seed import seed

QUESTION = "checkoutservice is throwing 5xx errors. What broke?"


@pytest.fixture(scope="module")
def g():
    """Seed a throwaway graph once for this module, then delete it."""
    graph = get_graph("incidents_app_test")
    seed(graph)
    yield graph
    graph.delete()


@pytest.fixture
def client(g, monkeypatch):
    """A test client whose routes use the throwaway graph and the rules instead of an LLM."""
    monkeypatch.setenv("AUTO_RESEED", "0")
    monkeypatch.setattr(app, "get_graph", lambda: g)

    def fake_investigate(question, g=None):
        """Same return shape as agent.investigate(), answered by the rules."""
        r = diagnose.rule_answer(g, question)
        return {"answer": r["answer"], "steps": r["steps"], "queries": [], "path": r["path"],
                "provider": "rules", "model": None, "llm_calls": 0,
                "tokens": {"prompt": 0, "completion": 0},
                "check": {"problems": [], "retried": False, "by_rules": True},
                "fallbacks": [], "seconds": 0.0}
    monkeypatch.setattr(agent, "investigate", fake_investigate)
    return TestClient(app.app)


def test_index_serves_the_ui(client):
    """"/" returns the one-page UI."""
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Incident Detective</title>" in r.text


def test_graph_has_story_nodes_and_consistent_edges(client):
    """The drawn graph has every service, the Story A deploy, health and owners, and no dangling edges."""
    data = client.get("/api/graph").json()
    by_id = {n["id"]: n for n in data["nodes"]}
    assert data["counts"]["services"] == 22
    assert by_id["deploy:fraud-check@v3.1.0"]["by"] == "Daniel Okafor"
    db = by_id["svc:payments-db"]
    assert db["health"] == "critical"
    assert db["alerts"][0]["message"].startswith("CPU 98%")
    assert by_id["svc:fraud-check"]["team"] == "Payments"
    assert by_id["svc:fraud-check"]["on_call"] == "Priya Nair"
    assert by_id["svc:emailservice"]["health"] is None
    assert all(e["from"] in by_id and e["to"] in by_id for e in data["edges"])


def test_ask_returns_summary_and_timeline(client):
    """Story A: the summary names the path, the blamed deploy, the unrelated service and the page."""
    out = client.post("/api/ask", json={"question": QUESTION}).json()
    s = out["summary"]
    assert s["path"] == ["checkoutservice", "paymentservice", "fraud-check"]
    assert s["root_cause"] == {"service": "fraud-check", "version": "v3.1.0", "by": "Daniel Okafor"}
    assert s["unrelated"] == ["adservice"]
    assert s["page"] == {"service": "fraud-check", "team": "Payments", "on_call": "Priya Nair"}
    assert "payments-db" in s["related"]

    t = out["timeline"]
    assert [e["minutes_ago"] for e in t] == sorted((e["minutes_ago"] for e in t), reverse=True)
    assert {e["service"] for e in t if not e["related"]} == {"adservice"}


def test_question_length_is_checked(client):
    """Too-short and too-long questions are rejected before the agent runs."""
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422


def test_summary_ignores_unrelated_claims_about_the_chain(g):
    """A service the answer wrongly calls unrelated is not drawn as unrelated if the tools put it on the chain."""
    r = diagnose.rule_answer(g, QUESTION)
    out = {"answer": "Root cause: fraud-check v3.1.0\nUnrelated: paymentservice, adservice",
           "steps": r["steps"], "path": r["path"]}
    s = app.summarize(out, verify.graph_vocabulary(g))
    assert s["unrelated"] == ["adservice"]
    assert s["root_cause"]["service"] == "fraud-check"


def test_client_error_is_logged(client, caplog):
    """Browser errors posted to /api/client-error show up in the server log."""
    with caplog.at_level("ERROR", logger="detective.app"):
        r = client.post("/api/client-error", json={"message": "boom", "where": "index.html:1:2"})
    assert r.status_code == 204
    assert "browser: boom at index.html:1:2" in caplog.text
