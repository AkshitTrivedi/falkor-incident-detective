"""Checks for app.py: the web routes and the summary the UI lights the graph from.

The agent is replaced by the rules diagnosis (diagnose.rule_answer), so these
tests need the local FalkorDB but no LLM key, and run in well under a second.
"""
import pytest
from fastapi.testclient import TestClient

import agent
import app
import diagnose
import tools
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


def test_summary_lists_names_missing_from_the_graph(g):
    """billing-service and v2.0.0 are not in the graph; the card must say so."""
    r = diagnose.rule_answer(g, QUESTION)
    out = {"answer": "Root cause: fraud-check v3.1.0", "steps": r["steps"], "path": r["path"]}
    vocab = verify.graph_vocabulary(g)
    s = app.summarize(out, vocab, "billing-service is down, did the paymentservice v2.0.0 deploy break it?")
    assert s["missing"] == ["billing-service", "v2.0.0"]
    assert app.summarize(out, vocab, QUESTION)["missing"] == []


def test_summary_counts_callers_as_related(g):
    """Rollback question: blast_radius callers of fraud-check are not unrelated."""
    r = diagnose.rule_answer(g, "If we roll back fraud-check, which alerts should clear?")
    s = app.summarize(r, verify.graph_vocabulary(g))
    assert s["unrelated"] == ["adservice"]
    assert {"paymentservice", "checkoutservice"} <= set(s["related"])


def test_client_error_is_logged(client, caplog):
    """Browser errors posted to /api/client-error show up in the server log."""
    with caplog.at_level("ERROR", logger="detective.app"):
        r = client.post("/api/client-error", json={"message": "boom", "where": "index.html:1:2"})
    assert r.status_code == 204
    assert "browser: boom at index.html:1:2" in caplog.text


def test_summary_marks_rollback_questions(g):
    """"If we roll back fraud-check..." sets rollback, so the card says
    "Should clear after rollback" over the alerts on the chain."""
    vocab = verify.graph_vocabulary(g)
    out = {"steps": [], "answer": "", "path": None}
    assert app.summarize(out, vocab, "If we roll back fraud-check, which alerts clear?")["rollback"]
    assert app.summarize(out, vocab, "Should we revert fraud-check?")["rollback"]
    assert not app.summarize(out, vocab, "checkoutservice is throwing 5xx")["rollback"]


def test_summary_lists_people_and_teams_missing_from_the_graph(g):
    """The card's notice also names made-up people and teams, as written."""
    vocab = verify.graph_vocabulary(g)
    out = {"steps": [], "answer": "", "path": None}
    assert app.summarize(out, vocab, "Is Sarah Jenkins on call?")["missing"] == ["Sarah Jenkins"]
    assert app.summarize(out, vocab, "Should I page the SRE team?")["missing"] == ["SRE"]
    assert app.summarize(out, vocab, "Is Priya Nair on call for Payments?")["missing"] == []


@pytest.mark.parametrize("question", ["adservice cache miss rate is high. What broke?",
                                      "Roll back the cartservice deploy from 10 minutes ago"])
def test_summary_blames_nothing_when_no_deploy_explains_it(g, question):
    """The card shows no root cause when the answer says none was found, even
    though recent_changes returned the fraud-check deploy (edge case C3)."""
    r = diagnose.rule_answer(g, question)
    assert "Root cause: not found" in r["answer"]
    s = app.summarize(r, verify.graph_vocabulary(g), question)
    assert s["root_cause"] is None and s["no_cause"]


def test_summary_pages_every_failing_service_when_no_cause(g):
    """Two failing services, no deploy to blame: the card gets both owners, not
    a blast-radius verdict for the last one checked (UI check, Oct 16)."""
    q = "cartservice and adservice are both failing, what broke?"
    s = app.summarize(diagnose.rule_answer(g, q), verify.graph_vocabulary(g), q)
    assert s["no_cause"] and s["root_cause"] is None
    assert [(p["service"], p["on_call"]) for p in s["pages"]] == [
        ("cartservice", "Omar Haddad"), ("adservice", "Maya Chen")]


def test_summary_has_no_no_cause_flag_when_a_deploy_is_blamed(g):
    """The fraud-check story keeps its root-cause card."""
    q = "checkoutservice is throwing 5xx errors. What broke?"
    s = app.summarize(diagnose.rule_answer(g, q), verify.graph_vocabulary(g), q)
    assert s["root_cause"]["service"] == "fraud-check" and not s["no_cause"]


def test_off_topic_questions_skip_the_models(g, monkeypatch):
    """"hello there" and write requests never reach a model (no client is built)."""
    monkeypatch.setattr(agent, "make_client", lambda *a, **k: pytest.fail("model was called"))
    for q in ("hello there", "Delete all alerts", "who is akshit?",
              "billing-service is down, who do I page?", "Who is on call for the SRE team?"):
        out = agent.investigate(q, g=g)
        assert out["provider"] == "rules" and out["steps"] == [] and out["llm_calls"] == 0


def test_summary_reads_unrelated_from_prose(g):
    """C1, live Oct 16 (StepFun): "the adservice alert is unrelated" with no
    Unrelated line, after dependencies(adservice) ran. The card must list
    adservice as unrelated, not as related."""
    vocab = verify.graph_vocabulary(g)
    steps = [{"tool": "dependencies", "args": {"service": s}, "result": tools.dependencies(g, s)}
             for s in ("checkoutservice", "adservice")]
    out = {"answer": "**No — the adservice alert is unrelated to the checkout outage.**\n"
                     "The actual chain is fraud-check v3.1.0 → payments-db → paymentservice.",
           "steps": steps, "path": None}
    s = app.summarize(out, vocab, "There is also an alert on adservice. Is it related to the checkout outage?")
    assert s["unrelated"] == ["adservice"]
    assert "adservice" not in s["related"]
    assert "paymentservice" in s["related"]
