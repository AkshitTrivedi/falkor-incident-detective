"""Checks for actions.py and the /api/actions routes: opening an incident, rolling
back, and the agent remembering both through recent_changes().

Each test seeds its own throwaway graph, because the actions change it.
Needs the local FalkorDB; no LLM key.
"""
import pytest
from fastapi.testclient import TestClient

import actions
import app
import diagnose
import tools
from db import get_graph
from seed import seed

CAUSE = ("fraud-check", "v3.1.0")
CHAIN_ALERTS = ["checkoutservice", "payments-db", "paymentservice"]


@pytest.fixture
def g():
    """A freshly seeded graph per test, deleted afterwards."""
    graph = get_graph("incidents_actions_test")
    seed(graph)
    yield graph
    graph.delete()


def test_open_incident_pages_the_owner_and_links_the_alerting_chain(g):
    inc = actions.open_incident(g, *CAUSE)
    assert inc["id"] == "INC-1" and not inc["existing"]
    # Who to page comes from the graph: fraud-check is owned by Payments.
    assert (inc["team"], inc["paged"]) == ("Payments", "Priya Nair")
    # adservice alerts too, but has no path to fraud-check: not affected.
    assert inc["affects"] == CHAIN_ALERTS


def test_open_incident_twice_does_not_page_twice(g):
    first = actions.open_incident(g, *CAUSE)
    again = actions.open_incident(g, *CAUSE)
    assert again["existing"] and again["id"] == first["id"]
    assert tools._run(g, "MATCH (:Incident)-[r:PAGED]->() RETURN count(r)")[0][0] == 1


def test_unknown_deploy_is_refused_and_writes_nothing(g):
    assert "error" in actions.open_incident(g, "fraud-check", "v9.9.9")
    assert "error" in actions.rollback(g, "paymentservice", "v3.1.0")  # right version, wrong service
    assert tools._run(g, "MATCH (i:Incident) RETURN count(i)")[0][0] == 0


def test_rollback_clears_only_the_chain_alerts(g):
    res = actions.rollback(g, *CAUSE)
    assert res["rolled_back"]
    assert sorted(a["service"] for a in res["resolved"]) == CHAIN_ALERTS
    assert [a["service"] for a in res["still_open"]] == ["adservice"]
    assert res["incident"]["status"] == "mitigated"
    # A second rollback changes nothing.
    assert not actions.rollback(g, *CAUSE)["rolled_back"]


def test_recent_changes_remembers_what_was_done(g):
    assert tools.recent_changes(g)["incidents"] == []
    actions.rollback(g, *CAUSE)
    r = tools.recent_changes(g)
    (inc,) = r["incidents"]
    assert inc["cause"] == "fraud-check v3.1.0" and inc["paged"] == "Priya Nair"
    assert inc["status"] == "mitigated" and inc["mitigation"] == "rolled back fraud-check v3.1.0"
    assert "rolled_back_minutes_ago" in r["deploys"][0]
    assert [a["service"] for a in r["alerts"]] == ["adservice"]


def test_rules_answer_says_it_was_already_handled(g):
    q = "checkoutservice is throwing 5xx errors. What broke and who should I page?"
    assert "Already handled" not in diagnose.rule_answer(g, q)["answer"]
    actions.rollback(g, *CAUSE)
    answer = diagnose.rule_answer(g, q)["answer"]
    assert "Already handled: INC-1" in answer and "rolled back" in answer
    assert "already paged for INC-1" in answer


@pytest.fixture
def client(g, monkeypatch):
    """A test client whose routes use the throwaway graph."""
    monkeypatch.setattr(app, "get_graph", lambda: g)
    return TestClient(app.app)


def test_action_routes(client):
    r = client.post("/api/actions/incident", json={"service": "fraud-check", "version": "v3.1.0"})
    assert r.status_code == 200 and r.json()["paged"] == "Priya Nair"
    r = client.post("/api/actions/rollback", json={"service": "fraud-check", "version": "v3.1.0"})
    assert r.status_code == 200 and r.json()["still_open"][0]["service"] == "adservice"
    assert client.post("/api/actions/rollback",
                       json={"service": "fraud-check", "version": "v0"}).status_code == 404


@pytest.mark.parametrize("body", [
    {"service": "fraud-check'}) DETACH DELETE n //", "version": "v3.1.0"},
    {"service": "fraud-check", "version": ""},
    {"service": "fraud-check"},
])
def test_action_routes_reject_bad_input(client, body):
    assert client.post("/api/actions/incident", json=body).status_code == 422
