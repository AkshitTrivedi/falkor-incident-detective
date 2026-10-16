"""Checks for diagnose.py: the rules-based answer, run on a freshly seeded graph.

No LLM is involved, so these are fast and free. They need the local FalkorDB.
"""
import pytest

import diagnose
from db import get_graph
from seed import seed


@pytest.fixture(scope="module")
def g():
    """Seed a throwaway graph once for this module, then delete it."""
    graph = get_graph("incidents_diagnose_test")
    seed(graph)
    yield graph
    graph.delete()


def test_story_a_is_diagnosed(g):
    """The rules reach the same answer as a good model on the demo question."""
    out = diagnose.rule_answer(g, "checkoutservice is throwing 5xx errors. What broke?")
    a = out["answer"]
    assert out["decided"]
    assert "Root cause: fraud-check v3.1.0, deployed by Daniel Okafor 14 min ago" in a
    assert "checkoutservice → paymentservice → fraud-check" in a
    evidence = a.split("Evidence:")[1].split("Unrelated:")[0]
    assert all(s in evidence for s in ("payments-db", "paymentservice", "checkoutservice"))
    assert "adservice" not in evidence
    assert "Unrelated: adservice" in a
    assert a.endswith("Page: Payments team, on call Priya Nair.")
    assert out["path"] == ["checkoutservice", "paymentservice", "fraud-check"]
    assert [s["tool"] for s in out["steps"]] == [
        "recent_changes", "dependencies", "dependency_path", "who_to_page"]


def test_deploy_to_the_named_service_itself(g):
    """When the named service is the one deployed, there is no path to show."""
    out = diagnose.rule_answer(g, "fraud-check is failing, what changed?")
    assert out["decided"] and out["path"] is None
    assert "the deploy is to fraud-check itself" in out["answer"]


@pytest.mark.parametrize("question, reason", [
    ("Customers can't complete orders. Who do I wake up?", "no service is named"),
    ("Is the adservice alert related to checkoutservice?", "names several services"),
    ("cartservice is slow, what broke?", "no recent deploy to cartservice"),
])
def test_rules_do_not_guess(g, question, reason):
    """Without exactly one named service and one suspect deploy, the rules say why."""
    out = diagnose.rule_answer(g, question)
    assert not out["decided"] and reason in out["answer"]
    assert "Root cause" not in out["answer"]
    assert "- Deploy: fraud-check v3.1.0" in out["answer"]  # recent rows still shown


def test_undecided_shows_the_models_rows(g):
    """If the model already ran tools, their rows are shown, not the rules' own."""
    model_steps = [{"tool": "critical_services", "args": {}, "ms": 1, "result": {
        "metric": "betweenness", "ranking": [{"service": "paymentservice", "score": 65.0}]}}]
    out = diagnose.rule_answer(g, "Which services are chokepoints?", model_steps)
    assert "Top by betweenness: paymentservice (65)" in out["answer"]


def test_unknown_service_is_not_matched(g):
    """billing-service is not in the graph, so it is not taken as the failing service."""
    assert diagnose.failing_services(g, "What depends on billing-service?") == []
