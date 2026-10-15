import pytest

from db import get_graph
from seed import seed
import tools


@pytest.fixture(scope="module")
def ctx():
    g = get_graph("incidents_tools_test")
    now = seed(g)
    yield g, now
    g.delete()


def names(items):
    return [i["service"] for i in items]


def test_story_a_recent_changes(ctx):
    g, now = ctx
    r = tools.recent_changes(g, 60, now_ms=now)
    assert r["deploys"] == [{"service": "fraud-check", "version": "v3.1.0",
                             "by": "Daniel Okafor", "minutes_ago": 14}]
    assert names(r["alerts"]) == ["checkoutservice", "paymentservice", "adservice", "payments-db"]


def test_checkout_dependencies_include_root_cause(ctx):
    g, _ = ctx
    deps = names(tools.dependencies(g, "checkoutservice")["depends_on"])
    assert {"paymentservice", "fraud-check", "payments-db"} <= set(deps)
    assert "adservice" not in deps


def test_payments_db_blast_radius(ctx):
    g, _ = ctx
    r = tools.blast_radius(g, "payments-db")
    assert set(names(r["affected"])) == {"paymentservice", "fraud-check", "checkoutservice",
                                         "frontend", "api-gateway"}


def test_path_checkout_to_payments_db(ctx):
    g, _ = ctx
    r = tools.dependency_path(g, "payments-db", "checkoutservice")  # reverse order on purpose
    assert r["path"] == ["checkoutservice", "paymentservice", "payments-db"]
    assert r["latency_ms"] == 36


def test_adservice_unrelated(ctx):
    g, _ = ctx
    assert tools.dependency_path(g, "adservice", "checkoutservice")["path"] is None


def test_critical_services(ctx):
    g, _ = ctx
    assert names(tools.critical_services(g, "betweenness", 3)["ranking"]) == [
        "paymentservice", "checkoutservice", "fraud-check"]
    assert names(tools.critical_services(g, "pagerank", 1)["ranking"]) == ["inventory-db"]


def test_who_to_page(ctx):
    g, _ = ctx
    assert tools.who_to_page(g, "fraud-check") == {
        "service": "fraud-check", "team": "Payments", "on_call": "Priya Nair",
        "cypher": tools.who_to_page(g, "fraud-check")["cypher"]}


def test_unknown_service_lists_known(ctx):
    g, _ = ctx
    r = tools.blast_radius(g, "payment-db")
    assert "error" in r and "payments-db" in r["known_services"]
