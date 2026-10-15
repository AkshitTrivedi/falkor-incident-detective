"""Checks each agent tool returns the answers the demo stories expect.

Runs against local FalkorDB on a throwaway graph that is deleted afterwards.
"""
import pytest

from db import get_graph
from seed import seed
import tools


@pytest.fixture(scope="module")
def ctx():
    """Seed a test graph once; yield it with the `now` the data is anchored to."""
    g = get_graph("incidents_tools_test")
    now = seed(g)
    yield g, now
    g.delete()


def names(items):
    """Service names from a list of tool result items."""
    return [i["service"] for i in items]


def test_story_a_recent_changes(ctx):
    """Story A: one fresh deploy, and the alert cascade newest first."""
    g, now = ctx
    r = tools.recent_changes(g, 60, now_ms=now)
    assert r["deploys"] == [{"service": "fraud-check", "version": "v3.1.0",
                             "by": "Daniel Okafor", "minutes_ago": 14}]
    assert names(r["alerts"]) == ["checkoutservice", "paymentservice", "adservice", "payments-db"]


def test_checkout_dependencies_include_root_cause(ctx):
    """The deployed service and the hot database are downstream of checkout."""
    g, _ = ctx
    deps = names(tools.dependencies(g, "checkoutservice")["depends_on"])
    assert {"paymentservice", "fraud-check", "payments-db"} <= set(deps)
    assert "adservice" not in deps


def test_deepest_dependency_is_found(ctx):
    """feature-store is 6 hops below api-gateway; a too-small MAX_HOPS would hide it."""
    g, _ = ctx
    deps = tools.dependencies(g, "api-gateway")["depends_on"]
    assert {"service": "feature-store", "hops": 6} in deps


def test_payments_db_blast_radius(ctx):
    """If payments-db fails, the whole payment chain up to the edge is affected."""
    g, _ = ctx
    r = tools.blast_radius(g, "payments-db")
    assert set(names(r["affected"])) == {"paymentservice", "fraud-check", "checkoutservice",
                                         "frontend", "api-gateway"}


def test_path_checkout_to_payments_db(ctx):
    """Lowest-latency path is found even when the arguments are in reverse order."""
    g, _ = ctx
    r = tools.dependency_path(g, "payments-db", "checkoutservice")  # reverse order on purpose
    assert r["path"] == ["checkoutservice", "paymentservice", "payments-db"]
    assert r["latency_ms"] == 36  # 30 ms + 6 ms edge weights


def test_adservice_unrelated(ctx):
    """Story C: no path, so the adservice alert is not part of the incident."""
    g, _ = ctx
    assert tools.dependency_path(g, "adservice", "checkoutservice")["path"] is None


def test_critical_services(ctx):
    """Story B: both rankings are stable and match the seeded topology."""
    g, _ = ctx
    assert names(tools.critical_services(g, "betweenness", 3)["ranking"]) == [
        "paymentservice", "checkoutservice", "fraud-check"]
    assert names(tools.critical_services(g, "pagerank", 1)["ranking"]) == ["inventory-db"]


def test_who_to_page(ctx):
    """Story A ends by paging the Payments on-call."""
    g, _ = ctx
    r = tools.who_to_page(g, "fraud-check")
    assert (r["team"], r["on_call"]) == ("Payments", "Priya Nair")


def test_unknown_service_lists_known(ctx):
    """A typo returns the valid names so the LLM can retry with the right one."""
    g, _ = ctx
    r = tools.blast_radius(g, "payment-db")
    assert "error" in r and "payments-db" in r["known_services"]
