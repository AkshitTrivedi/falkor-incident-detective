"""Checks that seed.py builds the graph the demo stories rely on.

Runs against local FalkorDB (see db.py) on a throwaway graph that is deleted
afterwards, so the real "incidents" graph is never touched.
"""
import time

import pytest

from db import get_graph
from seed import HOUR, MIN, ensure_fresh, seed
from tools import MAX_HOPS


@pytest.fixture(scope="module")
def g():
    """Seed a separate test graph once for this module, then delete it."""
    graph = get_graph("incidents_test")
    seed(graph)
    yield graph
    graph.delete()


def rows(g, q):
    """Run a read-only query and return its rows."""
    return g.ro_query(q, timeout=3000).result_set


def test_counts(g):
    """Node counts per label match the seed definition."""
    counts = dict(rows(g, "MATCH (n) RETURN labels(n)[0], count(n)"))
    assert counts == {"Service": 22, "Team": 5, "Person": 15, "Deploy": 41, "Alert": 14}


def test_only_story_a_deploy_in_last_hour(g):
    """Background deploys are older than 2h, so Story A's deploy stands alone."""
    now = rows(g, "MATCH (d:Deploy) RETURN max(d.ts)")[0][0]
    recent = rows(g, f"MATCH (d:Deploy)-[:TO]->(s:Service) WHERE d.ts >= {now - HOUR} "
                     "RETURN s.name, d.version")
    assert recent == [["fraud-check", "v3.1.0"]]


def test_each_team_has_one_on_call(g):
    """who_to_page always has exactly one person to return."""
    on_call = rows(g, "MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t:Team) "
                      "RETURN t.name, count(p)")
    assert len(on_call) == 5 and all(c == 1 for _, c in on_call)


def test_payments_db_blast_radius(g):
    """Everything upstream of payments-db, up to the edge services."""
    callers = rows(g, "MATCH (:Service {name: 'payments-db'})<-[:DEPENDS_ON*1..5]-(x:Service) "
                      "RETURN collect(DISTINCT x.name)")[0][0]
    assert set(callers) == {"paymentservice", "fraud-check", "checkoutservice",
                            "frontend", "api-gateway"}


def test_adservice_not_on_checkout_chain(g):
    """Story C: no path either way, so the adservice alert is a red herring."""
    paths = rows(g, "MATCH (a:Service {name: 'adservice'}), (c:Service {name: 'checkoutservice'}) "
                    "RETURN shortestPath((a)-[:DEPENDS_ON*]->(c)), "
                    "shortestPath((c)-[:DEPENDS_ON*]->(a))")
    assert paths == [[None, None]]


def test_betweenness_top_chokepoints(g):
    """Story B: the payment chain services are the top chokepoints."""
    top = rows(g, "CALL algo.betweenness({nodeLabels: ['Service'], "
                  "relationshipTypes: ['DEPENDS_ON'], samplingSeed: 42}) "
                  "YIELD node, score RETURN node.name ORDER BY score DESC LIMIT 3")
    assert [r[0] for r in top] == ["paymentservice", "checkoutservice", "fraud-check"]


def test_longest_chain_fits_hop_cap(g):
    """The tools cap walks at MAX_HOPS; the deepest chain must fit under it.

    Uses an unbounded pattern on purpose to measure the true depth. The seeded
    graph has no DEPENDS_ON cycles, so this is safe here.
    """
    longest = rows(g, "MATCH p = (:Service)-[:DEPENDS_ON*]->(:Service) "
                      "RETURN max(length(p))")[0][0]
    assert longest <= MAX_HOPS


def test_ensure_fresh_reseeds_only_when_stale():
    """A new or old graph is re-seeded; one seeded 10 minutes ago is left alone."""
    graph = get_graph("incidents_fresh_test")
    try:
        graph.delete()
    except Exception:
        pass  # did not exist
    try:
        assert ensure_fresh(graph)  # no graph yet -> seeded
        assert not ensure_fresh(graph)  # just seeded -> fresh
        seed(graph, now_ms=int(time.time() * 1000) - 10 * MIN)
        assert not ensure_fresh(graph)  # 10 min old, under the 30 min limit
        seed(graph, now_ms=int(time.time() * 1000) - 45 * MIN)
        assert ensure_fresh(graph)  # 45 min old -> re-seeded
    finally:
        graph.delete()
