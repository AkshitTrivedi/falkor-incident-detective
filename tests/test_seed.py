import pytest

from db import get_graph
from seed import HOUR, seed


@pytest.fixture(scope="module")
def g():
    graph = get_graph("incidents_test")
    seed(graph)
    yield graph
    graph.delete()


def rows(g, q):
    return g.ro_query(q, timeout=3000).result_set


def test_counts(g):
    counts = dict(rows(g, "MATCH (n) RETURN labels(n)[0], count(n)"))
    assert counts == {"Service": 22, "Team": 5, "Person": 15, "Deploy": 41, "Alert": 14}


def test_only_story_a_deploy_in_last_hour(g):
    now = rows(g, "MATCH (d:Deploy) RETURN max(d.ts)")[0][0]
    recent = rows(g, f"MATCH (d:Deploy)-[:TO]->(s:Service) WHERE d.ts >= {now - HOUR} "
                     "RETURN s.name, d.version")
    assert recent == [["fraud-check", "v3.1.0"]]


def test_each_team_has_one_on_call(g):
    on_call = rows(g, "MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t:Team) "
                      "RETURN t.name, count(p)")
    assert len(on_call) == 5 and all(c == 1 for _, c in on_call)


def test_payments_db_blast_radius(g):
    callers = rows(g, "MATCH (:Service {name: 'payments-db'})<-[:DEPENDS_ON*1..5]-(x:Service) "
                      "RETURN collect(DISTINCT x.name)")[0][0]
    assert set(callers) == {"paymentservice", "fraud-check", "checkoutservice",
                            "frontend", "api-gateway"}


def test_adservice_not_on_checkout_chain(g):
    paths = rows(g, "MATCH (a:Service {name: 'adservice'}), (c:Service {name: 'checkoutservice'}) "
                    "RETURN shortestPath((a)-[:DEPENDS_ON*]->(c)), "
                    "shortestPath((c)-[:DEPENDS_ON*]->(a))")
    assert paths == [[None, None]]


def test_betweenness_top_chokepoints(g):
    top = rows(g, "CALL algo.betweenness({nodeLabels: ['Service'], "
                  "relationshipTypes: ['DEPENDS_ON'], samplingSeed: 42}) "
                  "YIELD node, score RETURN node.name ORDER BY score DESC LIMIT 3")
    assert [r[0] for r in top] == ["paymentservice", "checkoutservice", "fraud-check"]
