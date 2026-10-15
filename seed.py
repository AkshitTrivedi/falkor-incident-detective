"""Build the Incident Detective graph from scratch.

Topology is based on Google's public Online Boutique demo plus synthetic services.
All people, deploys and alerts are synthetic. Times are relative to "now" so the
demo incident is always recent.

Usage:
  python seed.py                                    # local Docker
  FALKORDB_URL=$FALKORDB_CLOUD_URL python seed.py   # FalkorDB Cloud
"""
import random
import time

from redis.exceptions import ResponseError

from db import get_graph

MIN = 60_000
HOUR = 60 * MIN
DAY = 24 * HOUR

# name, tier, criticality (1 = highest)
SERVICES = [
    ("api-gateway", "edge", 1),
    ("frontend", "edge", 1),
    ("adservice", "backend", 3),
    ("recommendation", "backend", 3),
    ("checkoutservice", "backend", 1),
    ("cartservice", "backend", 2),
    ("productcatalog", "backend", 2),
    ("currencyservice", "backend", 2),
    ("shippingservice", "backend", 2),
    ("emailservice", "backend", 3),
    ("paymentservice", "backend", 1),
    ("fraud-check", "backend", 1),
    ("risk-model", "backend", 2),
    ("auth-service", "backend", 1),
    ("card-gateway", "external", 1),
    ("redis-cart", "data", 2),
    ("payments-db", "data", 1),
    ("users-db", "data", 1),
    ("orders-db", "data", 2),
    ("inventory-db", "data", 2),
    ("feature-store", "data", 2),
    ("session-cache", "data", 2),
]

# caller -> callee, weight = typical latency in ms (used by algo.SPpaths)
DEPENDS_ON = [
    # Online Boutique core (from kubernetes-manifests *_ADDR env vars)
    ("frontend", "adservice", 15),
    ("frontend", "cartservice", 10),
    ("frontend", "productcatalog", 8),
    ("frontend", "currencyservice", 3),
    ("frontend", "recommendation", 20),
    ("frontend", "shippingservice", 12),
    ("frontend", "checkoutservice", 40),
    ("checkoutservice", "cartservice", 10),
    ("checkoutservice", "productcatalog", 8),
    ("checkoutservice", "currencyservice", 3),
    ("checkoutservice", "shippingservice", 12),
    ("checkoutservice", "emailservice", 25),
    ("checkoutservice", "paymentservice", 30),
    ("recommendation", "productcatalog", 8),
    ("cartservice", "redis-cart", 2),
    # Synthetic extras for multi-hop depth
    ("api-gateway", "frontend", 5),
    ("api-gateway", "auth-service", 6),
    ("auth-service", "users-db", 4),
    ("auth-service", "session-cache", 1),
    ("paymentservice", "fraud-check", 18),
    ("paymentservice", "payments-db", 6),
    ("paymentservice", "card-gateway", 120),
    ("fraud-check", "risk-model", 22),
    ("fraud-check", "payments-db", 6),
    ("risk-model", "feature-store", 9),
    ("productcatalog", "inventory-db", 5),
    ("shippingservice", "orders-db", 5),
]

TEAMS = {
    "Storefront": ["api-gateway", "frontend", "adservice", "recommendation"],
    "Checkout": ["checkoutservice", "cartservice", "redis-cart", "emailservice"],
    "Payments": ["paymentservice", "fraud-check", "card-gateway", "payments-db"],
    "Catalog & Logistics": ["productcatalog", "currencyservice", "shippingservice",
                            "inventory-db", "orders-db"],
    "Platform/Data": ["auth-service", "users-db", "session-cache", "risk-model",
                      "feature-store"],
}

# Made-up names. First person in each team is on call.
PEOPLE = {
    "Storefront": ["Maya Chen", "Arjun Mehta", "Lena Fischer"],
    "Checkout": ["Omar Haddad", "Sofia Rossi", "Kenji Sato"],
    "Payments": ["Priya Nair", "Daniel Okafor", "Elena Petrova"],
    "Catalog & Logistics": ["Ravi Kumar", "Hannah Berg", "Lucas Silva"],
    "Platform/Data": ["Aisha Bello", "Tom Walsh", "Mei Lin"],
}


def build_rows(now_ms):
    rng = random.Random(42)  # fixed seed: same graph every run
    owner = {s: t for t, svcs in TEAMS.items() for s in svcs}

    deploys, alerts = [], []
    # Background: ~40 successful deploys between 7 days and 2 hours ago,
    # so the last hour only holds the Story A deploy.
    for i in range(40):
        svc = rng.choice(SERVICES)[0]
        deploys.append({
            "id": f"dep-{i:03d}", "service": svc,
            "version": f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}",
            "ts": now_ms - rng.randint(2 * HOUR, 7 * DAY), "status": "success",
            "by": rng.choice(PEOPLE[owner[svc]]),
        })
    # Background: ~10 low-severity alerts that already resolved.
    for i in range(10):
        svc = rng.choice(SERVICES)[0]
        alerts.append({
            "id": f"alt-{i:03d}", "service": svc, "severity": "low",
            "message": rng.choice(["p95 latency above baseline", "pod restarted",
                                   "error rate blip", "disk usage 70%"]),
            "ts": now_ms - rng.randint(2 * HOUR, 7 * DAY), "resolved": True,
        })

    # Story A: bad deploy cascades up the payment chain.
    deploys.append({"id": "dep-story-a", "service": "fraud-check", "version": "v3.1.0",
                    "ts": now_ms - 14 * MIN, "status": "success", "by": "Daniel Okafor"})
    alerts += [
        {"id": "alt-a1", "service": "payments-db", "severity": "critical",
         "message": "CPU 98%: slow query from fraud-check (no index)",
         "ts": now_ms - 10 * MIN, "resolved": False},
        {"id": "alt-a2", "service": "paymentservice", "severity": "high",
         "message": "p99 latency 4.2s", "ts": now_ms - 6 * MIN, "resolved": False},
        {"id": "alt-a3", "service": "checkoutservice", "severity": "critical",
         "message": "HTTP 5xx rate 23%", "ts": now_ms - 2 * MIN, "resolved": False},
        # Story C: red herring, not on the checkout failure chain.
        {"id": "alt-c1", "service": "adservice", "severity": "medium",
         "message": "ad inventory cache miss rate high",
         "ts": now_ms - 8 * MIN, "resolved": False},
    ]
    return deploys, alerts


def seed(g, now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    deploys, alerts = build_rows(now_ms)
    try:
        g.delete()
    except ResponseError:
        pass  # graph did not exist yet

    for label, prop in [("Service", "name"), ("Team", "name"), ("Person", "name"),
                        ("Deploy", "ts"), ("Alert", "ts")]:
        g.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")

    g.query("UNWIND $rows AS r CREATE (:Service {name: r[0], tier: r[1], criticality: r[2]})",
            {"rows": [list(s) for s in SERVICES]})
    g.query("""UNWIND $rows AS r
               MATCH (a:Service {name: r[0]}), (b:Service {name: r[1]})
               CREATE (a)-[:DEPENDS_ON {weight: r[2]}]->(b)""",
            {"rows": [list(e) for e in DEPENDS_ON]})
    g.query("""UNWIND $rows AS r
               CREATE (t:Team {name: r.team})
               WITH t, r UNWIND r.services AS s
               MATCH (svc:Service {name: s}) CREATE (t)-[:OWNS]->(svc)""",
            {"rows": [{"team": t, "services": s} for t, s in TEAMS.items()]})
    g.query("""UNWIND $rows AS r
               MATCH (t:Team {name: r.team})
               CREATE (:Person {name: r.name, on_call: r.on_call})-[:MEMBER_OF]->(t)""",
            {"rows": [{"team": t, "name": p, "on_call": i == 0}
                      for t, ps in PEOPLE.items() for i, p in enumerate(ps)]})
    g.query("""UNWIND $rows AS r
               MATCH (s:Service {name: r.service}), (p:Person {name: r.by})
               CREATE (p)-[:DEPLOYED]->(:Deploy {id: r.id, version: r.version, ts: r.ts,
                                                 status: r.status})-[:TO]->(s)""",
            {"rows": deploys})
    g.query("""UNWIND $rows AS r
               MATCH (s:Service {name: r.service})
               CREATE (:Alert {id: r.id, severity: r.severity, message: r.message,
                               ts: r.ts, resolved: r.resolved})-[:FIRED_ON]->(s)""",
            {"rows": alerts})

    # Sanity check: every node must be connected to the rest of the graph.
    components = g.ro_query("CALL algo.WCC(null) YIELD componentId "
                            "RETURN count(DISTINCT componentId)").result_set[0][0]
    if components != 1:
        raise RuntimeError(f"seeded graph has {components} components, expected 1")
    return now_ms


if __name__ == "__main__":
    g = get_graph()
    seed(g)
    counts = g.ro_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) ORDER BY label").result_set
    print(f"Seeded graph '{g.name}':", ", ".join(f"{c} {l}" for l, c in counts))
