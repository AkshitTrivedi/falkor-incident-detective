"""Build the Incident Detective graph from scratch.

Topology is based on Google's public Online Boutique demo plus synthetic services.
All people, deploys and alerts are synthetic. Times are relative to "now" so the
demo incident is always recent.

Graph model:
  (:Service {name, tier, criticality})
  (:Service)-[:DEPENDS_ON {weight}]->(:Service)   caller -> callee, weight = latency ms
  (:Team {name})-[:OWNS]->(:Service)
  (:Person {name, on_call})-[:MEMBER_OF]->(:Team)
  (:Person)-[:DEPLOYED]->(:Deploy {id, version, ts, status})-[:TO]->(:Service)
  (:Alert {id, severity, message, ts, resolved})-[:FIRED_ON]->(:Service)
All `ts` values are integer epoch milliseconds.

Demo stories built into the data:
  A  fraud-check v3.1.0 deployed 14 min ago -> payments-db CPU -> paymentservice
     latency -> checkoutservice 5xx. Root cause: the deploy. Page Payments on-call.
  B  "Which services are chokepoints?" answered by betweenness / PageRank.
  C  adservice alert at the same time is a red herring: no path to checkout.

Running this script deletes and rebuilds the graph.

Usage:
  python seed.py                                    # local Docker
  python seed.py --scale 300                        # 300-service graph, "incidents_scale"
  FALKORDB_URL=$FALKORDB_CLOUD_URL python seed.py   # FalkorDB Cloud
"""
import logging
import random
import time

from redis.exceptions import ResponseError

import verify
from db import get_graph

# Time units in milliseconds, matching the epoch-ms timestamps in the graph.
MIN = 60_000
HOUR = 60 * MIN
DAY = 24 * HOUR

# How long before seeding the Story A deploy happened. It is always the newest
# deploy in the graph (background deploys are 2h+ old), so the newest deploy's
# timestamp tells ensure_fresh() when the graph was seeded.
STORY_DEPLOY_AGO = 14 * MIN

# The tools look back 60 minutes by default, so after about 46 minutes the
# Story A deploy drops out of recent_changes(). Re-seed well before that.
MAX_STORY_AGE_MIN = 30

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

# caller -> callee, weight = typical latency in ms (used by algo.SPpaths).
# The longest chain is 6 hops; tools.MAX_HOPS must stay at or above it.
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

# Team -> services it owns. Every service has exactly one owner.
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


# --- Scale mode: a larger synthetic platform around the same demo services ---
#
# `python seed.py --scale 300` adds synthetic services until the graph has 300,
# with teams, people and 90 days of deploys and resolved alerts. The demo
# graph above is kept exactly: synthetic services only ever CALL demo services,
# never the other way round, so every demo service depends on exactly what it
# did before and Stories A-C give the same answers. What changes is what a
# bigger graph changes: more callers in each blast radius, a centrality
# ranking over 300 services, and much more history to search.
#
# Rules the generator keeps, each one a question in evals/edge_cases.py:
#   - nothing calls api-gateway (it stays the entry point);
#   - layers only call downwards, so the graph has no cycles, and the longest
#     chain stays within tools.MAX_HOPS (8);
#   - no name the demo questions use as a trap: billing-service, ledger, SRE,
#     a second Priya, Omar or Daniel, version v2.0.0 or a second v3.1.0;
#   - every background deploy and alert is older than 2 hours and every alert
#     is resolved, so "the last hour" still holds only the story.

SCALE_DOMAINS = [
    "search", "loyalty", "pricing", "tax", "reviews", "wishlist", "subscription", "refunds",
    "promotions", "analytics", "audit", "geo", "media", "notifications", "returns", "warehouse",
    "fulfilment", "identity", "consent", "experiments", "seller", "marketplace", "giftcards",
    "support", "chat", "recsys", "ranking", "fx", "invoices", "partners", "affiliates",
    "content", "localization", "compliance", "risk-scoring",
]
# suffix -> (tier, level). Level 0 is storage; a service only calls lower levels.
SCALE_ROLES = [("-db", "data", 0), ("-cache", "data", 0), ("-queue", "data", 0),
               ("-worker", "backend", 1), ("-service", "backend", 2), ("-api", "backend", 3),
               ("-bff", "edge", 4)]
SCALE_FIRST = ["Aiden", "Bea", "Carlos", "Dana", "Emeka", "Farah", "Gus", "Hiro", "Ines", "Jonas",
               "Kira", "Leo", "Marta", "Nikhil", "Olga", "Pablo", "Quinn", "Rosa", "Sven", "Tara",
               "Uma", "Viktor", "Wen", "Ximena", "Yusuf", "Zoe", "Anika", "Bruno", "Chloe", "Dev"]
SCALE_LAST = ["Abbott", "Banerjee", "Costa", "Dubois", "Eriksen", "Fujita", "Garcia", "Huang",
              "Ivanova", "Jensen", "Kowalski", "Larsen", "Moreau", "Novak", "Okoro", "Park",
              "Quintero", "Reyes", "Schmidt", "Tanaka", "Uddin", "Varga", "Weber", "Yilmaz"]
SCALE_HISTORY_DAYS = 90


def scale_topology(n_services):
    """Return the synthetic services, edges, teams and people for scale mode.

    Deterministic (fixed seed 7), so the same `n_services` always gives the
    same graph. Demo services and edges are not included; seed() adds these
    on top of SERVICES, DEPENDS_ON, TEAMS and PEOPLE.

    Args:
        n_services: Total services wanted, demo services included (> 22).

    Returns:
        (services, edges, teams, people):
          services: [(name, tier, criticality)]
          edges:    [(caller, callee, weight_ms)]
          teams:    {team: [service, ...]}
          people:   {team: [name, ...]}   the first one is on call
    """
    rng = random.Random(7)
    extra = n_services - len(SERVICES)
    # Names: domain + role, in domain order, so each domain is a small stack
    # (its db, cache, worker, service, api, bff) owned by one team.
    names = [(d, d + suffix, tier, level) for d in SCALE_DOMAINS
             for suffix, tier, level in SCALE_ROLES]
    if extra > len(names):  # more than 245 extra: number the domains ("search-2-api")
        names += [(f"{d}-{k}", f"{d}-{k}{suffix}", tier, level)
                  for k in range(2, extra // len(names) + 3) for d in SCALE_DOMAINS
                  for suffix, tier, level in SCALE_ROLES]
    names = names[:extra]
    services = [(name, tier, rng.choice([1, 2, 2, 3, 3])) for _, name, tier, _ in names]
    level = {name: lvl for _, name, _, lvl in names}

    # Demo services a synthetic service may call, by the level it calls from.
    # Only edge services (level 4) call frontend, so the longest chain is
    # bff -> frontend -> checkout -> payment -> fraud-check -> risk-model ->
    # feature-store (6 hops); api -> service -> worker -> checkoutservice -> ...
    # is at most 8.
    demo_backend = [s for s, tier, _ in SERVICES if tier == "backend"]
    demo_data = [s for s, tier, _ in SERVICES if tier in ("data", "external")]
    by_level = {lvl: [n for n, l in level.items() if l == lvl] for lvl in range(5)}
    edges = set()
    for name, lvl in level.items():
        if lvl == 0:
            continue
        lower = [n for l in range(lvl) for n in by_level[l]]
        same_domain = [n for n in lower if n.rsplit("-", 1)[0] == name.rsplit("-", 1)[0]]
        # Its own stack first (an api calls its service, a service its db), then
        # a few other domains, then sometimes the demo platform.
        picks = same_domain[:2] + rng.sample(lower, min(len(lower), rng.randint(4, 10)))
        if lvl >= 2 and rng.random() < 0.35:
            picks.append(rng.choice(demo_backend))
        if rng.random() < 0.25:
            picks.append(rng.choice(demo_data))
        if lvl == 4 and rng.random() < 0.3:
            picks.append("frontend")
        for callee in picks:
            if callee != name:
                edges.add((name, callee, rng.randint(1, 60)))
    # Every storage service needs a caller, or it would hang off its team only.
    called = {e[1] for e in edges}
    for name in by_level[0]:
        if name not in called:
            caller = rng.choice([n for l in (1, 2) for n in by_level[l]])
            edges.add((caller, name, rng.randint(1, 20)))

    # Teams: one per domain stack (numbered domains share their base team).
    teams = {}
    for domain, name, _, _ in names:
        base = domain.rsplit("-", 1)[0] if domain[-1].isdigit() else domain
        teams.setdefault(base.replace("-", " ").title(), []).append(name)
    taken = {p.split()[0] for ps in PEOPLE.values() for p in ps} | {"Sarah", "Akshit"}
    pool = [f"{f} {l}" for l in SCALE_LAST for f in SCALE_FIRST if f not in taken]
    rng.shuffle(pool)
    people = {t: [pool.pop() for _ in range(4)] for t in teams}
    return services, sorted(edges), teams, people


def build_rows(now_ms):
    """Generate the deploy and alert rows, with times relative to `now_ms`.

    Background noise comes first (random but repeatable, fixed seed 42), then
    the hand-written story events. The background is kept older than 2 hours so
    the "last hour" only shows the story.

    Args:
        now_ms: Reference time in epoch milliseconds.

    Returns:
        (deploys, alerts):
          deploys: [{"id", "service", "version", "ts", "status", "by"}, ...]  41 rows
          alerts:  [{"id", "service", "severity", "message", "ts", "resolved"}, ...]  14 rows
    """
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
                    "ts": now_ms - STORY_DEPLOY_AGO, "status": "success", "by": "Daniel Okafor"})
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


def scale_history(now_ms, teams, people):
    """Generate 90 days of background deploys and resolved alerts for scale mode.

    Covers every service, demo ones included. All rows are older than 2 hours
    and every alert is resolved, so the last hour still holds only the story.
    Versions v2.0.0 and v3.1.0 are never used: questions rely on the first not
    existing and the second being only the Story A deploy.

    Args:
        now_ms: Reference time in epoch milliseconds.
        teams: {team: [service, ...]} for every team, demo and synthetic.
        people: {team: [person, ...]} for every team.

    Returns:
        (deploys, alerts) shaped like build_rows() output.
    """
    rng = random.Random(11)
    deploys, alerts = [], []
    span = (2 * HOUR, SCALE_HISTORY_DAYS * DAY)
    for team, svcs in sorted(teams.items()):
        for svc in svcs:
            for i in range(rng.randint(10, 40)):  # about one deploy every 2-9 days
                version = "v2.0.0"
                while version in ("v2.0.0", "v3.1.0"):
                    version = f"v{rng.randint(1, 6)}.{rng.randint(0, 12)}.{rng.randint(0, 30)}"
                deploys.append({"id": f"sdep-{svc}-{i}", "service": svc, "version": version,
                                "ts": now_ms - rng.randint(*span),
                                "status": rng.choice(["success"] * 9 + ["rolled_back"]),
                                "by": rng.choice(people[team])})
            for i in range(rng.randint(2, 15)):
                alerts.append({"id": f"salt-{svc}-{i}", "service": svc,
                               "severity": rng.choice(["low", "low", "medium", "high"]),
                               "message": rng.choice(["p95 latency above baseline", "pod restarted",
                                                      "error rate blip", "disk usage 70%",
                                                      "queue depth high", "cache hit ratio low",
                                                      "certificate expires in 14 days"]),
                               "ts": now_ms - rng.randint(*span), "resolved": True})
    return deploys, alerts


def seed(g, now_ms=None, scale=None):
    """Delete graph `g` and rebuild it from scratch.

    Each node type is created in one UNWIND query (one round trip per type),
    which builds the whole graph in well under a second, even on Cloud.

    Args:
        g: FalkorDB Graph handle (any name; tests use throwaway graphs).
        now_ms: Time the story is anchored to, in epoch ms. Defaults to now.
        scale: Total number of services for scale mode (see scale_topology());
            None or 22 or less builds the demo graph only.

    Returns:
        The `now_ms` used, so callers (tests, re-seed logic) can compute
        "minutes ago" against the same reference.

    Raises:
        RuntimeError: if the built graph is not one connected component.
    """
    now_ms = now_ms or int(time.time() * 1000)
    deploys, alerts = build_rows(now_ms)
    services, edges = list(SERVICES), list(DEPENDS_ON)
    teams, people = dict(TEAMS), dict(PEOPLE)
    if scale and scale > len(SERVICES):
        more_services, more_edges, more_teams, more_people = scale_topology(scale)
        services += more_services
        edges += more_edges
        teams.update(more_teams)
        people.update(more_people)
        old_deploys, old_alerts = scale_history(now_ms, teams, people)
        deploys = old_deploys + deploys
        alerts = old_alerts + alerts
    try:
        g.delete()
    except ResponseError:
        pass  # graph did not exist yet

    # Indexes on the lookup keys: names for MATCH by name, ts for time-range filters.
    for label, prop in [("Service", "name"), ("Team", "name"), ("Person", "name"),
                        ("Deploy", "ts"), ("Alert", "ts")]:
        g.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")

    # Services, then the dependency edges between them.
    g.query("UNWIND $rows AS r CREATE (:Service {name: r[0], tier: r[1], criticality: r[2]})",
            {"rows": [list(s) for s in services]})
    g.query("""UNWIND $rows AS r
               MATCH (a:Service {name: r[0]}), (b:Service {name: r[1]})
               CREATE (a)-[:DEPENDS_ON {weight: r[2]}]->(b)""",
            {"rows": [list(e) for e in edges]})
    # Teams and the services they own.
    g.query("""UNWIND $rows AS r
               CREATE (t:Team {name: r.team})
               WITH t, r UNWIND r.services AS s
               MATCH (svc:Service {name: s}) CREATE (t)-[:OWNS]->(svc)""",
            {"rows": [{"team": t, "services": s} for t, s in teams.items()]})
    # People; the first person listed in each team is on call.
    g.query("""UNWIND $rows AS r
               MATCH (t:Team {name: r.team})
               CREATE (:Person {name: r.name, on_call: r.on_call})-[:MEMBER_OF]->(t)""",
            {"rows": [{"team": t, "name": p, "on_call": i == 0}
                      for t, ps in people.items() for i, p in enumerate(ps)]})
    # Deploys: who deployed which version of which service, and when.
    g.query("""UNWIND $rows AS r
               MATCH (s:Service {name: r.service}), (p:Person {name: r.by})
               CREATE (p)-[:DEPLOYED]->(:Deploy {id: r.id, version: r.version, ts: r.ts,
                                                 status: r.status})-[:TO]->(s)""",
            {"rows": deploys})
    # Alerts, each attached to the service it fired on.
    g.query("""UNWIND $rows AS r
               MATCH (s:Service {name: r.service})
               CREATE (:Alert {id: r.id, severity: r.severity, message: r.message,
                               ts: r.ts, resolved: r.resolved})-[:FIRED_ON]->(s)""",
            {"rows": alerts})

    # Sanity check: every node must be connected to the rest of the graph.
    # WCC = weakly connected components; a typo in a name above would leave an
    # orphan node and show up here as a second component.
    components = g.ro_query("CALL algo.WCC(null) YIELD componentId "
                            "RETURN count(DISTINCT componentId)").result_set[0][0]
    if components != 1:
        raise RuntimeError(f"seeded graph has {components} components, expected 1")
    verify.clear_cache()  # name lists cached by verify.py are now stale
    return now_ms


def ensure_fresh(g, max_age_min=MAX_STORY_AGE_MIN):
    """Re-seed graph `g` if it is empty or was seeded more than `max_age_min` ago.

    The demo story is anchored to seed time, so an old graph shows "no recent
    deploys" and the agent blames the wrong service (seen in testing after an
    hour). Only for the synthetic demo graph: a real graph is never re-seeded.

    Args:
        g: FalkorDB Graph handle.
        max_age_min: Oldest acceptable seed age, in minutes.

    Returns:
        True if the graph was re-seeded, False if it was fresh enough.
    """
    try:
        newest, n_services = g.ro_query(
            "MATCH (d:Deploy) WITH max(d.ts) AS newest "
            "MATCH (s:Service) RETURN newest, count(s)").result_set[0]
    except (ResponseError, IndexError):
        newest, n_services = None, 0  # graph does not exist yet
    now_ms = int(time.time() * 1000)
    if newest is not None and now_ms - (newest + STORY_DEPLOY_AGO) <= max_age_min * MIN:
        return False
    # A scale-mode graph is re-seeded at the same size, not shrunk to the demo.
    scale = n_services if n_services > len(SERVICES) else None
    logging.getLogger("detective.seed").info(
        "re-seeding the demo graph (%s%s)", "empty" if newest is None else "story too old",
        f", {scale} services" if scale else "")
    seed(g, now_ms, scale)
    return True


if __name__ == "__main__":
    # Seed the demo graph, or with --scale N a graph of N services (default
    # 300) in its own graph, "incidents_scale" unless GRAPH_NAME is set, so the
    # demo graph is never overwritten by accident.
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build the Incident Detective graph.")
    parser.add_argument("--scale", nargs="?", type=int, const=300, default=None,
                        help="total services for scale mode (default 300)")
    args = parser.parse_args()
    name = os.environ.get("GRAPH_NAME") or ("incidents_scale" if args.scale else "incidents")
    g = get_graph(name)
    started = time.time()
    seed(g, scale=args.scale)
    counts = g.ro_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) ORDER BY label").result_set
    edges = g.ro_query("MATCH ()-[r:DEPENDS_ON]->() RETURN count(r)").result_set[0][0]
    print(f"Seeded graph '{g.name}' in {time.time() - started:.1f}s:",
          ", ".join(f"{c} {l}" for l, c in counts) + f", {edges} DEPENDS_ON edges")
    if args.scale:
        print(f"Run the app on it: GRAPH_NAME={g.name} uvicorn app:app --port 8765")
