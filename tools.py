"""Graph tools the agent can call. Each one is a fixed, parameterized, read-only query.

Safety model:
  * The LLM only picks a tool and its arguments; it never writes Cypher itself.
  * Every query runs through RO_QUERY, so FalkorDB rejects any write.
  * Fixed queries also keep out things RO_QUERY still allows, such as LOAD CSV
    fetching an outside URL.
  * Every query has a timeout, so one slow call cannot hang the agent.

Every tool returns a plain dict (JSON-ready for the LLM and the API). The dict
includes the Cypher that ran, so the UI can show users exactly how each answer
was found. Unknown service names return an error plus the list of valid names,
so the LLM can correct itself and retry.
"""
import time

# Per-query timeout in milliseconds. Our graph answers in a few ms; this only
# guards against a runaway query on the 100MB FalkorDB Cloud free tier.
TIMEOUT_MS = 3000

# Upper bound for variable-length DEPENDS_ON walks (the "*1..N" in the patterns).
# This is our own choice, not a FalkorDB rule: FalkorDB's default upper bound is
# unlimited. A cap keeps the number of paths the query explores bounded if the
# graph grows or ever gains a cycle. The longest chain in the seeded graph is 6
# hops (api-gateway -> ... -> feature-store), so 8 leaves headroom.
# tests/test_seed.py fails if the seed ever outgrows this cap.
MAX_HOPS = 8


def _run(g, cypher, params=None):
    """Run a read-only query with the standard timeout and return its rows."""
    return g.ro_query(cypher, params, timeout=TIMEOUT_MS).result_set


def _check_service(g, name):
    """Return None if `name` is a known service, else an error dict for the LLM.

    The error lists every valid name, so a typo like "payment-db" can be fixed
    by the model on its next call instead of ending the investigation.
    """
    if _run(g, "MATCH (s:Service {name: $name}) RETURN count(s)", {"name": name})[0][0]:
        return None
    known = [r[0] for r in _run(g, "MATCH (s:Service) RETURN s.name ORDER BY s.name")]
    return {"error": f"unknown service '{name}'", "known_services": known}


def dependencies(g, service):
    """Services that `service` calls, directly or indirectly: possible root causes.

    Follows DEPENDS_ON edges outward. `hops` is the shortest distance, so a
    service reachable by several routes appears once.
    """
    if err := _check_service(g, service):
        return err
    # The path variable `p` is used in length(p), so every edge is really
    # walked (see the "unused edge alias" limitation in the FalkorDB docs).
    cypher = (f"MATCH p = (s:Service {{name: $name}})-[r:DEPENDS_ON*1..{MAX_HOPS}]->(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    rows = _run(g, cypher, {"name": service})
    return {"service": service, "depends_on": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher}


def blast_radius(g, service):
    """Services that call `service`, directly or indirectly: what breaks if it fails.

    Same walk as dependencies(), but along incoming DEPENDS_ON edges.
    """
    if err := _check_service(g, service):
        return err
    cypher = (f"MATCH p = (s:Service {{name: $name}})<-[r:DEPENDS_ON*1..{MAX_HOPS}]-(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    rows = _run(g, cypher, {"name": service})
    return {"service": service, "affected": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher}


def recent_changes(g, minutes=60, now_ms=None):
    """Deploys and unresolved alerts in the last `minutes`, newest first.

    Timestamps are stored as integer epoch milliseconds (indexed in seed.py).
    `now_ms` can be fixed by tests so results do not depend on the wall clock.
    """
    now_ms = now_ms or int(time.time() * 1000)
    params = {"since": now_ms - minutes * 60_000}
    deploy_q = ("MATCH (p:Person)-[:DEPLOYED]->(d:Deploy)-[:TO]->(s:Service) WHERE d.ts >= $since "
                "RETURN s.name, d.version, p.name, d.ts ORDER BY d.ts DESC LIMIT 20")
    alert_q = ("MATCH (a:Alert)-[:FIRED_ON]->(s:Service) WHERE a.ts >= $since AND NOT a.resolved "
               "RETURN s.name, a.severity, a.message, a.ts ORDER BY a.ts DESC LIMIT 20")
    # Minutes ago reads better for the LLM and for people than raw epoch values.
    ago = lambda ts: round((now_ms - ts) / 60_000)
    return {
        "deploys": [{"service": s, "version": v, "by": p, "minutes_ago": ago(ts)}
                    for s, v, p, ts in _run(g, deploy_q, params)],
        "alerts": [{"service": s, "severity": sev, "message": m, "minutes_ago": ago(ts)}
                   for s, sev, m, ts in _run(g, alert_q, params)],
        "cypher": [deploy_q, alert_q],
    }


def dependency_path(g, from_service, to_service):
    """Lowest-latency call path between two services (either direction), or none.

    Uses algo.SPpaths with each edge's `weight` (typical latency in ms). Calls
    only go one way, so we try a -> b first and then b -> a; the LLM does not
    need to know which service is the caller. "No path" is a useful answer: it
    shows an alert is unrelated to the incident (e.g. adservice in Story C).
    """
    for name in (from_service, to_service):
        if err := _check_service(g, name):
            return err
    cypher = ("MATCH (a:Service {name: $a}), (b:Service {name: $b}) "
              "CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['DEPENDS_ON'], "
              "weightProp: 'weight'}) YIELD path, pathWeight "
              "RETURN [n IN nodes(path) | n.name], pathWeight")
    for a, b in ((from_service, to_service), (to_service, from_service)):
        rows = _run(g, cypher, {"a": a, "b": b})
        if rows:
            return {"path": rows[0][0], "latency_ms": rows[0][1], "cypher": cypher}
    return {"path": None, "note": f"no dependency path between {from_service} and "
                                  f"{to_service} in either direction", "cypher": cypher}


def critical_services(g, metric="betweenness", limit=5):
    """Rank services by graph importance.

    'betweenness': chokepoints, services that many call paths pass through.
    'pagerank':    services that many others depend on, directly or not.
    """
    if metric == "pagerank":
        cypher = ("CALL algo.pageRank('Service', 'DEPENDS_ON') YIELD node, score "
                  "RETURN node.name, score ORDER BY score DESC LIMIT $limit")
    elif metric == "betweenness":
        # Betweenness samples start nodes; without a fixed seed it uses the
        # clock, and the ranking could change between runs.
        cypher = ("CALL algo.betweenness({nodeLabels: ['Service'], relationshipTypes: "
                  "['DEPENDS_ON'], samplingSeed: 42}) YIELD node, score "
                  "RETURN node.name, score ORDER BY score DESC LIMIT $limit")
    else:
        return {"error": "metric must be 'betweenness' or 'pagerank'"}
    # Clamp the LLM-supplied limit to the number of services in the seed.
    rows = _run(g, cypher, {"limit": min(int(limit), 22)})
    return {"metric": metric, "ranking": [{"service": n, "score": round(s, 4)} for n, s in rows],
            "cypher": cypher}


def who_to_page(g, service):
    """Owning team and its on-call person for `service`.

    OPTIONAL MATCH keeps the team in the answer even if nobody is on call.
    """
    if err := _check_service(g, service):
        return err
    cypher = ("MATCH (s:Service {name: $name})<-[:OWNS]-(t:Team) "
              "OPTIONAL MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t) "
              "RETURN t.name, p.name")
    team, person = _run(g, cypher, {"name": service})[0]
    return {"service": service, "team": team, "on_call": person, "cypher": cypher}


# Name -> function lookup used by the agent to dispatch the LLM's tool calls.
TOOLS = {f.__name__: f for f in
         (dependencies, blast_radius, recent_changes, dependency_path, critical_services,
          who_to_page)}
