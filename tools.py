"""Graph tools the agent can call. Each one is a fixed, parameterized, read-only query.

Safety model:
  * The LLM only picks a tool and its arguments; it never writes Cypher itself.
  * Every query runs through RO_QUERY, so FalkorDB rejects any write.
  * Fixed queries also keep out things RO_QUERY still allows, such as LOAD CSV
    fetching an outside URL.
  * Every query has a timeout, so one slow call cannot hang the agent.

Every tool returns a plain dict (JSON-ready for the LLM and the API). The dict
includes the Cypher that ran and its parameter values ("cypher", "params"), so
the UI can show users exactly how each answer was found; runnable() joins the
two into a query that can be pasted into FalkorDB Browser. Unknown service names return an error plus the list of valid names,
so the LLM can correct itself and retry.

All tools take `g`, a FalkorDB Graph handle from db.get_graph().
"""
import json
import time

# Per-query timeout in milliseconds. Our graph answers in a few ms; this only
# guards against a runaway query on the 100MB FalkorDB Cloud free tier.
TIMEOUT_MS = 3000

# Upper bound for variable-length DEPENDS_ON walks (the "*1..N" in the patterns).
# This is our own choice, not a FalkorDB rule: FalkorDB's default upper bound is
# unlimited. A cap keeps the number of paths the query explores bounded if the
# graph grows or ever gains a cycle. The longest chain in the seeded graph is 6 hops:
#   api-gateway -> frontend -> checkoutservice -> paymentservice
#               -> fraud-check -> risk-model -> feature-store
# so 8 leaves headroom. tests/test_seed.py fails if the seed ever outgrows this cap.
MAX_HOPS = 8

# Result keys that trace how a result was found. They are for people (the UI),
# so they are left out of what the LLM and the answer checker read.
TRACE_KEYS = {"cypher", "params"}


def runnable(cypher, params=None):
    """Return `cypher` with its parameter values written in as a CYPHER prefix.

    FalkorDB reads `CYPHER name="x" MATCH ... $name ...` as a parameterised
    query, so the result runs as pasted into FalkorDB Browser or redis-cli
    GRAPH.QUERY, with the same values the tool used.

    Args:
        cypher: Query text with $params.
        params: Dict of parameter values (str, int, float, bool), or None.

    Returns:
        The query text, prefixed with "CYPHER k=v ... " when there are params.

    Example:
        runnable("MATCH (s:Service {name: $name}) RETURN s", {"name": "frontend"})
        -> 'CYPHER name="frontend" MATCH (s:Service {name: $name}) RETURN s'
    """
    if not params:
        return cypher
    # json.dumps gives a double-quoted, escaped string literal, which Cypher accepts.
    values = " ".join(f"{k}={json.dumps(v)}" for k, v in params.items())
    return f"CYPHER {values} {cypher}"


def _run(g, cypher, params=None):
    """Run one read-only query with the standard timeout.

    Args:
        g: FalkorDB Graph handle.
        cypher: Query text. Values must be passed as $params, never formatted in.
        params: Dict of query parameters, or None.

    Returns:
        List of result rows; each row is a list of column values.

    Raises:
        redis.exceptions.ResponseError: if the query tries to write, has a
        syntax error, or runs past TIMEOUT_MS.
    """
    return g.ro_query(cypher, params, timeout=TIMEOUT_MS).result_set


def _check_service(g, name):
    """Check that a Service node called `name` exists.

    The error lists every valid name, so a typo like "payment-db" can be fixed
    by the model on its next call instead of ending the investigation.

    Args:
        g: FalkorDB Graph handle.
        name: Service name as given by the LLM.

    Returns:
        None if the service exists, otherwise
        {"error": str, "known_services": [str, ...]}.
    """
    if _run(g, "MATCH (s:Service {name: $name}) RETURN count(s)", {"name": name})[0][0]:
        return None
    known = [r[0] for r in _run(g, "MATCH (s:Service) RETURN s.name ORDER BY s.name")]
    return {"error": f"unknown service '{name}'", "known_services": known}


def _minutes_ago(now_ms, ts):
    """Convert an epoch-ms timestamp to whole minutes before `now_ms`.

    "14 minutes ago" is easier for the LLM and for people than a raw epoch value.
    """
    return round((now_ms - ts) / 60_000)


def dependencies(g, service):
    """Find every service that `service` calls, directly or indirectly.

    These are the possible root causes when `service` is failing. Follows
    DEPENDS_ON edges outward up to MAX_HOPS. A service reachable by several
    routes appears once, with its shortest distance.

    Args:
        g: FalkorDB Graph handle.
        service: Name of the failing service, e.g. "checkoutservice".

    Returns:
        {"service": str,
         "depends_on": [{"service": str, "hops": int}, ...]  # nearest first
         "cypher": str}
        or the _check_service error dict for an unknown name.

    Example:
        dependencies(g, "checkoutservice")["depends_on"][:2]
        -> [{"service": "cartservice", "hops": 1}, {"service": "currencyservice", "hops": 1}]
    """
    if err := _check_service(g, service):
        return err
    # The path variable `p` is used in length(p), so every edge is really
    # walked (see the "unused edge alias" limitation in the FalkorDB docs).
    cypher = (f"MATCH p = (s:Service {{name: $name}})-[r:DEPENDS_ON*1..{MAX_HOPS}]->(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    params = {"name": service}
    rows = _run(g, cypher, params)
    return {"service": service, "depends_on": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher, "params": params}


def blast_radius(g, service):
    """Find every service that calls `service`, directly or indirectly.

    This is what breaks if `service` fails. Same walk as dependencies(), but
    along incoming DEPENDS_ON edges.

    Args:
        g: FalkorDB Graph handle.
        service: Name of the service that might fail, e.g. "payments-db".

    Returns:
        {"service": str,
         "affected": [{"service": str, "hops": int}, ...]  # nearest first
         "cypher": str}
        or the _check_service error dict for an unknown name.

    Example:
        blast_radius(g, "payments-db")["affected"][:2]
        -> [{"service": "fraud-check", "hops": 1}, {"service": "paymentservice", "hops": 1}]
    """
    if err := _check_service(g, service):
        return err
    cypher = (f"MATCH p = (s:Service {{name: $name}})<-[r:DEPENDS_ON*1..{MAX_HOPS}]-(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    params = {"name": service}
    rows = _run(g, cypher, params)
    return {"service": service, "affected": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher, "params": params}


def recent_changes(g, minutes=60, now_ms=None):
    """List deploys and still-open alerts from the last `minutes`, newest first.

    Usually the agent's first call: "what changed just before things broke?"
    Timestamps are integer epoch milliseconds (indexed in seed.py). Each list
    is capped at 20 rows to keep the LLM prompt small.

    Args:
        g: FalkorDB Graph handle.
        minutes: How far back to look. Default 60.
        now_ms: Reference time in epoch ms. Defaults to the current time;
            tests pass a fixed value so results do not depend on the clock.

    Returns:
        {"deploys": [{"service", "version", "by", "minutes_ago"}, ...],
         "alerts":  [{"service", "severity", "message", "minutes_ago"}, ...],
         "cypher":  [deploy query, alert query]}

    Example (Story A):
        recent_changes(g)["deploys"]
        -> [{"service": "fraud-check", "version": "v3.1.0",
             "by": "Daniel Okafor", "minutes_ago": 14}]
    """
    now_ms = now_ms or int(time.time() * 1000)
    params = {"since": now_ms - minutes * 60_000}
    deploy_q = ("MATCH (p:Person)-[:DEPLOYED]->(d:Deploy)-[:TO]->(s:Service) WHERE d.ts >= $since "
                "RETURN s.name, d.version, p.name, d.ts ORDER BY d.ts DESC LIMIT 20")
    # Resolved alerts are left out: they are background noise, not the incident.
    alert_q = ("MATCH (a:Alert)-[:FIRED_ON]->(s:Service) WHERE a.ts >= $since AND NOT a.resolved "
               "RETURN s.name, a.severity, a.message, a.ts ORDER BY a.ts DESC LIMIT 20")
    return {
        "deploys": [{"service": s, "version": v, "by": p, "minutes_ago": _minutes_ago(now_ms, ts)}
                    for s, v, p, ts in _run(g, deploy_q, params)],
        "alerts": [{"service": s, "severity": sev, "message": m,
                    "minutes_ago": _minutes_ago(now_ms, ts)}
                   for s, sev, m, ts in _run(g, alert_q, params)],
        "cypher": [deploy_q, alert_q],
        "params": params,   # the same values for both queries
    }


def dependency_path(g, from_service, to_service):
    """Find the lowest-latency call path between two services, in either direction.

    Uses algo.SPpaths with each edge's `weight` (typical latency in ms). Calls
    only go one way, so we try from -> to first and then to -> from; the LLM
    does not need to know which service is the caller. "No path" is a useful
    answer: it shows an alert is unrelated to the incident (Story C, adservice).

    Args:
        g: FalkorDB Graph handle.
        from_service: One service name, e.g. "checkoutservice".
        to_service: The other service name, e.g. "payments-db".

    Returns:
        {"path": [str, ...], "latency_ms": float, "cypher": str} when connected,
        {"path": None, "note": str, "cypher": str} when not,
        or the _check_service error dict for an unknown name.

    Example:
        dependency_path(g, "checkoutservice", "payments-db")
        -> {"path": ["checkoutservice", "paymentservice", "payments-db"],
            "latency_ms": 36.0, ...}
    """
    for name in (from_service, to_service):
        if err := _check_service(g, name):
            return err
    cypher = ("MATCH (a:Service {name: $a}), (b:Service {name: $b}) "
              "CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['DEPENDS_ON'], "
              "weightProp: 'weight'}) YIELD path, pathWeight "
              "RETURN [n IN nodes(path) | n.name], pathWeight")
    # Try the given order first, then the reverse.
    for a, b in ((from_service, to_service), (to_service, from_service)):
        params = {"a": a, "b": b}
        rows = _run(g, cypher, params)
        if rows:
            return {"path": rows[0][0], "latency_ms": rows[0][1], "cypher": cypher,
                    "params": params}
    return {"path": None, "note": f"no dependency path between {from_service} and "
                                  f"{to_service} in either direction", "cypher": cypher,
            "params": {"a": from_service, "b": to_service}}


def critical_services(g, metric="betweenness", limit=5):
    """Rank services by how important they are in the dependency graph.

    Metrics:
        "betweenness": chokepoints, services that many call paths pass through.
            If one fails, many request flows break.
        "pagerank": services that many others depend on, directly or not
            (often shared databases).

    Args:
        g: FalkorDB Graph handle.
        metric: "betweenness" (default) or "pagerank".
        limit: How many services to return (clamped to the 22 in the graph).

    Returns:
        {"metric": str, "ranking": [{"service": str, "score": float}, ...], "cypher": str}
        or {"error": str} for an unknown metric.

    Example (Story B):
        critical_services(g, "betweenness", 3)["ranking"]
        -> paymentservice, checkoutservice, fraud-check (highest first)
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
    params = {"limit": min(int(limit), 22)}
    rows = _run(g, cypher, params)
    return {"metric": metric, "ranking": [{"service": n, "score": round(s, 4)} for n, s in rows],
            "cypher": cypher, "params": params}


def who_to_page(g, service):
    """Find the team that owns `service` and who on that team is on call.

    Usually the agent's last call, once it has found the root-cause service.
    OPTIONAL MATCH keeps the team in the answer even if nobody is on call.

    Args:
        g: FalkorDB Graph handle.
        service: Name of the root-cause service, e.g. "fraud-check".

    Returns:
        {"service": str, "team": str, "on_call": str or None, "cypher": str}
        or the _check_service error dict for an unknown name.

    Example (Story A):
        who_to_page(g, "fraud-check") -> team "Payments", on_call "Priya Nair"
    """
    if err := _check_service(g, service):
        return err
    cypher = ("MATCH (s:Service {name: $name})<-[:OWNS]-(t:Team) "
              "OPTIONAL MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t) "
              "RETURN t.name, p.name")
    params = {"name": service}
    team, person = _run(g, cypher, params)[0]
    return {"service": service, "team": team, "on_call": person, "cypher": cypher,
            "params": params}


# Name -> function lookup used by the agent to dispatch the LLM's tool calls.
TOOLS = {f.__name__: f for f in
         (dependencies, blast_radius, recent_changes, dependency_path, critical_services,
          who_to_page)}
