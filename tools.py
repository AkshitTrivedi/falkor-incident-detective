"""Graph tools the agent can call. Each one is a fixed, parameterized, read-only query.

The LLM only chooses a tool and its arguments; it never writes Cypher. RO_QUERY blocks
writes, and fixed queries also keep out things RO_QUERY allows (e.g. LOAD CSV fetching URLs).
Every result includes the Cypher that ran, so the UI can show it.
"""
import time

TIMEOUT_MS = 3000
MAX_HOPS = 5


def _run(g, cypher, params=None):
    return g.ro_query(cypher, params, timeout=TIMEOUT_MS).result_set


def _check_service(g, name):
    if _run(g, "MATCH (s:Service {name: $name}) RETURN count(s)", {"name": name})[0][0]:
        return None
    known = [r[0] for r in _run(g, "MATCH (s:Service) RETURN s.name ORDER BY s.name")]
    return {"error": f"unknown service '{name}'", "known_services": known}


def dependencies(g, service):
    """Services that `service` calls, directly or indirectly: possible root causes."""
    if err := _check_service(g, service):
        return err
    cypher = (f"MATCH p = (s:Service {{name: $name}})-[r:DEPENDS_ON*1..{MAX_HOPS}]->(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    rows = _run(g, cypher, {"name": service})
    return {"service": service, "depends_on": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher}


def blast_radius(g, service):
    """Services that call `service`, directly or indirectly: what breaks if it fails."""
    if err := _check_service(g, service):
        return err
    cypher = (f"MATCH p = (s:Service {{name: $name}})<-[r:DEPENDS_ON*1..{MAX_HOPS}]-(x:Service) "
              "RETURN x.name, min(length(p)) AS hops ORDER BY hops, x.name")
    rows = _run(g, cypher, {"name": service})
    return {"service": service, "affected": [{"service": n, "hops": h} for n, h in rows],
            "cypher": cypher}


def recent_changes(g, minutes=60, now_ms=None):
    """Deploys and unresolved alerts in the last `minutes`, newest first."""
    now_ms = now_ms or int(time.time() * 1000)
    params = {"since": now_ms - minutes * 60_000}
    deploy_q = ("MATCH (p:Person)-[:DEPLOYED]->(d:Deploy)-[:TO]->(s:Service) WHERE d.ts >= $since "
                "RETURN s.name, d.version, p.name, d.ts ORDER BY d.ts DESC LIMIT 20")
    alert_q = ("MATCH (a:Alert)-[:FIRED_ON]->(s:Service) WHERE a.ts >= $since AND NOT a.resolved "
               "RETURN s.name, a.severity, a.message, a.ts ORDER BY a.ts DESC LIMIT 20")
    ago = lambda ts: round((now_ms - ts) / 60_000)
    return {
        "deploys": [{"service": s, "version": v, "by": p, "minutes_ago": ago(ts)}
                    for s, v, p, ts in _run(g, deploy_q, params)],
        "alerts": [{"service": s, "severity": sev, "message": m, "minutes_ago": ago(ts)}
                   for s, sev, m, ts in _run(g, alert_q, params)],
        "cypher": [deploy_q, alert_q],
    }


def dependency_path(g, from_service, to_service):
    """Lowest-latency call path between two services (either direction), or none."""
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
    """Rank services: 'betweenness' = chokepoints, 'pagerank' = most depended on."""
    if metric == "pagerank":
        cypher = ("CALL algo.pageRank('Service', 'DEPENDS_ON') YIELD node, score "
                  "RETURN node.name, score ORDER BY score DESC LIMIT $limit")
    elif metric == "betweenness":
        # Fixed seed so the ranking is the same on every run.
        cypher = ("CALL algo.betweenness({nodeLabels: ['Service'], relationshipTypes: "
                  "['DEPENDS_ON'], samplingSeed: 42}) YIELD node, score "
                  "RETURN node.name, score ORDER BY score DESC LIMIT $limit")
    else:
        return {"error": "metric must be 'betweenness' or 'pagerank'"}
    rows = _run(g, cypher, {"limit": min(int(limit), 22)})
    return {"metric": metric, "ranking": [{"service": n, "score": round(s, 4)} for n, s in rows],
            "cypher": cypher}


def who_to_page(g, service):
    """Owning team and its on-call person for `service`."""
    if err := _check_service(g, service):
        return err
    cypher = ("MATCH (s:Service {name: $name})<-[:OWNS]-(t:Team) "
              "OPTIONAL MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t) "
              "RETURN t.name, p.name")
    team, person = _run(g, cypher, {"name": service})[0]
    return {"service": service, "team": team, "on_call": person, "cypher": cypher}


TOOLS = {f.__name__: f for f in
         (dependencies, blast_radius, recent_changes, dependency_path, critical_services,
          who_to_page)}
