"""Actions the on-call person can take on the agent's finding: open an incident
(and page its owner) and roll back the deploy that caused it.

This is the only module besides seed.py that writes to the graph, and it is
kept apart from tools.py on purpose:
  * The model never calls these. They run only when a person clicks the button
    the UI shows under an answer (POST /api/actions/...), or from the MCP server.
  * Each one takes just the root-cause deploy (service + version). Who to page
    and which alerts clear are worked out from the graph here, never taken from
    the caller, so a request cannot page a stranger or clear an unrelated alert.
  * Every query is fixed and parameterised, like the read tools.

What gets written, so the next question (or the next session) can see it:
  (:Incident {id, status, opened_ts, title})-[:CAUSED_BY]->(:Deploy)
  (:Incident)-[:AFFECTS]->(:Service)      every alerting service on the deploy's chain
  (:Incident)-[:PAGED {ts}]->(:Person)    the on-call person of the deploy's owner team
  rollback: Deploy.status = "rolled_back", Incident.status = "mitigated", and the
            chain's open alerts resolved, each tagged with what cleared it.

The data is synthetic, so a rollback here changes the graph only; no real
service is touched. tools.recent_changes() returns incidents, which is how the
agent remembers them.
"""
import time

import tools
import verify

# Writes get the same timeout as reads: none of these touches more than a few nodes.
TIMEOUT_MS = tools.TIMEOUT_MS


def _write(g, cypher, params=None):
    """Run one write query with the standard timeout and return its rows."""
    return g.query(cypher, params, timeout=TIMEOUT_MS).result_set


def _deploy(g, service, version):
    """Return (deploy ts, status) for `service` `version`, or None if there is no such deploy."""
    rows = tools._run(g, "MATCH (d:Deploy {version: $v})-[:TO]->(:Service {name: $s}) "
                         "RETURN d.ts, d.status ORDER BY d.ts DESC LIMIT 1",
                      {"s": service, "v": version})
    return rows[0] if rows else None


def _chain(g, service):
    """`service`, everything it depends on, and everything that depends on it."""
    return ({service}
            | {d["service"] for d in tools.dependencies(g, service)["depends_on"]}
            | {d["service"] for d in tools.blast_radius(g, service)["affected"]})


def _incident(g, service, version):
    """The newest incident caused by this deploy, as a dict, or None.

    Returns:
        {"id", "status", "opened_ts", "paged", "affects": [service, ...]} or None.
    """
    rows = tools._run(g, "MATCH (i:Incident)-[:CAUSED_BY]->(d:Deploy {version: $v})"
                         "-[:TO]->(:Service {name: $s}) "
                         "OPTIONAL MATCH (i)-[:PAGED]->(p:Person) "
                         "OPTIONAL MATCH (i)-[:AFFECTS]->(a:Service) "
                         "RETURN i.id, i.status, i.opened_ts, p.name, collect(DISTINCT a.name) "
                         "ORDER BY i.opened_ts DESC LIMIT 1",
                      {"s": service, "v": version})
    if not rows:
        return None
    i, status, ts, paged, affects = rows[0]
    return {"id": i, "status": status, "opened_ts": ts, "paged": paged, "affects": sorted(affects)}


def open_incident(g, service, version, now_ms=None):
    """Open an incident for the `service` `version` deploy and page its owner's on-call.

    Idempotent: if this deploy already has an incident, that one is returned
    (with "existing": True) and nobody is paged twice.

    Args:
        g: FalkorDB Graph handle.
        service: Service the root-cause deploy went to, e.g. "fraud-check".
        version: The deploy's version, e.g. "v3.1.0".
        now_ms: Time to record, in epoch ms; tests pass a fixed value.

    Returns:
        {"id": "INC-1", "status": "open", "existing": bool, "service", "version",
         "team", "paged": person or None, "affects": [alerting services on the chain]}
        or {"error": str} if there is no such deploy.

    Example:
        open_incident(g, "fraud-check", "v3.1.0")
        -> {"id": "INC-1", "paged": "Priya Nair",
            "affects": ["checkoutservice", "payments-db", "paymentservice"], ...}
    """
    if not _deploy(g, service, version):
        return {"error": f"no {service} {version} deploy in the graph"}
    owner = tools.who_to_page(g, service)
    if found := _incident(g, service, version):
        return {**found, "existing": True, "service": service, "version": version,
                "team": owner["team"]}
    now_ms = now_ms or int(time.time() * 1000)
    chain = _chain(g, service)
    affects = sorted({a["service"] for a in tools.recent_changes(g, now_ms=now_ms)["alerts"]}
                     & chain)
    n = tools._run(g, "MATCH (i:Incident) RETURN count(i)")[0][0] + 1
    params = {"s": service, "v": version, "id": f"INC-{n}", "now": now_ms,
              "title": f"{service} {version}", "person": owner["on_call"] or "",
              "affects": affects}
    # One query: the incident, its cause, who was paged, and what it affects.
    # OPTIONAL MATCH + FOREACH skip the PAGED edge when the team has nobody on call.
    _write(g, "MATCH (d:Deploy {version: $v})-[:TO]->(:Service {name: $s}) "
              "WITH d ORDER BY d.ts DESC LIMIT 1 "
              "CREATE (i:Incident {id: $id, status: 'open', opened_ts: $now, title: $title})"
              "-[:CAUSED_BY]->(d) "
              "WITH i OPTIONAL MATCH (p:Person {name: $person}) "
              "FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | "
              "  CREATE (i)-[:PAGED {ts: $now}]->(p)) "
              "WITH i UNWIND $affects AS name "
              "MATCH (a:Service {name: name}) CREATE (i)-[:AFFECTS]->(a)", params)
    verify.clear_cache()
    return {"id": params["id"], "status": "open", "opened_ts": now_ms, "existing": False,
            "service": service, "version": version, "team": owner["team"],
            "paged": owner["on_call"], "affects": affects}


def rollback(g, service, version, now_ms=None):
    """Roll back the `service` `version` deploy and clear the alerts it caused.

    Opens the incident first if there is none, so the rollback is on record.
    Only alerts on the deploy's dependency chain that fired after the deploy
    are resolved: an alert with no path to the deploy (adservice) stays open.

    Args:
        g: FalkorDB Graph handle.
        service: Service the deploy went to, e.g. "fraud-check".
        version: The deploy's version, e.g. "v3.1.0".
        now_ms: Time to record, in epoch ms; tests pass a fixed value.

    Returns:
        {"incident": open_incident() result, "rolled_back": bool (False if it
         already was), "resolved": [{"service", "message"}, ...],
         "still_open": [{"service", "message"}, ...]}  open alerts it did not clear
        or {"error": str} if there is no such deploy.
    """
    found = _deploy(g, service, version)
    if not found:
        return {"error": f"no {service} {version} deploy in the graph"}
    deploy_ts, status = found
    now_ms = now_ms or int(time.time() * 1000)
    incident = open_incident(g, service, version, now_ms)
    resolved = []
    if status != "rolled_back":
        params = {"s": service, "v": version, "now": now_ms, "since": deploy_ts,
                  "chain": sorted(_chain(g, service)), "by": f"rollback of {service} {version}"}
        _write(g, "MATCH (d:Deploy {version: $v})-[:TO]->(:Service {name: $s}) "
                  "SET d.status = 'rolled_back', d.rolled_back_ts = $now", params)
        resolved = [{"service": s, "message": m} for s, m in _write(
            g, "MATCH (a:Alert)-[:FIRED_ON]->(x:Service) "
               "WHERE x.name IN $chain AND NOT a.resolved AND a.ts >= $since "
               "SET a.resolved = true, a.resolved_ts = $now, a.resolved_by = $by "
               "RETURN x.name, a.message ORDER BY x.name", params)]
        _write(g, "MATCH (i:Incident {id: $id}) SET i.status = 'mitigated', i.mitigated_ts = $now, "
                  "i.mitigation = $m",
               {"id": incident["id"], "now": now_ms, "m": f"rolled back {service} {version}"})
        verify.clear_cache()
    still_open = [{"service": a["service"], "message": a["message"]}
                  for a in tools.recent_changes(g, now_ms=now_ms)["alerts"]]
    return {"incident": {**incident, "status": "mitigated"}, "rolled_back": status != "rolled_back",
            "resolved": resolved, "still_open": still_open}
