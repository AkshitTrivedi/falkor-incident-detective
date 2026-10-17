"""Web app for Incident Detective: a FastAPI backend plus a one-page UI.

Routes:
  GET  /            the UI (static/index.html)
  GET  /api/graph   the topology to draw: services, dependencies, teams, on-call
                    people, and the deploys and open alerts from the last hour
  POST /api/ask     {"question": str} -> the agent's answer, its tool steps and
                    Cypher, plus a "summary" the UI uses to light up the graph
  GET  /api/health  whether FalkorDB answers
  POST /api/client-error   browser JS errors, printed in the server terminal
  GET  /docs        FastAPI's interactive API docs

Run locally:
  graph-hacks-env/bin/uvicorn app:app --reload
then open http://127.0.0.1:8000.

Everything the UI shows comes from the graph or from agent.investigate(); this
module only reshapes it. All queries here are read-only, except seed.ensure_fresh(),
which re-seeds the demo graph when its story has gone stale.
"""
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import diagnose
import seed
import tools
import verify
from db import get_graph

STATIC = Path(__file__).parent / "static"

# Terminal logging for the whole app (agent, seed, browser errors). Default
# INFO shows every model, LLM call and tool call; LOG_LEVEL=DEBUG adds more,
# LOG_LEVEL=WARNING shows only problems (use that when deployed).
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("detective.app")
# The HTTP client logs every LLM request; the agent's own lines already say the
# same, so show these only when debugging.
if logging.getLogger().level > logging.DEBUG:
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# Questions longer than this are almost surely pasted logs, not questions; the
# limit also caps how much text one request can push into the LLM prompt.
MAX_QUESTION_CHARS = 500

app = FastAPI(title="Incident Detective",
              description="On-call agent over a FalkorDB graph of services, deploys, "
                          "alerts and owners.")


class ClientError(BaseModel):
    """Body of POST /api/client-error: one JavaScript error from the browser."""
    message: str = Field(max_length=2000)
    where: str = Field("", max_length=500)
    stack: str = Field("", max_length=4000)


class Ask(BaseModel):
    """Body of POST /api/ask."""
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS,
                          examples=["checkoutservice is throwing 5xx errors. What broke?"])


def _fresh_graph():
    """Return the graph handle, re-seeding the demo story first if it is stale.

    AUTO_RESEED=0 turns the re-seed off (same switch as agent.investigate()).
    """
    g = get_graph()
    if os.getenv("AUTO_RESEED", "1") == "1":
        seed.ensure_fresh(g)
    return g


@app.get("/", include_in_schema=False)
def index():
    """Serve the one-page UI."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    """Report whether FalkorDB answers a trivial read-only query."""
    try:
        get_graph().ro_query("RETURN 1")
        return {"ok": True}
    except Exception as e:  # any connection or server error means "not healthy"
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/graph")
def graph(minutes: int = 60):
    """Return the nodes and edges the UI draws.

    Deploys and alerts are limited to the last `minutes` (open alerts only), the
    same window recent_changes() uses, so the picture matches what the agent sees.
    Alerts are not separate nodes: each service carries its open alerts and a
    "health" (its worst alert severity), which the UI shows as the node's border.
    Teams and people are not drawn up front either; each service carries its
    owner team and on-call person, and the UI adds them when who_to_page runs.

    Returns:
        {"nodes": [service and deploy nodes, see build_graph()],
         "edges": [{"from", "to", "rel", "latency_ms"?}, ...],
         "counts": {"services", "deploys", "alerts"}}
    """
    return build_graph(_fresh_graph(), minutes)


# Alert severities from worst to mildest; a service's health is its worst open alert.
SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def build_graph(g, minutes=60):
    """Read the drawable topology from `g`. See graph() for the overall shape.

    Node shapes:
      service: {"id": "svc:<name>", "kind": "service", "label", "tier", "criticality",
                "team", "on_call", "alerts": [{"severity", "message", "minutes_ago"}],
                "health": worst severity or None}
      deploy:  {"id": "deploy:<service>@<version>", "kind": "deploy", "label",
                "service", "version", "by", "minutes_ago"}
    Edges: DEPENDS_ON (caller -> callee, with latency_ms) and TO (deploy -> service).
    """
    def rows(cypher, params=None):
        return g.ro_query(cypher, params, timeout=tools.TIMEOUT_MS).result_set

    recent = tools.recent_changes(g, minutes)
    alerts = {}
    for a in recent["alerts"]:
        alerts.setdefault(a["service"], []).append(
            {k: a[k] for k in ("severity", "message", "minutes_ago")})

    nodes, edges = [], []
    services = rows("MATCH (s:Service) "
                    "OPTIONAL MATCH (t:Team)-[:OWNS]->(s) "
                    "OPTIONAL MATCH (p:Person {on_call: true})-[:MEMBER_OF]->(t) "
                    "RETURN s.name, s.tier, s.criticality, t.name, p.name ORDER BY s.name")
    for name, tier, crit, team, on_call in services:
        mine = alerts.get(name, [])
        worst = min((SEVERITY_ORDER.index(a["severity"]) for a in mine
                     if a["severity"] in SEVERITY_ORDER), default=None)
        nodes.append({"id": f"svc:{name}", "label": name, "kind": "service", "tier": tier,
                      "criticality": crit, "team": team, "on_call": on_call, "alerts": mine,
                      "health": None if worst is None else SEVERITY_ORDER[worst]})
    for a, b, w in rows("MATCH (a:Service)-[r:DEPENDS_ON]->(b:Service) "
                        "RETURN a.name, b.name, r.weight"):
        edges.append({"from": f"svc:{a}", "to": f"svc:{b}", "rel": "DEPENDS_ON", "latency_ms": w})
    for d in recent["deploys"]:
        node = f"deploy:{d['service']}@{d['version']}"
        nodes.append({"id": node, "label": d["version"], "kind": "deploy", "service": d["service"],
                      "version": d["version"], "by": d["by"], "minutes_ago": d["minutes_ago"]})
        edges.append({"from": node, "to": f"svc:{d['service']}", "rel": "TO"})

    return {"nodes": nodes, "edges": edges,
            "counts": {"services": len(services), "deploys": len(recent["deploys"]),
                       "alerts": len(recent["alerts"])}}


@app.post("/api/client-error", status_code=204)
def client_error(body: ClientError):
    """Print a browser JavaScript error in the server terminal.

    The page sends uncaught errors and failed requests here, so problems show
    up next to the server's own logs instead of only in the browser console.
    """
    log.error("browser: %s%s%s", body.message, f" at {body.where}" if body.where else "",
              f"\n{body.stack}" if body.stack else "")


@app.post("/api/ask")
def ask(body: Ask):
    """Answer one incident question with the agent.

    Plain `def`, so FastAPI runs it in a worker thread: one slow model call does
    not block other requests.

    Returns:
        Everything agent.investigate() returns, plus
        "summary": see summarize(), and
        "timeline": recent deploys and alerts, oldest first, see timeline().

    Raises:
        HTTPException 503: no LLM key is set and FalkorDB is unreachable, so not
        even the rules diagnosis can answer.
    """
    g = _fresh_graph()
    try:
        out = agent.investigate(body.question, g=g)
    except RuntimeError as e:  # no API key set at all
        log.error("ask failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    vocab = verify.graph_vocabulary(g)
    out["summary"] = summarize(out, vocab, body.question)
    out["timeline"] = timeline(g, out["summary"])
    return out


def summarize(out, vocab, question=""):
    """Pull the facts the UI highlights out of one investigate() result.

    Uses only tool rows, plus the answer text to find which services the answer
    called unrelated and which deploy it blamed; anything the answer names that
    the tools did not return is ignored.

    Args:
        out: Return value of agent.investigate().
        vocab: verify.graph_vocabulary(g).
        question: The user's question, for "missing" below.

    Returns:
        {"path": [service, ...] or None,     last dependency path found
         "related": [service, ...],          services the failing service depends on, plus the path
         "root_cause": {"service", "version", "by"} or None,
         "unrelated": [service, ...],        named under "Unrelated" in the answer
         "page": {"service", "team", "on_call"} or None,  service = the one who_to_page was asked about
         "pages": [page, ...],               every who_to_page result, in order (several
                                             failing services, one owner each)
         "no_cause": bool,                   the answer says no root cause was found, so the
                                             card says that instead of a blast radius
         "missing": [name, ...],             services, versions, people and teams the question
                                             names that the graph does not have, e.g.
                                             "billing-service", "Sarah Jenkins", "SRE"
         "rollback": bool}                   the question asks what a rollback would fix
    """
    steps = out["steps"]
    path = out.get("path")
    related = set(path or [])
    # Services a tool returned as a dependency, a caller or a path node. The
    # service a tool was asked about is in `related` but not here: asking
    # dependencies(adservice) does not put adservice on the failure chain.
    chain = set(related)
    deploys, page, pages = [], None, []
    for s in steps:
        r = s["result"]
        if "error" in r:
            continue
        if s["tool"] == "dependencies":
            related.add(r["service"])
            related.update(d["service"] for d in r["depends_on"])
            chain.update(d["service"] for d in r["depends_on"])
        elif s["tool"] == "blast_radius":  # callers are effects, not unrelated
            related.add(r["service"])
            related.update(d["service"] for d in r["affected"])
            chain.update(d["service"] for d in r["affected"])
        elif s["tool"] == "recent_changes":
            deploys = r["deploys"]
        elif s["tool"] == "who_to_page":
            page = {"service": r["service"], "team": r["team"], "on_call": r["on_call"]}
            if page not in pages:
                pages.append(page)

    # The blamed deploy: one the tools returned whose version and service the
    # Root cause section names. If several match, the first (newest) wins.
    cause_text = verify._section(out["answer"].lower(), "root cause")
    no_cause = bool(re.match(verify._NO_CAUSE, cause_text))
    if no_cause:
        cause_text = ""  # "Root cause: not found in the graph." blames nothing
    # A sentence that clears a deploy ("fraud-check v3.1.0 ... has no
    # dependency path to or from adservice") does not blame it (live C3).
    cause_text = " ".join(sent for sent in re.split(r"(?<=[.!?])\s+|\n", cause_text)
                          if not re.search(verify._CLEARS, sent))
    root = next((d for d in deploys
                 if d["version"].lower() in cause_text and d["service"].lower() in cause_text), None)

    answer = out["answer"].lower()
    unrelated_text = verify._section(answer, "unrelated")
    named = verify._mentions(unrelated_text, vocab["services"])
    if not named:
        # No "Unrelated:" line, so read prose: "the adservice alert is
        # unrelated to the checkout outage" names adservice before the word.
        for sent in re.split(r"(?<=[.!?])\s+|\n", answer):
            m = re.search(r"\b(?:unrelated|not related|separate incident)\b", sent)
            if m:
                named |= verify._mentions(sent[:m.start()], vocab["services"])
    # The answer's verdict wins for a service a tool was only asked about,
    # but not for one the tools put on the chain or the blamed deploy's.
    keep = chain | ({root["service"]} if root else set())
    # Anything that calls the blamed service, or that it calls, is on the
    # incident chain even if no tool returned it (live F4: checkoutservice
    # was shown as unrelated to a fraud-check root cause).
    reach = vocab.get("reach") or {}
    if root:
        keep |= reach.get(root["service"], set())
        keep |= {svc for svc, deps in reach.items() if root["service"] in deps}
    if no_cause:
        # Live C3: "adservice cache miss rate is high" found no cause, and the
        # card listed adservice itself as "Not related" to its own question.
        keep |= verify._asked_services(question.lower(), vocab["services"])
    unrelated = sorted(named - keep)
    related -= set(unrelated)
    # Names in the question the graph lacks, so the card can say so up front
    # instead of leading with an answer about something else.
    q = question.lower()
    missing = [diagnose._as_written(question, name)
               for _, name in verify.unknown_names(question, vocab)]
    return {"path": path, "related": sorted(related),
            "root_cause": root and {k: root[k] for k in ("service", "version", "by")},
            "unrelated": unrelated, "page": page, "pages": pages,
            "no_cause": no_cause and not root, "missing": missing,
            "rollback": bool(re.search(r"\broll(?:ing|ed)?[\s-]*back|\brevert", q))}


def timeline(g, summary, minutes=60):
    """List recent deploys and open alerts, oldest first, for the timeline panel.

    Each event is marked "related" when its service is on the failure chain
    (summary["related"]), so the UI can grey out the rest.

    Returns:
        [{"kind": "deploy"|"alert", "service", "text", "minutes_ago",
          "severity" (alerts only), "related": bool}, ...]
    """
    recent = tools.recent_changes(g, minutes)
    related = set(summary["related"])
    events = [{"kind": "deploy", "service": d["service"],
               "text": f"{d['service']} {d['version']} deployed by {d['by']}",
               "minutes_ago": d["minutes_ago"], "related": d["service"] in related}
              for d in recent["deploys"]]
    events += [{"kind": "alert", "service": a["service"], "text": a["message"],
                "severity": a["severity"], "minutes_ago": a["minutes_ago"],
                "related": a["service"] in related}
               for a in recent["alerts"]]
    return sorted(events, key=lambda e: -e["minutes_ago"])
