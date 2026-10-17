"""Check an LLM's final answer against the tool results it was given.

The agent's tools only return facts from the graph, but the model writes the
final answer itself, and it can still state things no tool returned. The worst
case seen in testing (docs/MODEL_EVALUATION.md, question O): Codestral blamed
"a paymentservice deploy by Priya Nair". Both names exist in the graph, yet that
deploy does not; the real one is fraud-check v3.1.0 by Daniel Okafor.

check_answer() returns a list of problems; an empty list means the answer
passed. The rules are deliberately narrow, so a correct answer is not flagged:

  1. A person or deploy version named anywhere must appear in a tool result.
  2. A word that looks like a service name ("payment-service", "orders-db")
     must be a real service, or be quoted from the question.
  3. Root cause: a version named must be one of the returned deploys, and if
     people are named, at least one must be the deployer of a returned deploy.
  4. Page: if people are named, at least one must be a returned on-call person.
  5. Unrelated: must not list an alerting service that the tools showed the
     failing service depends on (its dependencies() list or the dependency
     path). Only services with an open alert count, so naming the root cause
     for comparison ("unrelated to fraud-check") is fine.

agent.py uses the problems to ask the model for one corrected answer. If that
one fails too, it shows plain_answer() instead: the tool rows, with no model text.
"""
import json
import re

import tools

# Section labels in the answer format the system prompt asks for, e.g.
# "Root cause:", "**Page:**", "- Evidence (graph):". Same pattern as bakeoff.py.
_LABEL = r"(?:^|\n)[*_#\s-]*(?:{})[^:\n]{{0,20}}:"
_SECTION_NAMES = "root cause|evidence|unrelated|page"

# Looks like a service name: ends in "service", "-service", "-db" or "-api".
# Ordinary hyphenated words ("on-call") and real names without those endings
# ("fraud-check") don't match, so only rule 1's people/versions cover those.
_SERVICE_LIKE = re.compile(r"(?<![\w-])([a-z][a-z0-9]*(?:-[a-z0-9]+)*?(?:-?service|-db|-api))(?![\w-])")
# Words that match _SERVICE_LIKE but are not service names. The hyphenated ones
# came up in real answers ("cross-service calls", Qwen, Story B).
_NOT_SERVICES = {"microservice", "micro-service", "cross-service", "inter-service",
                 "multi-service", "per-service", "service-to-service", "web-service"}


def graph_vocabulary(g):
    """Fetch every service name, person name and deploy version in the graph.

    Args:
        g: FalkorDB Graph handle.

    Returns:
        {"services": set, "people": set, "versions": set}, all lower-case.
    """
    def column(cypher):
        return {row[0].lower() for row in g.ro_query(cypher).result_set if row[0]}
    return {"services": column("MATCH (s:Service) RETURN s.name"),
            "people": column("MATCH (p:Person) RETURN p.name"),
            "versions": column("MATCH (d:Deploy) RETURN DISTINCT d.version")}


def _section(text, name):
    """Return the text of section `name` ("root cause", "page"...), or "" if absent."""
    start = re.search(_LABEL.format(name), text)
    if not start:
        return ""
    rest = text[start.end():]
    end = re.search(_LABEL.format(_SECTION_NAMES), rest)
    return rest[:end.start()] if end else rest


def _mentions(text, names):
    """Return the names from `names` that appear in `text` as whole words.

    "Whole word" means not glued to letters, digits or hyphens, so
    "paymentservice" does not match inside "paymentservices" and "v3.1.0" does
    not match inside "v3.1.0-rc".
    """
    return {n for n in names
            if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text)}


def _returned(steps):
    """Collect the facts the tools returned, for the root-cause and page rules.

    Args:
        steps: The agent's steps, [{"tool", "args", "result", "ms"}, ...].

    Returns:
        {"text": all results as one lower-case string,
         "deployers": set, "versions": set, "on_call": set,
         "chains": [(service, {services it depends on}), ...],
         "alerting": services with an open alert}, all lower-case.
    """
    facts = {"text": "", "deployers": set(), "versions": set(), "on_call": set(),
             "chains": [], "alerting": set()}
    for step in steps:
        result = {k: v for k, v in step["result"].items() if k not in tools.TRACE_KEYS}
        facts["text"] += json.dumps(result).lower() + "\n"
        for d in result.get("deploys", []):
            facts["deployers"].add(str(d.get("by", "")).lower())
            facts["versions"].add(str(d.get("version", "")).lower())
        for al in result.get("alerts", []):
            facts["alerting"].add(str(al.get("service", "")).lower())
        if result.get("on_call"):
            facts["on_call"].add(result["on_call"].lower())
        # (service, the services it depends on), for rule 5.
        if "depends_on" in result:
            facts["chains"].append((result["service"].lower(),
                                    {d["service"].lower() for d in result["depends_on"]}))
        if result.get("path"):
            path = [p.lower() for p in result["path"]]
            facts["chains"].append((path[0], set(path[1:])))
    return facts


def check_answer(answer, steps, question, vocab):
    """List the claims in `answer` that the tool results do not support.

    Args:
        answer: The model's final answer.
        steps: The tool steps that led to it (agent.investigate()'s "steps").
        question: The user's question; names quoted from it are allowed.
        vocab: graph_vocabulary() for the graph the tools ran on.

    Returns:
        [str, ...] one plain-English problem per unsupported claim; [] if none.
    """
    a = answer.lower().replace("’", "'").replace("‑", "-")
    q = question.lower()
    facts = _returned(steps)
    problems = []

    # Rule 1: people and versions must come from a tool result.
    for name in sorted(_mentions(a, vocab["people"] | vocab["versions"])):
        if name not in facts["text"]:
            problems.append(f"'{name}' is named, but no tool returned it")

    # Rule 2: service-like words must be real services (or come from the question).
    for word in sorted(set(_SERVICE_LIKE.findall(a))):
        if word not in vocab["services"] and word not in _NOT_SERVICES and word not in q:
            problems.append(f"'{word}' is not a service in the graph")

    # Rule 3: the root cause must be a deploy the tools returned.
    cause = _section(a, "root cause")
    for version in sorted(_mentions(cause, vocab["versions"]) - facts["versions"]):
        problems.append(f"root cause names version '{version}', "
                        "which is not a recent deploy returned by recent_changes")
    cause_people = _mentions(cause, vocab["people"])
    if cause_people and not cause_people & facts["deployers"]:
        problems.append(f"root cause names {sorted(cause_people)}, but none of them "
                        "deployed a change returned by recent_changes")

    # Rule 4: the person to page must be an on-call person who_to_page returned.
    page_people = _mentions(_section(a, "page"), vocab["people"])
    if page_people and not page_people & facts["on_call"]:
        problems.append(f"page names {sorted(page_people)}, but who_to_page did not "
                        "return them as on call")

    # Rule 5: "Unrelated" must not list an alerting dependency of a service the question names.
    # The named service itself is left out, as answers say "X is not a dependency of it".
    asked = _mentions(q, vocab["services"])
    on_chain = set().union(*[deps for svc, deps in facts["chains"] if svc in asked]) - asked
    on_chain &= facts["alerting"]
    wrongly_unrelated = _mentions(_section(a, "unrelated"), on_chain)
    if wrongly_unrelated:
        problems.append(f"unrelated lists {sorted(wrongly_unrelated)}, but the tools show "
                        "the failing service depends on them")
    return problems


def plain_answer(steps):
    """Build an answer from the tool results alone, with no model text.

    Used when the model's answer fails check_answer() twice. It does not pick a
    root cause; it lists what the graph returned so the on-call engineer can
    judge. Every line comes straight from a tool result.

    Args:
        steps: The agent's steps, [{"tool", "args", "result", "ms"}, ...].

    Returns:
        str: Markdown text. Says so plainly if the tools returned nothing.
    """
    lines = ["The model's answer did not match the graph results, so here is "
             "what the graph returned, with no model text:"]
    for step in steps:
        r = step["result"]
        if r.get("error"):
            lines.append(f"- {step['tool']}: {r['error']}")
        for d in r.get("deploys", []):
            lines.append(f"- Deploy: {d['service']} {d['version']} by {d['by']}, "
                         f"{d['minutes_ago']} min ago")
        for a in r.get("alerts", []):
            lines.append(f"- Alert ({a['severity']}): {a['service']}: {a['message']}, "
                         f"{a['minutes_ago']} min ago")
        if r.get("path"):
            lines.append("- Path: " + " → ".join(r["path"]))
        elif r.get("note"):
            lines.append(f"- {step['tool']}: {r['note']}")
        if "depends_on" in r:
            lines.append(f"- {r['service']} depends on: "
                         + (", ".join(d["service"] for d in r["depends_on"]) or "nothing"))
        if "affected" in r:
            lines.append(f"- If {r['service']} fails, affected: "
                         + (", ".join(d["service"] for d in r["affected"]) or "nothing"))
        if r.get("ranking"):
            lines.append(f"- Top by {r['metric']}: "
                         + ", ".join(f"{x['service']} ({x['score']:g})" for x in r["ranking"]))
        if r.get("on_call"):
            lines.append(f"- Page: {r['team']} team, on call {r['on_call']} "
                         f"(for {r['service']})")
    if len(lines) == 1:
        lines.append("- No tool returned any rows.")
    return "\n".join(lines)
