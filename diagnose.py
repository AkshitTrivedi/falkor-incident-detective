"""Diagnose an incident with fixed rules, no LLM.

Used when no model answer can be trusted: the model's answer failed
verify.check_answer() twice, every model in the chain failed, or the question
ran out of time. The rules are the same steps the system prompt in agent.py
asks the model to follow, so for a "what broke?" question they reach the same
answer:

  1. The failing service is the one service named in the question.
  2. Root cause: the one recent deploy to that service or to something it
     depends on (tools.dependencies).
  3. Evidence: the dependency path to that deploy, and the open alerts on the
     failing service and its dependencies.
  4. Unrelated: open alerts on services outside its dependencies.
  5. Page: who_to_page on the root-cause service.

Two other question types get their own rule, so the fallback still answers
the demo's other questions:

  * "What breaks if X goes down?" (blast radius): tools.blast_radius on X,
    then who owns X.
  * "Which services are most critical?": tools.critical_services ranking.

When the rules cannot decide (no service or several services in the
question, no recent deploy among the dependencies, or more than one), the
answer says why and lists the tool rows instead (verify.plain_answer), so it
never guesses.
"""
import re
import time

import tools
import verify

# Question types the rules recognise besides "what broke?". Kept narrow on
# purpose: a question that matches neither is treated as "what broke?".
_IMPACT = re.compile(r"\bgo(?:es)? down\b|\bbreaks? if\b|\bblast radius\b|\bwho depends on\b"
                     r"|\bwhat (?:else )?(?:is|would be|gets) (?:impacted|affected)\b")
_CRITICAL = re.compile(r"\b(?:most|which)\b.{0,30}\b(?:critical|important|central)\b"
                       r"|\bsingle points? of failure\b")


def _step(steps, name, *args):
    """Run one tool, record it in `steps` like the agent does, and return its result."""
    t0 = time.time()
    result = tools.TOOLS[name](*args)
    arg_names = {"recent_changes": [], "dependencies": ["service"], "blast_radius": ["service"],
                 "critical_services": [],
                 "dependency_path": ["from_service", "to_service"],
                 "who_to_page": ["service"]}[name]
    steps.append({"tool": name, "args": dict(zip(arg_names, args[1:])), "result": result,
                  "ms": round((time.time() - t0) * 1000)})
    return result


def failing_services(g, question):
    """Return the real service names that appear in `question`, as whole words.

    Args:
        g: FalkorDB Graph handle.
        question: The user's question.

    Returns:
        Sorted list of service names (exact graph spelling).
    """
    q = question.lower()
    names = [row[0] for row in g.ro_query("MATCH (s:Service) RETURN s.name").result_set]
    return sorted(n for n in names
                  if re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", q))


def unknown_services(g, question):
    """Return service-like words in `question` that are not services in the graph.

    Uses the same pattern as the answer checker (verify._SERVICE_LIKE), so
    "billing-service" or "auth-api" count, but "microservice" does not.

    Args:
        g: FalkorDB Graph handle.
        question: The user's question.

    Returns:
        Sorted list of the unknown names, as written in the question (lower case).
    """
    q = question.lower()
    real = {row[0].lower() for row in g.ro_query("MATCH (s:Service) RETURN s.name").result_set}
    return sorted({w for w in verify._SERVICE_LIKE.findall(q)
                   if w not in real and w not in verify._NOT_SERVICES})


def rule_answer(g, question, model_steps=()):
    """Answer `question` with the rules in the module docstring.

    Args:
        g: FalkorDB Graph handle.
        question: The user's question.
        model_steps: Tool steps the model already ran, if any. Shown when the
            rules cannot decide, since they match what the question asked.

    Returns:
        {"answer": str, "decided": bool, "steps": [...], "path": [...] or None}
        "steps" are the tool calls the rules made, in the agent's step format.
    """
    steps = []
    q = question.lower()
    if _CRITICAL.search(q):
        return _critical(g, steps)
    named = failing_services(g, question)
    if len(named) == 1 and _IMPACT.search(q):
        return _blast(g, named[0], steps)
    if len(named) != 1:
        unknown = unknown_services(g, question) if not named else []
        reason = (f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not a service "
                  "in the graph" if unknown else
                  "no service is named in the question" if not named else
                  f"the question names several services ({', '.join(named)})")
        return _undecided(g, reason, steps, model_steps)

    failing = named[0]
    changes = _step(steps, "recent_changes", g)
    deps = {d["service"] for d in _step(steps, "dependencies", g, failing)["depends_on"]}
    suspects = [d for d in changes["deploys"] if d["service"] in deps | {failing}]
    if len(suspects) != 1:
        reason = (f"no recent deploy to {failing} or anything it depends on" if not suspects
                  else f"{len(suspects)} recent deploys could explain it")
        return _undecided(g, reason, steps, model_steps)

    cause = suspects[0]
    path = None
    if cause["service"] != failing:
        path = _step(steps, "dependency_path", g, failing, cause["service"]).get("path")
    page = _step(steps, "who_to_page", g, cause["service"])
    related = [a for a in changes["alerts"] if a["service"] in deps | {failing}]
    unrelated = [a for a in changes["alerts"] if a["service"] not in deps | {failing}]

    def alert(a):
        """One alert as 'service "message" (N min ago)'."""
        return f'{a["service"]} "{a["message"]}" ({a["minutes_ago"]} min ago)'

    lines = [
        "Diagnosis by rules (no model):",
        f"Root cause: {cause['service']} {cause['version']}, deployed by {cause['by']} "
        f"{cause['minutes_ago']} min ago. It is the only recent deploy to {failing} "
        "or a service it depends on.",
        "Evidence: " + (" → ".join(path) if path else f"the deploy is to {failing} itself")
        + (". Alerts on it and its dependencies: " + "; ".join(map(alert, related))
           if related else ". No open alerts on it or its dependencies."),
        "Unrelated: " + ("; ".join(alert(a) + f", {a['service']} is not a dependency "
                                   f"of {failing}" for a in unrelated) or "none"),
        f"Page: {page['team']} team, on call {page['on_call'] or 'nobody listed'}.",
    ]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": path}


def _blast(g, service, steps):
    """Answer "what breaks if `service` goes down?" with blast_radius and its owner."""
    affected = _step(steps, "blast_radius", g, service)["affected"]
    page = _step(steps, "who_to_page", g, service)
    hit = ", ".join(f"{a['service']} ({a['hops']} hop{'s' if a['hops'] > 1 else ''})"
                    for a in affected)
    lines = ["Diagnosis by rules (no model):",
             f"Blast radius: if {service} goes down, {len(affected)} service"
             f"{'' if len(affected) == 1 else 's'} that depend on it can fail: {hit or 'none'}.",
             f"Page: {page['team']} team owns {service}, on call {page['on_call'] or 'nobody listed'}."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _critical(g, steps):
    """Answer "which services are most critical?" with the betweenness ranking."""
    ranking = _step(steps, "critical_services", g)["ranking"]
    top = "; ".join(f"{i}. {r['service']} ({r['score']:g})" for i, r in enumerate(ranking, 1))
    lines = ["Diagnosis by rules (no model):",
             "Most critical services, by betweenness (how many dependency paths run "
             f"through them): {top}."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _undecided(g, reason, steps, model_steps):
    """Build rule_answer()'s result when the rules cannot pick a root cause.

    Shows the model's tool rows if it ran any, else the rules' own; if neither
    ran anything yet, fetches recent_changes so there is something to look at.
    """
    if not model_steps and not steps:
        _step(steps, "recent_changes", g)
    rows = verify.plain_answer(list(model_steps) or steps).split("\n", 1)[1]
    answer = (f"No trusted model answer, and the rules could not decide: {reason}. "
              f"What the graph returned:\n{rows}")
    return {"answer": answer, "decided": False, "steps": steps, "path": None}
