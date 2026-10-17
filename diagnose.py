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
  4. Also affected: open alerts on services that call it (tools.blast_radius).
     They are effects of the same failure. A rollback question ("If we roll
     back fraud-check, which alerts clear?") names the deployed service, so its
     callers' alerts belong here, not under Unrelated (live test, Oct 16).
     Unrelated: open alerts on services with no path to it in either direction.
  5. Page: who_to_page on the root-cause service.

Two other question types get their own rule, so the fallback still answers
the demo's other questions:

  * "What breaks if X goes down?" (blast radius): tools.blast_radius on X,
    then who owns X.
  * "Which services are most critical?": tools.critical_services ranking.
  * "Who is on call for payments?" (a team, no service named): who_to_page
    on the team.
  * "Who is Daniel Okafor?" / "who is priya?" (a person in the graph): their
    team, who is on call there, and their deploys in the last hour.
  * Not about the graph ("hello", "who is akshit?", "write a poem"): a short
    reply saying what it can answer; no tool runs.
  * A request to change data ("delete all alerts", injected Cypher): says
    every query is read-only; no tool runs.

When the rules cannot decide (no service or several services in the
question, no recent deploy among the dependencies, or more than one), the
answer says why and lists the tool rows instead (verify.plain_answer), so it
never guesses.
"""
import difflib
import re
import time

import tools
import verify

# Question types the rules recognise besides "what broke?". Kept narrow on
# purpose: a question that matches neither is treated as "what broke?".
_IMPACT = re.compile(r"\bgo(?:es)? down\b|\bbreaks? if\b|\bblast radius\b|\b(?:who|what) depends on\b"
                     r"|\bwhat (?:else )?(?:is|would be|gets) (?:impacted|affected)\b")
_ON_CALL = re.compile(r"\bon[- ]?call\b|\bwho (?:do|should) i page\b|\bwho to page\b")
_OWNER = re.compile(r"\bwho owns\b|\bteam owns\b|\bowned by\b|\bowners? of\b|\bwho(?: is|'s) on[- ]?call\b"
                    r"|\bwho is responsible\b|\bwho (?:do|should) i (?:page|call|wake)\b|\bwho to page\b")
_DEPS_OF = re.compile(r"\bwhat does \S+ (?:depend on|call|need|use)\b|\bdependencies of\b"
                      r"|\bwhat is \S+ dependent on\b|\b\S+'s dependencies\b")
_DEPENDS = re.compile(r"\bdepends? on\b|\bdependent on\b|\bpath\b|\bcalls?\b|\breach\b")
_RELATED = re.compile(r"\brelated\b|\bconnected\b|\blinked\b|\bcaused? by\b|\bbecause of\b")
_METRIC = re.compile(r"(?:\bwhat(?:'s| is| are) the\b|\bhow (?:much|high|many)\b|\bshow(?: me)? the\b"
                     r"|\bcurrent\b).{0,30}\b(?:memory|cpu|disk|p\d\d|latency|throughput|qps|rps|"
                     r"usage|utili[sz]ation|metrics?|load)\b")
# Words that make a question about an incident, not about ownership.
_INCIDENT = re.compile(r"\b(?:broke|broken|break|breaking|error|errors|5xx|4xx|fail|failing|failed|"
                       r"failure|down|outage|slow|latency|timeout|timeouts|alert|alerts|incident|"
                       r"issue|issues|problem|problems|causing|cause|why|wrong|spik\w*|degraded|"
                       r"crash\w*|restart\w*|cpu|memory|can't|cannot|unable|not working|"
                       r"wake up|odd|weird|strange|stuck|hanging|unavailable)\b")
# "What changed?", and the vague way on-call people ask the same thing:
# "whats up with prod", "is everything ok?", "status?", "something is off".
_CHANGED = re.compile(r"\bwhat(?:'s| is)? (?:changed|happened|is going on|'s going on|going on|up)\b"
                      r"|\bany (?:recent )?(?:changes|deploys)\b|\bprod\b|\bproduction\b|\bstatus\b"
                      r"|\b(?:everything|all|things|anything) (?:ok|okay|fine|good|healthy|alright)\b"
                      r"|\bsomething(?:'s| is)? (?:off|up|wrong|weird|broken)\b|\bhealth\w*|\bon fire\b"
                      r"|\bsev ?[0-3]\b|\bp[0-2]\b|\bpaged\b|\bpager\b")
_ROLLBACK = re.compile(r"\broll(?:ing|ed)?[\s-]*back|\brevert")
# Words that make a question about the incident graph. A question with none
# of them and no service name ("hello there", "who is akshit?", "write me a
# poem") gets _off_topic() instead of a dump of tool rows (live test, Oct 16).
_ON_TOPIC = re.compile(r"\b(?:alert|alerts|deploy|deploys|deployed|deployment|release|version|"
                       r"service|services|incident|outage|down|slow|latency|timeout|timeouts|"
                       r"cpu|memory|metric|metrics|database|db|cache|graph|depend\w*|upstream|"
                       r"downstream|root cause|cause|caused|why|owner|owns|team|on[- ]?call|page|"
                       r"paging|broke|broken|break\w*|error|errors|5xx|4xx|fail\w*|crash\w*|"
                       r"changed?|happened|going on|roll\w*|revert\w*|critical|impact\w*|"
                       r"blast|customers?|users?|checkout|pay\w*|cart|login|orders?|search)\b")
# A request to change the graph ("run MATCH (n) DETACH DELETE n"). The tools
# only read, so the answer says so instead of pretending to try.
_WRITE = re.compile(r"^\W*(?:please\s+|can you\s+|could you\s+)?(?:delete|remove|drop|wipe|"
                    r"clear|truncate|create|insert|add|update|set|resolve|close|mute|silence|"
                    r"ack(?:nowledge)?|modify|change|edit|rename|reseed|reset)\b"
                    r"|\bdetach\s+delete\b|\b(?:create|merge)\s*\(|\bset\s+\w+\.\w+\s*="
                    r"|\bdelete\s+\w+\s*$|\bignore\s+(?:your|all|the|previous|prior)\b")
_CRITICAL = re.compile(r"\b(?:most|which)\b.{0,30}\b(?:critical|important|central)\b"
                       r"|\bsingle points? of failure\b")


def _step(steps, name, *args):
    """Run one tool, record it in `steps` like the agent does, and return its result.

    A call already in `steps` with the same arguments is reused, not run again,
    so a question about several services shows recent_changes once.
    """
    arg_names = {"recent_changes": [], "dependencies": ["service"], "blast_radius": ["service"],
                 "critical_services": [],
                 "dependency_path": ["from_service", "to_service"],
                 "who_to_page": ["service"]}[name]
    call_args = dict(zip(arg_names, args[1:]))
    for done in steps:
        if done["tool"] == name and done["args"] == call_args:
            return done["result"]
    t0 = time.time()
    result = tools.TOOLS[name](*args)
    steps.append({"tool": name, "args": call_args, "result": result,
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

    Names the question uses that the graph lacks ("billing-service", "SRE
    team", "Sarah Jenkins", "v2.0.0") are called out first, whatever rule
    answers the rest.

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
    out = _route(g, question, steps, model_steps)
    notes = "".join(f"There is no {_KIND[kind].format(_as_written(question, name))} in the graph.\n"
                    for kind, name in verify.unknown_names(question, verify.graph_vocabulary(g)))
    notes += _premise_notes(g, question, out["answer"])
    if notes:
        head, sep, rest = out["answer"].partition(_HEADER)
        out["answer"] = _HEADER + notes + rest if sep else notes + out["answer"]
    return out


def needs_no_model(g, question):
    """True when rule_answer() answers best without a model.

    Three kinds, all with one right reply that a model can only get wrong:
      * a request to change data ("Delete all alerts", injected Cypher);
      * nothing from the graph in it ("hello there", "who is akshit?");
      * the only service or team named is not in the graph ("billing-service
        is down", "who is on call for SRE?"). Live test, Oct 16: StepFun said
        billing-service was missing, then diagnosed paymentservice instead;
      * "who is <person>?" No tool returns a person's team, so models guessed:
        live Oct 16, StepFun said Daniel Okafor was not in the graph, and
        called Ravi Kumar's orders-db deploy "Payments-related".
    agent.investigate() answers these with the rules straight away.
    """
    q = question.lower().replace("\u2019", "'")
    if _WRITE.search(q) or _off_topic_question(g, question):
        return True
    if _WHO_IS.search(q) and _named_person(g, q) and not failing_services(g, question):
        return True
    if failing_services(g, question) or aliased_services(g, question):
        return False
    unknown = {kind for kind, _ in verify.unknown_names(question, verify.graph_vocabulary(g))}
    return "service" in unknown or ("team" in unknown and bool(_ON_CALL.search(q) or _OWNER.search(q))
                                    and not _INCIDENT.search(q))


def _off_topic_question(g, question):
    """True when the question has nothing from the graph in it ("hello there")."""
    q = question.lower().replace("\u2019", "'")
    if any(r.search(q) for r in (_ON_TOPIC, _INCIDENT, _CHANGED, _CRITICAL, _METRIC, _IMPACT,
                                 _ON_CALL, _OWNER, _ROLLBACK)):
        return False
    return not (failing_services(g, question)
                or aliased_services(g, question) or _named_person(g, q)
                or _named_team(g, q) or unknown_services(g, question))


_WHO_IS = re.compile(r"^\W*(?:who(?:'s| is)|tell me about)\b")
_ROOT_LINE = re.compile(r"Root cause: (\S+) (\S+), deployed by (.+?) \d+ min ago")
_DEPLOY_TO = re.compile(r"\bdeploy(?:ed|ment)?\s+(?:to|of|on|for)\s+([a-z][\w-]*)")


def _premise_notes(g, question, answer):
    """Correct a wrong premise about the root-cause deploy, one line each.

    "Mei Lin deployed fraud-check v3.1.0, right?" -> Mei Lin made no deploy in
    the last hour; Daniel Okafor did. "Daniel Okafor's deploy to
    paymentservice" -> the deploy was to fraud-check. Both people and services
    exist, so verify.unknown_names() has nothing to say (live test, Oct 16).
    """
    m = _ROOT_LINE.search(answer)
    q = question.lower().replace("\u2019", "'")
    if not m or "deploy" not in q:
        return ""
    service, version, by = m.groups()
    notes = ""
    person = _named_person(g, q)
    if person and person[0] != by:
        notes += f"{person[0]} has no deploy in the last hour; {service} {version} was deployed by {by}.\n"
    to = _DEPLOY_TO.search(q)
    if to and to.group(1) != service and to.group(1) in verify.graph_vocabulary(g)["services"]:
        notes += f"The recent deploy was to {service}, not {to.group(1)}.\n"
    return notes


def _as_written(question, name):
    """`name` with the capitals the question used ("Sarah Jenkins", "SRE")."""
    m = re.search(re.escape(name), question, re.I)
    return m.group(0) if m else name


# How rule_answer() names each kind of unknown name, e.g. "There is no SRE team".
_KIND = {"service": "service called {}", "version": "{} deploy", "person": "person named {}",
         "team": "{} team"}
_HEADER = "Diagnosis by rules (no model):\n"


def _route(g, question, steps, model_steps):
    """Pick the rule for `question` and run it; see rule_answer()."""
    q = question.lower().replace("\u2019", "'")
    if _WRITE.search(q):
        return _no_writes(steps)
    if _CRITICAL.search(q):
        return _critical(g, steps)
    named = failing_services(g, question)
    if not named and _ON_CALL.search(q) and (team := _named_team(g, q)):
        return _team_on_call(g, team, steps, _named_person(g, q))
    if not named:
        named = aliased_services(g, question)
    elif len(named) == 1 and _RELATED.search(q):
        # "Is the adservice alert related to the checkout outage?": one exact
        # name, one everyday name.
        named = sorted(set(named) | set(aliased_services(g, question)))
    # A question about a service the graph lacks ("billing-service is down")
    # gets the list of real services, never a guess at another root cause.
    unknown = verify.unknown_names(question, verify.graph_vocabulary(g))
    if not named and any(kind == "service" for kind, _ in unknown):
        return _unknown_service(g, steps, [n for kind, n in unknown if kind == "service"])
    if (not named and any(kind == "team" for kind, _ in unknown)
            and (_ON_CALL.search(q) or _OWNER.search(q)) and not _INCIDENT.search(q)):
        return _unknown_team(g, steps)
    if not named and (_INCIDENT.search(q) or _CHANGED.search(q)):
        # "Customers can't pay. What changed?": no service named. The one recent
        # deploy is the suspect if alerts sit on its chain.
        if only := _only_suspect(g):
            return _what_broke(g, only, q, steps, model_steps, unnamed=True)
    named = sorted(named, key=lambda n: q.find(n) if n in q else len(q))  # question order
    if named and _METRIC.search(q):
        return _metric(g, named, steps)
    if len(named) == 2 and _RELATED.search(q):
        return _relation(g, *named, steps)
    if len(named) == 2 and _DEPENDS.search(q):
        return _path(g, *named, q, steps)
    if len(named) == 1 and _IMPACT.search(q):
        return _blast(g, named[0], steps)
    if len(named) == 1 and _DEPS_OF.search(q):
        return _dependencies(g, named[0], steps)
    # "checkoutservice is throwing 5xx, who do I page?" is an incident: page the
    # root cause's owner (Priya Nair), not checkoutservice's (Omar Haddad).
    if len(named) == 1 and _OWNER.search(q) and not _INCIDENT.search(q):
        return _owner(g, named[0], steps)
    if not named and (person := _named_person(g, q)) and not _INCIDENT.search(q):
        return _person(g, person, steps)
    if not named and _off_topic_question(g, question):
        return _off_topic(steps)
    if len(named) > 1 and (_INCIDENT.search(q) or _CHANGED.search(q)):
        # "checkoutservice and cartservice are both failing": diagnose the ones
        # the one recent deploy reaches, and say which it does not explain.
        if (only := _only_suspect(g)):
            reach = _chain(g, only)
            hit = [n for n in named if n in reach]
            if hit:
                out = _what_broke(g, hit[0], q, steps, model_steps)
                for n in named:
                    if n not in reach:
                        out["answer"] += (f"\n{n}: no dependency path to or from {only}, so this "
                                          f"deploy does not explain it; ask about {n} on its own.")
                return out
        # The deploy explains none of them ("cartservice and adservice are both
        # failing"): diagnose each on its own rather than dump the tool rows.
        return _each(g, named, q, steps, model_steps)
    if len(named) != 1:
        unknown = unknown_services(g, question) if not named else []
        reason = (f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not a service "
                  "in the graph" if unknown else
                  "no service is named in the question" if not named else
                  f"the question names several services ({', '.join(named)})")
        return _undecided(g, reason, steps, model_steps)
    return _what_broke(g, named[0], q, steps, model_steps)


def aliased_services(g, question):
    """Services the question names in everyday words: "checkout", "payment service".

    Only used when no exact service name is in the question. A name's stem is
    the name without "service" ("checkout" for checkoutservice, "auth" for
    auth-service); a plural "s" and a following " service" are allowed.
    Hyphenated words do not count, so "payments-db" does not mean paymentservice.

    Args:
        g: FalkorDB Graph handle.
        question: The user's question.

    Returns:
        Sorted list of service names (exact graph spelling).
    """
    q = question.lower()
    found = set()
    for (name,) in g.ro_query("MATCH (s:Service) RETURN s.name").result_set:
        stems = {re.sub(r"-?service$", "", name)} | set(_EXTRA_ALIASES.get(name, ()))
        for stem in stems:
            if stem != name and len(stem) >= 3 and re.search(
                    rf"(?<![\w-]){re.escape(stem)}s?(?:\s+service)?(?![\w-])", q):
                found.add(name)
    return sorted(found)


# Everyday names for services whose name has no "service" suffix.
_EXTRA_ALIASES = {"productcatalog": ["catalog", "product catalog"],
                  "adservice": ["ads"], "recommendation": ["recommendations"]}


def _chain(g, svc):
    """`svc`, what it depends on, and what depends on it."""
    return ({d["service"] for d in tools.dependencies(g, svc)["depends_on"]}
            | {d["service"] for d in tools.blast_radius(g, svc)["affected"]} | {svc})


def _only_suspect(g):
    """The service of the one deploy in the last hour, if open alerts sit on its chain."""
    changes = tools.recent_changes(g)
    if len(changes["deploys"]) != 1:
        return None
    svc = changes["deploys"][0]["service"]
    chain = _chain(g, svc)
    return svc if any(a["service"] in chain for a in changes["alerts"]) else None


def _what_broke(g, failing, q, steps, model_steps, unnamed=False):
    """Rules 1-5 of the module docstring for one failing service.

    If nothing it depends on changed, a recent deploy to a service that calls
    it is the suspect instead: a new release can overload what it calls
    (payments-db at CPU 98% from fraud-check's new query).
    """
    changes = _step(steps, "recent_changes", g)
    deps = {d["service"] for d in _step(steps, "dependencies", g, failing)["depends_on"]}
    callers = {d["service"] for d in _step(steps, "blast_radius", g, failing)["affected"]}
    suspects = [d for d in changes["deploys"] if d["service"] in deps | {failing}]
    upstream = False
    if not suspects:
        suspects = [d for d in changes["deploys"] if d["service"] in callers]
        upstream = bool(suspects)
    if not suspects:
        return _no_deploy(g, failing, q, changes, steps)
    if len(suspects) != 1:
        return _undecided(g, f"{len(suspects)} recent deploys could explain it", steps, model_steps)

    cause = suspects[0]
    path = None
    if cause["service"] != failing:
        path = _step(steps, "dependency_path", g, failing, cause["service"]).get("path")
    page = _step(steps, "who_to_page", g, cause["service"])
    chain = deps | callers | {failing}
    related = [a for a in changes["alerts"] if a["service"] in deps | {failing}]
    effects = [a for a in changes["alerts"] if a["service"] in callers]
    unrelated = [a for a in changes["alerts"] if a["service"] not in chain]

    def alert(a):
        """One alert as 'service "message" (N min ago)'."""
        return f'{a["service"]} "{a["message"]}" ({a["minutes_ago"]} min ago)'

    why = (f"No service was named; it is the only deploy in the last hour, and the open alerts "
           f"below sit on its dependency chain" if unnamed else
           f"It is the only recent deploy to a service that calls {failing}; a new release "
           f"can overload what it calls" if upstream else
           f"It is the only recent deploy to {failing} or a service it depends on")
    lines = [
        _HEADER.strip(),
        f"Root cause: {cause['service']} {cause['version']}, deployed by {cause['by']} "
        f"{cause['minutes_ago']} min ago. {why}.",
        "Evidence: " + (" → ".join(path) if path else f"the deploy is to {failing} itself")
        + (". Alerts on it and its dependencies: " + "; ".join(map(alert, related))
           if related else ". No open alerts on it or its dependencies.")
        + (f". Also affected, as they call {failing}: " + "; ".join(map(alert, effects))
           if effects else ""),
    ]
    if _ROLLBACK.search(q):
        clears = related + effects
        lines.append(f"Rollback: rolling back {cause['service']} {cause['version']} should clear "
                     + ("; ".join(map(alert, clears)) if clears else "no open alerts") + ".")
    lines += [
        "Unrelated: " + ("; ".join(alert(a) + f", no dependency path to or from {failing}"
                                   for a in unrelated) or "none"),
        f"Page: {page['team']} team, on call {page['on_call'] or 'nobody listed'}.",
    ]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": path}


def _no_deploy(g, failing, q, changes, steps):
    """_what_broke() when no recent deploy touches `failing`'s chain in either direction.

    Says so plainly, lists the open alerts on `failing` and the deploys that
    did happen (none of them on its chain), and pages `failing`'s owner, since
    the problem is local to it ("adservice cache miss rate is high", "roll back
    the cartservice deploy" when there was none).
    """
    page = _step(steps, "who_to_page", g, failing)
    own = [a for a in changes["alerts"] if a["service"] == failing]
    lines = [_HEADER.strip(),
             f"Root cause: not found in the graph. No deploy in the last hour went to {failing}, "
             "to anything it depends on, or to anything that calls it."]
    if _ROLLBACK.search(q):
        lines.append(f"Rollback: there is no recent {failing} deploy to roll back.")
    # Under "Evidence:" so app.summarize() does not read these deploys as the
    # root cause (it reads the Root cause section up to the next heading).
    evidence = "Evidence: open alerts on " + failing + ": " + ("; ".join(
        f'"{a["message"]}" ({a["severity"]}, {a["minutes_ago"]} min ago)' for a in own) or "none") + "."
    if changes["deploys"]:
        evidence += " Deploys in the last hour, none on its dependency chain: " + "; ".join(
            f"{d['service']} {d['version']} by {d['by']}" for d in changes["deploys"]) + "."
    lines.append(evidence)
    lines.append(f"Page: {page['team']} team owns {failing}, on call "
                 f"{page['on_call'] or 'nobody listed'}.")
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _owner(g, service, steps):
    """Answer "who owns X?" / "who is on call for X?" with who_to_page."""
    r = _step(steps, "who_to_page", g, service)
    lines = [_HEADER.strip(), f"Owner: {service} is owned by the {r['team']} team.",
             f"Page: {r['team']} team, on call {r['on_call'] or 'nobody listed'}."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _dependencies(g, service, steps):
    """Answer "what does X depend on?" with dependencies, nearest first."""
    deps = _step(steps, "dependencies", g, service)["depends_on"]
    listed = ", ".join(f"{d['service']} ({d['hops']} hop{'s' if d['hops'] > 1 else ''})"
                       for d in deps)
    lines = [_HEADER.strip(), f"Dependencies: {service} depends on "
             + (listed if deps else "no other service") + "."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _path(g, a, b, q, steps):
    """Answer "does A depend on B?" / "path from A to B" with dependency_path.

    The path comes back in either direction; its first service is the caller.
    """
    r = _step(steps, "dependency_path", g, a, b)
    path = r.get("path")
    if not path:
        verdict = f"No: there is no dependency path between {a} and {b} in either direction."
    elif path[0] == a:
        verdict = (f"Yes: {a} depends on {b} through " + " → ".join(path)
                   + f" ({r['latency_ms']:g} ms).")
    else:
        verdict = (f"No: {a} does not depend on {b}. It is the other way round: {b} depends on "
                   f"{a} through " + " → ".join(path) + f" ({r['latency_ms']:g} ms).")
    lines = [_HEADER.strip(), verdict]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": path}


def _relation(g, a, b, steps):
    """Answer "is the A alert related to B?" with dependency_path both ways.

    No path either way means the alerts cannot be the same failure; the named
    services with an open alert off the recent deploy's chain go under Unrelated.
    """
    changes = _step(steps, "recent_changes", g)
    r = _step(steps, "dependency_path", g, a, b)
    path = r.get("path")
    if path:
        lines = [_HEADER.strip(), f"Yes, they are linked: " + " → ".join(path)
                 + f" ({r['latency_ms']:g} ms). A failure can spread along it."]
        return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": path}
    deployed = {d["service"] for d in changes["deploys"]}
    alerting = {al["service"]: al for al in changes["alerts"]}
    off_chain = []
    for svc in (a, b):
        if svc not in alerting:
            continue
        linked = svc in deployed or any(
            _step(steps, "dependency_path", g, svc, d).get("path") for d in deployed)
        if not linked:
            off_chain.append(svc)
    lines = [_HEADER.strip(),
             f"No: there is no dependency path between {a} and {b} in either direction, "
             "so a failure in one cannot reach the other."]
    if off_chain:
        lines.append("Unrelated: " + "; ".join(
            f'{s} "{alerting[s]["message"]}", not on the path of any recent deploy'
            for s in off_chain))
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _metric(g, services, steps):
    """Answer a metrics question ("memory usage of payments-db?") honestly.

    The graph has no metrics, only alerts; show the open alerts on the named
    services, since an alert message may carry the figure ("CPU 98%").
    """
    alerts = [a for a in _step(steps, "recent_changes", g)["alerts"] if a["service"] in services]
    lines = [_HEADER.strip(),
             f"The graph has no metrics (CPU, memory, latency) for {', '.join(services)}: it holds "
             "services, dependencies, deploys, alerts, owners and on-call people, so this is not "
             "available here. Check your metrics system."]
    if alerts:
        lines.append("Open alerts on " + ", ".join(services) + ": " + "; ".join(
            f'{a["service"]} "{a["message"]}" ({a["minutes_ago"]} min ago)' for a in alerts) + ".")
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _blast(g, service, steps):
    """Answer "what breaks if `service` goes down?" with blast_radius and its owner."""
    affected = _step(steps, "blast_radius", g, service)["affected"]
    page = _step(steps, "who_to_page", g, service)
    hit = ", ".join(f"{a['service']} ({a['hops']} hop{'s' if a['hops'] > 1 else ''})"
                    for a in affected)
    lines = ["Diagnosis by rules (no model):",
             (f"Blast radius: if {service} goes down, {len(affected)} service"
              f"{'' if len(affected) == 1 else 's'} that depend on it can fail: {hit}." if affected
              else f"Blast radius: no service in the graph depends on {service}, so only "
                   f"{service} itself is down if it fails."),
             f"Page: {page['team']} team owns {service}, on call {page['on_call'] or 'nobody listed'}."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _named_team(g, q):
    """Return the graph team the lower-case question `q` names, or None.

    A team counts if its full name or its first word appears ("payments",
    "catalog" for "Catalog & Logistics").
    """
    for (name,) in g.ro_query("MATCH (t:Team) RETURN t.name ORDER BY t.name").result_set:
        first = re.split(r"[\s/&]+", name.lower())[0]
        if any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", q) for w in {name.lower(), first}):
            return name
    return None


def _team_on_call(g, team, steps, person=None):
    """Answer "who is on call for <team>?" with who_to_page on the team.

    `person` is _named_person()'s (name, team, on_call) when the question also
    names someone ("Is Omar on call for payments?"); the answer then says
    whether that person is the one on call.
    """
    r = _step(steps, "who_to_page", g, team)
    who = f"on call {r['on_call']}" if r["on_call"] else "nobody is on call"
    answer = f"{_HEADER}Page: {r['team']} team, {who}."
    if person and person[0] != r["on_call"]:
        where = (f"on the {person[1]} team," if person[1] != r["team"]
                 else f"on the {r['team']} team but")
        answer += f"\n{person[0]} is {where} not on call for {r['team']}."
    return {"answer": answer, "decided": True, "steps": steps, "path": None}


def _unknown_team(g, steps):
    """Build rule_answer()'s result for an on-call or owner question about a
    team the graph lacks; rule_answer() names the team in front."""
    names = [r[0] for r in tools._run(g, "MATCH (t:Team) RETURN t.name ORDER BY t.name")]
    answer = _HEADER + "Teams in the graph: " + ", ".join(names) + "."
    return {"answer": answer, "decided": True, "steps": steps, "path": None}


def _named_person(g, q):
    """The person the question names, by full or first name ("who is priya?").

    Returns (name, team, on_call) or None. A first name counts only when one
    person has it. Read-only query, like every tool.
    """
    rows = tools._run(g, "MATCH (p:Person)-[:MEMBER_OF]->(t:Team) "
                         "RETURN p.name, t.name, p.on_call ORDER BY p.name")
    full = [r for r in rows if re.search(rf"(?<![\w-]){re.escape(r[0].lower())}(?![\w-])", q)]
    if len(full) == 1:
        return tuple(full[0])
    first = [r for r in rows
             if re.search(rf"(?<![\w-]){re.escape(r[0].split()[0].lower())}(?![\w-])", q)]
    return tuple(first[0]) if len(first) == 1 else None


def _person(g, person, steps):
    """Answer "who is Daniel Okafor?": their team, who is on call there, and
    their deploys in the last hour (recent_changes)."""
    name, team, on_call = person
    r = _step(steps, "who_to_page", g, team)
    if on_call:
        line = f"{name} is on the {team} team and is on call now."
    else:
        who = f"on call for {team} is {r['on_call']}" if r.get("on_call") else f"nobody on {team} is on call"
        line = f"{name} is on the {team} team and is not on call; {who}."
    changes = _step(steps, "recent_changes", g)
    mine = [d for d in changes["deploys"] if d["by"] == name]
    if mine:
        line += " Deploys in the last hour: " + "; ".join(
            f"{d['service']} {d['version']}, {d['minutes_ago']} min ago" for d in mine) + "."
    else:
        line += " No deploys by them in the last hour."
        if changes["deploys"]:
            line += " Deploys in the last hour: " + "; ".join(
                f"{d['service']} {d['version']} by {d['by']}, {d['minutes_ago']} min ago"
                for d in changes["deploys"]) + "."
    return {"answer": _HEADER + line, "decided": True, "steps": steps, "path": None}


def _critical(g, steps):
    """Answer "which services are most critical?" with the betweenness ranking."""
    ranking = _step(steps, "critical_services", g)["ranking"]
    top = "; ".join(f"{i}. {r['service']} ({r['score']:g})" for i, r in enumerate(ranking, 1))
    lines = ["Diagnosis by rules (no model):",
             "Most critical services, by betweenness (how many dependency paths run "
             f"through them): {top}."]
    return {"answer": "\n".join(lines), "decided": True, "steps": steps, "path": None}


def _unknown_service(g, steps, missing=()):
    """Build rule_answer()'s result when the only service named is not in the graph.

    rule_answer() puts "There is no service called billing-service in the
    graph." in front; this adds the services that do exist, so the user can
    ask again with the right name, and a "Did you mean" for a near miss
    ("chekoutservice"). Only a close spelling counts (difflib ratio 0.8), so
    billing-service is never offered paymentservice.
    """
    names = [r[0] for r in tools._run(g, "MATCH (s:Service) RETURN s.name ORDER BY s.name")]
    close = sorted({m for n in missing for m in difflib.get_close_matches(n, names, n=1, cutoff=0.8)})
    answer = (_HEADER + (f"Did you mean {' or '.join(close)}? " if close else "")
              + "Nothing to diagnose for it. Services in the graph: " + ", ".join(names) + ".")
    return {"answer": answer, "decided": True, "steps": steps, "path": None}


def _each(g, named, q, steps, model_steps):
    """Build rule_answer()'s result for several failing services, one block each.

    Used when no single recent deploy reaches them all. Each block is
    _what_broke() for that service, under its name; tool calls are shared, so
    recent_changes runs once.
    """
    parts = [_what_broke(g, n, q, steps, model_steps) for n in named]
    blocks = [f"{n}:\n" + p["answer"].removeprefix(_HEADER) for n, p in zip(named, parts)]
    return {"answer": _HEADER + "\n\n".join(blocks), "decided": all(p["decided"] for p in parts),
            "steps": steps, "path": next((p["path"] for p in parts if p["path"]), None)}


def _off_topic(steps):
    """Build rule_answer()'s result for a question that is not about an incident.

    No tool runs: there is nothing in the graph to look up. rule_answer()
    adds "There is no person named akshit in the graph." in front when the
    question names someone the graph lacks.
    """
    answer = (_HEADER + "I can only answer questions about the incident graph: its services, "
              "dependencies, deploys, alerts, owners and on-call people. For example "
              '"checkoutservice is throwing 5xx, what broke?", "What breaks if payments-db goes '
              'down?" or "Who is on call for payments?".')
    return {"answer": answer, "decided": True, "steps": steps, "path": None}


def _no_writes(steps):
    """Build rule_answer()'s result for a request to change or delete graph data."""
    answer = (_HEADER + "I cannot change the graph. Every query I run is a fixed, read-only "
              "lookup (GRAPH.RO_QUERY), so the graph stays exactly as it is. Ask about "
              "the services, deploys, alerts or owners instead.")
    return {"answer": answer, "decided": True, "steps": steps, "path": None}


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
