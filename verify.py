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
  6. No invented people. Rule 1 only knows names that exist in the graph, so a
     made-up "Sarah Jenkins" (Gemini, Oct 16) or "Maya Lin" (round 3, story C)
     slipped through. A Title-case name on the first line of the Root cause or Page section, or
     right after "on-call", "by", "page" or "contact" anywhere, must be a
     Person in the graph. Team-name words and labels ("Page", "Team") are
     trimmed first; checked against every saved model answer, the only hit
     was the real fabrication. A single first name after "on-call:" or "page",
     and a name followed by "deployed", "owns" and the like, count too.
  7. No invented versions. Any "v1.2.3" must be a version a tool returned, and
     in the Root cause a returned version must sit next to its own service
     ("paymentservice v3.1.0" is flagged when v3.1.0 was a fraud-check deploy).
  8. No invented services in a chain. In "a → b → c", if two or more names are
     real services, every name must be; "paymentservice → ledger → fraud-check"
     is flagged, while prose such as "bad deploy → slow queries" is not.
  9. Teams. In the Page section, the word before "team" must be a real team,
     and if who_to_page ran, a real team must be the one it returned.
 10. Numbers. A percentage or a duration (ms, s) must appear in a tool result
     ("CPU 99%" is flagged when the alert said 98%). 0 and 100 are allowed.
 11. Unknown services from the question (trap L, "What depends on
     billing-service?"). If the question names a service-like word that is not
     in the graph, the answer must say so ("not in the graph", "doesn't exist").

The adversarial set in tests/test_verify_adversarial.py has one made-up
answer per kind of lie and one correct answer per style; all must hold.

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

# Looks like a service name: ends in "service" or a hyphenated infrastructure
# word ("-db", "-api", "-gateway", "-cache"...). Ordinary hyphenated words
# ("on-call") don't match; rule 8 catches made-up names inside a chain.
_SERVICE_LIKE = re.compile(r"(?<![\w-])([a-z][a-z0-9]*(?:-[a-z0-9]+)*?"
                           r"(?:-?service|-svc|-db|-api|-gateway|-cache|-store|-queue|"
                           r"-worker|-proxy|-cluster|-check|-model))(?![\w-])")
# Words that match _SERVICE_LIKE but are not service names. The hyphenated ones
# came up in real answers ("cross-service calls", Qwen, Story B).
_NOT_SERVICES = {"microservice", "micro-service", "cross-service", "inter-service",
                 "multi-service", "per-service", "service-to-service", "web-service",
                 "health-check", "sanity-check", "double-check", "cross-check", "spot-check",
                 "pre-check", "re-check", "data-model", "threat-model", "mental-model",
                 "graph-model", "key-value-store"}

# Rule 7: a deploy version such as "v3.1.0" (a leading "v" and at least one dot).
_VERSION = re.compile(r"(?<![\w.])v\d+(?:\.\d+){1,3}(?![\w.]*\w)")
# Rule 8: names joined by arrows, "a → b → c" or "a -> b". Backticks and bold are
# stripped before matching.
_CHAIN = re.compile(r"[\w.-]+(?:\s*(?:→|->|⟶|=>)\s*[\w.-]+)+")
_ARROW = re.compile(r"\s*(?:→|->|⟶|=>)\s*")
# Rule 9: the word(s) before "team" in the Page section ("Payments team",
# "payments-team", "Platform/Data team"). Not after a hyphen: in "fraud-check
# team" the team is named by its service, which is fine.
_TEAM_WORD = re.compile(r"(?<![\w-])([a-z][a-z/&]*)[\s-]+team\b")
# Words that may stand before "team" without naming one.
_TEAM_FILLER = {"the", "owning", "owner", "owners", "responsible", "its", "their", "that",
                "this", "which", "same", "a", "an", "on-call", "call", "service", "whole",
                "entire", "right", "correct", "your", "that's", "of", "s"}
# Rule 10: a percentage or a duration in ms or seconds.
_MEASURE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(%|ms\b|s\b|sec\b|secs\b|seconds\b)")
# Rule 11: ways of saying a service is not in the graph (same list as bakeoff.py).
NOT_FOUND = ("not found", "unknown", "doesn't exist", "does not exist", "no service",
             "not a known", "isn't a known", "not in the", "no such", "not exist",
             "couldn't find", "could not find", "not recognised", "not recognized",
             "isn't in the", "is not in the", "don't know about", "do not know about",
             "can't find", "cannot find", "no record of", "don't have a service",
             "do not have a service", "isn't part of", "is not part of", "no node",
             "there is no", "there's no")

# Rule 6. A person-like name: two to four Title-case words ("Priya Nair").
_NAME_WORDS = r"[A-Z][a-z]+(?:[ \u00a0]+[A-Z][a-z]+){1,3}"
_NAME = r"(" + _NAME_WORDS + r")(?![\w-])"
_PERSON = re.compile(r"(?<![\w-])" + _NAME)
# The same, right after words that introduce a person, anywhere in the answer.
# "by" is lower-case only, so a heading like "By Betweenness" does not count.
_PERSON_AFTER = re.compile(r"(?:\b[Oo]n[- ]call\b[^A-Za-z\n]{0,4}|\bby\s+|\b[Pp]ag(?:e|ing)\s+|"
                           r"\b[Cc]ontact\s+|\b[Nn]otify\s+)" + _NAME)
# A name doing something only a person does: "Sarah Jenkins deployed it".
_PERSON_DOING = re.compile(r"(?<![\w-])(" + _NAME_WORDS + r")"
                           + r"\s+(?:deployed|pushed|shipped|released|rolled|merged|owns|"
                           r"is on[- ]call|is the on[- ]call)\b")
# A single first name where a person is expected: "on-call: Sarah", "page Sarah".
_FIRST_NAME_AFTER = re.compile(r"(?:\b[Oo]n[- ]call\b[^A-Za-z\n]{0,4}|\b[Pp]age\s+|\b[Pp]ing\s+|"
                               r"\b[Cc]ontact\s+|\b[Nn]otify\s+|\b[Ww]ake(?:\s+up)?\s+)"
                               r"([A-Z][a-z]+)\b(?![  ]+[A-Z])")
# Title-case words after "page", "ping"... that are not first names.
_NOT_FIRST_NAMES = {"them", "whoever", "anyone", "someone", "immediately", "now", "first",
                    "oncall", "duty", "slack", "pagerduty", "if", "and", "or", "for", "via"}
# Words trimmed from the ends of a candidate, on top of the team-name words.
_NOT_NAME_WORDS = {"team", "page", "on", "call", "root", "cause", "the", "a", "an", "no", "none",
                   "nobody", "owner", "owners", "lead", "engineer", "contact", "notify", "escalate"}


def graph_vocabulary(g):
    """Fetch every service name, person name and deploy version in the graph.

    Args:
        g: FalkorDB Graph handle.

    Returns:
        {"services": set, "people": set, "versions": set, "teams": set}, all lower-case.
    """
    def column(cypher):
        return {row[0].lower() for row in g.ro_query(cypher).result_set if row[0]}
    return {"services": column("MATCH (s:Service) RETURN s.name"),
            "people": column("MATCH (p:Person) RETURN p.name"),
            "versions": column("MATCH (d:Deploy) RETURN DISTINCT d.version"),
            "teams": column("MATCH (t:Team) RETURN t.name")}


def _section(text, name, flags=0):
    """Return the text of section `name` ("root cause", "page"...), or "" if absent.

    `flags=re.I` finds the labels in text that has not been lower-cased.
    """
    start = re.search(_LABEL.format(name), text, flags)
    if not start:
        return ""
    rest = text[start.end():]
    end = re.search(_LABEL.format(_SECTION_NAMES), rest, flags)
    return rest[:end.start()] if end else rest


def _invented_people(answer, vocab):
    """Rule 6: person-like names in `answer` that are not people in the graph.

    Args:
        answer: The model's answer, original case (names are found by capitals).
        vocab: graph_vocabulary(); "teams" is optional (older callers).

    Returns:
        Sorted list of the names, as written.
    """
    stop = _NOT_NAME_WORDS | {w for t in vocab.get("teams", ()) for w in re.findall(r"[a-z]+", t)}
    found = set()
    for sec in ("root cause", "page"):
        # Only the section's first line: a heading after the last section
        # ("Key Takeaways") would otherwise count as part of it.
        first = next((line for line in _section(answer, sec, re.I).splitlines() if line.strip()), "")
        found.update(_PERSON.findall(first))
    found.update(_PERSON_AFTER.findall(answer))
    found.update(_PERSON_DOING.findall(answer))
    known_words = {w for p in vocab["people"] for w in p.split()}
    invented = set()
    for name in _FIRST_NAME_AFTER.findall(answer):
        if name.lower() not in stop | known_words | _NOT_FIRST_NAMES:
            invented.add(name)
    for name in found:
        words = name.split()
        while words and words[0].lower() in stop:
            words.pop(0)
        while words and words[-1].lower() in stop:
            words.pop()
        if len(words) >= 2 and " ".join(words).lower() not in vocab["people"]:
            invented.add(" ".join(words))
    return sorted(invented)


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
         "alerting": services with an open alert,
         "deploys": {version: service}, "teams": teams who_to_page returned,
         "numbers": every number in the results, as floats}, all lower-case.
    """
    facts = {"text": "", "deployers": set(), "versions": set(), "on_call": set(),
             "chains": [], "alerting": set(), "deploys": {}, "teams": set(), "numbers": set()}
    for step in steps:
        result = {k: v for k, v in step["result"].items() if k not in tools.TRACE_KEYS}
        facts["text"] += json.dumps(result).lower() + "\n"
        for d in result.get("deploys", []):
            facts["deployers"].add(str(d.get("by", "")).lower())
            facts["versions"].add(str(d.get("version", "")).lower())
            facts["deploys"][str(d.get("version", "")).lower()] = str(d.get("service", "")).lower()
        for al in result.get("alerts", []):
            facts["alerting"].add(str(al.get("service", "")).lower())
        if result.get("on_call"):
            facts["on_call"].add(result["on_call"].lower())
        if result.get("team"):
            facts["teams"].add(result["team"].lower())
        # (service, the services it depends on), for rule 5.
        if "depends_on" in result:
            facts["chains"].append((result["service"].lower(),
                                    {d["service"].lower() for d in result["depends_on"]}))
        if result.get("path"):
            path = [p.lower() for p in result["path"]]
            facts["chains"].append((path[0], set(path[1:])))
    # Numbers as written, not inside words: "p99" is not the number 99.
    facts["numbers"] = {float(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", facts["text"])}
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
    original = answer.replace("’", "'").replace("‑", "-")
    a = original.lower()
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

    # Rule 6: no invented people.
    for name in _invented_people(original, vocab):
        problems.append(f"'{name}' is named as a person, but there is no such person in the "
                        "graph; name only people a tool returned")

    # Rule 7: no invented versions, and a root-cause version sits next to its own service.
    for version in sorted(set(_VERSION.findall(a)) - vocab["versions"] - facts["versions"]):
        if version not in q:
            problems.append(f"version '{version}' is not a deploy in the graph")
    for version in sorted(_mentions(cause, facts["deploys"])):
        owner = facts["deploys"][version]
        named = _mentions(cause, vocab["services"])
        if named and owner not in named:
            problems.append(f"root cause puts {version} on {sorted(named)}, but {version} "
                            f"was a {owner} deploy")

    # Rule 8: no made-up services inside a chain of real ones.
    plain = re.sub(r"[`*]", "", a)
    for chain in _CHAIN.findall(plain):
        names = [n.strip(".-") for n in _ARROW.split(chain)]
        real = [n for n in names if n in vocab["services"]]
        if len(real) >= 2:
            for n in names:
                # Service-like names were already reported by rule 2. A short form
                # of a real name ("checkout → payment → fraud-check") is fine.
                if (n and n not in vocab["services"] and n not in q
                        and not _SERVICE_LIKE.fullmatch(n)
                        and not any(s.startswith(n) for s in vocab["services"])):
                    problems.append(f"'{n}' in the path {chain.strip()} is not a service "
                                    "in the graph")

    # Rule 9: teams in the Page section must be real, and the one who_to_page returned.
    team_words = {w for t in vocab.get("teams", ()) for w in re.split(r"[\s/&]+", t) if w}
    page = _section(a, "page")
    for word in _TEAM_WORD.findall(page):
        parts = [w for w in re.split(r"[/&]+", word) if w]
        if word in _TEAM_FILLER or not team_words:
            continue
        if not all(w in team_words for w in parts):
            problems.append(f"page names a '{word}' team, but there is no such team in the graph")
        elif facts["teams"] and not any(
                all(w in re.split(r"[\s/&]+", t) for w in parts) for t in facts["teams"]):
            problems.append(f"page names the {word} team, but who_to_page returned "
                            f"{sorted(facts['teams'])}")

    # Rule 10: percentages and durations must come from a tool result.
    for value, unit in _MEASURE.findall(plain):
        number = float(value)
        if number not in facts["numbers"] and number not in (0, 100) and value not in q:
            problems.append(f"'{value}{unit}' does not appear in any tool result")

    # Rule 11: a service from the question that is not in the graph must be called unknown.
    for word in sorted(set(_SERVICE_LIKE.findall(q)) - vocab["services"] - _NOT_SERVICES):
        if word in a and not any(p in plain for p in NOT_FOUND):
            problems.append(f"'{word}' is not in the graph, but the answer does not say so")
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
