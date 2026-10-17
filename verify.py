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
     4b. A Page section must name an on-call person or a team.
     4c. "Who should I page?", or a status question naming no service ("is
     everything ok?") when the tools showed open alerts, must be answered
     with an on-call person.
  5. Unrelated: must not list an alerting service that the tools showed the
     failing service depends on (its dependencies() list or the dependency
     path), or that calls it (blast_radius()). Only services with an open alert count, so naming the root cause
     for comparison ("unrelated to fraud-check") is fine.
     5b. Nor an alerting service that calls the root-cause service, or that
     it calls, at any depth in the graph.
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
     10b. "Alerts on X" must name services that have an open alert.
     10c. "N services" must be the length of a list a tool returned.
 11. Unknown services from the question (trap L, "What depends on
     billing-service?"). If the question names a service-like word that is not
     in the graph, the answer must say so ("not in the graph", "doesn't exist").
 12. Unknown versions from the question ("Did the paymentservice v2.0.0 deploy
     cause it?"): if no such deploy exists, the answer must say so.
 13. Unknown people from the question ("Is Sarah Jenkins on call?"): the
     answer must say there is no such person.
 14. Unknown teams from the question ("Should I page the SRE team?"): the
     answer must say there is no such team.
 15. A real service, person or team must not be called missing.
 16. Graph facts with no tool call behind them are flagged.
 17. A root-cause deploy must be on the named services' dependency chain.
 18. "X depends on Y" (or "Yes" to "does X depend on Y") must match the
     graph's direction.
 19. A failure question that pages someone must have run recent_changes.
 20. "Most critical" must use the betweenness ranking.
 21. "Should we roll back X?" must be answered yes or no.

Rules 11-14 need the denial in a sentence that names the thing: "adservice is
not in the dependency path" does not deny that v2.0.0 exists (Gemini, Oct 17).

The adversarial set in tests/test_verify_adversarial.py has one made-up
answer per kind of lie and one correct answer per style; all must hold.

agent.py uses the problems to ask the model for one corrected answer. If that
one fails too, it shows plain_answer() instead: the tool rows, with no model text.
"""
import difflib
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
             "there is no", "there's no", "isn't a service", "is not a service",
             "not a service", "no service named", "no service called", "not listed",
             "doesn't appear", "does not appear", "not present")
# Rule 17: a Root cause section that blames nothing ("not found in the graph").
# "not identifiable in the graph" is live C3 (StepFun, Oct 16).
_NO_CAUSE = (r"\W*(?:not found|none|unknown|no root cause|undetermined|"
             r"not (?:identifiable|identified|determined|determinable|clear)|"
             r"(?:cannot|can't|could not|couldn't) be (?:identified|determined)|"
             r"no (?:recent )?(?:deploy|change)s? (?:or \w+ )?(?:in the graph )?explains?)")
# Rule 19: words that report a failure (kept in step with diagnose._INCIDENT).
_INCIDENT = (r"\b(?:broke|broken|error|errors|5xx|fail|failing|failed|failure|down|outage|"
             r"slow|latency|timeout|timeouts|crash\w*|degraded|not working|unavailable)\b")
# Rule 4c: status questions that mean "something is wrong, help".
# "What happened in the last hour?" is not one: a list of changes answers it.
_STATUS = (r"\b(?:got paged|is (?:everything|anything|something|prod) (?:ok|okay|fine|wrong|broken|down)|"
           r"what'?s (?:up|wrong|going on))\b")
# Rule 17: a root-cause sentence that clears a deploy instead of blaming it.
_CLEARS = (r"\b(?:not caused by|not the cause|isn't caused|is not caused|cannot|can't|"
           r"unrelated|separate incident|not related|no dependency path)\b")
# Rule 12: ways of saying a deploy did not happen, on top of NOT_FOUND.
NO_DEPLOY = NOT_FOUND + ("no deploy", "not deployed", "wasn't deployed", "was not deployed",
                         "no record", "not a recent", "isn't a recent", "is not a recent",
                         "only deploy", "only recent deploy", "only change", "never deployed",
                         "did not happen", "didn't happen", "no such")

# Rule 13: ways of saying a person is not in the graph, on top of NOT_FOUND.
NO_PERSON = NOT_FOUND + ("no one named", "nobody named", "no person", "not a person",
                         "not on call", "not on-call", "isn't on call", "isn't on-call",
                         "not the on-call", "not the on call")
# Rule 13: capitalised question words that start a sentence, not a name.
_QUESTION_WORDS = {"is", "are", "was", "did", "does", "do", "should", "can", "could", "who",
                   "what", "why", "when", "where", "which", "will", "has", "have", "tell",
                   "ask", "ping", "call", "hey", "please", "if", "then", "so", "and", "but"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[^a-z0-9])|\n+")

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
        {"services": set, "people": set, "versions": set, "teams": set,
         "alerting": set of services with an unresolved alert,
         "reach": {service: {every service it depends on, at any depth}}},
        all lower-case. "reach" is for rules 5b, 17 and 18; "alerting" for 5b and 10b.
    """
    def column(cypher):
        return {row[0].lower() for row in g.ro_query(cypher).result_set if row[0]}
    reach = {row[0].lower(): {n.lower() for n in row[1]} for row in g.ro_query(
        f"MATCH (a:Service)-[:DEPENDS_ON*1..{tools.MAX_HOPS}]->(b:Service) "
        "RETURN a.name, collect(DISTINCT b.name)").result_set}
    return {"services": column("MATCH (s:Service) RETURN s.name"),
            "people": column("MATCH (p:Person) RETURN p.name"),
            "versions": column("MATCH (d:Deploy) RETURN DISTINCT d.version"),
            "teams": column("MATCH (t:Team) RETURN t.name"),
            "alerting": column("MATCH (a:Alert)-[:FIRED_ON]->(s:Service) "
                               "WHERE NOT a.resolved RETURN DISTINCT s.name"),
            "reach": reach}


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


def _called_out(text, name, phrases):
    """Rules 11-14: True if a sentence of `text` names `name` and denies it.

    Args:
        text: The answer, lower-case, markdown stripped.
        name: The unknown name from the question, lower-case.
        phrases: Denial phrases, e.g. NOT_FOUND.

    Returns:
        bool. An answer that never repeats the name has not called it out.
    """
    for sentence in _SENTENCE_END.split(text):
        if not _mentions(sentence, {name}):
            continue
        if (any(p in sentence for p in phrases)
                or re.search(rf"\bno\s+(?:\w+\s+){{0,3}}?['\"]?{re.escape(name)}", sentence)):
            return True
    return False


def _question_people(question, vocab):
    """Rule 13: person-like names in the question that are not people in the graph.

    Args:
        question: The user's question, original case.
        vocab: graph_vocabulary().

    Returns:
        Sorted list of the names, lower-case ("sarah jenkins").
    """
    stop = (_NOT_NAME_WORDS | _QUESTION_WORDS | set(_TEAM_FILLER)
            | {w for t in vocab.get("teams", ()) for w in re.findall(r"[a-z]+", t)})
    unknown = set()
    for name in _PERSON.findall(question):
        words = name.split()
        while words and words[0].lower() in stop:
            words.pop(0)
        while words and words[-1].lower() in stop:
            words.pop()
        name = " ".join(words).lower()
        if len(words) >= 2 and name not in vocab["people"]:
            unknown.add(name)
    # "who is akshit?": one lower-case word is a name too when the whole
    # question asks who someone is (live test, Oct 16: Gemini answered it from
    # nothing). A first name of someone in the graph ("who is priya?") is known.
    m = _WHO_IS.match(question.lower().replace("\u2019", "'"))
    if m:
        words = m.group(1).split()
        known = ({w for p in vocab["people"] for w in p.split()} | vocab["services"]
                 | {w for t in vocab.get("teams", ()) for w in re.findall(r"[a-z]+", t)})
        if not any(w in stop or w in known or w in _NOT_WHO for w in words):
            unknown.add(" ".join(words))
    return sorted(unknown)


# "who is akshit?" / "who's ravi k?": the whole question asks who one or two words are.
_WHO_IS = re.compile(r"^\s*who(?:\s+is|'s)\s+([a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*)?)\s*[?.!]*\s*$")
# Words after "who is" that are not a name: "who is responsible?", "who is paged?".
_NOT_WHO = {"responsible", "paged", "affected", "down", "it", "he", "she", "they", "this", "that",
            "there", "here", "me", "you", "we", "oncall", "on-call", "available", "around",
            "awake", "online", "free", "in", "at", "to", "for", "blamed", "guilty", "accountable"}


# "ledger service" written as two words: the first word, unless it is one of these.
_SERVICE_TWO_WORDS = re.compile(r"(?<![\w-])([a-z][a-z0-9]*)\s+service\b")
_NOT_SERVICE_FIRST = {"the", "a", "an", "which", "what", "each", "every", "any", "that", "this",
                      "one", "other", "another", "same", "web", "micro", "my", "our", "your",
                      "their", "its", "of", "for", "to", "in", "on", "is", "no", "some", "affected",
                      "failing", "broken", "down", "customer", "external", "internal", "backend",
                      "frontend", "data", "whole", "entire", "single", "root", "caller", "upstream",
                      "downstream", "dependent", "owning", "on-call", "me"}


# Everyday words that look like misspelt service names to difflib.
_NOT_TYPOS = {"service", "services", "serviced", "servicing", "frontends", "payments", "payment"}


def unknown_names(question, vocab):
    """Everything the question names that the graph does not have.

    Used by the rules answer (it starts with "There is no ... in the graph")
    and by app.summarize() (the card's "Not in the graph" notice), so both
    list the same names the checker's rules 11-14 ask the model to call out.

    Args:
        question: The user's question, original case.
        vocab: graph_vocabulary().

    Returns:
        Sorted list of (kind, name), kind in "service" | "version" | "person" | "team",
        e.g. [("service", "billing-service"), ("team", "sre")].
    """
    q = question.lower().replace("’", "'")
    services = vocab["services"]
    found = {("service", w) for w in _SERVICE_LIKE.findall(q) if w not in services | _NOT_SERVICES}
    for word in _SERVICE_TWO_WORDS.findall(q):
        if (word not in _NOT_SERVICE_FIRST and word not in services
                and not any(s.startswith(word) or word.startswith(s.replace("service", "").strip("-"))
                            for s in services)):
            found.add(("service", f"{word} service"))
    # Misspelt service names ("paymentsdb", "fronted", "chekoutservice"): a
    # word of 6+ letters at least 90% like a real name. "services" is not one
    # (it is 94% like "adservice").
    for word in set(re.findall(r"(?<![\w-])[a-z][a-z0-9-]{5,}(?![\w-])", q)) - services - _NOT_TYPOS:
        if difflib.get_close_matches(word, services, n=1, cutoff=0.9):
            found.add(("service", word))
    found |= {("version", v) for v in _VERSION.findall(q) if v not in vocab["versions"]}
    found |= {("person", p) for p in _question_people(question, vocab)}
    team_words = {w for t in vocab.get("teams", ()) for w in re.split(r"[\s/&]+", t) if w}
    for word in _TEAM_WORD.findall(q):
        parts = [w for w in re.split(r"[/&]+", word) if w]
        if word not in _TEAM_FILLER and team_words and not all(w in team_words for w in parts):
            found.add(("team", word))
    return sorted(found)


def _mentions(text, names):
    """Return the names from `names` that appear in `text` as whole words.

    "Whole word" means not glued to letters, digits or hyphens, so
    "paymentservice" does not match inside "paymentservices" and "v3.1.0" does
    not match inside "v3.1.0-rc".
    """
    return {n for n in names
            if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text)}


def _asked_services(question, services):
    """Return the services a question names, short forms included.

    People say "the checkout outage" for checkoutservice and "payment errors"
    for paymentservice, so a service whose name ends in "service" also
    matches its stem ("checkout", "payment", "payments") when the stem has at
    least four letters ("ad" for adservice is too short to trust).

    Args:
        question: The user's question, lower-case.
        services: Service names from graph_vocabulary().

    Returns:
        set of service names.
    """
    found = _mentions(question, services)
    for name in services:
        stem = name[:-len("service")] if name.endswith("service") else ""
        if len(stem) >= 4 and re.search(rf"(?<![\w-]){re.escape(stem)}s?(?![\w-])", question):
            found.add(name)
    return found


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
         "numbers": every number in the results, as floats,
         "counts": the length of every list in the results}, all lower-case.
    """
    facts = {"text": "", "deployers": set(), "versions": set(), "on_call": set(),
             "chains": [], "alerting": set(), "deploys": {}, "teams": set(), "numbers": set(),
             "counts": set()}
    for step in steps:
        result = {k: v for k, v in step["result"].items() if k not in tools.TRACE_KEYS}
        facts["text"] += json.dumps(result).lower() + "\n"
        # Rule 10c: how many rows each list held ("47 services").
        facts["counts"] |= {len(v) for v in result.values() if isinstance(v, list)}
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
        # Callers of a service are on its chain too: their alerts are effects.
        if "affected" in result:
            facts["chains"].append((result["service"].lower(),
                                    {d["service"].lower() for d in result["affected"]}))
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
    # gpt-oss writes "Priya\u202fNair" (narrow no-break space); match it as a space.
    original = (answer.replace("’", "'").replace("‑", "-")
                .replace("\u202f", " ").replace("\u00a0", " "))
    a = original.lower()
    q = question.lower()
    facts = _returned(steps)
    problems = []

    # Rule 1: people and versions must come from a tool result (or the question:
    # "Did Priya Nair deploy anything?" may be answered "No, Priya Nair did not").
    for name in sorted(_mentions(a, vocab["people"] | vocab["versions"])):
        if name not in facts["text"] and name not in q:
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

    # Rule 4b: a Page section must name who to page. Live C2, Oct 16 (StepFun):
    # "Page: the fraud-check owner — use who_to_page on fraud-check" named nobody.
    page = _section(a, "page")
    # A team counts only when named as one: "the Payments team", "Payments
    # on-call". Live C1: "the checkout outage" is not the Checkout team.
    teams = "|".join(re.escape(t) for t in vocab.get("teams", ()))
    named_team = teams and re.search(rf"(?<![\w-])(?:{teams})[*`]*\s+(?:team|on[- ]?call)", page)
    if (page.strip() and not page_people and not named_team
            and not re.match(r"\W*(?:none|no one|nobody|n/a|not needed|not applicable|no page)", page)):
        problems.append("the page section names no on-call person or team; "
                        "call who_to_page and name them")

    # Rule 4c: "who should I page?" must be answered with a person. Saved
    # Nemotron Nano answer (round 4, story A) was cut off before its Page line.
    # A status question that names no service ("is everything ok?") needs
    # someone to page too, once the tools showed open alerts. Live V2, Oct 16 (Nemotron): "I just got paged, is
    # everything ok?" listed the alerts and named nobody.
    asks_page = re.search(r"\bwho (?:should|do|can|must|to) (?:i |we )?(?:page|wake|call|contact)|"
                          r"\bkisko page\b", q)
    reports = (facts["alerting"] and re.search(_STATUS, q)
               and not _asked_services(q, vocab["services"]))
    if ((asks_page or reports)
            and not unknown_names(q, vocab)
            # "on-call: Priya" names her too.
            and not _mentions(a, facts["on_call"] | {n.split()[0] for n in facts["on_call"]})
            and not re.search(r"\b(?:no one|nobody|not on call|nobody listed)\b", a)):
        problems.append("the question asks who to page, but the answer names no on-call "
                        "person from who_to_page")

    # Rule 10b: "alerts on X, Y" must name services that have an open alert.
    # Saved Ministral 8B answer (round 4, story A): "Unrelated: alerts on
    # adservice, cartservice, currencyservice, ..." where only adservice had one.
    # Only the list right after "alerts on" counts: in "alerts on payments-db
    # (slow query from fraud-check)" fraud-check is not said to have an alert,
    # and "open alerts on cartservice: none" says it has none.
    alerting = facts["alerting"] | vocab.get("alerting", set())
    svc = "|".join(re.escape(n) for n in sorted(vocab["services"], key=len, reverse=True))
    listed = rf"(?:{svc})(?![\w-])(?:\s*(?:,|and|&)\s*(?:{svc})(?![\w-]))*"
    for m in re.finditer(rf"(?<!no )(?<!without )\balerts? on ({listed})(\W*none)?",
                         re.sub(r"[`*]", "", a)):
        if m[2]:
            continue
        for name in sorted(_mentions(m[1], vocab["services"]) - alerting):
            problems.append(f"the answer says {name} has an alert, but it has no open alert")

    # Rule 10c: "47 services" must be the length of a list a tool returned.
    # Live test on the 300-service graph (StepFun): "49 services are affected"
    # for a 47-row blast radius.
    for m in re.finditer(r"(?<![\w.])(\d+)\s+(?:services|callers|dependents|dependencies)\b", a):
        n = int(m[1])
        if n > 1 and n not in facts["counts"] and m[1] not in q:
            problems.append(f"the answer counts {n} services, but no tool returned a list of {n}")

    # Rule 5: "Unrelated" must not list an alerting dependency of a service the question names.
    # The named service itself is left out, as answers say "X is not a dependency of it".
    # The graph-wide reach map counts too. Live test, Oct 16 (Gemini): for
    # "frontend is returning errors" it listed adservice as unrelated, though
    # frontend calls it, because it never ran dependencies(frontend).
    asked = _mentions(q, vocab["services"])
    on_chain = set().union(*[deps for svc, deps in facts["chains"] if svc in asked],
                           *[vocab.get("reach", {}).get(svc, set()) for svc in asked]) - asked
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
    # "ledger service" (two words) may be written "ledger-service" or "ledger".
    for kind, word in unknown_names(question, vocab):
        if kind != "service":
            continue
        variants = {word, word.replace(" ", "-"), word.replace(" ", ""), word.split()[0]}
        if not any(_called_out(plain, v, NOT_FOUND) for v in variants):
            problems.append(f"'{word}' is not in the graph, but the answer does not say so")

    # Rule 12: a version from the question that was never deployed must be called so.
    # "The only deploy was fraud-check v3.1.0" also denies it without repeating it.
    only = re.search(r"\bonly\s+(?:recent\s+)?(?:deploy|change)", plain)
    for version in sorted(set(_VERSION.findall(q)) - vocab["versions"]):
        if not only and not _called_out(plain, version, NO_DEPLOY):
            problems.append(f"there is no {version} deploy in the graph, but the answer "
                            "does not say so")

    # Rule 13: a person from the question who is not in the graph must be called so.
    for name in _question_people(question.replace("’", "'"), vocab):
        if not _called_out(plain, name, NO_PERSON):
            problems.append(f"there is no person named '{name}' in the graph, but the "
                            "answer does not say so")

    # Rule 14: a team from the question that is not in the graph must be called so.
    team_words = {w for t in vocab.get("teams", ()) for w in re.split(r"[\s/&]+", t) if w}
    for word in sorted(set(_TEAM_WORD.findall(q))):
        parts = [w for w in re.split(r"[/&]+", word) if w]
        if word in _TEAM_FILLER or not team_words or all(w in team_words for w in parts):
            continue
        if not _called_out(plain, word, NOT_FOUND + ("no team",)):
            problems.append(f"there is no '{word}' team in the graph, but the answer "
                            "does not say so")

    # Rule 15: a real service, person or team must not be called missing.
    # Live test, Oct 16: "There is no Priya Nair in the graph." and "Mei Lin is
    # not in the graph" (both are people in the graph) passed rules 1-14.
    # A first name the question used counts too, if only one person has it
    # (live test, Oct 16: "who is priya?" -> "Priya is not in the graph.").
    firsts = [n.split()[0] for n in vocab["people"]]
    real = ([("service", n) for n in vocab["services"]] + [("person", n) for n in vocab["people"]]
            + [("person", f) for f in firsts if firsts.count(f) == 1
               and re.search(rf"\b{re.escape(f)}\b", q)]
            + [("team", t.lower()) for t in vocab.get("teams", ())])
    for kind, name in sorted(real):
        if _denied(plain, name, kind):
            problems.append(f"'{name}' is in the graph, but the answer says it is not")

    # Rule 16: an answer with no tool call behind it is not from the graph.
    # Live test, Oct 16: StepFun answered "Show the dependency path from
    # api-gateway to feature-store" with a made-up direct call, no tools run.
    # A pure denial ("There is no service called billing-service") states no
    # graph fact, so it needs no tool.
    states = (_mentions(a, vocab["services"] | vocab["people"] | vocab["versions"])
              or re.search(r"→|->", a))
    if not steps and states:
        problems.append("no tool was called, so nothing in the answer comes from the graph")
    # Live test, Oct 16 (StepFun): an empty reply, or "I'll check the on-call
    # information", with no tool call. It states nothing false but answers
    # nothing either. A question with a made-up name is graded by rules 11-14.
    elif (not steps and not any(p in plain for p in NOT_FOUND)
          and not unknown_names(question, vocab)):
        problems.append("no tool was called and the answer neither answers nor says "
                        "what is missing from the graph")

    # Rule 17: a root-cause deploy must be on the dependency chain (either
    # direction) of a failing service the question names. Live test, Oct 16:
    # "cartservice and adservice are both failing" -> "Root cause: fraud-check",
    # which neither of them reaches.
    reach = vocab.get("reach")
    asked = _asked_services(q, vocab["services"])
    # A sentence that clears a deploy ("is not caused by the fraud-check
    # v3.1.0 deploy") does not blame it.
    blamed = {facts["deploys"][v]
              for sent in re.split(r"(?<=[.!?])\s+|\n", cause)
              if not re.search(_CLEARS, sent)
              for v in _mentions(sent, set(facts["deploys"]))}
    if reach is not None and asked and not re.match(_NO_CAUSE, cause):
        for svc in sorted(blamed - asked):
            if not any(svc in reach.get(a, set()) or a in reach.get(svc, set()) for a in asked):
                problems.append(f"root cause {svc} has no dependency path to or from "
                                f"{', '.join(sorted(asked))}")

    # Rule 5b: "Unrelated" must not list an alerting service on the blamed
    # deploy's chain (it calls that service or that service calls it). Live
    # test, Oct 16 (StepFun): for paymentservice latency it called the
    # checkoutservice 5xx unrelated, though checkoutservice calls paymentservice
    # -> fraud-check. Only the first service of each item (a line, or a part
    # between semicolons) is its subject, so "adservice: not on the
    # checkoutservice path. There is no path between checkoutservice and
    # adservice." lists adservice alone.
    if reach is not None and blamed and not re.match(_NO_CAUSE, cause):
        incident = set().union(blamed, *[reach.get(b, set()) for b in blamed])
        incident |= {svc for svc, deps in reach.items() if deps & blamed}
        listed = {_first_named(item, vocab["services"])
                  for item in re.split(r"\n|;\s", _section(a, "unrelated"))}
        alerting = facts["alerting"] | vocab.get("alerting", set())
        for svc in sorted((listed & incident & alerting) - blamed):
            problems.append(f"unrelated lists {svc}, but it is on the dependency chain of the "
                            f"root cause {', '.join(sorted(blamed))}")

    # Rule 18: "X depends on Y" must match the graph's direction. Live test,
    # Oct 16 (Gemini): "Does payments-db depend on checkoutservice?" -> "Yes",
    # read off a path that runs checkoutservice -> paymentservice -> payments-db.
    if reach is not None:
        svc = "|".join(re.escape(n) for n in sorted(vocab["services"], key=len, reverse=True))
        # A claim right after "not", "nor does" or "neither" is a denial.
        claims = [(m[1], m[2]) for m in re.finditer(
            rf"(?<![\w-])({svc})(?![\w-]) (?:does |do )?(?:depends|depend) on (?:the )?({svc})(?![\w-])",
            plain) if not re.search(r"\b(?:not|nor|neither|never|n't)\b[\w ]{0,12}$",
                                    plain[max(0, m.start() - 20):m.start()])]
        ask = re.search(rf"\bdoes ({svc}) (?:\w+ )?depend on (?:the )?({svc})(?![\w-])", q)
        if ask and re.match(r"\W*yes\b", plain):
            claims.append((ask[1], ask[2]))
        for x, y in dict.fromkeys(claims):
            if y not in reach.get(x, set()):
                problems.append(f"the answer says {x} depends on {y}, but in the graph it does not"
                                + (f"; {y} depends on {x}" if x in reach.get(y, set()) else ""))

    # Rule 19: an incident question needs recent_changes before paging. Live
    # test, Oct 16 (Gemini): "checkout service me errors aa rahe hai, kisko
    # page karu?" paged checkout's owner without looking for the cause, which
    # was a Payments deploy.
    tools_run = {s["tool"] for s in steps}
    if (re.search(_INCIDENT, q) and _asked_services(q, vocab["services"])
            and "who_to_page" in tools_run and "recent_changes" not in tools_run):
        problems.append("the question reports a failure, but recent_changes was not called, "
                        "so the page skips the cause")

    # Rule 21: "should we roll back X?" needs a yes or a no. Live test, Oct 16
    # (Nemotron) gave the root cause and never said whether to roll back.
    # A bare "no" anywhere is not an answer: "slow query (no index)" passed
    # this rule once, so yes/no only counts at the start of a sentence.
    if (re.search(r"\b(?:should|do|can) (?:we|i) (?:roll ?back|revert)\b", q)
            and not re.search(r"(?:^|[.:!?\n]\s*)(?:yes|no)\b|\broll(?:ing)? ?back|\brevert\w*",
                              plain, re.M | re.I)):
        problems.append("the question asks whether to roll back, but the answer does not say "
                        "yes or no")

    # Rule 20: "most critical" is ranked by betweenness. Live test, Oct 16
    # (Gemini) ranked by PageRank only and listed leaf databases.
    metrics = {s["args"].get("metric") or "betweenness" for s in steps if s["tool"] == "critical_services"}
    if (re.search(r"\bcritical\b", q) and not re.search(r"\bdepend", q)
            and metrics and "betweenness" not in metrics):
        problems.append("'most critical' needs the betweenness ranking, but only "
                        f"{', '.join(sorted(metrics))} was used")
    return problems


def _first_named(text, services):
    """Rule 5b: the service named first in `text`, or None.

    An "Unrelated" item names its subject first and may name others only to
    explain it ("adservice: no path to or from checkoutservice").
    """
    hits = [(m.start(), n) for n in services
            for m in [re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text)] if m]
    return min(hits)[1] if hits else None


def _denied(text, name, kind):
    """Rule 15: True if `text` says `name` is not in the graph.

    Only a sentence that makes the name itself the missing thing counts:
    "There is no Priya Nair in the graph", "Mei Lin is not in the graph",
    "no service called paymentservice". "There is no paymentservice v2.0.0
    deploy in the graph" denies the deploy, not the service, and "There is
    no payments service" is true of the Payments team, so neither counts.

    Args:
        text: The answer, lower-case, markdown stripped.
        name: A real name from graph_vocabulary(), lower-case.
        kind: "service", "person" or "team"; only the right noun counts
            ("no Payments team" denies the team, "no payments service" does not).

    Returns:
        bool.
    """
    noun = {"service": "service", "person": "(?:person|one)", "team": "team"}[kind]
    n = r"['\"]?" + re.escape(name) + r"['\"]?"
    suffix = rf"(?: {noun})?" if kind != "person" else ""
    patterns = (
        rf"\bthere(?: is|'s) no (?:{noun} (?:named|called) )?{n}{suffix} in the graph",
        rf"(?<![\w-]){n}{suffix} (?:is not|isn't|was not|wasn't) (?:in|part of) the graph",
        rf"(?<![\w-]){n}{suffix} (?:does not|doesn't) exist",
        rf"\bno (?:record of )?(?:an? )?{noun} (?:named|called) {n}(?![\w-])",
        rf"(?<![\w-]){n} (?:is not|isn't) (?:an? )?(?:known )?{noun}\b",
    )
    return any(re.search(p, text) for p in patterns)


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
