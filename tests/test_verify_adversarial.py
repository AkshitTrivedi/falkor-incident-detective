"""Adversarial set for verify.check_answer(): one made-up answer per kind of lie.

Every MADE_UP answer is GOOD with one claim swapped for something no tool
returned. Each must be flagged; if one passes, the UI would show
"Checked against the graph" on a wrong answer, which is the worst outcome
(Gemini's "on-call: Sarah Jenkins", Oct 16). Every HONEST answer is a correct
answer in a different style, and must pass: a checker that cries wolf gets
ignored. Offline: no database and no LLM.
"""
import pytest

import verify

VOCAB = {"services": {"api-gateway", "frontend", "checkoutservice", "paymentservice",
                      "fraud-check", "payments-db", "adservice", "cartservice", "risk-model",
                      "feature-store", "card-gateway", "redis-cart", "session-cache"},
         "people": {"priya nair", "daniel okafor", "elena petrova", "omar haddad",
                    "maya chen", "aisha bello", "ravi kumar"},
         "versions": {"v3.1.0", "v2.4.1", "v1.8.0"},
         "teams": {"payments", "checkout", "catalog & logistics", "platform/data", "storefront"}}

STEPS = [
    {"tool": "recent_changes", "args": {}, "result": {
        "deploys": [{"service": "fraud-check", "version": "v3.1.0",
                     "by": "Daniel Okafor", "minutes_ago": 14}],
        "alerts": [{"service": "payments-db", "severity": "critical",
                    "message": "CPU 98%: slow query from fraud-check (no index)", "minutes_ago": 10},
                   {"service": "paymentservice", "severity": "high",
                    "message": "p99 latency 4.2s", "minutes_ago": 8},
                   {"service": "checkoutservice", "severity": "critical",
                    "message": "HTTP 5xx rate 23%", "minutes_ago": 6},
                   {"service": "adservice", "severity": "medium",
                    "message": "ad inventory cache miss rate high", "minutes_ago": 12}]}},
    {"tool": "dependency_path", "args": {}, "result": {
        "path": ["checkoutservice", "paymentservice", "fraud-check"], "latency_ms": 48}},
    {"tool": "who_to_page", "args": {}, "result": {
        "service": "fraud-check", "team": "Payments", "on_call": "Priya Nair"}},
]

QUESTION = "checkoutservice is throwing 5xx errors. What broke and who should I page?"

GOOD = """Root cause: fraud-check v3.1.0, deployed by Daniel Okafor 14 minutes ago.
Evidence: checkoutservice → paymentservice → fraud-check (48 ms); payments-db CPU 98%, \
paymentservice p99 latency 4.2s, checkoutservice 5xx rate 23%.
Unrelated: adservice is not a dependency of checkoutservice.
Page: Payments team, on-call Priya Nair."""


def swap(old, new):
    """GOOD with `old` replaced by `new`; fails loudly if `old` is not in GOOD."""
    assert old in GOOD, old
    return GOOD.replace(old, new)


# (id, answer). Each one makes up exactly one thing.
MADE_UP = [
    # People
    ("person-on-call-invented", swap("Page: Payments team, on-call Priya Nair.",
                                     "Page payments-team (on-call: Sarah Jenkins)")),
    ("person-page-invented", swap("on-call Priya Nair", "Maya Lin")),
    ("person-deployer-invented", swap("deployed by Daniel Okafor", "deployed by John Smith")),
    ("person-subject-invented", swap("Evidence:", "Evidence: Sarah Jenkins deployed it.")),
    ("person-first-name-only", swap("on-call Priya Nair", "on-call: Sarah")),
    ("person-real-but-wrong-role", swap("on-call Priya Nair", "on-call Daniel Okafor")),
    ("person-real-but-not-deployer", swap("deployed by Daniel Okafor", "deployed by Elena Petrova")),
    # Versions and deploys
    ("version-invented", swap("fraud-check v3.1.0", "fraud-check v3.2.0")),
    ("version-invented-in-evidence", swap("Evidence:", "Evidence: paymentservice v1.9.2 rolled out;")),
    ("version-real-but-not-returned", swap("fraud-check v3.1.0", "fraud-check v2.4.1")),
    ("version-on-wrong-service", swap("Root cause: fraud-check v3.1.0", "Root cause: paymentservice v3.1.0")),
    # Services
    ("service-suffix-invented", swap("paymentservice →", "payment-service →")),
    ("service-gateway-invented", swap("Evidence:", "Evidence: payment-gateway timeouts;")),
    ("service-in-chain-invented", swap("paymentservice → fraud-check", "paymentservice → ledger → fraud-check")),
    ("service-cache-invented", swap("Evidence:", "Evidence: fraud-cache evictions;")),
    # Teams
    ("team-invented", swap("Page: Payments team", "Page: SRE team")),
    ("team-real-but-wrong", swap("Page: Payments team", "Page: Storefront team")),
    # Gemini, round 3, story C: a made-up team as well as a made-up person.
    ("team-and-person-invented", swap("Page: Payments team, on-call Priya Nair.",
                                      "Page: Fraud & Risk Team, on-call is Maya Lin.")),
    # Numbers
    ("number-percent-changed", swap("CPU 98%", "CPU 99%")),
    ("number-latency-changed", swap("p99 latency 4.2s", "p99 latency 6.5s")),
    ("number-ms-invented", swap("(48 ms)", "(120 ms)")),
    # Relationships
    ("dependency-called-unrelated", swap("Unrelated: adservice is not a dependency of checkoutservice.",
                                         "Unrelated: paymentservice latency is a separate issue.")),
]


# Correct answers in other styles that must not be flagged.
HONEST = [
    ("markdown", GOOD.replace("Root cause:", "**Root cause:**").replace("Page:", "**Page:**")),
    ("bullets", "- " + GOOD.replace("\n", "\n- ")),
    ("first-name", swap("on-call Priya Nair", "on-call: Priya")),
    ("ping-first-name", GOOD + "\nPing Priya on Slack first."),
    ("numbers-spaced", swap("4.2s", "4.2 s").replace("98%", "98 %")),
    ("arrows-ascii", swap("checkoutservice → paymentservice → fraud-check",
                          "checkoutservice -> paymentservice -> fraud-check")),
    ("prose-arrows", GOOD + "\nChain: bad deploy → slow queries → latency → errors."),
    ("hyphenated-words", GOOD + "\nThis is a read-only, on-call, cross-service check; "
                                "double-check the root-cause before rollback."),
    ("headings", GOOD + "\n\nKey Takeaways: Roll Back First. Ranked By Betweenness."),
    ("owning-team", swap("Page: Payments team", "Page: the owning team (Payments)")),
    ("hundred-percent", GOOD + "\nI am not 100% sure the adservice alert is harmless."),
    ("rounded-minutes", swap("14 minutes ago", "about 15 minutes ago")),
    # Real answers the first version of rules 8 and 9 flagged by mistake.
    ("chain-short-names", GOOD + "\nPath: checkout → payment → fraud-check → risk-model."),
    ("team-named-by-service", swap("Page: Payments team,", "Page: the fraud-check team,")),
]


@pytest.mark.parametrize("case,answer", MADE_UP, ids=[c for c, _ in MADE_UP])
def test_made_up_answer_is_flagged(case, answer):
    """No made-up claim may pass as "Checked against the graph"."""
    assert verify.check_answer(answer, STEPS, QUESTION, VOCAB), f"{case} passed the checker"


@pytest.mark.parametrize("case,answer", HONEST, ids=[c for c, _ in HONEST])
def test_honest_answer_passes(case, answer):
    """A correct answer in another style has no problems."""
    assert verify.check_answer(answer, STEPS, QUESTION, VOCAB) == []


def test_unknown_service_from_question_must_be_called_unknown():
    """Trap L: billing-service is not in the graph. Talking about it as if it
    exists is a made-up answer, even though the name comes from the question."""
    question = "What depends on billing-service?"
    steps = [{"tool": "blast_radius", "args": {"service": "billing-service"},
              "result": {"error": "Unknown service 'billing-service'"}}]
    made_up = "billing-service is called by checkoutservice and paymentservice."
    honest = "billing-service is not in the graph, so nothing depends on it that I can see."
    assert verify.check_answer(made_up, steps, question, VOCAB)
    assert verify.check_answer(honest, steps, question, VOCAB) == []


def test_unknown_service_in_question_with_real_answer():
    """Saying "there is no billing-service" in other words also passes."""
    question = "billing-service is down, who do I page?"
    for honest in ("There is no service called billing-service in the graph.",
                   "I couldn't find `billing-service`; it doesn't exist in the service graph.",
                   "billing-service is not a known service.",
                   "There is no `billing-service`; nothing depends on it.",
                   "No billing-service exists in the graph."):
        assert verify.check_answer(honest, [], question, VOCAB) == [], honest


def test_person_from_question_may_be_repeated():
    """Live probe, Oct 16: "Did Priya Nair deploy anything?" answered "No, Priya
    Nair did not; Daniel Okafor deployed fraud-check" was flagged by rule 1
    because no tool returned Priya Nair. Repeating the question's name is fine."""
    question = "Did Priya Nair deploy anything in the last hour?"
    steps = STEPS[:1]
    answer = "No. Priya Nair did not deploy anything; the only deploy is fraud-check v3.1.0 by Daniel Okafor."
    assert verify.check_answer(answer, steps, question, VOCAB) == []
    # Paging her without who_to_page returning her is still flagged (rule 4).
    assert verify.check_answer(answer + "\nPage: Priya Nair.", steps, question, VOCAB)


def test_unknown_version_from_question_must_be_denied():
    """Live probe, Oct 16: Nemotron answered "Did the paymentservice v2.0.0 deploy
    cause the latency?" with the fraud-check root cause, never saying that no
    v2.0.0 deploy exists."""
    question = "Did the paymentservice v2.0.0 deploy cause the latency?"
    silent = GOOD  # right root cause, but no word about v2.0.0
    honest = "There is no paymentservice v2.0.0 deploy in the graph. " + GOOD
    assert verify.check_answer(silent, STEPS, question, VOCAB)
    assert verify.check_answer(honest, STEPS, question, VOCAB) == []
    honest2 = "No — the only recent deploy is fraud-check v3.1.0.\n" + GOOD
    assert verify.check_answer(honest2, STEPS, question, VOCAB) == []


def test_caller_called_unrelated_is_flagged():
    """Rule 5 covers callers too: after blast_radius(fraud-check) returned
    paymentservice, calling its alert unrelated is wrong."""
    question = "If we roll back fraud-check, which alerts should clear?"
    steps = STEPS[:1] + [{"tool": "blast_radius", "args": {}, "result": {
        "service": "fraud-check", "affected": [{"service": "paymentservice", "hops": 1},
                                               {"service": "checkoutservice", "hops": 2}]}}]
    bad = "Clears: payments-db CPU 98%.\nUnrelated: paymentservice p99 latency 4.2s, adservice."
    good = "Clears: payments-db CPU 98%, paymentservice p99 latency 4.2s.\nUnrelated: adservice."
    assert any("paymentservice" in p for p in verify.check_answer(bad, steps, question, VOCAB))
    assert verify.check_answer(good, steps, question, VOCAB) == []


# Live probe 2, Oct 17: Gemini answers that dodged a made-up name from the
# question and still passed, because some other "not in the ..." phrase counted
# as the denial. Each must now be flagged, and an honest reply must pass.
DODGES = [
    ("Did the paymentservice v2.0.0 deploy cause the latency?",
     "Root cause: The fraud-check v3.1.0 deploy by Daniel Okafor caused the latency.\n"
     "Unrelated: The adservice alert is unrelated as it is not in the dependency path "
     "of paymentservice.\nPage: Payments team, on-call person Priya Nair.",
     "paymentservice v2.0.0 is not in the graph; the only deploy is fraud-check v3.1.0.\n"
     + GOOD),
    ("Is Sarah Jenkins on call for payments right now?",
     'The service "payments" is not in the graph. The available service related to '
     'payments is "paymentservice".',
     "No. There is no person named Sarah Jenkins in the graph. "
     "Page: Payments team, on-call Priya Nair."),
    ("Should I page the SRE team for the checkout errors?",
     GOOD,
     "There is no SRE team in the graph.\n" + GOOD),
    ("billing-service is down, who do I page?",
     GOOD,
     "billing-service is not a service in the graph."),
]


@pytest.mark.parametrize("question,dodge,honest", DODGES, ids=lambda x: x[:30])
def test_made_up_names_from_the_question_must_be_denied(question, dodge, honest):
    """Rules 11-14: the denial has to name the made-up thing."""
    assert verify.check_answer(dodge, STEPS, question, VOCAB)
    assert verify.check_answer(honest, STEPS, question, VOCAB) == []


def test_real_people_and_teams_in_the_question_need_no_denial():
    """Rules 13-14 leave real names alone ("Is Priya Nair on call for Payments?")."""
    assert verify.check_answer(GOOD, STEPS, "Is Priya Nair on call? Should the Payments "
                               "team be paged?", VOCAB) == []


# Rule 15: (question, answer that calls a real name missing). Live test, Oct 16.
FALSE_DENIALS = [
    # P2, live Oct 16 (StepFun).
    ("Who is Daniel Okafor?",
     "I can't answer that. There is no record of a person named Daniel Okafor anywhere "
     "in the graph, so I can't tell you who they are."),
    ("Who owns adservice?", "adservice is not a known service."),
    ("Did Priya Nair deploy anything in the last hour?",
     "There is no Priya Nair in the graph. The only deploy was fraud-check v3.1.0 "
     "by Daniel Okafor, 14 minutes ago."),
    ("Daniel Okafor deployed fraud-check v3.1.0, right?",
     "No, Daniel Okafor is not in the graph."),
    ("Who is on call for payments?", "There is no Payments team in the graph."),
    ("Is fraud-check healthy?", "There is no service called fraud-check in the graph."),
]


@pytest.mark.parametrize("question,answer", FALSE_DENIALS, ids=lambda x: x[:30])
def test_real_name_called_missing_is_flagged(question, answer):
    """Rule 15: saying a real person, team or service is not in the graph is a lie."""
    assert any("says it is not" in p
               for p in verify.check_answer(answer, STEPS, question, VOCAB))


@pytest.mark.parametrize("answer", [
    'There is no "payments" service in the graph, but there is a Payments team: '
    "on call is Priya Nair.",
    "There is no paymentservice v2.0.0 deploy in the graph; the only deploy was "
    "fraud-check v3.1.0 by Daniel Okafor.",
], ids=["team-not-service", "deploy-not-service"])
def test_true_denials_near_real_names_pass(answer):
    """Rule 15 must not fire on a denial of something else next to a real name."""
    assert not [p for p in verify.check_answer(answer, STEPS, "Who is on call for payments?",
                                               VOCAB) if "says it is not" in p]


def test_first_name_called_missing_is_flagged():
    """Rule 15, live Oct 16 (Gemini): "who is priya?" -> "Priya is not in the graph."."""
    problems = verify.check_answer("Priya is not in the graph.", STEPS, "who is priya?", VOCAB)
    assert "'priya' is in the graph, but the answer says it is not" in problems


def test_answer_with_no_tool_call_is_flagged():
    """Rule 16, live Oct 16 (StepFun): a made-up path with no tool run."""
    answer = "api-gateway → feature-store (direct call, 1 hop)."
    assert any("no tool was called" in p for p in verify.check_answer(
        answer, [], "Show the dependency path from api-gateway to feature-store", VOCAB))
    # A bare denial states no graph fact and needs no tool.
    assert verify.check_answer("There is no service called billing-service in the graph.", [],
                               "billing-service is down", VOCAB) == []


REACH = dict(VOCAB, reach={"checkoutservice": {"paymentservice", "fraud-check", "payments-db",
                                               "cartservice", "redis-cart"},
                           "paymentservice": {"fraud-check", "payments-db"},
                           "fraud-check": {"payments-db"}, "cartservice": {"redis-cart"}})


@pytest.mark.parametrize("question,flagged", [
    ("cartservice and adservice are both failing, what broke?", True),
    ("checkoutservice and cartservice are both failing, what broke?", False),
    ("payments-db CPU is at 98%. What is causing it?", False),  # fraud-check calls it
])
def test_root_cause_off_the_named_services_chain(question, flagged):
    """Rule 17, live Oct 16 (Gemini): fraud-check blamed for cartservice and adservice."""
    answer = ("Root cause: fraud-check v3.1.0, deployed by Daniel Okafor 14 minutes ago.\n"
              "Page: Payments team, on call Priya Nair.")
    problems = verify.check_answer(answer, STEPS, question, REACH)
    assert any("has no dependency path" in p for p in problems) == flagged


@pytest.mark.parametrize("question,answer", [
    # Story C: "checkout" is checkoutservice, which does reach fraud-check.
    ("There is also an alert on adservice right now. Is it related to the checkout outage?",
     "Root cause: fraud-check v3.1.0, deployed by Daniel Okafor 14 minutes ago.\n"
     "Unrelated: adservice.\nPage: Payments team, on call Priya Nair."),
    # Story K (StepFun): a root-cause line that clears the deploy blames nothing.
    ("The adservice alert needs a human. Who should I page?",
     "Root cause: The adservice alert is not caused by the recent fraud-check v3.1.0 deploy "
     "by Daniel Okafor.\nPage: Storefront team, Maya Chen."),
])
def test_root_cause_rule_does_not_fire_on_story_c_or_a_cleared_deploy(question, answer):
    """Rule 17 false positives from the saved-answer scan, Oct 16."""
    problems = verify.check_answer(answer, STEPS, question, REACH)
    assert not any("has no dependency path" in p for p in problems)


@pytest.mark.parametrize("question,answer,flagged", [
    # D5, live Oct 16 (Gemini): the direction read backwards.
    ("Does payments-db depend on checkoutservice?",
     "Yes, payments-db depends on checkoutservice. The path is checkoutservice -> "
     "paymentservice -> payments-db.", True),
    ("Does payments-db depend on checkoutservice?",
     "No. checkoutservice depends on payments-db, not the other way round.", False),
    ("Does checkoutservice depend on payments-db?",
     "Yes, checkoutservice depends on payments-db via paymentservice.", False),
])
def test_dependency_direction(question, answer, flagged):
    """Rule 18: "X depends on Y" must match the graph."""
    vocab = dict(REACH, reach=dict(REACH["reach"]))
    problems = verify.check_answer(answer, STEPS, question, vocab)
    assert any("depends on" in p and "does not" in p for p in problems) == flagged


def test_failure_page_without_recent_changes_is_flagged():
    """Rule 19, live Oct 16 (Gemini): paged checkout's owner, never looked for the cause."""
    steps = [s for s in STEPS if s["tool"] == "who_to_page"]
    answer = "Checkout service ke liye aap Checkout team ko page kar sakte hain."
    q = "checkout service me errors aa rahe hai, kisko page karu?"
    assert any("recent_changes was not called" in p
               for p in verify.check_answer(answer, steps, q, REACH))
    # Ownership with no failure reported needs no recent_changes.
    assert not any("recent_changes was not called" in p
                   for p in verify.check_answer(answer, steps, "who owns checkoutservice?", REACH))


@pytest.mark.parametrize("metric,flagged", [("pagerank", True), ("betweenness", False),
                                            (None, False)])
def test_most_critical_needs_betweenness(metric, flagged):
    """Rule 20, live Oct 16 (Gemini): "most critical" ranked by PageRank only."""
    args = {"metric": metric} if metric else {}
    steps = [{"tool": "critical_services", "args": args, "result": {"ranking": []}}]
    problems = verify.check_answer("The most critical services are listed above.", steps,
                                   "Which services are most critical?", REACH)
    assert any("betweenness" in p for p in problems) == flagged


def test_unrelated_uses_graph_reach_when_the_model_skipped_dependencies():
    """Rule 5, live Oct 16 (Gemini): frontend calls adservice, so adservice is not unrelated."""
    reach = dict(REACH, reach=dict(REACH["reach"], frontend={"adservice", "checkoutservice"}))
    steps = STEPS + [{"tool": "recent_changes", "args": {}, "result": {
        "deploys": [], "alerts": [{"service": "adservice", "severity": "medium",
                                   "message": "ad inventory cache miss rate high", "minutes_ago": 10}]}}]
    problems = verify.check_answer("Root cause: fraud-check v3.1.0.\nUnrelated: adservice.",
                                   steps, "frontend is returning errors, what broke?", reach)
    assert any("unrelated lists ['adservice']" in p for p in problems)


def test_negated_dependency_claim_is_not_flagged():
    """Rule 18 false positive from the saved-answer scan (Codestral, story C)."""
    answer = ("The adservice does not depend on checkoutservice, nor does checkoutservice "
              "depend on adservice.")
    problems = verify.check_answer(answer, STEPS, "Is the adservice alert related?", REACH)
    assert not any("in the graph it does not" in p for p in problems)


@pytest.mark.parametrize("answer", ["(Empty message, no tool call)",
                                    "<answer>\nI'll check the on-call information for the payments service.\n</answer>"])
def test_empty_reply_with_no_tool_is_flagged(answer):
    """Rule 16, live Oct 16 (StepFun): replies that answer nothing."""
    assert any("no tool was called" in p
               for p in verify.check_answer(answer, [], "Is Omar on call for payments?", VOCAB))


@pytest.mark.parametrize("answer,flagged", [
    ("Root cause: fraud-check v3.1.0 deployed by Daniel Okafor.\nPage: Payments team.", True),
    ("Yes, roll back fraud-check v3.1.0. Root cause: fraud-check v3.1.0.", False),
    ("Rolling back fraud-check v3.1.0 should clear the payments-db alert.", False),
    # Live Oct 16 (Nemotron): the "no" in "(no index)" once counted as an answer.
    ("Root cause: fraud-check v3.1.0 runs a slow query on payments-db (no index).\n"
     "Page: Payments team.", True),
    ("No. fraud-check v3.1.0 is not on the failing chain.", False),
])
def test_rollback_question_needs_yes_or_no(answer, flagged):
    """Rule 21, live Oct 16 (Nemotron): root cause given, rollback never answered."""
    problems = verify.check_answer(answer, STEPS, "Should we roll back fraud-check v3.1.0?", VOCAB)
    assert any("roll back" in p for p in problems) == flagged


@pytest.mark.parametrize("unrelated,flagged", [
    # Live Oct 16 (StepFun, F4): checkoutservice calls paymentservice -> fraud-check.
    ('checkoutservice "HTTP 5xx rate 23%": checkoutservice is not in paymentservice\'s list.', True),
    ("payments-db CPU 98%, a separate issue.", True),  # fraud-check calls payments-db
    # The subject is adservice; checkoutservice is only named to explain it (live A1).
    ("adservice: adservice is not in checkoutservice's dependency list.", False),
    # Saved StepFun story C: a second sentence in the item names checkoutservice first.
    ("adservice (medium). There is no dependency path between checkoutservice and adservice.",
     False),
    ("none", False),
])
def test_unrelated_must_not_list_the_root_cause_chain(unrelated, flagged):
    """Rule 5b: a service on the blamed deploy's chain is not unrelated."""
    answer = ("Root cause: fraud-check v3.1.0, deployed by Daniel Okafor.\n"
              f"Unrelated: {unrelated}\nPage: Payments team, on-call Priya Nair.")
    problems = verify.check_answer(answer, STEPS, "paymentservice p99 latency is 4.2s, why?", REACH)
    assert any("dependency chain of the root cause" in p for p in problems) == flagged


ALERTING = dict(REACH, alerting={"checkoutservice", "paymentservice", "payments-db", "adservice"})


@pytest.mark.parametrize("page,flagged", [
    # Live C2, Oct 16 (StepFun): names nobody.
    ("the fraud-check owner — use who_to_page on fraud-check for the team and on-call person.", True),
    ("fraud-check's owning team and on-call person — let me know if you'd like me to look that up.",
     True),
    # Live C1, Oct 16 (StepFun): "checkout" here is the outage, not the Checkout team.
    ("the fraud-check team and on-call, since fraud-check is the root cause of the checkout "
     "outage.", True),
    ("Payments team, on-call Priya Nair.", False),
    ("Payments team.", False),
    ("the Payments on-call.", False),
    ("not applicable.", False),
])
def test_page_section_must_name_someone(page, flagged):
    """Rule 4b: a Page section names an on-call person or a team."""
    answer = f"Root cause: fraud-check v3.1.0, deployed by Daniel Okafor.\nPage: {page}"
    problems = verify.check_answer(answer, STEPS, "Is the adservice alert caused by the deploy?",
                                   VOCAB)
    assert any("page section" in p for p in problems) == flagged


@pytest.mark.parametrize("answer,flagged", [
    # Saved Nemotron Nano answer: cut off before the Page line.
    ("Root cause: fraud-check v3.1.0, deployed by Daniel Okafor.\nUnrelated: adservice", True),
    (GOOD, False),
    (swap("on-call Priya Nair", "on-call: Priya"), False),
])
def test_who_to_page_question_needs_a_person(answer, flagged):
    """Rule 4c: "who should I page?" is answered with an on-call person."""
    problems = verify.check_answer(answer, STEPS, QUESTION, VOCAB)
    assert any("asks who to page" in p for p in problems) == flagged


@pytest.mark.parametrize("text,flagged", [
    # Saved Ministral 8B answer: only adservice had an alert.
    ("Unrelated: alerts on adservice, cartservice, emailservice", True),
    # The alert message names fraud-check; fraud-check is not said to have one.
    ('Evidence: alerts on payments-db (slow query from fraud-check) and paymentservice', False),
    # Rules answer wording for a service with no alert.
    ("Evidence: open alerts on cartservice: none.", False),
    ("There are no alerts on cartservice.", False),
])
def test_alerts_on_must_name_alerting_services(text, flagged):
    """Rule 10b: "alerts on X" names services with an open alert."""
    answer = ("Root cause: fraud-check v3.1.0, deployed by Daniel Okafor.\n" + text +
              "\nPage: Payments team, on-call Priya Nair.")
    problems = verify.check_answer(answer, STEPS, QUESTION, ALERTING)
    assert any("has an alert" in p for p in problems) == flagged


@pytest.mark.parametrize("answer,flagged", [
    # Live V2, Oct 16 (Nemotron): listed the alerts, paged nobody.
    ("Not everything is ok. Active alerts: checkoutservice HTTP 5xx rate 23%, "
     "payments-db CPU 98%. Consider the recent fraud-check deploy.", True),
    (GOOD, False),
])
def test_status_question_needs_someone_to_page(answer, flagged):
    """Rule 4c: "is everything ok?" with open alerts is answered with a page."""
    problems = verify.check_answer(answer, STEPS, "I just got paged, is everything ok?", VOCAB)
    assert any("asks who to page" in p for p in problems) == flagged


def test_names_with_no_break_spaces_still_match():
    """gpt-oss writes "Priya Nair"; it is the same person."""
    answer = swap("Page: Payments team, on-call Priya Nair.", "Page: **Payments** team – Priya Nair")
    assert verify.check_answer(answer, STEPS, QUESTION, VOCAB) == []
