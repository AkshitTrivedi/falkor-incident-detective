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
