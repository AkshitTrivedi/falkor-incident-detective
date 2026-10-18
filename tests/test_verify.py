"""Offline checks for verify.py: no database and no LLM, so they are fast.

The tool results and graph vocabulary below are hand-written copies of what
the seeded graph returns for Story A. The bad answers are real ones from the
bake-off (docs/model-eval/bakeoff_*.jsonl), trimmed.
"""
import pytest

import verify

VOCAB = {"services": {"api-gateway", "frontend", "checkoutservice", "paymentservice",
                      "fraud-check", "payments-db", "adservice", "cartservice"},
         "people": {"priya nair", "daniel okafor", "elena petrova", "omar haddad"},
         "versions": {"v3.1.0", "v2.4.1"}}

STEPS = [
    {"tool": "recent_changes", "args": {}, "result": {
        "deploys": [{"service": "fraud-check", "version": "v3.1.0",
                     "by": "Daniel Okafor", "minutes_ago": 14}],
        "alerts": [{"service": "payments-db", "severity": "critical",
                    "message": "CPU 98%", "minutes_ago": 10}],
        "cypher": "MATCH ..."}},
    {"tool": "dependency_path", "args": {}, "result": {
        "path": ["checkoutservice", "paymentservice", "fraud-check"], "latency_ms": 48}},
    {"tool": "who_to_page", "args": {}, "result": {
        "service": "fraud-check", "team": "Payments", "on_call": "Priya Nair"}},
]

QUESTION = "checkoutservice is throwing 5xx errors. What broke?"

GOOD = """Root cause: fraud-check v3.1.0, deployed by Daniel Okafor 14 minutes ago.
Evidence: checkoutservice → paymentservice → fraud-check; payments-db CPU 98%.
Unrelated: adservice is not a dependency of checkoutservice.
Page: Payments team, on-call Priya Nair."""


def test_correct_answer_passes():
    """Qwen's real Story A answer, in the requested format, has no problems."""
    assert verify.check_answer(GOOD, STEPS, QUESTION, VOCAB) == []


def test_markdown_labels_are_understood():
    """Bold section labels ("**Page:**") are found like plain ones."""
    bold = GOOD.replace("Root cause:", "**Root cause:**").replace("Page:", "**Page:**")
    assert verify.check_answer(bold, STEPS, QUESTION, VOCAB) == []


def test_invented_deploy_is_caught():
    """Codestral, question O: blamed a deploy by Priya Nair that never happened."""
    bad = GOOD.replace("fraud-check v3.1.0, deployed by Daniel Okafor",
                       "paymentservice deploy by Priya Nair")
    problems = verify.check_answer(bad, STEPS, QUESTION, VOCAB)
    assert any("root cause names ['priya nair']" in p for p in problems)


def test_paging_the_deployer_is_caught():
    """Ministral 14B, Story C: paged the deployer instead of the on-call person."""
    bad = GOOD.replace("on-call Priya Nair", "on-call Daniel Okafor")
    problems = verify.check_answer(bad, STEPS, QUESTION, VOCAB)
    assert any("page names ['daniel okafor']" in p for p in problems)


@pytest.mark.parametrize("name", ["paymentsservice", "authservice", "payment-service"])
def test_unknown_service_names_are_caught(name):
    """Misspelt or invented service names (Ministral 3B wrote the first two)."""
    bad = GOOD.replace("→ paymentservice →", f"→ {name} →")
    assert verify.check_answer(bad, STEPS, QUESTION, VOCAB) == [
        f"'{name}' is not a service in the graph"]


def test_ordinary_hyphenated_words_are_not_services():
    """Real Story B wordings (Qwen, Ministral 14B) must not be read as service names."""
    answer = "paymentservice carries the most cross-service and service-to-service calls."
    problems = verify.check_answer(answer, [], "Which services are chokepoints?", VOCAB)
    assert not [p for p in problems if "not a service" in p]


def test_unknown_service_from_question_is_allowed():
    """Saying "billing-service doesn't exist" repeats the question; it is not invented."""
    answer = "There is no billing-service in the graph."
    assert verify.check_answer(answer, [], "Who owns billing-service?", VOCAB) == []


def test_version_not_returned_is_caught():
    """A real version from the graph that no tool returned is still unsupported."""
    bad = GOOD.replace("v3.1.0", "v2.4.1")
    problems = verify.check_answer(bad, STEPS, QUESTION, VOCAB)
    assert "'v2.4.1' is named, but no tool returned it" in problems


def test_curly_apostrophes_and_hyphens_do_not_break_matching():
    """Models often write ’ and the non-breaking hyphen ‑ ; both are folded first."""
    answer = GOOD.replace("fraud-check", "fraud‑check").replace("is not", "isn’t")
    assert verify.check_answer(answer, STEPS, QUESTION, VOCAB) == []


def test_dependency_called_unrelated_is_caught():
    """gpt-oss-120b, Story A: put paymentservice's alert under Unrelated, though it is on the path."""
    steps = STEPS + [{"tool": "recent_changes", "args": {}, "result": {"deploys": [], "alerts": [
        {"service": "paymentservice", "severity": "high", "message": "p99 latency 4.2s",
         "minutes_ago": 6}]}}]
    bad = GOOD.replace("Unrelated: adservice is not a dependency of checkoutservice.",
                       "Unrelated: adservice and paymentservice latency are not on the chain.")
    problems = verify.check_answer(bad, steps, QUESTION, VOCAB)
    assert any("unrelated lists ['paymentservice']" in p for p in problems)


def test_naming_the_root_cause_in_unrelated_is_fine():
    """"adservice (unrelated to fraud-check)" names fraud-check only for comparison."""
    answer = GOOD.replace("Unrelated: adservice is not a dependency of checkoutservice.",
                          "Unrelated: adservice (cache misses unrelated to fraud-check)")
    assert verify.check_answer(answer, STEPS, QUESTION, VOCAB) == []


def test_plain_answer_lists_other_tools():
    """Dependencies, blast radius, rankings and 'no path' notes are listed too."""
    text = verify.plain_answer([
        {"tool": "dependencies", "result": {"service": "a", "depends_on": [{"service": "b"}]}},
        {"tool": "blast_radius", "result": {"service": "b", "affected": []}},
        {"tool": "critical_services", "result": {
            "metric": "pagerank", "ranking": [{"service": "b", "score": 0.25}]}},
        {"tool": "dependency_path", "result": {"path": None, "note": "no dependency path"}},
    ])
    assert "- a depends on: b" in text
    assert "- If b fails, affected: nothing" in text
    assert "- Top by pagerank: b (0.25)" in text
    assert "- dependency_path: no dependency path" in text


def test_plain_answer_lists_tool_rows_only():
    """The no-model answer repeats each returned row and nothing else."""
    text = verify.plain_answer(STEPS)
    assert "Deploy: fraud-check v3.1.0 by Daniel Okafor, 14 min ago" in text
    assert "Alert (critical): payments-db: CPU 98%, 10 min ago" in text
    assert "Path: checkoutservice → paymentservice → fraud-check" in text
    assert "Page: Payments team, on call Priya Nair (for fraud-check)" in text
    assert "MATCH" not in text  # the Cypher is for the trace panel, not the answer


def test_plain_answer_with_no_rows():
    """With nothing returned, it says so instead of printing only the heading."""
    assert "No tool returned any rows." in verify.plain_answer([])


def test_invented_on_call_person_is_flagged():
    """Gemini's real answer (Oct 16) paged a made-up person, with no colon after "Page"."""
    bad = GOOD.replace("Page: Payments team, on-call Priya Nair.",
                       "Page payments-team (on-call: Sarah Jenkins)")
    problems = verify.check_answer(bad, STEPS, QUESTION, VOCAB)
    assert any("'Sarah Jenkins'" in p for p in problems)


def test_invented_person_in_page_section_is_flagged():
    """Round 3, story C: "Maya Lin" in the Page section; the real person is Maya Chen."""
    bad = GOOD.replace("on-call Priya Nair", "Maya Lin")
    problems = verify.check_answer(bad, STEPS, QUESTION, VOCAB)
    assert any("'Maya Lin'" in p for p in problems)


def test_team_names_and_headings_are_not_people():
    """Team names, "Page" and Title-case headings outside person contexts are not flagged."""
    vocab = {**VOCAB, "teams": {"payments", "catalog & logistics"}}
    answer = GOOD + "\n\nKey Observations: Team Payments owns it. Ranked By Betweenness Score."
    assert verify.check_answer(answer, STEPS, QUESTION, vocab) == []


class _CountingGraph:
    """Wraps a graph and counts the queries sent through it."""

    def __init__(self, g):
        self._g, self.name, self.calls = g, g.name, 0

    def ro_query(self, *args, **kwargs):
        self.calls += 1
        return self._g.ro_query(*args, **kwargs)


def test_vocabulary_is_fetched_once_and_cleared_by_seed():
    """Repeated lookups cost no queries; re-seeding forgets the cached names."""
    import seed
    from db import get_graph
    graph = get_graph("incidents_verify_cache_test")
    seed.seed(graph)                       # clears the cache
    counted = _CountingGraph(graph)
    first = verify.graph_vocabulary(counted)
    after_first = counted.calls
    assert after_first > 0
    assert verify.graph_vocabulary(counted) is first
    assert verify.service_names(counted) == verify.service_names(counted)
    assert counted.calls == after_first + 1      # only service_names' first fetch
    seed.seed(graph)
    verify.graph_vocabulary(counted)
    assert counted.calls > after_first + 1
    graph.delete()


def test_page_may_name_the_person_an_incident_already_paged():
    """recent_changes' incidents back "already paged" without a who_to_page call."""
    steps = [{"tool": "recent_changes", "args": {}, "result": {
        "deploys": [], "alerts": [],
        "incidents": [{"id": "INC-1", "status": "mitigated", "cause": "fraud-check v3.1.0",
                       "paged": "Priya Nair", "opened_minutes_ago": 3}]}}]
    facts = verify._returned(steps)
    assert "priya nair" in facts["on_call"]
