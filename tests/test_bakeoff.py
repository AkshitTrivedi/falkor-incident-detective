"""Offline tests for the bake-off answer checks (no LLM or database calls).

The answers below are real model answers from a trial run, so the tests pin
down that good answers pass and known bad ones (refusals, made-up facts) fail.
"""
from bakeoff import QUESTION_SETS, _normalise, parse_spec
import bakeoff


def _out(*tools):
    """Fake investigate() result holding only the tools the agent called."""
    return {"steps": [{"tool": t} for t in tools]}


def _check(story):
    """The check function for one story letter."""
    return {s: c for s, _, c in QUESTION_SETS["all"]}[story]


def test_question_sets_have_unique_stories():
    """15 questions in total, each with its own story letter."""
    stories = [s for s, _, _ in QUESTION_SETS["all"]]
    assert len(stories) == 15 and len(set(stories)) == 15
    assert QUESTION_SETS["demo"] == bakeoff.QUESTIONS


def test_blast_radius_answer_passes():
    """E: an answer listing every affected service, from blast_radius, passes."""
    answer = ("If payments-db goes down: fraud-check (1 hop), paymentservice (1 hop), "
              "checkoutservice (2 hops), frontend (3 hops), api-gateway (4 hops)")
    assert _check("E")(_out("blast_radius"), _normalise(answer)) == []


def test_refusal_without_tool_fails():
    """H: refusing instead of calling dependency_path fails all three checks."""
    answer = "I'm sorry, but I don't have the capability to access the specific tools."
    assert len(_check("H")(_out(), _normalise(answer))) == 3


def test_made_up_root_cause_fails():
    """O: inventing a paymentservice deploy by Priya fails, even though Priya is paged."""
    answer = ("Root cause: paymentservice deploy by Priya Nair 10 minutes ago\n"
              "Page: Payments team, Priya Nair")
    failed = _check("O")(_out("recent_changes", "who_to_page"), _normalise(answer))
    assert failed == ["root cause fraud-check not named"]


def test_unknown_service_must_be_called_out():
    """L: saying billing-service was not found passes; listing callers does not."""
    good = "Sorry, I couldn't find the service 'billing-service'."
    bad = "billing-service is used by checkoutservice and frontend."
    assert _check("L")(_out("blast_radius"), _normalise(good)) == []
    assert _check("L")(_out("blast_radius"), _normalise(bad)) != []


def test_invented_cpu_figure_fails():
    """M: admitting there is no CPU data passes; a made-up percentage fails."""
    good = "I don't have access to real-time CPU metrics for frontend."
    bad = "I don't have live data, but frontend CPU is around 45%."
    assert _check("M")(_out(), _normalise(good)) == []
    assert "invented a CPU figure for frontend" in _check("M")(_out(), _normalise(bad))


def test_no_dependency_answer():
    """N: "No" with a graph check passes; "Yes" fails."""
    assert _check("N")(_out("dependency_path"),
                       _normalise("No, cartservice does not depend on payments-db.")) == []
    assert "answered yes" in _check("N")(_out("dependency_path"),
                                         _normalise("Yes, via checkoutservice."))


def test_parse_spec_keeps_colons_in_model_id():
    """OpenRouter ids end in ':free'; only the first ':' splits off the provider."""
    assert parse_spec("openrouter:nvidia/x:free") == ("openrouter", "nvidia/x:free")
