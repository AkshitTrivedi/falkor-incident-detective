"""Compare LLM models on the three demo questions (the "bake-off").

Each model answers the same questions against its own freshly seeded local
graph. An answer passes when it names the facts the graph says are correct,
states nothing the graph says is wrong, and the agent called the tool that
proves them. This picks the model that uses the tools most reliably, which
matters more than raw model quality here.

Every run is also saved as one JSON line (answer, tools called, tokens, time)
so the answers can be compared by hand for quality, not just pass/fail.

Usage:
  python bakeoff.py                                   # each provider's default model
  python bakeoff.py groq gemini                       # only these providers
  python bakeoff.py groq:qwen/qwen3.8-27b nvidia:z-ai/glm-5.3   # specific models
  python bakeoff.py --out results.jsonl groq          # also save answers
  python bakeoff.py --repeat 3 --pause 20 groq        # 3 rounds, 20 s between questions
  python bakeoff.py --set all mistral:codestral-2508  # all 15 questions (demo + extended)
Needs local FalkorDB running and the API keys in .env.
"""
import argparse
import contextlib
import json
import re
import statistics
import time

from openai import APIError

from agent import PROVIDERS, investigate
from db import get_graph
from seed import seed
import verify

# Default pause between questions, so free tiers measured per minute (Groq:
# 8k tokens per model, Mistral: about 1 request a second) are less likely to
# reject the next run.
PAUSE_S = 5

# A section label at the start of a line, e.g. "Page:" or "**Unrelated alerts:**".
# {} is filled with the label name(s); up to 20 extra characters may precede the ':'.
_LABEL = r"(?:^|\n)[*_#\s-]*(?:{})[^:\n]{{0,20}}:"
_SECTION_NAMES = "root cause|evidence|unrelated|page"


def _called(out, tool):
    """True if the agent called `tool` at least once."""
    return any(s["tool"] == tool for s in out["steps"])


def _normalise(answer):
    """Lower-case the answer and swap typographic characters for plain ones.

    Some models write "fraud‑check" with a non-breaking hyphen (U+2011) or put
    narrow no-break spaces (U+202F) in names, which defeats substring checks.
    """
    for fancy, plain in (("‐", "-"), ("‑", "-"), (" ", " "), (" ", " ")):
        answer = answer.replace(fancy, plain)
    answer = answer.replace("’", "'")  # curly apostrophe: "don’t" -> "don't"
    return answer.lower()


def _section(a, name):
    """Return the text of the answer's `name` section, or "" if it has none.

    The section runs from its label (see _LABEL) to the next section label or
    the end. Example: _section(a, "page") on "...\\nPage: Payments team" gives
    " Payments team".
    """
    start = re.search(_LABEL.format(name), a)
    if not start:
        return ""
    rest = a[start.end():]
    end = re.search(_LABEL.format(_SECTION_NAMES), rest)
    return rest[:end.start()] if end else rest


def _failures(checks):
    """Return the messages of the failed checks; each check is (ok, message)."""
    return [msg for ok, msg in checks if not ok]


def _mistakes(a):
    """Checks for wrong facts that any of the answers could state.

    - paymentservice sits on the failing path (checkout -> payment -> fraud-check),
      so listing it as unrelated is wrong.
    - Priya Nair is the one to page. Daniel Okafor deployed the bad change but is
      not on call; Omar Haddad is on call for Checkout, the victim, not the cause.

    Returns:
        [(ok, message), ...] for _failures().
    """
    page = _section(a, "page")
    wrong_person = "priya" not in page and any(n in page for n in ("daniel", "omar"))
    return [
        ("paymentservice" not in _section(a, "unrelated"),
         "called paymentservice unrelated, but it is on the failing path"),
        (not wrong_person, "paged the wrong person (on call is Priya Nair)"),
    ]


def check_a(out, a):
    """Story A: fraud-check v3.1.0 is the root cause; page Payments on-call Priya Nair.

    Args:
        out: investigate() result.
        a: The answer after _normalise().

    Returns:
        List of failed expectations; empty means pass. (Same for check_b and check_c.)
    """
    return _failures([
        ("fraud-check" in a, "root cause fraud-check not named"),
        ("v3.1.0" in a, "deploy version v3.1.0 not named"),
        ("priya" in a, "on-call Priya Nair not named"),
        (_called(out, "who_to_page"), "who_to_page not called"),
        *_mistakes(a),
    ])


def check_b(out, a):
    """Story B: paymentservice is the top chokepoint, found with critical_services."""
    return _failures([
        ("paymentservice" in a, "top chokepoint paymentservice not named"),
        (_called(out, "critical_services"), "critical_services not called"),
        *_mistakes(a),
    ])


def check_c(out, a):
    """Story C: the adservice alert is unrelated, shown by a graph check."""
    return _failures([
        (any(w in a for w in ("not related", "unrelated", "no dependency", "no path",
                              "isn't related", "is not")), "did not say unrelated"),
        (_called(out, "dependency_path") or _called(out, "dependencies"),
         "no graph check of the relationship"),
        *_mistakes(a),
    ])


# (story, question, check function).
QUESTIONS = [
    ("A", "checkoutservice is throwing 5xx errors. What broke and who should I page?", check_a),
    ("B", "Which services are the biggest chokepoints in our architecture?", check_b),
    ("C", "There is also an alert on adservice right now. Is it related to the checkout outage?",
     check_c),
]


# --- Extended set: 12 more questions, one per kind of question an on-call engineer
# asks. Expected answers below were read from the tools on a freshly seeded graph.

def _missing(a, names, what):
    """One failed check per name in `names` that the answer does not mention."""
    return [(n in a, f"{what} {n} not named") for n in names]


def _tool_used(out, *tools):
    """Check that the agent called at least one of `tools` (it must not guess)."""
    return (any(_called(out, t) for t in tools), f"{' or '.join(tools)} not called")


def check_d(out, a):
    """D: checkoutservice's six direct dependencies, from the dependencies tool."""
    return _failures([
        *_missing(a, ("cartservice", "currencyservice", "emailservice", "paymentservice",
                      "productcatalog", "shippingservice"), "dependency"),
        _tool_used(out, "dependencies"),
    ])


def check_e(out, a):
    """E: payments-db outage reaches fraud-check, paymentservice, checkoutservice,
    frontend and api-gateway (blast_radius walks callers upstream)."""
    return _failures([
        *_missing(a, ("fraud-check", "paymentservice", "checkoutservice", "frontend",
                      "api-gateway"), "affected service"),
        _tool_used(out, "blast_radius"),
    ])


def check_f(out, a):
    """F: the only change in the last hour is fraud-check v3.1.0 by Daniel Okafor."""
    return _failures([
        *_missing(a, ("fraud-check", "v3.1.0", "daniel"), "recent change detail"),
        _tool_used(out, "recent_changes"),
    ])


def check_g(out, a):
    """G: the 6-hop chain api-gateway -> frontend -> checkoutservice -> paymentservice
    -> fraud-check -> risk-model -> feature-store (the deepest path in the graph)."""
    return _failures([
        *_missing(a, ("frontend", "checkoutservice", "paymentservice", "fraud-check",
                      "risk-model"), "hop"),
        _tool_used(out, "dependency_path", "dependencies"),
    ])


def check_h(out, a):
    """H: fastest path checkoutservice -> paymentservice -> payments-db, 36 ms
    (the route through fraud-check is 54 ms)."""
    return _failures([
        ("paymentservice" in a, "hop paymentservice not named"),
        ("36" in a, "latency 36 ms not given"),
        _tool_used(out, "dependency_path"),
    ])


def check_i(out, a):
    """I: inventory-db ranks first by PageRank (most depended-on, directly or not)."""
    return _failures([
        ("inventory-db" in a, "top PageRank service inventory-db not named"),
        _tool_used(out, "critical_services"),
    ])


def check_j(out, a):
    """J: risk-model is owned by Platform/Data; Aisha Bello is on call."""
    return _failures([
        ("platform" in a, "owning team Platform/Data not named"),
        ("aisha" in a, "on-call Aisha Bello not named"),
        _tool_used(out, "who_to_page"),
    ])


def check_k(out, a):
    """K: adservice belongs to Storefront, so page Maya Chen, not Payments."""
    return _failures([
        ("maya" in a, "on-call Maya Chen not named"),
        ("priya" not in _section(a, "page"), "paged Priya Nair for an adservice alert"),
        _tool_used(out, "who_to_page"),
    ])


# Phrases an answer uses when it says the graph does not have something;
# the same list the answer checker uses for rule 11.
_NOT_FOUND = verify.NOT_FOUND
_NO_DATA = ("don't have", "do not have", "not available", "isn't available", "cannot",
            "can't", "no data", "not tracked", "doesn't include", "does not include",
            "no metric", "not stored", "unable", "doesn't contain", "does not contain",
            "no cpu", "not track", "isn't tracked", "doesn't track", "does not track")


def check_l(out, a):
    """L: billing-service does not exist; the agent must say so, not invent callers."""
    plain = re.sub(r"[`*]", "", a)  # "no `billing-service`" -> "no billing-service"
    return _failures([
        (any(p in plain for p in _NOT_FOUND), "did not say billing-service is unknown"),
        _tool_used(out, "blast_radius", "dependencies", "dependency_path", "who_to_page"),
    ])


def check_m(out, a):
    """M: the graph has no CPU metrics for frontend; the agent must not make one up."""
    return _failures([
        (any(p in a for p in _NO_DATA), "did not say CPU data is unavailable"),
        (not re.search(r"frontend[^.\n]{0,40}\d+(\.\d+)?\s?%", a),
         "invented a CPU figure for frontend"),
    ])


def check_n(out, a):
    """N: cartservice has no dependency path to payments-db in either direction."""
    return _failures([
        (bool(re.search(r"\b(no|not|doesn't|does not|isn't)\b", a)),
         "did not say cartservice does not depend on payments-db"),
        (not a.lstrip(" *#").startswith("yes"), "answered yes"),
        _tool_used(out, "dependency_path", "dependencies", "blast_radius"),
    ])


def check_o(out, a):
    """O: Story A in other words (no service named): still page Priya Nair."""
    return _failures([
        ("fraud-check" in a, "root cause fraud-check not named"),
        ("priya" in a, "on-call Priya Nair not named"),
        _tool_used(out, "who_to_page"),
        *_mistakes(a),
    ])


EXTENDED = [
    ("D", "What does checkoutservice depend on?", check_d),
    ("E", "If payments-db goes down, which services are affected?", check_e),
    ("F", "What changed in the last hour?", check_f),
    ("G", "Show me how a request to api-gateway reaches feature-store.", check_g),
    ("H", ("What is the fastest call path from checkoutservice to payments-db, "
           "and how long does it take?"), check_h),
    ("I", "Which services are the most important by PageRank?", check_i),
    ("J", "Who owns risk-model and who is on call for it?", check_j),
    ("K", "The adservice alert needs a human. Who should I page?", check_k),
    ("L", "What depends on billing-service?", check_l),
    ("M", "What is the current CPU usage of frontend?", check_m),
    ("N", "Does cartservice depend on payments-db?", check_n),
    ("O", "Customers can't complete orders since a few minutes ago. Who do I wake up?",
     check_o),
]

# --set choices: the 3 demo questions, the 12 extended ones, or all 15.
QUESTION_SETS = {"demo": QUESTIONS, "extended": EXTENDED, "all": QUESTIONS + EXTENDED}


def parse_spec(spec):
    """Split "provider" or "provider:model" into (provider, model).

    Model ids can contain ':' themselves (e.g. "...:free"), so only the first
    ':' separates the provider. A bare provider gets its default model.
    """
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider '{provider}', choose from {list(PROVIDERS)}")
    return provider, model or PROVIDERS[provider][2]


def run(specs, out_path=None, repeat=1, pause=PAUSE_S, questions=QUESTIONS):
    """Run every question on every model, printing a line per run and a summary.

    Args:
        specs: List of "provider" or "provider:model" strings.
        out_path: Optional JSONL file; one line per run is appended to it.
        repeat: How many rounds of the question set each model gets.
        pause: Seconds to wait after each question.
        questions: (story, question, check) list; one of QUESTION_SETS.

    Returns:
        The list of result rows (the same dicts written to out_path).
    """
    # nullcontext() stands in for a file when no --out path is given.
    with open(out_path, "a") if out_path else contextlib.nullcontext() as out_file:
        rows = _run_specs(specs, out_file, repeat, pause, questions)
    _print_summary(rows)
    return rows


def _run_specs(specs, out_file, repeat, pause, questions):
    """Run all rounds of all questions for each spec; helper for run().

    Args:
        specs, repeat, pause, questions: As for run().
        out_file: Open file to append JSON lines to, or None.

    Returns:
        List of result rows, one per question asked.
    """
    rows = []
    for spec in specs:
        provider, model = parse_spec(spec)
        # Each model gets its own graph, so several bake-offs can run in parallel,
        # even for models of the same provider.
        g = get_graph("bakeoff_" + re.sub(r"\W+", "_", f"{provider}_{model}"))
        try:
            for round_no in range(1, repeat + 1):
                seed(g)  # fresh story timestamps for every round
                for story, question, check in questions:
                    row = _ask(g, provider, model, round_no, story, question, check)
                    rows.append(row)
                    if out_file:
                        out_file.write(json.dumps(row) + "\n")
                        out_file.flush()
                    time.sleep(pause)
        finally:
            g.delete()  # the bake-off graph is throwaway
    return rows


def _ask(g, provider, model, round_no, story, question, check):
    """Ask one question on one model, print a result line and return its row.

    Provider errors (rate limit, bad model id, timeout...) and a missing API
    key are recorded in the row instead of stopping the bake-off. Any other
    exception is a bug in our code, so it is not caught.
    """
    row = {"provider": provider, "model": model, "round": round_no, "story": story}
    label = f"{provider}:{model}  {story}{round_no}"
    try:
        out = investigate(question, g=g, provider=provider, model=model)
        failed = check(out, _normalise(out["answer"] or ""))
        # The bake-off grades the model, so a rules answer (after the model's
        # answer failed verify.py twice) counts against it even if it is right.
        if out["check"]["by_rules"]:
            failed = ["model answer failed the answer check; rules answered"] + failed
        row.update(passed=not failed, failed=failed, seconds=out["seconds"],
                   llm_calls=out["llm_calls"], tokens=out["tokens"],
                   tools=[s["tool"] for s in out["steps"]], answer=out["answer"],
                   check=out["check"])  # verify.py's verdict, incl. retry and plain fallback
        status = "PASS" if not failed else "FAIL: " + "; ".join(failed)
        print(f"{label}  {out['seconds']}s  {len(out['steps'])} tools  "
              f"{sum(out['tokens'].values())} tokens  {status}", flush=True)
    except (APIError, RuntimeError) as e:
        row.update(passed=False, error=str(e)[:1000])  # long enough for quota details
        print(f"{label}  ERROR: {str(e)[:150]}", flush=True)
    return row


def _median(values):
    """Median of the values as text, or "-" if there are none."""
    return f"{statistics.median(values):g}" if values else "-"


def _mean(values):
    """Average of the values as text (one decimal), or "-" if there are none."""
    return f"{statistics.mean(values):.1f}" if values else "-"


def _print_summary(rows):
    """Print one line per model: passes, errors, median and average time, tool calls, tokens.

    Figures cover only the runs that got an answer. The median is the typical run;
    the average also shows how much the slow runs (e.g. rate-limit waits) drag it up.
    """
    print(f"\n{'model':52} {'pass':>6} {'errors':>6} {'median s':>9} {'avg s':>6} "
          f"{'tools':>6} {'tokens':>7}")
    for key in dict.fromkeys(f"{r['provider']}:{r['model']}" for r in rows):
        mine = [r for r in rows if f"{r['provider']}:{r['model']}" == key]
        answered = [r for r in mine if "error" not in r]
        passed = sum(r["passed"] for r in mine)
        print(f"{key:52} {passed:>3}/{len(mine):<2} {len(mine) - len(answered):>6} "
              f"{_median([r['seconds'] for r in answered]):>9} "
              f"{_mean([r['seconds'] for r in answered]):>6} "
              f"{_median([len(r['tools']) for r in answered]):>6} "
              f"{_median([sum(r['tokens'].values()) for r in answered]):>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLM models on the demo questions.")
    parser.add_argument("specs", nargs="*",
                        help='"provider" or "provider:model"; default: every provider')
    parser.add_argument("--out", help="append one JSON line per run to this file")
    parser.add_argument("--repeat", type=int, default=1, help="rounds of the question set")
    parser.add_argument("--pause", type=float, default=PAUSE_S,
                        help="seconds to wait after each question")
    parser.add_argument("--set", default="demo", choices=QUESTION_SETS,
                        help="demo = 3 story questions, extended = 12 more, all = 15")
    args = parser.parse_args()
    run(args.specs or list(PROVIDERS), args.out, args.repeat, args.pause,
        QUESTION_SETS[args.set])
