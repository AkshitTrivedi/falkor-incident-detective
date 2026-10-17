"""The Incident Detective agent: an LLM that investigates outages with graph tools.

How it works:
  1. The user asks a question, e.g. "checkoutservice is throwing 5xx, what broke?"
  2. The LLM sees the six tools from tools.py (as OpenAI function schemas) and
     picks one. We run it against FalkorDB and send the result back.
  3. Steps repeat until the LLM writes a final answer or MAX_STEPS is reached.
  4. verify.check_answer() compares the answer with the tool results. If it
     states anything the tools do not support, the LLM gets one chance to fix
     it; if the fix fails too, the user gets diagnose.rule_answer(), worked out
     by fixed rules with no model.
  5. The same rules answer when every model fails or the question runs past
     QUESTION_DEADLINE_S, so the user always gets an answer from the graph.

The LLM never sees or writes Cypher; it can only call the fixed tools. Every
step is logged with the Cypher that ran, so the answer can be traced back to
the graph.

Providers: any OpenAI-compatible API. By default a question goes to the first
model in FALLBACK_CHAIN; if that model is rate-limited, too slow or down, the
question restarts on the next one. To use one model only, with no fallback:
  LLM_PROVIDER  groq | gemini | mistral | openrouter | nvidia
  LLM_MODEL     optional; overrides the provider's default model
API keys are read from .env (see PROVIDERS below for the variable names).

Usage:
  python agent.py "checkoutservice is throwing 5xx errors, what broke?"
  LLM_PROVIDER=gemini python agent.py "which services are chokepoints?"
"""
import difflib
import json
import logging
import os
import re
import sys
import threading
import time

import httpx2  # the HTTP library openai uses; needed to build its timeout error
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI

import diagnose
import seed
import tools
import verify
from db import get_graph

# Progress and errors for the terminal: each model tried, each LLM call and
# tool call, check failures, fallbacks. app.py sets the level from LOG_LEVEL
# (INFO by default; WARNING keeps only problems, e.g. when deployed).
log = logging.getLogger("detective.agent")

load_dotenv()

# Upper limit on tool-calling rounds per question. Story A needs about 4-5
# (recent changes, dependencies, path, who to page); the cap stops a confused
# model from looping forever and burning free-tier quota.
MAX_STEPS = 8

# provider -> (OpenAI-compatible base URL, env var holding the key, default model).
# Default models were chosen from each provider's live model list on 2026-10-16.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               "gemini-3.1-flash-lite"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-medium-latest"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "nvidia/nemotron-3-super-120b-a12b:free"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
               "nvidia/nemotron-3-super-120b-a12b"),
}

# Models tried in order for each question, picked by the 15-question test on
# 2026-10-16 (docs/MODEL_EVALUATION.md). Each entry is a different provider, so
# one provider's outage or quota can't take out two links of the chain.
# Codestral (fastest, 1.9 s) is left out on purpose: it blamed a deploy that
# doesn't exist on question O in all 3 rounds.
FALLBACK_CHAIN = [
    # 45/45 correct, median 8.6 s (includes Groq's 8k tokens/min waits). Groq
    # marks it "Preview"; if it is withdrawn, the next model answers.
    ("groq", "qwen/qwen3.8-27b"),
    # Every answer it gave was correct (37/37), but 8 of 45 calls hit Google's
    # 503 "overloaded"; a 503 simply moves the question down the chain.
    ("gemini", "gemini-3.1-flash-lite"),
    # 43/45, median 5.7 s; once left out "unrelated", once skipped the path tool.
    ("nvidia", "stepfun-ai/step-5-preview"),
    # 45/45 but slow (median 8.1 s, average 19.8 s, up to 92 s): last resort.
    ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
]

# Seconds to wait for one LLM reply before giving up. Without it the SDK waits
# up to 10 minutes and retries silently, which hides an overloaded provider.
LLM_TIMEOUT_S = 60
# Shorter wait for a model that has a fallback after it: a reply this slow
# means the provider is struggling, and the next model is likely faster.
FAILOVER_TIMEOUT_S = 20
# Hard limits on wall-clock time. The timeouts above are per network read, so a
# provider that trickles bytes can pass them: in testing one question took 946 s.
# QUESTION_DEADLINE_S covers the whole question across every model; after it,
# diagnose.rule_answer() answers. MODEL_BUDGET_S caps one model's whole
# attempt when another model follows it.
QUESTION_DEADLINE_S = 90
MODEL_BUDGET_S = 30

# The investigation procedure the LLM follows. Kept short: free tiers limit
# tokens per minute, and this prompt is resent on every step.
SYSTEM_PROMPT = """You are Incident Detective, an on-call assistant for a microservice platform.
You investigate incidents ONLY with the provided tools, which query a graph of services,
dependencies, deploys, alerts, teams and on-call people. Never invent services, people, teams,
versions or numbers: every name and figure in your answer must appear in a tool result.
Write for the on-call engineer: say "the graph", never "the tool results above".
If the question names a service, person, team or version the graph does not have, start your
answer with a sentence that names it and says it is not in the graph ("There is no SRE team in
the graph."). If no tool can answer (for example CPU or memory metrics), say so.
Put your whole answer in your final message, not next to a tool call.

For "what broke?" questions:
1. Call recent_changes to see recent deploys and open alerts.
2. Call dependencies on the failing service to find what it relies on.
3. A recent deploy to a service the failing one depends on is the prime suspect. Confirm the
   link with dependency_path.
4. Alerts on services that are on that path are symptoms of the same incident. Alerts on
   services that call the failing service (blast_radius) are its effects. An alert on a service
   with no dependency path to or from the failing service is unrelated.
5. Call who_to_page on the root-cause service.

Final answer format for "what broke?" questions (plain text, short):
Root cause: <service and change, with who deployed it and when>
Evidence: <the dependency path and the alerts along it>
Unrelated: <alerts on services outside the dependencies list, if any>
Page: <team and on-call person>
Other questions (owners, on-call, dependencies, blast radius) get a short plain answer
without these labels."""

# Sent after an answer fails verify.check_answer(); {problems} is one per line.
RETRY_PROMPT = """Your answer states things the tool results do not support:
{problems}
Rewrite the answer in the same format using only services, people, teams, versions and numbers
that appear in the tool results above. If something the question names is not in the graph,
start with a sentence that names it and says it is not in the graph. Do not call any more tools."""

# OpenAI function-calling schemas for tools.TOOLS. Descriptions tell the LLM
# when to use each tool; names must match the functions in tools.py.
_SERVICE = {"type": "string", "description": "Exact service name, e.g. 'checkoutservice'"}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "recent_changes",
        "description": "Deploys and unresolved alerts in the last N minutes, newest first. "
                       "Start here for any 'what broke' question.",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer", "description": "Look-back window, default 60"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "dependencies",
        "description": "All services the given service calls, directly or indirectly, with "
                       "hop distance. These are the possible root causes of its failure.",
        "parameters": {"type": "object", "properties": {"service": _SERVICE},
                       "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "blast_radius",
        "description": "All services that call the given service, directly or indirectly: "
                       "what breaks if it fails.",
        "parameters": {"type": "object", "properties": {"service": _SERVICE},
                       "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "dependency_path",
        "description": "Lowest-latency call path between two services, either direction. "
                       "path=null means they are unrelated. 'direction' says which one "
                       "depends on the other.",
        "parameters": {"type": "object", "properties": {
            "from_service": _SERVICE, "to_service": _SERVICE},
            "required": ["from_service", "to_service"]}}},
    {"type": "function", "function": {
        "name": "critical_services",
        "description": "Rank services by graph importance. 'betweenness' = chokepoints many "
                       "call paths pass through; use it for 'most critical'. 'pagerank' = "
                       "most depended on; use it only when asked what others depend on most.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["betweenness", "pagerank"]},
            "limit": {"type": "integer", "description": "How many to return, default 5"}},
            "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "who_to_page",
        "description": "Owning team and current on-call person for a service. "
                       "A team name (e.g. Payments) also works.",
        "parameters": {"type": "object", "properties": {"service": _SERVICE},
                       "required": ["service"]}}},
]


def make_client(provider, model=None, fail_fast=False):
    """Create an OpenAI-compatible client for one provider.

    Args:
        provider: Key of PROVIDERS, e.g. "groq".
        model: Model id. Defaults to the provider's default model.
        fail_fast: True when another model will take over if this one fails:
            short timeout and no retries, so the switch happens quickly.
            False waits longer and retries once (the SDK honours the
            retry-after header of a 429), for when there is no fallback.

    Returns:
        (client, model) ready for chat.completions.create().

    Raises:
        ValueError: unknown provider.
        RuntimeError: the provider's API key is missing from the environment.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}', choose from {list(PROVIDERS)}")
    base_url, key_var, default_model = PROVIDERS[provider]
    if not os.environ.get(key_var):
        raise RuntimeError(f"{key_var} is not set (add it to .env)")
    client = OpenAI(base_url=base_url, api_key=os.environ[key_var],
                    timeout=FAILOVER_TIMEOUT_S if fail_fast else LLM_TIMEOUT_S,
                    max_retries=0 if fail_fast else 1)
    return client, model or default_model


def model_chain(provider=None, model=None):
    """List the (provider, model) pairs to try for one question, in order.

    Args:
        provider: Use only this provider, with no fallback. Defaults to env
            LLM_PROVIDER. The bake-off relies on this to measure exactly the
            model it asked for.
        model: Model id for that provider. Defaults to env LLM_MODEL, then
            the provider's default (model None means "the default").

    Returns:
        [(provider, model)] when a provider is chosen. Otherwise
        FALLBACK_CHAIN without the providers whose API key is not set, which
        can be empty.
    """
    provider = provider or os.environ.get("LLM_PROVIDER")
    if provider:
        return [(provider, model or os.environ.get("LLM_MODEL"))]
    return [(p, m) for p, m in FALLBACK_CHAIN if os.environ.get(PROVIDERS[p][1])]


def run_tool(g, name, raw_args):
    """Run one tool call requested by the LLM, never raising.

    Any problem (unknown tool, bad JSON, wrong arguments, query error) comes
    back as {"error": ...} so the LLM can read it and try again.

    Args:
        g: FalkorDB Graph handle.
        name: Tool name chosen by the LLM.
        raw_args: JSON string of arguments from the LLM.

    Returns:
        (args, result): the parsed arguments (or {} if unparseable) and the
        tool's result dict.
    """
    if name not in tools.TOOLS:
        return {}, {"error": f"unknown tool '{name}', choose from {list(tools.TOOLS)}"}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return {}, {"error": f"arguments are not valid JSON: {e}"}
    try:
        return args, tools.TOOLS[name](g, **args)
    except TypeError as e:  # missing or unexpected argument names
        return args, {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # e.g. a query timeout; report it instead of crashing
        log.exception("tool %s(%s) raised", name, args)   # full traceback in the terminal
        return args, {"error": f"{name} failed: {e}"}


def _for_llm(result):
    """Return the tool result without its Cypher and params, to save LLM tokens.

    The Cypher is for people (shown in the UI); the LLM only needs the data.
    """
    return json.dumps({k: v for k, v in result.items() if k not in tools.TRACE_KEYS})


def investigate(question, g=None, provider=None, model=None, max_steps=MAX_STEPS):
    """Answer an incident question by letting the LLM call graph tools in a loop.

    Tries the models from model_chain() in order. If one fails (rate limit,
    timeout, outage, model withdrawn), the question restarts from scratch on
    the next, so every answer comes from a single model. If every model fails
    or QUESTION_DEADLINE_S passes, diagnose.rule_answer() answers instead
    (provider "rules").

    The demo graph is re-seeded first if its story has gone stale
    (seed.ensure_fresh); set AUTO_RESEED=0 to turn that off.

    Args:
        question: The user's question in plain English.
        g: FalkorDB Graph handle. Defaults to db.get_graph().
        provider, model: Use only this model, with no fallback. See model_chain().
        max_steps: Maximum number of LLM rounds per model.

    Returns:
        {"answer": str,             final text from the LLM, or from the rules
         "steps": [{"tool", "args", "result", "ms"}, ...],   in call order
         "queries": [str, ...],     every Cypher query that ran
         "path": [str, ...] or None, the last dependency path found, for the UI
         "provider": str, "model": str,   the model that answered ("rules", None if none)
         "llm_calls": int,          LLM requests made by that model
         "tokens": {"prompt": int, "completion": int},   summed over those requests
         "check": {"problems": [str, ...], "retried": bool, "by_rules": bool},
                                    see _check(); problems are from the first answer
         "fallbacks": [{"provider", "model", "error"}, ...],   models that failed first
         "seconds": float}          total time, including failed attempts

    Raises:
        RuntimeError: one model was asked for (provider/model) and its API key
            is not set. With no key at all and no pinned model, the rules answer.
        openai.APIError: only when one model was asked for (provider/model, as
            the bake-off does) and it failed; the bake-off records the error.
    """
    g = g or get_graph()
    if os.getenv("AUTO_RESEED", "1") == "1":
        seed.ensure_fresh(g)
    pinned = bool(provider or os.environ.get("LLM_PROVIDER"))
    chain = model_chain(provider, model)
    started = time.time()
    if not chain:
        # Only an unpinned chain can be empty. Nobody set a key: the rules diagnosis still answers from the graph.
        log.warning("no LLM API key is set; answering with the rules")
        return _rules_result(g, question, [], started)
    deadline = started + QUESTION_DEADLINE_S
    fallbacks = []
    log.info("question: %r", question)
    if not pinned and diagnose.needs_no_model(g, question):
        # "Delete all alerts", "hello there": nothing to look up, so the rules
        # reply at once (see diagnose.needs_no_model).
        log.info("no model needed; answering with the rules")
        return _rules_result(g, question, fallbacks, started)
    for i, (p, m) in enumerate(chain):
        is_last = i == len(chain) - 1
        if time.time() >= deadline:
            log.warning("skip %s/%s: question deadline (%ss) passed", p, m, QUESTION_DEADLINE_S)
            fallbacks.append({"provider": p, "model": m, "error": "skipped: question deadline"})
            continue
        model_deadline = deadline if is_last else min(deadline, time.time() + MODEL_BUDGET_S)
        log.info("trying %s/%s", p, m)
        try:
            out = _investigate_once(question, g, p, m, max_steps, fail_fast=not is_last,
                                    deadline=model_deadline)
        except APIError as e:  # includes timeouts and connection errors
            log.warning("%s/%s failed: %s: %s", p, m, type(e).__name__, str(e)[:300])
            if pinned:
                raise
            fallbacks.append({"provider": p, "model": m,
                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        out.update(fallbacks=fallbacks, seconds=round(time.time() - started, 1))
        log.info("answered by %s/%s in %ss: %d LLM calls, %d tokens, %d tool calls%s",
                 out["provider"], out["model"], out["seconds"], out["llm_calls"],
                 sum(out["tokens"].values()), len(out["steps"]),
                 ", by rules after failed check" if out["check"]["by_rules"] else "")
        return out

    # Every model failed or time ran out: answer from the graph with the rules.
    log.warning("no model answered; using the rules diagnosis")
    return _rules_result(g, question, fallbacks, started)


def _rules_result(g, question, fallbacks, started):
    """investigate()'s result, answered by diagnose.rule_answer() (provider "rules")."""
    rules = diagnose.rule_answer(g, question)
    return {"answer": rules["answer"], "steps": rules["steps"],
            "queries": _queries(rules["steps"]), "path": rules["path"],
            "provider": "rules", "model": None, "llm_calls": 0,
            "tokens": {"prompt": 0, "completion": 0},
            "check": {"problems": [], "retried": False, "by_rules": True},
            "fallbacks": fallbacks, "seconds": round(time.time() - started, 1)}


def _queries(steps):
    """Return every Cypher query that ran in `steps`, in order (for the UI trace).

    Each query has its parameter values written in (tools.runnable), so it can
    be copied from the UI and run as is in FalkorDB Browser.
    """
    out = []
    for step in steps:
        cypher, params = step["result"].get("cypher"), step["result"].get("params")
        for c in cypher if isinstance(cypher, list) else [cypher] if cypher else []:
            out.append(tools.runnable(c, params))
    return out


def _investigate_once(question, g, provider, model, max_steps, fail_fast, deadline=None):
    """Run the tool-calling loop on one model; helper for investigate().

    Args:
        question, g, max_steps: As for investigate().
        provider, model: The model to use (model None = the provider's default).
        fail_fast: Passed to make_client().
        deadline: time.time() value by which this model must finish, or None.

    Returns:
        investigate()'s result, without "fallbacks" and "seconds".

    Raises:
        openai.APIError: the provider failed (rate limit, timeout, outage...),
        or openai.APITimeoutError when `deadline` passed.
    """
    client, model = make_client(provider, model, fail_fast)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    steps, path = [], None
    tokens = {"prompt": 0, "completion": 0}
    answer, llm_calls = None, 0
    draft = ""  # the longest text the model wrote next to a tool call (see _merge_draft)

    for _ in range(max_steps):
        t0 = time.time()
        reply = _ask(client, model, messages, tokens, deadline)
        llm_calls += 1
        log.info("  LLM call %d: %.1fs, %s", llm_calls, time.time() - t0,
                 ", ".join(tc.function.name for tc in reply.tool_calls) if reply.tool_calls
                 else "final answer")

        # No tool calls means the LLM has written its final answer.
        if not reply.tool_calls:
            answer = _merge_draft(draft, reply.content)
            break
        if reply.content and len(reply.content.strip()) >= DRAFT_MIN_CHARS:
            draft = reply.content

        # Keep the assistant's tool-call message in the history, as the API requires.
        messages.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [tc.model_dump() for tc in reply.tool_calls]})

        # Run every tool the LLM asked for this round and send back the results.
        for tc in reply.tool_calls:
            t0 = time.time()
            args, result = run_tool(g, tc.function.name, tc.function.arguments)
            steps.append({"tool": tc.function.name, "args": args, "result": result,
                          "ms": round((time.time() - t0) * 1000)})
            if "error" in result:
                log.warning("    tool %s(%s) -> error: %s", tc.function.name, args, result["error"])
            else:
                log.info("    tool %s(%s) %d ms", tc.function.name, args, steps[-1]["ms"])
            if tc.function.name == "dependency_path" and result.get("path"):
                path = result["path"]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": _for_llm(result)})

    if answer is None:  # ran out of steps: treat like an answer that failed the check
        log.warning("  no final answer after %d steps; using the rules", max_steps)
        check = {"problems": [f"no final answer after {max_steps} steps"],
                 "retried": False, "by_rules": True}
        answer, rule_steps, rule_path = _rules(g, question, steps)
    else:
        answer = _drop_empty_sections(answer)
        answer, check, rule_steps, rule_path = _check(
            question, answer, steps, messages, g, client, model, tokens, deadline)
        llm_calls += check["retried"]
    steps += rule_steps  # the rules' own tool calls, so the trace shows them too
    path = rule_path or path
    return {"answer": answer, "steps": steps, "queries": _queries(steps), "path": path,
            "provider": provider, "model": model, "llm_calls": llm_calls, "tokens": tokens,
            "check": check}


# Text written next to a tool call shorter than this is narration ("Let me check
# the dependencies."), not a draft answer.
DRAFT_MIN_CHARS = 200
# Section labels other than "Page" (the part a fragment final message tends to keep).
_BODY_LABEL = verify._LABEL.format("root cause|evidence|unrelated")
_PAGE_LABEL = verify._LABEL.format("page")


# A section line with nothing in it: "Root cause: N/A — this is an ownership question".
_EMPTY_SECTION = re.compile(r"^[ \t>*_-]*(?:root cause|evidence|unrelated|page)[*_]*\s*:[*_]*\s*"
                            r"(?:n/?a|none needed|not applicable)\b.*(?:\n|$)", re.I | re.M)


def _drop_empty_sections(answer):
    """Remove "Root cause: N/A" style lines a model adds to a non-incident answer.

    Live test, Oct 16 (StepFun, "Who owns adservice and who is on call for
    it?"): four "N/A" lines came before the real one-line answer. "Unrelated:
    none" is kept; it says something.

    Args:
        answer: The model's answer.

    Returns:
        str: The answer without those lines; unchanged if that would leave nothing.
    """
    kept = _EMPTY_SECTION.sub("", answer).strip()
    return kept or answer


def _merge_draft(draft, final):
    """Join a draft answer written next to a tool call with a fragment final reply.

    StepFun (live probe, Oct 16, "If we roll back fraud-check, which alerts
    should clear?") wrote its whole answer in the same message as its
    who_to_page call, then sent only "Page: Payments team, Priya Nair" as the
    final message, so the UI showed one line. When the final reply has no body
    sections but the draft does, or the final reply is empty, the draft is kept
    and the final reply's Page line replaces the draft's.

    Args:
        draft: The latest text of DRAFT_MIN_CHARS or more sent with tool calls, or "".
        final: The final reply's text (may be None).

    Returns:
        str: The answer to check and show.
    """
    final = (final or "").strip()
    if not draft or re.search(_BODY_LABEL, final, re.I):
        return final
    if not final:
        return draft.strip()
    if not re.search(_BODY_LABEL, draft, re.I) and len(final) >= DRAFT_MIN_CHARS:
        return final  # a full final answer in free form; the draft was narration
    page = re.search(_PAGE_LABEL, draft, re.I)
    if page and re.search(_PAGE_LABEL, final, re.I):
        draft = draft[:page.start()]
    return draft.rstrip() + "\n\n" + final


def _ask(client, model, messages, tokens, deadline=None):
    """Send one chat request with the tool schemas; helper for _investigate_once().

    Args:
        client, model: From make_client().
        messages: The conversation so far.
        tokens: {"prompt": int, "completion": int}, updated in place.
        deadline: time.time() value to give up at, or None for no limit.

    Returns:
        The reply message (openai ChatCompletionMessage).

    Raises:
        openai.APIError: the provider failed; APITimeoutError if `deadline` passed.
    """
    response = _within(deadline, lambda: client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SCHEMAS, temperature=0))
    if response.usage:  # token counts; a few providers leave them out
        tokens["prompt"] += response.usage.prompt_tokens or 0
        tokens["completion"] += response.usage.completion_tokens or 0
    return response.choices[0].message


def _within(deadline, call):
    """Run `call()` but give up when `deadline` (a time.time() value) passes.

    The openai client's own timeout is per network read, so it cannot stop a
    reply that keeps trickling in. The call runs in a daemon thread; on
    timeout it is abandoned (it can't be killed) and its result ignored.

    Raises:
        openai.APITimeoutError: the deadline passed first.
        Anything `call()` raises.
    """
    if deadline is None:
        return call()
    box = {}

    def run():
        """Store call()'s result or exception in `box`."""
        try:
            box["value"] = call()
        except BaseException as e:  # passed on to the caller below
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(max(0.0, deadline - time.time()))
    if worker.is_alive():
        raise APITimeoutError(request=httpx2.Request("POST", "https://llm.deadline/"))
    if "error" in box:
        raise box["error"]
    return box["value"]


def _rules(g, question, steps):
    """Answer with diagnose.rule_answer(); returns (answer, rule steps, path)."""
    rules = diagnose.rule_answer(g, question, steps)
    return rules["answer"], rules["steps"], rules["path"]


def _check(question, answer, steps, messages, g, client, model, tokens, deadline=None):
    """Check the final answer against the tool results; retry once if it fails.

    Order: the model's answer -> one corrected answer from the same model ->
    diagnose.rule_answer(), worked out by rules with no model. A provider
    error or timeout during the retry also goes to the rules: the tool results
    are in hand, so restarting on another model would only cost time (tested:
    2.7 s this way vs 22-37 s restarting).

    Args:
        question, answer, steps: The question, the model's answer, its tool steps.
        messages: The conversation so far; the retry is appended to it.
        g: FalkorDB Graph handle, for verify.graph_vocabulary() and the rules.
        client, model, tokens, deadline: As for _ask().

    Returns:
        (answer, check, rule_steps, rule_path): the answer to show;
        {"problems": first answer's problems, "retried": bool, "by_rules": bool};
        and the rules' tool steps and path ([] and None if they did not run).
    """
    vocab = verify.graph_vocabulary(g)
    answer = answer.strip()  # StepFun sometimes starts with blank lines
    problems = verify.check_answer(answer, steps, question, vocab)
    check = {"problems": problems, "retried": False, "by_rules": False}
    if not problems:
        return _missing_notes(question, answer, vocab) + answer, check, [], None

    if not steps:
        # Rule 16: no tool ran, so a rewrite "using only the tool results"
        # has nothing to use. The rules look the facts up instead.
        log.warning("  check failed: %s; using the rules", "; ".join(problems))
        check["by_rules"] = True
        answer, rule_steps, rule_path = _rules(g, question, steps)
        return answer, check, rule_steps, rule_path

    log.warning("  check failed: %s; asking the model to fix it", "; ".join(problems))
    check["retried"] = True
    messages += [{"role": "assistant", "content": answer},
                 {"role": "user", "content": RETRY_PROMPT.format(
                     problems="\n".join(f"- {p}" for p in problems))}]
    try:
        reply = _ask(client, model, messages, tokens, deadline)
        fixed = None if reply.tool_calls else reply.content
    except APIError as e:
        log.warning("  retry failed: %s: %s", type(e).__name__, str(e)[:300])
        fixed = None
    if fixed and not (still := verify.check_answer(fixed, steps, question, vocab)):
        log.info("  retry passed the check")
        return _missing_notes(question, fixed, vocab) + fixed, check, [], None
    if fixed:
        log.warning("  retry still failed the check: %s", "; ".join(still))
    log.warning("  using the rules diagnosis")
    check["by_rules"] = True
    answer, rule_steps, rule_path = _rules(g, question, steps)
    return answer, check, rule_steps, rule_path


def _missing_notes(question, answer, vocab):
    """Lines saying which names from the question are not in the graph.

    verify.check_answer() accepts "Sarah Jenkins is not on call" for "Is Sarah
    Jenkins on call?" (live test, Oct 16), which is true but lets the reader
    think she exists. Each name the answer does not plainly call missing gets
    the rules' own line ("There is no person named Sarah Jenkins in the
    graph."), so model and rules answers say it the same way.

    Args:
        question: The user's question.
        answer: A model answer that passed verify.check_answer().
        vocab: verify.graph_vocabulary().

    Returns:
        str: One line per such name, each ending in a newline; "" if none.
    """
    plain = re.sub(r"[`*]", "", answer.lower().replace("\u2019", "'"))
    notes = ""
    for kind, name in verify.unknown_names(question, vocab):
        phrases = verify.NO_DEPLOY if kind == "version" else verify.NOT_FOUND
        if not verify._called_out(plain, name, phrases):
            close = (difflib.get_close_matches(name, sorted(vocab["services"]), n=1, cutoff=0.8)
                     if kind == "service" else [])
            notes += (f"There is no {diagnose._KIND[kind].format(diagnose._as_written(question, name))}"
                      " in the graph." + (f" Did you mean {close[0]}?" if close else "") + "\n")
    return notes


if __name__ == "__main__":
    # Command-line use: print skipped models, each tool step, then the answer.
    q = " ".join(sys.argv[1:]) or "checkoutservice is throwing 5xx errors. What broke?"
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(levelname)s %(message)s")
    out = investigate(q)
    for f in out["fallbacks"]:
        print(f"skipped {f['provider']} / {f['model']}: {f['error']}")
    print(f"[{out['provider']} / {out['model']}, {out['seconds']}s, "
          f"{out['llm_calls']} LLM calls, {sum(out['tokens'].values())} tokens]\n")
    for i, s in enumerate(out["steps"], 1):
        print(f"step {i}: {s['tool']}({json.dumps(s['args'])})  {s['ms']} ms")
    print("\n" + out["answer"])
    if out["check"]["problems"]:  # the first answer failed verify.check_answer()
        print("\n[answer check] first answer had: " + "; ".join(out["check"]["problems"]))
        print("[answer check] " + ("answered by rules" if out["check"]["by_rules"]
                                   else "fixed on retry"))
    elif out["provider"] == "rules":
        print("\n[answer check] no model answered in time; answered by rules")
