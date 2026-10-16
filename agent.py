"""The Incident Detective agent: an LLM that investigates outages with graph tools.

How it works:
  1. The user asks a question, e.g. "checkoutservice is throwing 5xx, what broke?"
  2. The LLM sees the six tools from tools.py (as OpenAI function schemas) and
     picks one. We run it against FalkorDB and send the result back.
  3. Steps repeat until the LLM writes a final answer or MAX_STEPS is reached.

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
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import APIError, OpenAI

import tools
from db import get_graph

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

# The investigation procedure the LLM follows. Kept short: free tiers limit
# tokens per minute, and this prompt is resent on every step.
SYSTEM_PROMPT = """You are Incident Detective, an on-call assistant for a microservice platform.
You investigate incidents ONLY with the provided tools, which query a graph of services,
dependencies, deploys, alerts, teams and on-call people. Never invent services, people or facts.

For "what broke?" questions:
1. Call recent_changes to see recent deploys and open alerts.
2. Call dependencies on the failing service to find what it relies on.
3. A recent deploy to a service the failing one depends on is the prime suspect. Confirm the
   link with dependency_path.
4. Alerts on services that are on that path are symptoms of the same incident. An alert on a
   service that is NOT in the dependencies list is unrelated: it cannot affect the failing service.
5. Call who_to_page on the root-cause service.

Final answer format (plain text, short):
Root cause: <service and change, with who deployed it and when>
Evidence: <the dependency path and the alerts along it>
Unrelated: <alerts on services outside the dependencies list, if any>
Page: <team and on-call person>"""

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
                       "path=null means they are unrelated.",
        "parameters": {"type": "object", "properties": {
            "from_service": _SERVICE, "to_service": _SERVICE},
            "required": ["from_service", "to_service"]}}},
    {"type": "function", "function": {
        "name": "critical_services",
        "description": "Rank services by graph importance. 'betweenness' = chokepoints many "
                       "call paths pass through; 'pagerank' = most depended on.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["betweenness", "pagerank"]},
            "limit": {"type": "integer", "description": "How many to return, default 5"}},
            "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "who_to_page",
        "description": "Owning team and current on-call person for a service.",
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
        return args, {"error": f"{name} failed: {e}"}


def _for_llm(result):
    """Return the tool result without its Cypher, to save LLM tokens.

    The Cypher is for people (shown in the UI); the LLM only needs the data.
    """
    return json.dumps({k: v for k, v in result.items() if k != "cypher"})


def investigate(question, g=None, provider=None, model=None, max_steps=MAX_STEPS):
    """Answer an incident question by letting the LLM call graph tools in a loop.

    Tries the models from model_chain() in order. If one fails (rate limit,
    timeout, outage, model withdrawn), the question restarts from scratch on
    the next, so every answer comes from a single model.

    Args:
        question: The user's question in plain English.
        g: FalkorDB Graph handle. Defaults to db.get_graph().
        provider, model: Use only this model, with no fallback. See model_chain().
        max_steps: Maximum number of LLM rounds per model.

    Returns:
        {"answer": str,             final text from the LLM (or a stop message)
         "steps": [{"tool", "args", "result", "ms"}, ...],   in call order
         "queries": [str, ...],     every Cypher query that ran
         "path": [str, ...] or None, the last dependency path found, for the UI
         "provider": str, "model": str,   the model that answered
         "llm_calls": int,          LLM requests made by that model
         "tokens": {"prompt": int, "completion": int},   summed over those requests
         "fallbacks": [{"provider", "model", "error"}, ...],   models that failed first
         "seconds": float}          total time, including failed attempts

    Raises:
        RuntimeError: no API key is set for any model in the chain.
        openai.APIError: the last model in the chain failed too.
    """
    g = g or get_graph()
    chain = model_chain(provider, model)
    if not chain:
        raise RuntimeError("no LLM API key is set; add one of the keys in PROVIDERS to .env")
    started = time.time()
    fallbacks = []
    for i, (p, m) in enumerate(chain):
        is_last = i == len(chain) - 1
        try:
            out = _investigate_once(question, g, p, m, max_steps, fail_fast=not is_last)
        except APIError as e:  # includes timeouts and connection errors
            if is_last:
                raise
            fallbacks.append({"provider": p, "model": m,
                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        out.update(fallbacks=fallbacks, seconds=round(time.time() - started, 1))
        return out


def _investigate_once(question, g, provider, model, max_steps, fail_fast):
    """Run the tool-calling loop on one model; helper for investigate().

    Args:
        question, g, max_steps: As for investigate().
        provider, model: The model to use (model None = the provider's default).
        fail_fast: Passed to make_client().

    Returns:
        investigate()'s result, without "fallbacks" and "seconds".

    Raises:
        openai.APIError: the provider failed (rate limit, timeout, outage...).
    """
    client, model = make_client(provider, model, fail_fast)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    steps, queries, path = [], [], None
    tokens = {"prompt": 0, "completion": 0}
    answer, llm_calls = None, 0

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, temperature=0)
        llm_calls += 1
        if response.usage:  # token counts; a few providers leave them out
            tokens["prompt"] += response.usage.prompt_tokens or 0
            tokens["completion"] += response.usage.completion_tokens or 0
        reply = response.choices[0].message

        # No tool calls means the LLM has written its final answer.
        if not reply.tool_calls:
            answer = reply.content
            break

        # Keep the assistant's tool-call message in the history, as the API requires.
        messages.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [tc.model_dump() for tc in reply.tool_calls]})

        # Run every tool the LLM asked for this round and send back the results.
        for tc in reply.tool_calls:
            t0 = time.time()
            args, result = run_tool(g, tc.function.name, tc.function.arguments)
            steps.append({"tool": tc.function.name, "args": args, "result": result,
                          "ms": round((time.time() - t0) * 1000)})
            cypher = result.get("cypher")
            queries += cypher if isinstance(cypher, list) else [cypher] if cypher else []
            if tc.function.name == "dependency_path" and result.get("path"):
                path = result["path"]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": _for_llm(result)})

    if answer is None:
        answer = f"Stopped after {max_steps} steps without a final answer."
    return {"answer": answer, "steps": steps, "queries": queries, "path": path,
            "provider": provider, "model": model, "llm_calls": llm_calls, "tokens": tokens}


if __name__ == "__main__":
    # Command-line use: print skipped models, each tool step, then the answer.
    q = " ".join(sys.argv[1:]) or "checkoutservice is throwing 5xx errors. What broke?"
    out = investigate(q)
    for f in out["fallbacks"]:
        print(f"skipped {f['provider']} / {f['model']}: {f['error']}")
    print(f"[{out['provider']} / {out['model']}, {out['seconds']}s, "
          f"{out['llm_calls']} LLM calls, {sum(out['tokens'].values())} tokens]\n")
    for i, s in enumerate(out["steps"], 1):
        print(f"step {i}: {s['tool']}({json.dumps(s['args'])})  {s['ms']} ms")
    print("\n" + out["answer"])
