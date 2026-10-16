"""Offline checks for agent.py: no LLM calls, so they are free and fast.

A fake LLM (FakeLLM below) replays scripted replies, to test the tool loop and
the fallback chain. Real models are checked by bakeoff.py, which needs API keys.
"""
import inspect

import httpx2  # the HTTP library openai uses; needed to build its error objects
import openai
import pytest
from openai.types.chat import ChatCompletion

import agent
import tools
from db import get_graph
from seed import seed

# Every provider's API key variable, so tests can control which keys are "set".
KEY_VARS = [key_var for _, key_var, _ in agent.PROVIDERS.values()]


def _reply(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=20):
    """Build a chat completion like a provider returns: text, or tool calls."""
    return ChatCompletion.model_validate({
        "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}})


def _tool_call(name, arguments):
    """One tool call in the shape the API returns it (arguments as JSON text)."""
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _rate_limited():
    """A 429 error, as the openai SDK raises it when a provider is rate-limited."""
    request = httpx2.Request("POST", "https://llm.invalid/v1/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx2.Response(429, request=request),
                                 body=None)


class FakeLLM:
    """Stands in for an OpenAI client: returns (or raises) scripted replies in order.

    Only the part investigate() uses is faked: client.chat.completions.create().
    """

    def __init__(self, *replies):
        """Store the replies; each is a ChatCompletion to return or an error to raise."""
        self.replies = list(replies)
        self.chat = self  # so that client.chat.completions.create() resolves here
        self.completions = self

    def create(self, **kwargs):
        """Return or raise the next scripted reply, ignoring the request."""
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake_llms(monkeypatch):
    """Replace the real models with FakeLLMs for one test.

    Usage: fake_llms({"groq": FakeLLM(...), "gemini": FakeLLM(...)}). The
    fallback chain becomes these providers in dict order, with model ids
    "groq-model", "gemini-model"... Every API key is set to a dummy value, and
    LLM_PROVIDER / LLM_MODEL are cleared so the chain is used.
    """
    def install(fakes):
        """Make `fakes` (provider -> FakeLLM) the fallback chain."""
        def fake_make_client(provider, model=None, fail_fast=False):
            """Return the provider's FakeLLM instead of a real API client."""
            return fakes[provider], model

        for key_var in KEY_VARS:
            monkeypatch.setenv(key_var, "test-key")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(agent, "FALLBACK_CHAIN", [(p, f"{p}-model") for p in fakes])
        monkeypatch.setattr(agent, "make_client", fake_make_client)
    return install


@pytest.fixture(scope="module")
def g():
    """Seed a throwaway graph once for this module, then delete it."""
    graph = get_graph("incidents_agent_test")
    seed(graph)
    yield graph
    graph.delete()


def test_schemas_match_tools():
    """Every schema names a real tool, and its parameters match the function's."""
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in agent.TOOL_SCHEMAS}
    assert set(schemas) == set(tools.TOOLS)
    for name, params in schemas.items():
        fn_params = set(inspect.signature(tools.TOOLS[name]).parameters) - {"g", "now_ms"}
        assert set(params["properties"]) == fn_params, name


def test_run_tool_success(g):
    """A valid call returns the parsed args and the tool's result."""
    args, result = agent.run_tool(g, "who_to_page", '{"service": "fraud-check"}')
    assert args == {"service": "fraud-check"} and result["on_call"] == "Priya Nair"


@pytest.mark.parametrize("name, raw, expected", [
    ("drop_database", "{}", "unknown tool"),
    ("who_to_page", "{not json", "not valid JSON"),
    ("who_to_page", '{"svc": "fraud-check"}', "bad arguments"),
])
def test_run_tool_errors_are_returned_not_raised(g, name, raw, expected):
    """Mistakes by the LLM come back as an error dict it can read and fix."""
    _, result = agent.run_tool(g, name, raw)
    assert expected in result["error"]


def test_cypher_is_hidden_from_llm():
    """The LLM gets the data only; the Cypher is kept for the UI."""
    assert "cypher" not in agent._for_llm({"path": ["a", "b"], "cypher": "MATCH ..."})


def test_unknown_provider():
    """A typo in LLM_PROVIDER fails clearly instead of calling the wrong API."""
    with pytest.raises(ValueError):
        agent.make_client("openai-typo")


def test_tool_loop_with_fake_llm(g, fake_llms):
    """The loop runs the tool the LLM asks for, sends the result back and sums tokens."""
    fake_llms({"groq": FakeLLM(
        _reply(tool_calls=[_tool_call("who_to_page", '{"service": "fraud-check"}')]),
        _reply(content="Page: Payments team, on-call Priya Nair."))})
    out = agent.investigate("Who do I page for fraud-check?", g=g)
    assert out["answer"] == "Page: Payments team, on-call Priya Nair."
    assert [s["tool"] for s in out["steps"]] == ["who_to_page"]
    assert out["steps"][0]["result"]["on_call"] == "Priya Nair"
    assert out["queries"]  # the Cypher that ran is kept for the UI
    assert out["llm_calls"] == 2
    assert out["tokens"] == {"prompt": 200, "completion": 40}
    assert out["fallbacks"] == []


def test_rate_limited_model_falls_back_to_next(g, fake_llms):
    """A 429 from the first model restarts the question on the next one."""
    fake_llms({"groq": FakeLLM(_rate_limited()),
               "gemini": FakeLLM(_reply(content="Root cause: fraud-check"))})
    out = agent.investigate("What broke?", g=g)
    assert (out["provider"], out["model"]) == ("gemini", "gemini-model")
    assert out["answer"] == "Root cause: fraud-check"
    assert [f["provider"] for f in out["fallbacks"]] == ["groq"]
    assert "RateLimitError" in out["fallbacks"][0]["error"]


def test_last_model_error_is_raised(g, fake_llms):
    """When every model in the chain fails, the last error reaches the caller."""
    fake_llms({"groq": FakeLLM(_rate_limited()), "gemini": FakeLLM(_rate_limited())})
    with pytest.raises(openai.RateLimitError):
        agent.investigate("What broke?", g=g)


def test_explicit_provider_has_no_fallback(g, fake_llms):
    """Asking for one provider, as the bake-off does, never switches to another."""
    fake_llms({"groq": FakeLLM(_rate_limited()), "gemini": FakeLLM(_reply(content="unused"))})
    with pytest.raises(openai.RateLimitError):
        agent.investigate("What broke?", g=g, provider="groq")


def test_env_provider_pins_one_model(monkeypatch):
    """LLM_PROVIDER and LLM_MODEL select a single model, with no fallback."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    assert agent.model_chain() == [("gemini", "gemini-test")]


def test_chain_skips_models_without_key(monkeypatch):
    """Models whose provider has no API key set are left out of the chain."""
    for key_var in KEY_VARS:
        monkeypatch.delenv(key_var, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert agent.model_chain() == [m for m in agent.FALLBACK_CHAIN if m[0] == "gemini"]


def test_no_api_key_at_all_is_a_clear_error(g, monkeypatch):
    """With no API key set, investigate() says so instead of failing obscurely."""
    for key_var in KEY_VARS:
        monkeypatch.delenv(key_var, raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="no LLM API key"):
        agent.investigate("What broke?", g=g)
