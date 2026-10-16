"""Stage 1 of the model evaluation: list every model a provider offers and probe it.

The probe is one tiny request per model ("Who should I page for checkoutservice?
Use the tool.") with a single tool available. It answers four questions cheaply,
before the full bake-off (bakeoff.py) spends real quota:
  1. Does the model exist for our key?        (404 / 403 means no)
  2. Is there free quota for it right now?     (429 / 503 means no)
  3. Does it call the tool with the right argument?
  4. What rate limits does the provider report? (x-ratelimit-* headers, if any)

Models that are clearly not chat models (embeddings, OCR, speech, image/video,
safety classifiers, code-completion) are recorded as skipped without a request.
OpenRouter is limited to its free models, so the probe never spends credits.

Usage:
  python -m scripts.probe_models groq gemini nvidia mistral openrouter
  python -m scripts.probe_models --pause 5 --out docs/model-eval mistral
Writes <out>/probe_<provider>.jsonl (one line per model). Needs the API keys in
.env. Never prints or saves API keys.
"""
import argparse
import json
import os
import re
import time
import urllib.request

from dotenv import load_dotenv
from openai import OpenAI

from agent import PROVIDERS

# Name fragments that mark a model as not a general chat model, so it is skipped.
NOT_CHAT = re.compile(
    r"embed|ocr|tts|transcribe|whisper|orpheus|guard|safety|moderation|reward|clip|lyria|veo|"
    r"image|nano-banana|parse|video|translate|deplot|kosmos|vila|neva|fuyu|aqa|live|robotics|"
    r"fim|deep-research|computer-use|nvclip|retriever|starcoder|codegemma|codellama|"
    r"deepseek-coder|calibration|diffusion|voxtral|recurrentgemma|gemma-2b|llama2")

# The one tool offered in the probe; a passing model calls it for checkoutservice.
PROBE_TOOL = [{"type": "function", "function": {
    "name": "who_to_page", "description": "On-call person for a service",
    "parameters": {"type": "object", "properties": {"service": {"type": "string"}},
                   "required": ["service"]}}}]
PROBE_QUESTION = "Who should I page for checkoutservice? Use the tool."


def list_models(provider):
    """Return every model id the provider lists for our key (OpenRouter: free ones only).

    Args:
        provider: A key of agent.PROVIDERS, e.g. "groq".

    Returns:
        Sorted list of model ids, without Gemini's "models/" prefix.
    """
    if provider == "openrouter":
        # OpenRouter's public catalogue includes prices; keep only zero-cost models.
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as resp:
            data = json.load(resp)["data"]
        return sorted(m["id"] for m in data
                      if float(m["pricing"]["prompt"]) == 0
                      and float(m["pricing"]["completion"]) == 0)
    base_url, key_env, _ = PROVIDERS[provider]
    client = OpenAI(base_url=base_url, api_key=os.environ[key_env], timeout=30)
    return sorted(m.id.removeprefix("models/") for m in client.models.list())


def _limit_headers(headers):
    """Keep only the rate-limit headers from an HTTP response (they hold no secrets)."""
    return {k: v for k, v in (headers or {}).items() if "ratelimit" in k.lower()}


def probe(client, provider, model):
    """Send the probe question to one model and describe what happened.

    Args:
        client: OpenAI-compatible client for the provider (no retries).
        provider: Provider name, stored in the result.
        model: Model id to probe.

    Returns:
        Dict with provider, model, status ("ok", "no tool call", "error <code> <type>"
        or "skipped (not a chat model)"), seconds, rate-limit headers and, on errors,
        the first 200 characters of the error.
    """
    rec = {"provider": provider, "model": model}
    if NOT_CHAT.search(model):
        rec["status"] = "skipped (not a chat model)"
        return rec
    start = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, tools=PROBE_TOOL, max_tokens=300,
            messages=[{"role": "user", "content": PROBE_QUESTION}])
        calls = raw.parse().choices[0].message.tool_calls
        ok = bool(calls) and "checkoutservice" in calls[0].function.arguments
        rec.update(status="ok" if ok else "no tool call", headers=_limit_headers(raw.headers))
    except Exception as e:  # a survey: record every failure instead of stopping
        response = getattr(e, "response", None)
        rec.update(status=f"error {getattr(e, 'status_code', '')} {type(e).__name__}",
                   detail=str(e)[:200],
                   headers=_limit_headers(getattr(response, "headers", None)))
    rec["seconds"] = round(time.time() - start, 1)
    return rec


def main():
    """Probe every model of the given providers, one at a time, and save the results."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("providers", nargs="+", choices=sorted(PROVIDERS))
    parser.add_argument("--pause", type=float, default=3,
                        help="seconds between requests, to stay under per-minute limits")
    parser.add_argument("--out", default="docs/model-eval", help="folder for the JSONL files")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for provider in args.providers:
        base_url, key_env, _ = PROVIDERS[provider]
        # max_retries=0: a 429 should be recorded as "no quota", not waited out.
        client = OpenAI(base_url=base_url, api_key=os.environ[key_env],
                        timeout=60, max_retries=0)
        with open(os.path.join(args.out, f"probe_{provider}.jsonl"), "w") as out:
            for model in list_models(provider):
                rec = probe(client, provider, model)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"{provider:10} {model:55} {rec['status']} {rec.get('seconds', '')}",
                      flush=True)
                if "seconds" in rec:
                    time.sleep(args.pause)


if __name__ == "__main__":
    main()
