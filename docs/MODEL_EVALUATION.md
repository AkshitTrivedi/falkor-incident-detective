# Model evaluation: which LLM runs the agent

Incident Detective needs an LLM that picks the right graph tool, reads the result correctly, and pages the right person. A model that does those three things on a free tier, quickly and every time, beats a "smarter" model that gets one of them wrong. So instead of picking a model by reputation, we tested **every model our five free API keys could reach**.

All tests ran on **16 Oct 2026** against free tiers. Raw results are in [`model-eval/`](model-eval/), and every number below can be regenerated from them (see [Reproduce](#reproduce)).

## At a glance

| | |
|---|---|
| Providers | Groq, Google Gemini, NVIDIA, Mistral, OpenRouter (all OpenAI-compatible) |
| Model ids listed | **218** |
| Chat models that called our tool correctly | **33** |
| Models put through the full test | **35** (257 questions asked, 194 answered; the rest hit quota or timeouts) |
| Models with every answer correct | **10** |

## How we tested

### Stage 1: probe every model ([`scripts/probe_models.py`](../scripts/probe_models.py))

One tiny request per model: *"Who should I page for checkoutservice? Use the tool."*, with one tool offered. This answers four questions cheaply before spending real quota:

1. Does the model exist for our key? (404/403 = no)
2. Is there free quota right now? (429/503 = no)
3. Does it call the tool with the right argument?
4. What rate limits does the provider report? (`x-ratelimit-*` headers, where sent)

Embeddings, OCR, speech, image/video and safety-classifier models were skipped without a request.

| Provider | Model ids | Not chat models | Gone / not enabled (404, 403, 400) | No quota (429, 503) | Timed out | **Works** |
|---|---|---|---|---|---|---|
| Groq | 11 | 7 | 1 | 0 | 0 | **3** |
| Gemini | 63 | 36 | 9 | 11 | 1 | **6** |
| NVIDIA | 81 | 34 | 32 | 1 | 8 | **6** |
| Mistral | 45 | 21 | 2 | 13 | 0 | **9** |
| OpenRouter (free models only) | 18 | 3 | 2 | 3 | 1 | **9** |
| **Total** | **218** | **101** | **46** | **28** | **10** | **33** |

### Stage 2: the bake-off ([`bakeoff.py`](../bakeoff.py))

Each model answers the three demo questions against its **own freshly seeded FalkorDB graph**, through the real agent loop (same prompt, same six read-only tools, up to 8 tool rounds). Most models did this **3 times** (9 answers). OpenRouter models got 1 round, because the free key allows about 50 requests a day.

| Story | Question | Correct only if the answer… |
|---|---|---|
| A | *checkoutservice is throwing 5xx errors. What broke and who should I page?* | names `fraud-check` and `v3.1.0`, pages **Priya Nair**, and the agent actually called `who_to_page` |
| B | *Which services are the biggest chokepoints in our architecture?* | names `paymentservice` (top betweenness score) and the agent called `critical_services` |
| C | *There is also an alert on adservice. Is it related to the checkout outage?* | says it is unrelated, backed by a graph check (`dependency_path` or `dependencies`) |

Every answer also **fails** if it states something the graph says is wrong:
- it calls `paymentservice` unrelated (it is on the failing path), or
- it pages Daniel Okafor (he deployed the change but isn't on call) or Omar Haddad (on call for the *victim* team, not the cause).

We added these "wrong fact" checks after reading round-1 answers by hand: three answers had passed the first, looser checks while paging the wrong person or calling a broken service unrelated. [`scripts/summarize_results.py`](../scripts/summarize_results.py) re-grades **every** saved answer with the current checks, so all rounds are judged by the same rules. We also read the top models' answers ourselves.

## Results

**Median** = the time a typical answer takes (half are faster, half slower).
**Average** = total time ÷ answers, so a few very slow answers pull it up.
A big gap between the two means the model is usually fast but sometimes gets stuck, which on Groq means waiting out its 8K tokens-per-minute limit.
*Tools* and *tokens* are medians per question.

### Every answer correct

| Model | Correct | Median s | Avg s | Slowest s | Tools | Tokens |
|---|---|---|---|---|---|---|
| Mistral `codestral-2508` | 9/9 | **1.7** | 2.0 | 3.8 | 2 | 1,979 |
| Mistral `ministral-3b-2512` | 9/9 | 2.2 | 2.7 | 6.0 | 3 | 3,638 |
| Groq `qwen/qwen3.8-27b` | **12/12** | 2.6 | 6.4 | 19.9 | 3 | 4,367 |
| Gemini `gemini-3.1-flash-lite-preview` | 9/9 | 7.2 | 15.9 | 56.0 | 1 | 1,786 |
| OpenRouter `cohere/north-mini-code:free` | 3/3 | 10.0 | 9.0 | 12.9 | 4 | 4,630 |
| NVIDIA `stepfun-ai/step-5-preview` | 9/9 | 10.6 | 10.9 | 19.4 | 5 | 5,083 |
| Gemini `gemini-3.1-flash-lite` | **12/12** | 12.8 | 15.3 | 34.3 | 1 | 1,786 |
| OpenRouter `dots-studio/dots-3-note-preview:free` | 3/3 | 20.4 | 20.3 | 25.3 | 4 | 4,666 |
| OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` | 3/3 | 34.4 | 31.3 | 44.7 | 4 | 8,999 |
| NVIDIA `nvidia/nemotron-3-super-120b-a12b` | **12/12** | 35.0 | 37.0 | 79.0 | 4 | 8,606 |

Read by hand: Codestral's answers are complete and precise (root cause, version, deployer, path with latencies, unrelated alert, on-call name). Ministral 3B gets every fact right but misspells `paymentsservice`, which would look sloppy in a live demo.

### Mostly correct (one wrong fact is enough to rule a model out for paging)

| Model | Correct | Median s | Avg s | Main mistake |
|---|---|---|---|---|
| Mistral `mistral-code-latest` | 8/9 | 2.2 | 2.4 | called paymentservice unrelated |
| Gemini `gemini-3.5-flash-lite` | 8/9 | 3.9 | 3.8 | paged the wrong person |
| NVIDIA `nemotron-3.5-super-vl-120b-a12b` | 8/9 | 21.9 | 23.9 | called paymentservice unrelated |
| Gemini `gemma-4-26b-a4b-it` | 8/9 | 27.2 | 34.1 | one quota error; up to 107 s |
| Groq `openai/gpt-oss-20b` | 10/12 | 3.9 | 4.6 | paged the wrong person |
| Groq `openai/gpt-oss-120b` | 9/12 | 4.0 | 5.7 | called paymentservice unrelated |
| Mistral `ministral-8b-2512` | 6/9 | 2.0 | 2.4 | left out the deploy version |
| NVIDIA `openai/gpt-oss-20b` | 6/9 | 13.7 | 14.8 | called paymentservice unrelated |
| Mistral `ministral-14b-2512` | 3/9 | 3.5 | 4.1 | paged the wrong person |

### Could not finish on the free tier

| Model | Correct | What happened |
|---|---|---|
| Gemini `gemini-3.5-flash`, `3.6-flash`, `3.7-flash`, `3-flash-preview`, `3.8-flash` | 1–4 of 3–12 | free quota ran out after 2–4 questions (429) |
| OpenRouter `poolside/laguna-s-2.1:free`, `nemotron-3-nano-omni:free` | 3/6, 1/2 | upstream 429s, or ran into the daily cap |
| NVIDIA `poolside/laguna-xs-2.1` | 2/9 | 503 "worker request limit reached" |
| NVIDIA `meta/llama-3.2-11b-vision-instruct` | 0/9 | supports only one tool call at a time; wrong answers |
| NVIDIA `glm-5.3`, `deepseek-v4.1-flash`, `kimi-k2.6`; Mistral `mistral-small-latest`; OpenRouter `gemma-4-31b-it:free`, `inkling:free` | 0/3 | timed out, no quota, or not enabled for our account |

## Free-tier limits we hit

| Provider | What our key got | Source |
|---|---|---|
| Groq | 8,000 tokens/min and 1,000 requests/day per model. One of our questions uses ~4–6K tokens, so back-to-back questions wait. | response headers |
| Mistral (small models, Codestral) | 125 requests/min and 625,000 tokens/min. Medium, Small and Magistral models: no free quota (429). | response headers |
| Gemini | Flash-Lite models ran all tests without a limit error; full Flash models ran out after a few questions; Pro models had none. No limit headers. | observed |
| NVIDIA | No limit headers. 32 listed models return 404 (not enabled for free accounts); several time out at 60 s. | observed |
| OpenRouter | Free tier: about 50 requests/day and 20/min for `:free` models; upstream providers also return 429s. | key info + docs |

Groq's limits were also checked against its [deprecations page](https://console.groq.com/docs/deprecations): models often quoted as having higher limits (`llama-3.1-8b-instant`, `llama-3.3-70b`, `llama-4-scout`, `qwen3-32b`, `kimi-k2`) were shut down between July and September 2026, so they couldn't be tested.

## What we use

The agent tries a **fallback chain**: if a model is rate-limited, slow or down, the same question goes to the next one. One question is always answered by one model; answers are never mixed.

Shortlist from this evaluation (all 100% correct, spread across providers so one outage or quota can't stop the demo):

1. Mistral `codestral-2508`: fastest, and the largest free limits by far
2. Groq `qwen/qwen3.8-27b`: most runs at 100%, very fast between rate-limit waits
3. Gemini `gemini-3.1-flash-lite`: fewest tokens per question, stable model
4. NVIDIA `stepfun-ai/step-5-preview`: correct every time, ~11 s
5. NVIDIA `nemotron-3-super-120b-a12b`: correct every time, slowest

**Next:** a wider test of 10–15 questions on this shortlist before the chain in [`agent.py`](../agent.py) is finalised. Three questions show a model can do the job; fifteen show whether it does it every time.

## Reproduce

Needs local FalkorDB (`docker run -p 6379:6379 falkordb/falkordb:v4.22.0`) and API keys in `.env`.

```bash
# Stage 1: probe every model of each provider (~10 min)
python -m scripts.probe_models --out my-eval groq gemini nvidia mistral openrouter

# Stage 2: full test on chosen models, 3 rounds, 10 s between questions
python bakeoff.py --repeat 3 --pause 10 --out my-eval/bakeoff.jsonl \
    mistral:codestral-2508 groq:qwen/qwen3.8-27b gemini:gemini-3.1-flash-lite

# Merge and re-grade every saved run into one table
python -m scripts.summarize_results my-eval/bakeoff*.jsonl
python -m scripts.summarize_results docs/model-eval/bakeoff_round*.jsonl   # our runs
```

## Caveats

- **Free tiers change.** Models, quotas and speeds were measured on 16 Oct 2026 from India; your numbers will differ.
- **Small sample.** 3–12 answers per model show who *can* do the job; the 10–15 question run checks consistency.
- **Grading is rule-based.** The checks look for the facts that matter (cause, version, person to page, wrong claims), not writing quality, so the top models' answers were also read by hand.
- **Time includes waiting.** Times are wall-clock from question to answer, including any rate-limit waits the SDK does. That is what a user would experience.
- **Round 1** (`bakeoff_round1.jsonl`) didn't record token counts.
