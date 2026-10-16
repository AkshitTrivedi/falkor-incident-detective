# Incident Detective

An on-call agent that investigates outages by walking a [FalkorDB](https://www.falkordb.com/) graph of services, deploys, alerts and owners. It answers **what broke, why, and who to page**, and shows the graph path behind every answer.

Built for Graph Hacks (WeMakeDevs × FalkorDB), Oct 15–18 2026.
**Track:** 01 — Agents That Act on Connected Data.

> Work in progress. Setup, graph model and queries are added as they are built.

## Choosing the model

We didn't pick the LLM by reputation. We tested **all 218 models** our five free API keys could reach (Groq, Gemini, NVIDIA, Mistral, OpenRouter): first a one-request probe of each, then **35 models** through the real agent on the three demo incidents, graded on whether they named the right root cause, paged the right person, and never stated a wrong fact.

**10 models answered every question correctly.** The best six then answered **15 questions × 3 rounds**, including trap questions where the right answer is "that service doesn't exist" or "the graph has no CPU data". Only Groq Qwen 3.8 27B and NVIDIA Nemotron 3 Super got all 45 right. The fastest model, Mistral Codestral (1.9 s), blamed a deploy that doesn't exist when the question didn't name a service, so it was left out.

The agent uses a fallback chain across providers (Qwen → Gemini 3.1 Flash-Lite → Step-5 → Nemotron), so one rate limit or outage can't stop it.

Full method, results, free-tier limits and how to rerun it: **[docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)**. Raw answers: [docs/model-eval/](docs/model-eval/).

## AI usage

Built with **Claude Code** (Anthropic) as a coding assistant for planning, research, code drafting and review. Commits it helped write carry a `Co-Authored-By: Claude` trailer.

What I did myself: idea and track choice, graph model, dataset design, demo scenarios and their expected answers, choosing and verifying the graph algorithms, and testing every query against FalkorDB.

Design notes were prepared before Oct 15; all code was written during the event.
