# Incident Detective

An on-call agent that investigates outages by walking a [FalkorDB](https://www.falkordb.com/) graph of services, deploys, alerts and owners. It answers **what broke, why, and who to page**, and shows the graph path behind every answer.

Built for Graph Hacks (WeMakeDevs × FalkorDB), Oct 15–18 2026.
**Track:** 01 — Agents That Act on Connected Data.

> Work in progress. Setup, graph model and queries are added as they are built.

## AI usage

Built with **Claude Code** (Anthropic) as a coding assistant for planning, research, code drafting and review. Commits it helped write carry a `Co-Authored-By: Claude` trailer.

What I did myself: idea and track choice, graph model, dataset design, demo scenarios and their expected answers, choosing and verifying the graph algorithms, and testing every query against FalkorDB.

Design notes were prepared before Oct 15; all code was written during the event.
