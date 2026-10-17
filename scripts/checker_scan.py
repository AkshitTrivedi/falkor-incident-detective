"""Run the answer checker over every saved model answer in docs/model-eval/.

Why: the checker must catch wrong answers without rejecting right ones. The
saved answers were graded by hand, so any flag on an answer that was graded
correct is a possible false positive, and any graded-wrong answer the checker
passes is a possible miss.

Tool results are rebuilt from the live local graph (every tool for every
service), so the checker is lenient on rules 3-5 and 9: any returned fact
counts. Needs the demo graph loaded (python seed.py).

Usage:
    python -m scripts.checker_scan

Prints a FLAG line per answer the checker rejects, a PASS-BUT-GRADED-FAIL line
per answer it accepts although a grader failed it, and a summary line.
"""
import glob
import itertools
import json

import bakeoff
import tools
import verify
from db import get_graph


def rebuilt_steps(g, services):
    """Return tool steps that together hold every fact the graph can return.

    Args:
        g: FalkorDB graph handle.
        services: service names to run the per-service tools for.

    Returns:
        List of {"tool", "args", "result"} dicts, shaped like agent steps.
    """
    steps = [{"tool": "recent_changes", "args": {}, "result": tools.recent_changes(g, 60)}]
    for metric in ("betweenness", "pagerank"):
        steps.append({"tool": "critical_services", "args": {},
                      "result": tools.critical_services(g, metric, 10)})
    for s in services:
        for fn in (tools.dependencies, tools.blast_radius, tools.who_to_page):
            steps.append({"tool": fn.__name__, "args": {}, "result": fn(g, s)})
    for a, b in itertools.permutations(services, 2):
        steps.append({"tool": "dependency_path", "args": {}, "result": tools.dependency_path(g, a, b)})
    return steps


def main():
    """Check every saved answer and print flags plus a summary line."""
    g = get_graph()
    vocab = verify.graph_vocabulary(g)
    steps = rebuilt_steps(g, sorted(vocab["services"]))
    questions = {s: q for s, q, _ in bakeoff.QUESTIONS + bakeoff.EXTENDED}
    total = flagged = missed = 0
    for path in sorted(glob.glob("docs/model-eval/*.jsonl")):
        name = path.split("/")[-1]
        for line in open(path):
            row = json.loads(line)
            answer = row.get("answer")
            if not answer:
                continue
            total += 1
            question = row.get("question") or questions.get(row.get("story"), "")
            problems = verify.check_answer(answer, steps, question, vocab)
            if problems:
                flagged += 1
                print(f"FLAG {name} {row.get('model')} {row.get('story')} "
                      f"passed={row.get('passed')}: {problems}")
            elif row.get("passed") is False:
                missed += 1
                print(f"PASS-BUT-GRADED-FAIL {name} {row.get('model')} {row.get('story')}: "
                      f"{row.get('failed')}")
    print(f"{total} answers; {flagged} flagged; {missed} graded fail but checker clean")


if __name__ == "__main__":
    main()
