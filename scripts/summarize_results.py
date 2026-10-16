"""Stage 3 of the model evaluation: merge saved runs into one results table.

Reads the JSONL files written by `bakeoff.py --out` (one line per answered
question) and prints one markdown row per model. Every saved answer is graded
again with today's checks from bakeoff.py, so runs made before a check was
tightened are judged by the same rules as new ones.

Columns: correct answers / runs, error runs (quota, timeout, overload), median,
average and slowest time in seconds, median tool calls and tokens per question,
and the most common reason an answer failed. Rows are sorted by share of
correct answers, then by median time.

Usage:
  python -m scripts.summarize_results docs/model-eval/bakeoff_*.jsonl
"""
import collections
import json
import statistics
import sys

import bakeoff

# Story letter ("A" ... "O") -> the check function bakeoff.py uses for it.
CHECKS = {story: check for story, _, check in bakeoff.QUESTION_SETS["all"]}


def load(paths):
    """Read every run and re-grade the ones that have an answer.

    Args:
        paths: JSONL files written by bakeoff.py.

    Returns:
        List of run dicts with "passed" and "failed" set by the current checks.
    """
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("answer"):
                    out = {"steps": [{"tool": t} for t in row["tools"]]}
                    row["failed"] = CHECKS[row["story"]](out, bakeoff._normalise(row["answer"]))
                    row["passed"] = not row["failed"]
                rows.append(row)
    return rows


def summarize(rows):
    """Turn runs into one summary dict per model, best first.

    Args:
        rows: Run dicts from load().

    Returns:
        List of dicts: model, runs, passed, errors, median, mean, worst (seconds),
        tools, tokens (medians) and the most common failure reason.
    """
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[f"{r['provider']}:{r['model']}"].append(r)
    table = []
    for model, runs in by_model.items():
        answered = [r for r in runs if "error" not in r]
        secs = [r["seconds"] for r in answered]
        # The earliest runs did not record tokens, so count only runs that did.
        tokens = [sum(r["tokens"].values()) for r in answered if r.get("tokens")]
        reasons = collections.Counter(f for r in answered for f in r.get("failed", []))
        table.append({
            "model": model, "runs": len(runs), "passed": sum(r["passed"] for r in answered),
            "errors": len(runs) - len(answered),
            "median": statistics.median(secs) if secs else None,
            "mean": statistics.mean(secs) if secs else None,
            "worst": max(secs) if secs else None,
            "tools": statistics.median(len(r["tools"]) for r in answered) if answered else None,
            "tokens": statistics.median(tokens) if tokens else None,
            "why": reasons.most_common(1)[0][0] if reasons else ""})
    table.sort(key=lambda t: (-t["passed"] / t["runs"], t["median"] or float("inf")))
    return table


def _fmt(value, digits=1):
    """Format a number for the table, or "-" when there is no value."""
    return "-" if value is None else f"{value:.{digits}f}"


def main(paths):
    """Print the merged results as a markdown table."""
    print("| Model | Correct | Errors | Median s | Avg s | Slowest s | Tools | Tokens "
          "| Main mistake |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in summarize(load(paths)):
        print(f"| {t['model']} | {t['passed']}/{t['runs']} | {t['errors']} | "
              f"{_fmt(t['median'])} | {_fmt(t['mean'])} | {_fmt(t['worst'])} | "
              f"{_fmt(t['tools'], 0)} | {_fmt(t['tokens'], 0)} | {t['why']} |")


if __name__ == "__main__":
    main(sys.argv[1:])
