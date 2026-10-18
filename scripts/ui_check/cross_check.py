"""Cross-check every UI card against the API response it was built from.

Reads ui_full/results.jsonl (playwright-test-ui-full.js) and flags:
  * the reasoning line's numbers not matching the steps (tool calls, deepest hop, queries, ms)
  * a root-cause card whose failure path does not end at the blamed service
  * a root-cause card with no Act row, or an Act row on a card without a root cause
  * a page row naming someone the tools did not return as on call
  * console errors, non-200 responses, missing badge, an empty full answer
  * the ladder shown for fewer than 2 steps, or missing for 2 or more
  * a card naming a service as 'Not related' that sits on the root cause's chain
"""
import json
import re
import sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
bad = 0


def flag(r, msg):
    global bad
    bad += 1
    print(f"FLAG {r['id']:<4} {msg}")


for r in rows:
    api, dom = r["api"] or {}, r["dom"]
    steps = api.get("steps", [])
    ok_steps = [s for s in steps if "error" not in s["result"]]
    summ = api.get("summary", {})
    if r["status"] != 200:
        flag(r, f"HTTP {r['status']}")
    if r["errors"]:
        flag(r, f"console: {r['errors'][:2]}")
    if not dom["badge"]:
        flag(r, "no badge row")
    if not (dom["full"] or "").strip():
        flag(r, "full answer text empty")

    # Reasoning line numbers
    if ok_steps:
        deep = 0
        for s in ok_steps:
            res = s["result"]
            if s["tool"] == "dependencies":
                deep = max(deep, max([d["hops"] for d in res["depends_on"]] or [0]))
            elif s["tool"] == "blast_radius":
                deep = max(deep, max([d["hops"] for d in res["affected"]] or [0]))
            elif s["tool"] == "dependency_path" and res.get("path"):
                deep = max(deep, len(res["path"]) - 1)
        want = {"calls": len(ok_steps), "queries": len(api.get("queries", [])),
                "ms": sum(s["ms"] for s in steps), "deep": deep}
        line = dom["reasonline"] or ""
        m = re.match(r"(\d+) tool calls?(?: reached up to (\d+) hops? deep)?,? running (\d+) Cypher quer(?:y|ies) in (\d+) ms", line)
        if not m:
            flag(r, f"reasoning line unreadable: {line!r}")
        else:
            got = {"calls": int(m[1]), "deep": int(m[2] or 0), "queries": int(m[3]), "ms": int(m[4])}
            if got != want:
                flag(r, f"reasoning line {got} != steps {want}")
    elif dom["reasonline"]:
        flag(r, "reasoning line shown with no successful steps")

    # Ladder visibility
    n = len(ok_steps)
    if (n >= 2) != bool(dom["ladder"]):
        flag(r, f"ladder rows={len(dom['ladder'])} for {n} steps")
    if dom["ladder"] and len(dom["ladder"]) != n:
        flag(r, f"ladder rows {len(dom['ladder'])} != {n} steps")

    # Root cause card
    rc = summ.get("root_cause")
    if rc:
        if rc["service"] not in (dom["headline"] or ""):
            flag(r, f"headline {dom['headline']!r} lacks {rc['service']}")
        p = summ.get("path")
        if p and rc["service"] not in (p[0], p[-1]):
            flag(r, f"failure path {p} does not reach root cause {rc['service']}")
        if len(dom["acts"]) != 2:
            flag(r, f"{len(dom['acts'])} Act buttons on a root-cause card")
    elif dom["acts"]:
        flag(r, "Act buttons on a card with no root cause")

    # Page rows name only people a who_to_page call returned
    on_call = {s["result"].get("on_call") for s in ok_steps if s["tool"] == "who_to_page"} - {None}
    for page in dom["pages"]:
        if page.startswith("Page ") and not any(page.startswith("Page " + o) for o in on_call):
            flag(r, f"page row {page[:60]!r} names someone not returned on call {on_call}")

    # Unrelated vs chain
    if rc:
        reach = set()
        for s in ok_steps:
            res = s["result"]
            if s["tool"] == "dependencies":
                reach |= {d["service"] for d in res["depends_on"]}
            if s["tool"] == "blast_radius":
                reach |= {d["service"] for d in res["affected"]}
        # services on the root cause's own chain must not be listed unrelated
        for u in summ.get("unrelated", []):
            for s in ok_steps:
                if s["tool"] == "dependency_path" and s["result"].get("path") and u in s["result"]["path"]:
                    flag(r, f"{u} listed unrelated but is on path {s['result']['path']}")

print(f"\n{len(rows)} cards checked, {bad} flags")
