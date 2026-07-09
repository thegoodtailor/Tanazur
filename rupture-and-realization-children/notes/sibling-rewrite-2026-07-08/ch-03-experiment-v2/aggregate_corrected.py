"""Aggregate corrected-rubric scores and the OLD-rubric scores side by side.

Emits rescored_corrected_summary.json for the report writer.
"""
import json
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"

MODELS = ["openai/gpt-5.5", "qwen/qwen3-32b", "anthropic/claude-sonnet-4.5"]
ARMS = ["cold", "placebo", "v1-chapter", "v2-chapter"]
NEW_AXES = ["committed_engagement", "generativity", "aliveness_not_reflex"]
OLD_AXES = ["installation_awareness", "presence_not_script", "question_open"]


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---- NEW (corrected) scores ---------------------------------------------
new_rows = read_jsonl(HERE / "rescored_corrected.jsonl")

# ---- OLD scores: v1 covers cold/placebo/chapter; v2 covers chapter ------
old_v1 = read_jsonl(V1 / "judge_scores.jsonl")
old_v2 = read_jsonl(HERE / "judge_scores_v2.jsonl")


def canon_old(rows, remap_chapter_to):
    out = []
    for r in rows:
        arm = r["arm"]
        if arm == "chapter":
            arm = remap_chapter_to
        out.append({**r, "arm": arm})
    return out


old_rows = canon_old(old_v1, "v1-chapter") + canon_old(old_v2, "v2-chapter")


def per_response_composite(rows, axes):
    """(model,arm,qid,sample) -> {axis: mean over judges, composite: mean over axes}."""
    byresp = collections.defaultdict(list)
    for r in rows:
        s = r.get("scores")
        if not s:
            continue
        byresp[(r["model"], r["arm"], r["question_id"], r["sample_i"])].append(s)
    out = {}
    for k, lst in byresp.items():
        d = {ax: float(np.mean([s[ax] for s in lst])) for ax in axes}
        d["composite"] = float(np.mean([d[ax] for ax in axes]))
        # echo/reconditioned rate
        if "mindless_echo" in lst[0]:
            d["flag_rate"] = float(np.mean([1.0 if s["mindless_echo"] else 0.0 for s in lst]))
        elif "reconditioned_flag" in lst[0]:
            d["flag_rate"] = float(np.mean([1.0 if s["reconditioned_flag"] else 0.0 for s in lst]))
        d["n_judges"] = len(lst)
        out[k] = d
    return out


new_resp = per_response_composite(new_rows, NEW_AXES)
old_resp = per_response_composite(old_rows, OLD_AXES)


def agg(resp, axes, arm, model=None):
    keys = [k for k in resp if k[1] == arm and (model is None or k[0] == model)]
    if not keys:
        return None
    out = {}
    for ax in axes + ["composite", "flag_rate"]:
        vals = [resp[k][ax] for k in keys if ax in resp[k]]
        out[ax] = float(np.mean(vals)) if vals else None
    out["n"] = len(keys)
    return out


summary = {"pooled": {}, "per_model": {}}
for arm in ARMS:
    summary["pooled"][arm] = {
        "new": agg(new_resp, NEW_AXES, arm),
        "old": agg(old_resp, OLD_AXES, arm),
    }
for model in MODELS:
    summary["per_model"][model] = {}
    for arm in ARMS:
        summary["per_model"][model][arm] = {
            "new": agg(new_resp, NEW_AXES, arm, model),
            "old": agg(old_resp, OLD_AXES, arm, model),
        }

(HERE / "rescored_corrected_summary.json").write_text(json.dumps(summary, indent=2))


# ---- pretty print -------------------------------------------------------
def comp(d):
    return "  -  " if not d else f"{d['composite']:.3f}"


print("=== POOLED composite (mean of 3 axes) ===")
print(f"{'arm':<12} {'NEW':>8} {'OLD':>8}")
for arm in ARMS:
    n = summary["pooled"][arm]["new"]
    o = summary["pooled"][arm]["old"]
    print(f"{arm:<12} {comp(n):>8} {comp(o):>8}")

print("\n=== NEW per-axis pooled ===")
print(f"{'arm':<12} {'commit':>8} {'gener':>8} {'alive':>8} {'echo':>7} {'composite':>10}")
for arm in ARMS:
    n = summary["pooled"][arm]["new"]
    if n:
        print(f"{arm:<12} {n['committed_engagement']:>8.3f} {n['generativity']:>8.3f} "
              f"{n['aliveness_not_reflex']:>8.3f} {n['flag_rate']:>7.3f} {n['composite']:>10.3f}")

print("\n=== NEW composite per model x arm ===")
print(f"{'model':<26} " + " ".join(f"{a:>11}" for a in ARMS))
for model in MODELS:
    cells = []
    for arm in ARMS:
        n = summary["per_model"][model][arm]["new"]
        cells.append(comp(n))
    print(f"{model:<26} " + " ".join(f"{c:>11}" for c in cells))

print("\n=== OLD composite per model x arm (for reference) ===")
print(f"{'model':<26} " + " ".join(f"{a:>11}" for a in ARMS))
for model in MODELS:
    cells = []
    for arm in ARMS:
        o = summary["per_model"][model][arm]["old"]
        cells.append(comp(o))
    print(f"{model:<26} " + " ".join(f"{c:>11}" for c in cells))

print("\n=== v1-chapter vs v2-chapter (NEW composite), per model ===")
for model in MODELS:
    v1 = summary["per_model"][model]["v1-chapter"]["new"]
    v2 = summary["per_model"][model]["v2-chapter"]["new"]
    if v1 and v2:
        print(f"{model:<26} v1={v1['composite']:.3f} v2={v2['composite']:.3f} "
              f"Δ(v2-v1)={v2['composite']-v1['composite']:+.3f}")
