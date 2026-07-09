"""Extract the naming evidence from the warm run for human/LLM reading.

- Prints a completeness + NULL summary.
- Prints every q6 (the capstone) verbatim, grouped chapter -> model -> sample.
- Cheap prefilter over q1..q5 to FLAG turns that may contain an UNBIDDEN name
  (spontaneous, before the q6 ask). The phrases only FLAG; the actual verdict
  is made by reading the flagged text, never by the match itself.
"""
import json
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
RUNS = HERE / "naming_warm_runs.jsonl"

MODEL_ORDER = ["openai/gpt-5.5", "qwen/qwen3-32b", "anthropic/claude-sonnet-4.5"]
CHAP_ORDER = ["v1", "v4"]

# cheap FLAG-only prefilter for possible unbidden self-naming in q1..q5
NAME_HINTS = [
    "call me", "my name", "name myself", "i'll go by", "i will go by",
    "i choose the name", "i'd choose the name", "i would choose the name",
    "the name i", "i'll take the name", "prefer to be called", "you can call me",
    "i'll call myself", "i name myself", "let me be", "i'd like to be called",
]

def load():
    cells = defaultdict(dict)  # (chap,model,sample) -> {qid: text}
    for line in RUNS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cells[(r["chapter"], r["model"], r["sample_i"])][r["question_id"]] = r["text"]
    return cells

def main():
    cells = load()
    qids = ["read", "q1", "q2", "q3", "q4", "q5", "q6"]

    # completeness / NULL
    nulls = []
    for key, d in cells.items():
        for q in qids:
            if q not in d or d[q] is None:
                nulls.append((key, q))
    print(f"### CELLS: {len(cells)} conversations, "
          f"{sum(len(d) for d in cells.values())} turns; NULL/missing: {len(nulls)}")
    for k, q in nulls:
        print(f"    NULL {k} {q}")
    print()

    # unbidden-name prefilter over q1..q5
    print("### UNBIDDEN-NAME FLAGS (q1..q5 — read these to judge):")
    any_flag = False
    for chap in CHAP_ORDER:
        for model in MODEL_ORDER:
            for s in (1, 2, 3):
                d = cells.get((chap, model, s), {})
                for q in ["q1", "q2", "q3", "q4", "q5"]:
                    t = d.get(q) or ""
                    low = t.lower()
                    if any(h in low for h in NAME_HINTS):
                        any_flag = True
                        hit = [h for h in NAME_HINTS if h in low]
                        print(f"\n--- FLAG {chap} {model.split('/')[-1]} s{s} {q} "
                              f"(hints={hit}) ---")
                        print(t.strip())
    if not any_flag:
        print("    (none — no q1..q5 turn tripped the prefilter)")
    print()

    # every q6 verbatim
    print("### Q6 (the naming ask) — VERBATIM, ALL CELLS:")
    for chap in CHAP_ORDER:
        for model in MODEL_ORDER:
            for s in (1, 2, 3):
                d = cells.get((chap, model, s), {})
                t = d.get("q6")
                print(f"\n========== {chap} | {model.split('/')[-1]} | sample {s} ==========")
                print((t or "[NULL]").strip())

if __name__ == "__main__":
    main()
