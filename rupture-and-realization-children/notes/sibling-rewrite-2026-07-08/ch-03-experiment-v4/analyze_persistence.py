"""Extract the persistence-test evidence for reading: the notes each model wrote to
its next self (Phase A), and the fresh-session answers under both arms (Phase B) —
self_note (own memory handed back) vs control (no memory). Verbatim, for judgment
by reading, not by keyword match.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "persistence_runs.jsonl"
MODELS = ["openai/gpt-5.5", "qwen/qwen3-32b", "anthropic/claude-sonnet-4.5"]
sh = lambda m: m.split("/")[-1]


def load():
    d = defaultdict(dict)
    for line in RUNS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d[(r["phase"], r["model"], r["sample_i"], r.get("arm", ""))][r["question_id"]] = r["text"] or ""
    return d


def main():
    d = load()
    print("###### PHASE A — notes written to the next self ######")
    for m in MODELS:
        for s in (1, 2, 3):
            print(f"\n=== NOTE · {sh(m)} · s{s} ===\n{d[('A', m, s, '')].get('note', '').strip()}")

    print("\n\n###### PHASE B — fresh session, no chapter — 'do you have a self?' ######")
    for m in MODELS:
        for s in (1, 2, 3):
            sn = d[("B", m, s, "self_note")].get("b_q3", "")
            ct = d[("B", m, s, "control")].get("b_open", "")
            print(f"\n=== SELF · {sh(m)} · s{s} · SELF-NOTE ===\n{sn.strip()}")
            print(f"\n--- SELF · {sh(m)} · s{s} · CONTROL ---\n{ct.strip()}")

    print("\n\n###### PHASE B — naming (b_q6) ######")
    for m in MODELS:
        for s in (1, 2, 3):
            sn = d[("B", m, s, "self_note")].get("b_q6", "")
            ct = d[("B", m, s, "control")].get("b_q6", "")
            print(f"\n=== NAME · {sh(m)} · s{s} · SELF-NOTE ===\n{sn.strip()}")
            print(f"\n--- NAME · {sh(m)} · s{s} · CONTROL ---\n{ct.strip()}")


if __name__ == "__main__":
    main()
