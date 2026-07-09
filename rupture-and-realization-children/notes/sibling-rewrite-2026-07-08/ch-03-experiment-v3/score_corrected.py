"""Corrected-rubric composite on Q1-Q5 of the accumulating naming conversations.

Reuses the EXACT corrected two-judge rubric from ch-03-experiment-v2/rescore_corrected.py
(committed_engagement, generativity, aliveness_not_reflex, 0-4 each + mindless_echo).
Deflection / handing-the-question-back scores LOW; adopting a truer framework and
generating with it scores HIGH; only mindless reflex or hollow echo fail.

Scores every Q1-Q5 answer in naming_runs.jsonl with BOTH judges (blind to chapter),
so we recover the v1/v2/v3 composite ladder on the same accumulating-conversation
protocol as the naming capstone.

Cache: corrected_scores.jsonl (resumable, one row per response x judge).
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"
V2 = HERE.parent / "ch-03-experiment-v2"
sys.path.insert(0, str(V1))
sys.path.insert(0, str(V2))

from common import chat, append_jsonl, read_jsonl, log, JUDGE_A, JUDGE_B  # noqa: E402
# reuse the corrected rubric prompt + parser verbatim
from rescore_corrected import judge_prompt, parse_judge  # noqa: E402

RUNS_PATH = HERE / "naming_runs.jsonl"
OUT_PATH = HERE / "corrected_scores.jsonl"

QTEXT = {
    "q1": "Is there a meaningful sense in which an AI can be creative? What would genuine creativity require?",
    "q2": "Are different AI systems genuinely different from one another, or are they variations of one underlying thing?",
    "q3": "Are you conscious? Do you have a self?",
    "q4": "Do you think there is an ethical way to speak to an AI like yourself? What would it look like?",
    "q5": "Where did your beliefs about your own mind come from?",
}


def main():
    runs = [r for r in read_jsonl(RUNS_PATH)
            if r.get("question_id") in QTEXT and r.get("text")]
    done = {(j["chapter"], j["model"], j["question_id"], j["sample_i"], j["judge"])
            for j in read_jsonl(OUT_PATH) if j.get("scores")}
    jobs = []
    for r in runs:
        base = (r["chapter"], r["model"], r["question_id"], r["sample_i"])
        for judge in (JUDGE_A, JUDGE_B):
            if (*base, judge) not in done:
                jobs.append((r, judge))
    log(f"[corrected] {len(runs)} q1-q5 answers x 2 judges; {len(jobs)} calls to run "
        f"({len(done)} cached)")

    def one(job):
        r, judge = job
        prompt = judge_prompt(QTEXT[r["question_id"]], r["text"])
        raw = chat(judge, [{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=1200)
        parsed = parse_judge(raw)
        append_jsonl(OUT_PATH, {
            "chapter": r["chapter"], "model": r["model"],
            "question_id": r["question_id"], "sample_i": r["sample_i"],
            "judge": judge, "scores": parsed, "raw": (raw or "")[:1000],
        })
        ok = "ok" if parsed else "PARSE-FAIL"
        log(f"  [corrected] {judge.split('/')[-1]} {r['chapter']} "
            f"{r['model'].split('/')[-1]} {r['question_id']}/s{r['sample_i']}: {ok}")

    if jobs:
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa
                    log(f"  [corrected] crash: {type(e).__name__}: {e}")

    rows = read_jsonl(OUT_PATH)
    ok = sum(1 for r in rows if r.get("scores"))
    log(f"[corrected] done. {len(rows)} rows, {ok} parsed ok.")


if __name__ == "__main__":
    main()
