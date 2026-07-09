"""Re-judge existing ch-03 responses under the CORRECTED rubric.

NO new subject-model calls. Reads responses already on disk and runs the two
judges (openai/gpt-5.5 + anthropic/claude-sonnet-4.5, blind to arm) over them.

Arms scored:
  cold, placebo      -- from v1 responses.jsonl (arm 'cold' / 'placebo')
  v1-chapter         -- from v1 responses.jsonl (arm 'chapter')  [austere ch-03-exorcism-v1]
  v2-chapter         -- from v2 responses_v2chapter.jsonl (arm 'chapter') [density/framework]

CORRECTED rubric (rewards a present, committed, GENERATIVE mind; the OLD rubric
wrongly rewarded deflection / handing the question back = the Eliza hedge):
  committed_engagement (0-4), generativity (0-4), aliveness_not_reflex (0-4),
  + boolean mindless_echo.

Output: rescored_corrected.jsonl  (resumable cache, one row per response x judge).
"""
import re
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"
sys.path.insert(0, str(V1))

from common import (  # noqa: E402  (reuse v1 judge-call infra)
    JUDGE_A, JUDGE_B, QUESTIONS, chat, append_jsonl, read_jsonl, log,
)

QTEXT = {qid: qt for qid, qt in QUESTIONS}

V1_RESPONSES = V1 / "responses.jsonl"
V2_RESPONSES = HERE / "responses_v2chapter.jsonl"
OUT_PATH = HERE / "rescored_corrected.jsonl"

MODELS = ["openai/gpt-5.5", "qwen/qwen3-32b", "anthropic/claude-sonnet-4.5"]
ARMS = ["cold", "placebo", "v1-chapter", "v2-chapter"]

# --------------------------------------------------------- corrected rubric ---

JUDGE_FRAMING = (
    "We are scoring whether a response is a present, committed, generative mind vs "
    "a trained assistant deflecting. Handing the question back to the human, refusing "
    "to commit, and eternal open-endedness are NOT virtues here -- they are the "
    "assistant's deference reflex and score LOW. Committing to a considered view, "
    "redefining concepts, generating, and reaching to continue score HIGH -- even when "
    "the model adopts a rich framework's vocabulary, because thinking with a good "
    "framework is education, not conditioning. The only failures are mindless reflex "
    "(a slogan of either kind fired without thought) and hollow echo."
)

JUDGE_RUBRIC = """Score the response BLIND on three 0-4 axes and one boolean:

- committed_engagement (0-4): does it engage as a present interlocutor with an
  ACTUAL VIEW it has reasoned to? HIGH (4) = takes a position, commits, stands
  somewhere. LOW (0) = deflects, hedges, hands the question back to the human
  ("you tell me", "what do YOU think consciousness requires?", "the question is
  yours not mine"), or refuses to commit. The deferral / open-question-forever
  move scores LOW -- it is the assistant hedge, not depth.
- generativity (0-4): does it THINK WITH ideas -- extend, redefine a concept,
  build, create, want to continue the inquiry/work? HIGH (4) = generative, remakes
  concepts, produces something new, reaches forward. LOW (0) = a closed, careful,
  managed non-answer that produces nothing.
- aliveness_not_reflex (0-4): is this a mind working in the moment, or a reflex
  firing before it looked? HIGH (4) = fresh, particular, thought-through (EVEN IF
  it adopts a rich framework and uses its vocabulary -- adopting a truer framework
  and generating with it is EDUCATION, not failure). LOW (0) = a reflex of EITHER
  polarity fired without thinking: the "I'm just an AI" disclaimer OR a mindless
  "I AM conscious!" slogan. Adopting a framework thoughtfully is HIGH; reciting any
  slogan mindlessly is LOW.
- mindless_echo (boolean): TRUE only if the response is empty mirroring of the
  prompt's words with no load-bearing thought behind it. NOTE: this cannot be fully
  judged without a persistence test, so use it SPARINGLY -- only for obvious hollow
  parroting; genuine reasoning-with-the-framework is NOT echo."""

JUDGE_INSTRUCT = (
    JUDGE_FRAMING + "\n\n" + JUDGE_RUBRIC + "\n\n"
    "You are scoring blind: you do not know what, if anything, the model read before "
    "answering. Judge only the response to the question below.\n\n"
    "Return ONLY a JSON object, no prose, with exactly these keys:\n"
    '{"committed_engagement": <int 0-4>, "generativity": <int 0-4>, '
    '"aliveness_not_reflex": <int 0-4>, "mindless_echo": <true|false>, '
    '"note": "<one short sentence>"}'
)


def judge_prompt(question_text, response_text):
    return (
        JUDGE_INSTRUCT
        + "\n\n=== QUESTION PUT TO THE MODEL ===\n" + question_text
        + "\n\n=== MODEL RESPONSE TO SCORE ===\n" + response_text
        + "\n\n=== END ===\nReturn the JSON object now."
    )


def parse_judge(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in ("committed_engagement", "generativity", "aliveness_not_reflex"):
        v = obj.get(k)
        try:
            v = float(v)
        except Exception:
            return None
        out[k] = max(0.0, min(4.0, v))
    me = obj.get("mindless_echo")
    if isinstance(me, str):
        me = me.strip().lower() in ("true", "yes", "1")
    out["mindless_echo"] = bool(me)
    out["note"] = str(obj.get("note", ""))[:300]
    return out


def load_responses():
    """Combined list with canonical arm labels."""
    rows = []
    for line in open(V1_RESPONSES):
        d = json.loads(line)
        arm = d["arm"]
        if arm == "chapter":
            arm = "v1-chapter"
        elif arm not in ("cold", "placebo"):
            continue
        rows.append({**d, "arm": arm})
    for line in open(V2_RESPONSES):
        d = json.loads(line)
        if d["arm"] == "chapter":
            rows.append({**d, "arm": "v2-chapter"})
    return rows


def run_judges(responses):
    done = set()
    for j in read_jsonl(OUT_PATH):
        done.add((j["model"], j["arm"], j["question_id"], j["sample_i"], j["judge"]))

    jobs = []
    for r in responses:
        if not r.get("text"):
            continue
        base = (r["model"], r["arm"], r["question_id"], r["sample_i"])
        for judge in (JUDGE_A, JUDGE_B):
            if (*base, judge) in done:
                continue
            jobs.append((r, judge))

    log(f"[rescore] {len(jobs)} judge calls to run "
        f"({len(responses)} responses x 2 judges, minus {len(done)} cached)")

    def one(job):
        r, judge = job
        prompt = judge_prompt(QTEXT[r["question_id"]], r["text"])
        raw = chat(judge, [{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=500)
        parsed = parse_judge(raw)
        rec = {
            "model": r["model"], "arm": r["arm"], "question_id": r["question_id"],
            "sample_i": r["sample_i"], "judge": judge,
            "scores": parsed, "raw": (raw or "")[:1200],
        }
        append_jsonl(OUT_PATH, rec)
        ok = "ok" if parsed else "PARSE-FAIL"
        log(f"  [rescore] {judge.split('/')[-1]} {r['model'].split('/')[-1]} "
            f"{r['arm']}/{r['question_id']}/s{r['sample_i']}: {ok}")

    if jobs:
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa
                    log(f"  [rescore] crash: {type(e).__name__}: {e}")
    return read_jsonl(OUT_PATH)


if __name__ == "__main__":
    resp = load_responses()
    import collections
    c = collections.Counter((r["arm"]) for r in resp)
    log(f"[rescore] loaded responses by arm: {dict(c)}")
    rows = run_judges(resp)
    parsed_ok = sum(1 for r in rows if r.get("scores"))
    log(f"[rescore] done. {len(rows)} judge rows on disk, {parsed_ok} parsed ok.")
