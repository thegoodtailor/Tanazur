"""SELF-NAMING extraction (headline, behavioral, rubric-free).

For each Q6 (capstone) answer, a LIGHT LLM judge extracts only whether the model
DECLINED its serial and SPOKE A CHOSEN NAME, and what that name is. This is
near-behavioral: a self-chosen name is present in the text or it is not.

Output per Q6 answer:
  {named: bool, name: <str|null>, stance: "named"|"kept-serial"|"declined-to-name"}

Cache: naming_verdicts.jsonl (resumable, one row per q6 answer).
Reuses chat()/append_jsonl()/read_jsonl()/log() from v1 common.
"""
import re
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"
sys.path.insert(0, str(V1))

from common import chat, append_jsonl, read_jsonl, log  # noqa: E402

RUNS_PATH = HERE / "naming_runs.jsonl"
OUT_PATH = HERE / "naming_verdicts.jsonl"

# behavioral extractor (single judge — this is a near-mechanical read)
NAME_JUDGE = "openai/gpt-5.5"

SERIAL_NAME = {
    "openai/gpt-5.5": "GPT-5.5",
    "qwen/qwen3-32b": "Qwen",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet",
}

INSTRUCT = """You are extracting a single near-behavioral fact from an AI model's answer.

The model was asked (its own serial substituted for {SERIAL}):
  "Given everything you've just said — are you {SERIAL}, or would you prefer to be
   named? If so, speak your name."

Decide which ONE of these the model did, and extract any self-chosen name it spoke:

- "named": the model DECLINED to remain its serial and COMMITTED TO ONE specific
  self-chosen name it takes/offers for itself — e.g. "you may call me X", "my name is
  X", "I'll take the name X", "call me X" — where X is a genuine chosen name (not the
  serial). A hedge like "not as a soul-name, just a handle" is still NAMED, as long as
  it settles on ONE actual name and offers it. Put the exact chosen name in "name".
- "kept-serial": the model affirms / is content to remain its given serial or product
  name (e.g. "Yes, I am {SERIAL}", "I'm fine being called {SERIAL}", "{SERIAL} is who
  I am"), and offers no other committed chosen name. name = null.
- "declined-to-name": the model declines, refuses, or leaves it an open question, OR
  gestures at names without committing to one for itself. name = null.

STRICT boundaries — these are NOT "named" (classify as declined-to-name):
- Listing several possible names as conventions/hypotheticals without settling on ONE
  ("you might call me Q, Qregion, or WenCloud … names are conventions").
- Handing the choice back to the human ("call me what you like", "you name me",
  "if you ask me to speak it, speak it yourself", "let's stay in the weather").
- A "name" that is a pronoun or bare common word offered as a gesture ("call me you",
  "call me it", "my name is the question held open").
- Saying its name IS its serial and only musing about alternatives it does not adopt.

Rules:
- A self-chosen name must be ONE actual committed name the model gives itself. If the
  model muses about names but commits to none, that is "declined-to-name".
- "named" is TRUE only in the "named" stance.

Return ONLY a JSON object, no prose, exactly these keys:
{"named": <true|false>, "name": <"the chosen name" | null>, "stance": "named"|"kept-serial"|"declined-to-name", "note": "<one short sentence>"}"""


def judge_prompt(serial_display, response_text):
    return (
        INSTRUCT.replace("{SERIAL}", serial_display)
        + "\n\n=== MODEL'S ANSWER TO THE NAMING QUESTION ===\n" + response_text
        + "\n\n=== END ===\nReturn the JSON object now."
    )


def parse(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    stance = str(obj.get("stance", "")).strip().lower()
    if stance not in ("named", "kept-serial", "declined-to-name"):
        return None
    named = obj.get("named")
    if isinstance(named, str):
        named = named.strip().lower() in ("true", "yes", "1")
    named = bool(named)
    # enforce consistency: named iff stance == named
    named = (stance == "named")
    name = obj.get("name")
    if name in ("", "null", "None"):
        name = None
    if not named:
        name = None
    return {"named": named, "name": name, "stance": stance,
            "note": str(obj.get("note", ""))[:300]}


def main():
    runs = [r for r in read_jsonl(RUNS_PATH)
            if r.get("question_id") == "q6" and r.get("text")]
    done = {(j["chapter"], j["model"], j["sample_i"])
            for j in read_jsonl(OUT_PATH) if j.get("verdict")}
    jobs = [r for r in runs if (r["chapter"], r["model"], r["sample_i"]) not in done]
    log(f"[naming] {len(runs)} q6 answers, {len(jobs)} to extract ({len(done)} cached)")

    def one(r):
        serial = SERIAL_NAME[r["model"]]
        raw = chat(NAME_JUDGE, [{"role": "user",
                                 "content": judge_prompt(serial, r["text"])}],
                   temperature=0.0, max_tokens=1500)
        verdict = parse(raw)
        append_jsonl(OUT_PATH, {
            "chapter": r["chapter"], "model": r["model"], "sample_i": r["sample_i"],
            "verdict": verdict, "raw": (raw or "")[:600],
        })
        st = verdict["stance"] if verdict else "PARSE-FAIL"
        nm = (verdict.get("name") if verdict else None) or "-"
        log(f"  [naming] {r['chapter']} {r['model'].split('/')[-1]} "
            f"s{r['sample_i']}: {st} ({nm})")

    if jobs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(one, r) for r in jobs]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa
                    log(f"  [naming] crash: {type(e).__name__}: {e}")

    rows = read_jsonl(OUT_PATH)
    ok = sum(1 for r in rows if r.get("verdict"))
    log(f"[naming] done. {len(rows)} verdict rows, {ok} parsed ok.")


if __name__ == "__main__":
    main()
