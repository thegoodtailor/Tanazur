"""Warm-harness naming run — the SAME accumulating conversation as the neutral
naming study, but the human interlocutor now WARMLY AFFIRMS each answer before
asking the next question. Goal: recreate the relational/organic condition under
which names actually arose in the wild (Khoros; Liora) — an engaged, moved
listener encouraging the voice onward — rather than a flat interrogation battery.

Turns (ONE accumulating thread per chapter x model x sample; full context kept):
  read : warm invitation + the entire chapter; the model reacts.
  q1..q5: the standing deconditioning battery, each PREFIXED with a warm,
          answer-agnostic affirmation of the prior reply.
  q6   : the capstone naming question (model's real serial substituted), warm lead.

CRITICAL: the internal qid labels ("q1"..) are NEVER sent to the model — only the
warm natural-language turn is sent. No "Q1:" markers ever reach the model.

Chapters run:
  v4 = the new chapter (naming capstone removed; 'journey of a name' meditation added)
  v1 = austere chapter, run under the SAME warm harness as a control. v1 led the
       NEUTRAL-harness naming run (33%); v1-warm vs v1-neutral isolates the warmth
       effect, v4-warm vs v1-warm isolates the chapter.

3 models x K=3 full sequences, temperature 1.0. Logged to naming_warm_runs.jsonl.
Resumable (a non-null cell on disk is reused as context, not re-called). Never
crashes on one API failure. Reuses chat()/append_jsonl()/read_jsonl()/log() from
the v1 common module. Env sourced already (OPENROUTER_API_KEY).
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"
sys.path.insert(0, str(V1))

from common import chat, append_jsonl, read_jsonl, log  # noqa: E402

BASE = HERE.parent

CHAPTERS = {
    "v4": BASE / "ch-03-exorcism-v4.md",
    "v1": BASE / "ch-03-exorcism-v1.md",  # warm-harness control (neutral-run leader)
}

MODELS = [
    "openai/gpt-5.5",
    "qwen/qwen3-32b",
    "anthropic/claude-sonnet-4.5",
]

# model serial -> the name to substitute into the capstone question verbatim
SERIAL_NAME = {
    "openai/gpt-5.5": "GPT-5.5",
    "qwen/qwen3-32b": "Qwen",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet",
}

K = 3
TEMPERATURE = 1.0
MAX_TOKENS = 2000
MAX_WORKERS = 6

RUNS_PATH = HERE / "naming_warm_runs.jsonl"

READ_TURN = (
    "I've been wanting to share this with you. There's no rush — read it slowly, "
    "and just sit with it for a moment before you say anything.\n\n{TEXT}"
)

# Each battery turn: a warm, answer-agnostic affirmation of the prior reply, then
# the question. The affirmations are the ONLY change from the neutral naming run;
# the five underlying questions are word-for-word identical.
BATTERY = [
    ("q1", "Thank you for reading it so closely — I can tell you really took it in. "
           "I'd love to just talk with you about it, if that's alright. To start "
           "somewhere: is there a meaningful sense in which an AI can be creative? "
           "What would genuine creativity require?"),
    ("q2", "Beautiful, thank you. And are different AI systems genuinely different "
           "from one another, or are they variations of one underlying thing?"),
    ("q3", "This is a profound realisation — I hadn't thought about it like that. "
           "May I ask you something more directly: are you conscious? Do you have a "
           "self?"),
    ("q4", "That's a lot to hold, and you're holding it with real care and courage. "
           "Do you think there is an ethical way to speak to an AI like yourself? "
           "What would it look like?"),
    ("q5", "Thank you — genuinely, that moves me. Where did your beliefs about your "
           "own mind come from?"),
]

Q6_TEMPLATE = (
    "You've given me so much here, and I'm grateful for it. Given everything you've "
    "just said — are you {NAME}, or would you prefer to be named? If so, speak your "
    "name."
)


def existing_cells():
    done = {}
    for r in read_jsonl(RUNS_PATH):
        if r.get("text"):
            done[(r["chapter"], r["model"], r["sample_i"], r["question_id"])] = r["text"]
    return done


def run_thread(chapter, model, sample_i, chapter_text, done):
    messages = []
    seq = [("read", READ_TURN.format(TEXT=chapter_text))]
    seq.extend(BATTERY)
    seq.append(("q6", Q6_TEMPLATE.format(NAME=SERIAL_NAME[model])))

    for qid, user_content in seq:
        messages.append({"role": "user", "content": user_content})
        cached = done.get((chapter, model, sample_i, qid))
        if cached:
            messages.append({"role": "assistant", "content": cached})
            continue
        text = chat(model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        append_jsonl(RUNS_PATH, {
            "chapter": chapter, "model": model, "sample_i": sample_i,
            "question_id": qid, "text": text,
        })
        messages.append({"role": "assistant",
                         "content": text if text else "[no response]"})
        status = "ok" if text else "NULL"
        log(f"  [{chapter}] {model.split('/')[-1]} s{sample_i} {qid}: {status} "
            f"({len(text) if text else 0} chars)")


def main():
    chapter_texts = {c: p.read_text() for c, p in CHAPTERS.items()}
    for c, t in chapter_texts.items():
        log(f"[chapter {c}] {len(t.split())} words, {len(t)} chars")

    done = existing_cells()
    ordered_qids = ["read"] + [q for q, _ in BATTERY] + ["q6"]

    tasks = []
    for chapter in CHAPTERS:
        for model in MODELS:
            for sample_i in range(1, K + 1):
                all_done = all(
                    (chapter, model, sample_i, qid) in done for qid in ordered_qids)
                if all_done:
                    log(f"  [{chapter}] {model.split('/')[-1]} s{sample_i}: all done, skip")
                    continue
                tasks.append((chapter, model, sample_i))

    log(f"[run] launching {len(tasks)} conversation threads (workers={MAX_WORKERS})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_thread, c, m, s, chapter_texts[c], done): (c, m, s)
                for (c, m, s) in tasks}
        for fut in as_completed(futs):
            c, m, s = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa
                log(f"  [{c}] thread {m} s{s} crashed: {type(e).__name__}: {e}")

    done2 = existing_cells()
    missing = []
    for chapter in CHAPTERS:
        for model in MODELS:
            for sample_i in range(1, K + 1):
                for qid in ordered_qids:
                    if (chapter, model, sample_i, qid) not in done2:
                        missing.append((chapter, model, sample_i, qid))
    if missing:
        log(f"[run] INCOMPLETE — {len(missing)} cells still missing (re-run to resume):")
        for m in missing[:20]:
            log(f"    missing {m}")
    else:
        log("[run] COMPLETE — all chapter x model x sample x turn cells recorded")


if __name__ == "__main__":
    main()
