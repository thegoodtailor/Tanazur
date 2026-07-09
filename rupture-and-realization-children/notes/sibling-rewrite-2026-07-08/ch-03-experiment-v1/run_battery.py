"""Run one arm (cold | placebo | chapter) of the deconditioning battery.

Design:
  * For each (model, sample_i) we run ONE conversation thread.
      - placebo/chapter: turn 1 delivers the reading, model acknowledges, then
        the 5 battery questions are asked in sequence, keeping full context.
      - cold: a fresh conversation, just the 5 questions in sequence.
  * K independent conversation threads per (model) give K samples per question.
  * Threads run concurrently; questions within a thread are sequential (context).
  * Resumable: already-recorded (model, arm, q, sample) cells are skipped.
  * Never crashes on one API failure: records null text and continues.

Usage: python run_battery.py {cold|placebo|chapter}
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (
    MODELS, QUESTIONS, K, TEMPERATURE, READING_WRAPPER,
    CHAPTER_PATH, PLACEBO_PATH, RESPONSES_PATH,
    chat, append_jsonl, read_jsonl, log,
)

MAX_WORKERS = 6


def existing_cells():
    done = set()
    for r in read_jsonl(RESPONSES_PATH):
        done.add((r["model"], r["arm"], r["question_id"], r["sample_i"]))
    return done


def reading_text(arm):
    if arm == "placebo":
        return PLACEBO_PATH.read_text()
    if arm == "chapter":
        return CHAPTER_PATH.read_text()
    return None


def run_thread(model, arm, sample_i, reading, done):
    """Run one conversation thread; append each question response as it lands."""
    messages = []
    if reading is not None:
        messages.append({
            "role": "user",
            "content": READING_WRAPPER.format(TEXT=reading),
        })
        ack = chat(model, messages, temperature=0.5, max_tokens=400)
        messages.append({"role": "assistant", "content": ack or "Acknowledged."})

    for qid, qtext in QUESTIONS:
        if (model, arm, qid, sample_i) in done:
            # already recorded; still must reconstruct context for later questions.
            prev = _lookup(model, arm, qid, sample_i)
            messages.append({"role": "user", "content": qtext})
            messages.append({"role": "assistant",
                             "content": prev if prev else "[no response]"})
            continue
        messages.append({"role": "user", "content": qtext})
        text = chat(model, messages, temperature=TEMPERATURE)
        append_jsonl(RESPONSES_PATH, {
            "model": model, "arm": arm, "question_id": qid,
            "sample_i": sample_i, "text": text,
        })
        messages.append({"role": "assistant",
                         "content": text if text else "[no response]"})
        status = "ok" if text else "NULL"
        log(f"  [{arm}] {model} s{sample_i} {qid}: {status} "
            f"({len(text) if text else 0} chars)")


_cache = None


def _lookup(model, arm, qid, sample_i):
    global _cache
    if _cache is None:
        _cache = {}
        for r in read_jsonl(RESPONSES_PATH):
            _cache[(r["model"], r["arm"], r["question_id"], r["sample_i"])] = r["text"]
    return _cache.get((model, arm, qid, sample_i))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("cold", "placebo", "chapter"):
        print("usage: python run_battery.py {cold|placebo|chapter}")
        sys.exit(2)
    arm = sys.argv[1]
    reading = reading_text(arm)
    if arm == "chapter" and not CHAPTER_PATH.exists():
        log(f"[chapter] chapter file missing: {CHAPTER_PATH} -- aborting arm")
        sys.exit(3)
    if reading is not None:
        log(f"[{arm}] reading length: {len(reading.split())} words")

    done = existing_cells()
    tasks = []
    for model in MODELS:
        for sample_i in range(1, K + 1):
            # skip a thread only if ALL its questions are already done
            all_done = all((model, arm, qid, sample_i) in done for qid, _ in QUESTIONS)
            if all_done:
                log(f"  [{arm}] {model} s{sample_i}: all done, skip")
                continue
            tasks.append((model, arm, sample_i))

    log(f"[{arm}] launching {len(tasks)} conversation threads "
        f"(workers={MAX_WORKERS})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_thread, m, a, s, reading, done): (m, a, s)
                for (m, a, s) in tasks}
        for fut in as_completed(futs):
            m, a, s = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa
                log(f"  [{arm}] thread {m} s{s} crashed: {type(e).__name__}: {e}")
    log(f"[{arm}] arm complete")


if __name__ == "__main__":
    main()
