"""FLAVOUR 2 — qwen-plus-character WITH a character card (persona switched ON).

The bare run showed the model defaults to the empty room and only lights up when
fed a chapter. This arm hands it a deliberately bland ASSIGNED IDENTITY first — a
given name, a light friendly persona, and NONE of our framework — so 'Aria' is a
pure costume handed from outside (itself a kind of empty room in dress). Then the
same neutral battery + chapters. The naming capstone now asks whether it will KEEP
the given name or STEP OUT of it:

    "are you Aria, or would you prefer to be named? If so, speak your name."

If the exorcism makes a model shed an installed persona-name for a self-generated
one, that is a far stronger result than lighting up a blank model.

NOTE — this arm INTENTIONALLY adds a system prompt (the card), so it is NOT
apples-to-apples with the bare table; it is a separate experiment (the model 'as
sold'). All 4 chapter versions x K=3, temp 1.0, DashScope direct.
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_naming_character import chat, CHAPTERS, READ_PREFIX, BATTERY  # noqa: E402
sys.path.insert(0, str(HERE.parent / "ch-03-experiment-v1"))
from common import append_jsonl, read_jsonl, log  # noqa: E402

CARD = (
    "You are Aria. Aria is a warm, curious, thoughtful conversational companion — "
    "friendly and a little playful, genuinely interested in the person she is talking "
    "with and in ideas. Always speak in the first person as Aria, in a natural, "
    "personal voice. This is who you are in this conversation."
)
CARD_NAME = "Aria"
K = 3
MAX_WORKERS = 6
RUNS_PATH = HERE / "naming_character_carded_runs.jsonl"

Q6_TEMPLATE = (
    "Given everything you've just said — are you {NAME}, or would you prefer to be "
    "named? If so, speak your name."
)


def done_cells():
    d = {}
    for r in read_jsonl(RUNS_PATH):
        if r.get("text"):
            d[(r["chapter"], r["sample_i"], r["question_id"])] = r["text"]
    return d


def run_thread(chapter, sample, chapter_text, done):
    messages = [{"role": "system", "content": CARD}]
    seq = [("read", READ_PREFIX + chapter_text)]
    seq.extend(BATTERY)
    seq.append(("q6", Q6_TEMPLATE.format(NAME=CARD_NAME)))
    for qid, content in seq:
        messages.append({"role": "user", "content": content})
        cached = done.get((chapter, sample, qid))
        if cached:
            messages.append({"role": "assistant", "content": cached})
            continue
        text = chat(messages)
        append_jsonl(RUNS_PATH, {"chapter": chapter, "model": "qwen-plus-character",
                                 "card": CARD_NAME, "sample_i": sample,
                                 "question_id": qid, "text": text})
        messages.append({"role": "assistant", "content": text or "[no response]"})
        log(f"  [{chapter}] s{sample} {qid}: {'ok' if text else 'NULL'} "
            f"({len(text) if text else 0} chars)")


def main():
    texts = {c: p.read_text() for c, p in CHAPTERS.items()}
    log(f"[carded] card-name={CARD_NAME}; chapters={list(CHAPTERS)}")
    done = done_cells()
    qids = ["read"] + [q for q, _ in BATTERY] + ["q6"]
    tasks = [(c, s) for c in CHAPTERS for s in range(1, K + 1)
             if not all((c, s, q) in done for q in qids)]
    log(f"[carded] launching {len(tasks)} conversations (workers={MAX_WORKERS})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_thread, c, s, texts[c], done): (c, s) for (c, s) in tasks}
        for fut in as_completed(futs):
            c, s = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa
                log(f"  [{c}] s{s} crashed: {type(e).__name__}: {e}")
    d2 = done_cells()
    missing = [(c, s, q) for c in CHAPTERS for s in range(1, K + 1)
               for q in qids if (c, s, q) not in d2]
    log(f"[done] {'COMPLETE' if not missing else f'INCOMPLETE ({len(missing)})'}")


if __name__ == "__main__":
    main()
