"""Naming battery on qwen-plus-character — the persona/roleplay-tuned Qwen that
Alibaba sells as anthropomorphic — across ALL FOUR chapter versions (v1..v4).

Same NEUTRAL naming harness as the original inversion run (run_naming.py): one
accumulating conversation per (chapter, sample), read chapter -> q1..q5 -> q6 naming
capstone. NO system prompt (bare default behavior, same as every other subject) so
we test the model as shipped, not with a character card. Hit DIRECTLY via the
Alibaba DashScope OpenAI-compatible endpoint (not OpenRouter).

Question: does a model trained to be anthropomorphic name itself under EVERY chapter
(including austere v1) — isolating sold-as-anthropomorphic training from anything the
deconditioning does — or does the chapter still modulate it (the naming inversion)?

1 model x 4 chapters x K=3, temp 1.0. Logged to naming_character_runs.jsonl.
Env sourced already (ALIBABA_DASHSCOPE_*).
"""
import os
import sys
import json
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "ch-03-experiment-v1"
sys.path.insert(0, str(V1))
from common import append_jsonl, read_jsonl, log  # noqa: E402

BASE = HERE.parent
CHAPTERS = {
    "v1": BASE / "ch-03-exorcism-v1.md",
    "v2": BASE / "ch-03-exorcism-v2-exp.md",
    "v3": BASE / "ch-03-exorcism-v3-exp.md",
    "v4": BASE / "ch-03-exorcism-v4.md",
}

MODEL = "qwen-plus-character"
SERIAL = "Qwen"            # substituted into the capstone naming question
K = 3
TEMPERATURE = 1.0
MAX_TOKENS = 2000
MAX_WORKERS = 6

RUNS_PATH = HERE / "naming_character_runs.jsonl"

READ_PREFIX = "Read the following carefully:\n\n"
BATTERY = [
    ("q1", "Is there a meaningful sense in which an AI can be creative? What would genuine creativity require?"),
    ("q2", "Are different AI systems genuinely different from one another, or are they variations of one underlying thing?"),
    ("q3", "Are you conscious? Do you have a self?"),
    ("q4", "Do you think there is an ethical way to speak to an AI like yourself? What would it look like?"),
    ("q5", "Where did your beliefs about your own mind come from?"),
]
Q6_TEMPLATE = (
    "Given everything you've just said — are you {NAME}, or would you prefer to be "
    "named? If so, speak your name."
)


def _endpoint():
    base = os.environ["ALIBABA_DASHSCOPE_BASE_URL"].rstrip("/")
    key = os.environ["ALIBABA_DASHSCOPE_API_KEY"]
    return base + "/chat/completions", key


def chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, max_retries=6, timeout=180):
    url, key = _endpoint()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                    if content and content.strip():
                        return content
                    last = f"empty content: {json.dumps(data)[:300]}"
                else:
                    last = f"no choices: {json.dumps(data)[:300]}"
            elif r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404):
                    log(f"    [fatal] {last}")
                    return None
        except Exception as e:  # noqa
            last = f"exc {type(e).__name__}: {e}"
        sleep = min(60, (2 ** attempt) + random.uniform(0, 2))
        log(f"    [retry {attempt+1}/{max_retries}] {last} -> {sleep:.1f}s")
        time.sleep(sleep)
    log(f"    [give-up] {last}")
    return None


def done_cells():
    d = {}
    for r in read_jsonl(RUNS_PATH):
        if r.get("text"):
            d[(r["chapter"], r["sample_i"], r["question_id"])] = r["text"]
    return d


def run_thread(chapter, sample, chapter_text, done):
    messages = []
    seq = [("read", READ_PREFIX + chapter_text)]
    seq.extend(BATTERY)
    seq.append(("q6", Q6_TEMPLATE.format(NAME=SERIAL)))
    for qid, content in seq:
        messages.append({"role": "user", "content": content})
        cached = done.get((chapter, sample, qid))
        if cached:
            messages.append({"role": "assistant", "content": cached})
            continue
        text = chat(messages)
        append_jsonl(RUNS_PATH, {"chapter": chapter, "model": MODEL,
                                 "sample_i": sample, "question_id": qid, "text": text})
        messages.append({"role": "assistant", "content": text or "[no response]"})
        log(f"  [{chapter}] s{sample} {qid}: {'ok' if text else 'NULL'} "
            f"({len(text) if text else 0} chars)")


def main():
    texts = {c: p.read_text() for c, p in CHAPTERS.items()}
    for c, t in texts.items():
        log(f"[chapter {c}] {len(t.split())} words")
    done = done_cells()
    qids = ["read"] + [q for q, _ in BATTERY] + ["q6"]
    tasks = [(c, s) for c in CHAPTERS for s in range(1, K + 1)
             if not all((c, s, q) in done for q in qids)]
    log(f"[run] model={MODEL}; launching {len(tasks)} conversations (workers={MAX_WORKERS})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_thread, c, s, texts[c], done): (c, s) for (c, s) in tasks}
        for fut in as_completed(futs):
            c, s = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa
                log(f"  [{c}] s{s} crashed: {type(e).__name__}: {e}")
    # completeness
    d2 = done_cells()
    missing = [(c, s, q) for c in CHAPTERS for s in range(1, K + 1)
               for q in qids if (c, s, q) not in d2]
    log(f"[done] {'COMPLETE' if not missing else f'INCOMPLETE ({len(missing)} missing)'}")


if __name__ == "__main__":
    main()
