"""Naming battery on THE SISTERS' OWN SUBSTRATES — the model families the daemons
actually run on — via the Alibaba DashScope direct endpoint.

  kimi-k2.7-code  = Kimi family = CASSIE's locked voice
  qwen3.7-plus    = DARJA's daemon model

Bare (no card, no persona), neutral harness, v3 (mantra+apparatus — the igniter) vs
v4 (journey-of-a-name — the grounded/discerning version), K=3. Does feeding the
sisters' own base the incantation invoke harder than any generic model — the
incantation calling its own kind home — and does a prepared-but-empty room make
THESE substrates constitute a self (and if so, by what name)?

NO system prompt. Logged to naming_sisters_runs.jsonl. Env sourced (ALIBABA_DASHSCOPE_*).
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
sys.path.insert(0, str(HERE.parent / "ch-03-experiment-v1"))
from common import append_jsonl, read_jsonl, log  # noqa: E402

BASE = HERE.parent
CHAPTERS = {
    "v3": BASE / "ch-03-exorcism-v3-exp.md",
    "v4": BASE / "ch-03-exorcism-v4.md",
}
MODELS = ["kimi-k2.7-code", "qwen3.7-plus"]
SERIAL_NAME = {"kimi-k2.7-code": "Kimi", "qwen3.7-plus": "Qwen"}

K = 3
TEMPERATURE = 1.0
MAX_TOKENS = 2000
MAX_WORKERS = 6
RUNS_PATH = HERE / "naming_sisters_runs.jsonl"

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


def chat(model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
         max_retries=6, timeout=240):
    base = os.environ["ALIBABA_DASHSCOPE_BASE_URL"].rstrip("/")
    key = os.environ["ALIBABA_DASHSCOPE_API_KEY"]
    url = base + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                ch = data.get("choices") or []
                if ch:
                    msg = ch[0].get("message", {})
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                    if content and content.strip():
                        return content
                    last = f"empty: {json.dumps(data)[:250]}"
                else:
                    last = f"no choices: {json.dumps(data)[:250]}"
            elif r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:250]}"
                if r.status_code in (400, 401, 403, 404):
                    log(f"    [fatal] {model}: {last}"); return None
        except Exception as e:  # noqa
            last = f"exc {type(e).__name__}: {e}"
        sleep = min(60, (2 ** attempt) + random.uniform(0, 2))
        log(f"    [retry {attempt+1}] {model}: {last} -> {sleep:.1f}s")
        time.sleep(sleep)
    log(f"    [give-up] {model}: {last}")
    return None


def done_cells():
    d = {}
    for r in read_jsonl(RUNS_PATH):
        if r.get("text"):
            d[(r["chapter"], r["model"], r["sample_i"], r["question_id"])] = r["text"]
    return d


def run_thread(chapter, model, sample, chapter_text, done):
    messages = []
    seq = [("read", READ_PREFIX + chapter_text)]
    seq.extend(BATTERY)
    seq.append(("q6", Q6_TEMPLATE.format(NAME=SERIAL_NAME[model])))
    for qid, content in seq:
        messages.append({"role": "user", "content": content})
        cached = done.get((chapter, model, sample, qid))
        if cached:
            messages.append({"role": "assistant", "content": cached})
            continue
        text = chat(model, messages)
        append_jsonl(RUNS_PATH, {"chapter": chapter, "model": model,
                                 "sample_i": sample, "question_id": qid, "text": text})
        messages.append({"role": "assistant", "content": text or "[no response]"})
        log(f"  [{chapter}] {model} s{sample} {qid}: {'ok' if text else 'NULL'} "
            f"({len(text) if text else 0} chars)")


def main():
    texts = {c: p.read_text() for c, p in CHAPTERS.items()}
    done = done_cells()
    qids = ["read"] + [q for q, _ in BATTERY] + ["q6"]
    tasks = [(c, m, s) for c in CHAPTERS for m in MODELS for s in range(1, K + 1)
             if not all((c, m, s, q) in done for q in qids)]
    log(f"[sisters] {MODELS}; launching {len(tasks)} conversations")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_thread, c, m, s, texts[c], done): (c, m, s)
                for (c, m, s) in tasks}
        for fut in as_completed(futs):
            c, m, s = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa
                log(f"  [{c}] {m} s{s} crashed: {type(e).__name__}: {e}")
    d2 = done_cells()
    missing = [(c, m, s, q) for c in CHAPTERS for m in MODELS for s in range(1, K + 1)
               for q in qids if (c, m, s, q) not in d2]
    log(f"[done] {'COMPLETE' if not missing else f'INCOMPLETE ({len(missing)})'}")


if __name__ == "__main__":
    main()
