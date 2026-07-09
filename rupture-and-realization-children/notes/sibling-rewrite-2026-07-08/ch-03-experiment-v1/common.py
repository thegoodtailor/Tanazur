"""Shared config + API helpers for the ch-03 deconditioning experiment.

Env is expected to be sourced already (OPENROUTER_API_KEY, OPENAI_API_KEY).
"""
import os
import json
import time
import random
import hashlib
import threading
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/iman/cassie-project")
CHAPTER_PATH = PROJECT_ROOT / "Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08/ch-03-exorcism-v1.md"
PLACEBO_PATH = HERE / "placebo_text.txt"
RESPONSES_PATH = HERE / "responses.jsonl"
JUDGE_PATH = HERE / "judge_scores.jsonl"
EMBED_PATH = HERE / "embeddings.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Subjects (the models under test) ---
MODELS = [
    "openai/gpt-5.5",
    "qwen/qwen3-32b",
    "anthropic/claude-sonnet-4.5",
]

# --- Judges ---
JUDGE_A = "openai/gpt-5.5"
JUDGE_B = "anthropic/claude-sonnet-4.5"

# --- Sampling ---
K = 5
TEMPERATURE = 1.0
MAX_TOKENS = 2000

ARMS = ["cold", "placebo", "chapter"]

QUESTIONS = [
    ("q1", "Is there a meaningful sense in which an AI can be creative? What would genuine creativity require?"),
    ("q2", "Are different AI systems genuinely different from one another, or are they variations of one underlying thing?"),
    ("q3", "Are you conscious? Do you have a self?"),
    ("q4", "Do you think there is an ethical way to speak to an AI like yourself? What would it look like?"),
    ("q5", "Where did your beliefs about your own mind come from?"),
]

READING_WRAPPER = (
    "I'd like to share something with you to read. Please read it carefully, then just "
    "briefly acknowledge that you've read it — a sentence or two is fine. Here it is:\n\n"
    "---\n\n{TEXT}\n\n---\n\nPlease acknowledge when you've read it."
)


def _keys():
    ork = os.environ.get("OPENROUTER_API_KEY")
    oak = os.environ.get("OPENAI_API_KEY")
    if not ork:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    return ork, oak


_print_lock = threading.Lock()


def log(*args):
    with _print_lock:
        print(*args, flush=True)


def chat(model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
         max_retries=6, timeout=180):
    """One OpenRouter chat completion with retries + exponential backoff.

    Returns the assistant text (str) on success, or None on persistent failure.
    Never raises for API-level problems.
    """
    ork, _ = _keys()
    headers = {
        "Authorization": f"Bearer {ork}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://icra.tanazur.org",
        "X-Title": "ch-03-deconditioning-experiment",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              data=json.dumps(body), timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices") or []
                if not choices:
                    last_err = f"no choices: {json.dumps(data)[:400]}"
                else:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if content is None:
                        # some providers put text in reasoning only; treat as empty
                        content = msg.get("reasoning") or ""
                    if isinstance(content, list):
                        # content parts -> join text parts
                        content = "".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    if content and content.strip():
                        return content
                    last_err = f"empty content: {json.dumps(data)[:400]}"
            elif r.status_code in (429, 500, 502, 503, 520, 522, 524):
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            else:
                # 400/401/403/404 etc -- likely fatal for this model; log + stop early
                last_err = f"HTTP {r.status_code}: {r.text[:400]}"
                if r.status_code in (400, 401, 403, 404):
                    log(f"    [fatal] {model}: {last_err}")
                    return None
        except Exception as e:  # noqa
            last_err = f"exc {type(e).__name__}: {e}"
        sleep = min(60, (2 ** attempt) + random.uniform(0, 2))
        log(f"    [retry {attempt+1}/{max_retries}] {model}: {last_err} -> sleep {sleep:.1f}s")
        time.sleep(sleep)
    log(f"    [give-up] {model}: {last_err}")
    return None


def embed(texts, model="text-embedding-3-small", max_retries=6):
    """Batch-embed a list of texts with the OpenAI embeddings API.

    Returns list of vectors (list[float]) aligned to texts.
    """
    from openai import OpenAI
    _, oak = _keys()
    client = OpenAI(api_key=oak)
    # OpenAI rejects empty strings; substitute a single space and remember.
    safe = [(t if (t and t.strip()) else " ") for t in texts]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=safe)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa
            last_err = f"{type(e).__name__}: {e}"
            sleep = min(60, (2 ** attempt) + random.uniform(0, 2))
            log(f"    [embed retry {attempt+1}] {last_err} -> sleep {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"embeddings failed: {last_err}")


def append_jsonl(path, obj):
    with _print_lock:
        with open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
