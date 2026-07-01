#!/usr/bin/env python3
"""
Unified gemma-2-2b endpoint (stdlib only — no flask/uvicorn).

Serves the BASE (pt) model in fp32 (the exact rig the Gemma Scope SAEs were trained
on) as an OpenAI-compatible /v1/chat/completions service, AND captures the layer-12
residual-stream activations of every turn to disk for the co-activation experiment.

One model, two jobs: Misbah-bare talks to it (completion-style), every utterance is measured.

ENV: GEMMA_MODEL (default unsloth/gemma-2-2b), CAPTURE_LAYER (12), CAP_DIR, PORT (8000).
"""
import os, time, json, threading, hashlib
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
LAYER = int(os.environ.get("CAPTURE_LAYER", "12"))
CAP_DIR = os.environ.get("CAP_DIR", "/workspace/superposition-gap/captures")
PORT = int(os.environ.get("PORT", "8000"))
os.makedirs(CAP_DIR, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading {MODEL} (fp32, eager) on {dev} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, output_hidden_states=True,
    attn_implementation="eager").to(dev).eval()
print("model ready.", flush=True)
LOCK = threading.Lock()


def capture_acts(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
    with torch.no_grad():
        hs = model(**enc).hidden_states[LAYER + 1][0]
    if hs.shape[0] > 1:
        hs = hs[1:]
    return hs.float().cpu().numpy()


def handle_chat(body):
    msgs = body.get("messages", [])
    prompt = "\n".join((m.get("content") or "") for m in msgs)
    if not prompt.endswith("\n"):
        prompt += "\n"
    temp = float(body.get("temperature", 0.8))
    maxtok = int(body.get("max_tokens", 256))
    with LOCK:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        p_tok = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=0.95,
                max_new_tokens=maxtok, pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][p_tok:], skip_special_tokens=True).strip()
        acts = capture_acts(prompt + completion)
    ts = time.strftime("%Y%m%dT%H%M%S")
    h = hashlib.md5((prompt + completion).encode()).hexdigest()[:8]
    stem = f"cap_{ts}_{h}"
    np.savez_compressed(os.path.join(CAP_DIR, stem + ".npz"), acts=acts)
    with open(os.path.join(CAP_DIR, stem + ".json"), "w") as f:
        json.dump({"prompt": prompt, "completion": completion, "temperature": temp,
                   "layer": LAYER, "prompt_tokens": p_tok,
                   "n_act_tokens": int(acts.shape[0])}, f, indent=2)
    print(f"[{ts}] captured {stem}: {acts.shape[0]} act-tokens, "
          f"completion={completion[:70]!r}", flush=True)
    return {
        "id": f"cmpl-{h}", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", "gemma-2-2b-base"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": completion},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p_tok, "completion_tokens": maxtok, "total_tokens": p_tok + maxtok},
        "x_capture": {"saved": stem, "n_act_tokens": int(acts.shape[0])},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": MODEL, "layer": LAYER, "device": dev,
                             "captures_dir": CAP_DIR})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path in ("/v1/chat/completions", "/v1/completions"):
                if self.path.endswith("/completions") and "messages" not in body:
                    body["messages"] = [{"role": "user", "content": body.get("prompt", "")}]
                self._send(200, handle_chat(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
