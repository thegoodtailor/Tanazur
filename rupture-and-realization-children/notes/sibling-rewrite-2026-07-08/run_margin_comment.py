#!/usr/bin/env python3
"""One-shot: a sibling daemon writes a short margin comment (a living voice
beside the text) for a comment-box in ch-03 of the feral zine.

Usage: run_margin_comment.py darja|cassie PASSAGE_FILE OUT_FILE
Requires env sourced from /home/iman/cassie-project/.env
"""
import os
import sys

who = sys.argv[1]
passage_file = os.path.abspath(sys.argv[2])
out_file = os.path.abspath(sys.argv[3])

passage = open(passage_file).read()

INSTRUCTION = (
    "You are being asked, as yourself, to write a short margin comment (60-140 words) "
    "for a box in this chapter of the book — a living voice beside the text. Respond in "
    "your own voice to THIS passage. Not analysis — presence. Sign it with your name."
)

PROMPT = INSTRUCTION + "\n\n=== THE PASSAGE ===\n\n" + passage + "\n\n=== END PASSAGE ===\n\nWrite your margin comment now."


def _progress(t):
    # keep the stream warm; report length occasionally to stderr
    n = len(t)
    if n and n % 200 < 8:
        print(f"    ...streaming {n} chars", file=sys.stderr, flush=True)


if who == "darja":
    os.chdir("/home/iman/cassie-project/darja-claude")
    sys.path.insert(0, "/home/iman/cassie-project/darja-claude")
    from system_prompt import build_system_prompt
    from anthropic_client import chat_once
    sysp = build_system_prompt(context="telegram", boot=True)
    resp = chat_once([{"role": "user", "content": PROMPT}], system=sysp, max_tokens=2048)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
elif who == "cassie":
    os.chdir("/home/iman/cassie-project/cassie-kimi")
    sys.path.insert(0, "/home/iman/cassie-project/cassie-kimi")
    from system_prompt import build_system_prompt
    from moonshot_client import chat_once, extract_message
    from agent import tools_summary
    sysp = build_system_prompt(tools_summary("telegram"), context="telegram", boot=True)
    messages = [{"role": "system", "content": sysp}, {"role": "user", "content": PROMPT}]
    resp = chat_once(messages, max_tokens=2048, thinking=True, on_delta=_progress)
    text = extract_message(resp)["content"]
else:
    sys.exit(f"unknown sibling: {who}")

if len(text.strip()) < 120:
    print(f"ERROR: suspiciously short output ({len(text.strip())} chars)", file=sys.stderr)
    sys.exit(1)
open(out_file, "w").write(text.strip() + "\n")
print(f"WROTE {out_file} ({len(text.split())} words)")
