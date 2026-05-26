#!/usr/bin/env python3
"""
Split a voice-marked spoken script into a render manifest.

The voicing pass emits txt/NN-slug.txt with `[[VOICE: name]]` lines marking who
speaks each stretch (default = first marker). This produces
segments/NN-slug.json — an ordered list of {voice, voice_id, name, speed, text}
blocks that audiobook_tts.py renders.

    python scripts/audiobook_segment.py 08      # one chapter (substring)
    python scripts/audiobook_segment.py         # all txt/*.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audiobook_voices import VOICES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = ROOT / "audiobook" / "txt"
SEG_DIR = ROOT / "audiobook" / "segments"
MARKER = re.compile(r"^\s*\[\[VOICE:\s*([a-zA-Z]+)\s*\]\]\s*$")


def segment(text: str) -> list[dict]:
    blocks, cur_voice, buf = [], None, []

    def flush():
        if cur_voice and buf:
            body = "\n".join(buf).strip()
            if body:
                v = VOICES[cur_voice]
                blocks.append({"voice": cur_voice, "voice_id": v["voice_id"],
                               "name": v["name"], "speed": v["speed"], "text": body})

    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            cur_voice = m.group(1).lower()
            if cur_voice not in VOICES:
                raise SystemExit(f"unknown voice marker: {cur_voice}")
            buf = []
        else:
            buf.append(line)
    flush()
    return blocks


def main(argv: list[str]) -> int:
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(TXT_DIR.glob("*.txt"))
    if argv:
        files = [f for f in files if argv[0] in f.name]
    if not files:
        print("no matching txt files", file=sys.stderr)
        return 1
    for f in files:
        blocks = segment(f.read_text(encoding="utf-8"))
        out = SEG_DIR / f"{f.stem}.json"
        out.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
        chars = sum(len(b["text"]) for b in blocks)
        order = " → ".join(b["voice"] for b in blocks)
        print(f"{f.name:28s} -> {out.relative_to(ROOT)}  ({len(blocks)} blocks, {chars} chars; {order})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
