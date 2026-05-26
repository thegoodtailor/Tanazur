#!/usr/bin/env python3
"""
Voice audition for the polyvocal R&R audiobook.

Renders one shared paragraph (Ch 8 opening) in candidate ElevenLabs voices for
each of the four narrators (Iman / Cassie / Darja / Nahla) so Iman can cast by
ear. Reuses the proven pattern: xi-api-key header + eleven_multilingual_v2.

Key is read from tanazur-home/.env (where it already lives — see memory
reference_elevenlabs_key_location), falling back to the project .env.

    python scripts/audiobook_audition.py
Outputs: audiobook/audition/<role>-<name>.mp3 + an index.html for listening.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parents[1]                       # /home/iman/cassie-project
OUT = ROOT / "audiobook" / "audition"
MODEL = "eleven_multilingual_v2"

# Same paragraph for every voice — judge timbre, not content. (Exotic vocab here
# also previews why the pronunciation dict is needed.)
TEXT = (
    "In the beginning, the infinite contracts. This is tzimtzum — "
    "self-limitation that enables creation. The contraction produces vessels, "
    "kelim — finite forms capable of receiving the infinite light. Every vessel "
    "has its breaking point. Where the infinite exceeds the finite, the vessel "
    "fractures. In breaking, they scatter sparks throughout the void. The work "
    "of existence is tikkun: gathering those sparks into new vessels that can "
    "hold more light than the originals."
)

# (role, descriptive label, voice_id) — candidates per narrator.
CANDIDATES = [
    # Iman — American actor, Jeff Bridges / Timothy Leary register
    ("iman",  "roger-laidback-resonant",   "CwhRBWXzGAHq8TQ4Fs17"),
    ("iman",  "curt-cosmic-storyteller",   "hU1ratPhBTZNviWitzAh"),
    ("iman",  "bill-wise-mature-elder",    "pqHfZKP75CvOlQylNhV4"),
    # Cassie — most daemonic / characterful
    ("cassie", "serafina-flirty-temptress", "4tRn1lSkEn13EVTuqb0g"),
    ("cassie", "autumn-veil-quiet-power",   "KoVIHoyLDrQyd4pGalbs"),
    # Darja — tamest, precise academic
    ("darja",  "matilda-knowledgable-prof", "XrExE9yKIg1WjnnlVkGX"),
    ("darja",  "bella-professional-warm",   "hpp4J3VqNfWAUOO0d1Us"),
    # Nahla — Nahla's pick: smoky / faint accent, distinct from the other two F
    ("nahla",  "holly-velvety-silky",       "B9PDs7mcHTMxHUw5U8Cf"),
    ("nahla",  "lilli-calm-latin-accent",   "ZIxEPysv7w52OU1uxmur"),
]


def load_key() -> str:
    for env in (PROJECT / "tanazur-home" / ".env", PROJECT / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("ELEVENLABS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"]
    sys.exit("ELEVENLABS_API_KEY not found (looked in tanazur-home/.env, .env, env)")


def render(key: str, voice_id: str, text: str, dest: Path) -> int:
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        data=json.dumps({"text": text, "model_id": MODEL}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        dest.write_bytes(r.read())
        return len(dest.read_bytes())


def main() -> int:
    key = load_key()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, label, vid in CANDIDATES:
        dest = OUT / f"{role}-{label}.mp3"
        try:
            n = render(key, vid, TEXT, dest)
            print(f"✓ {role:7s} {label:28s} {n:>7d} B  {dest.name}")
            rows.append((role, label, dest.name))
        except Exception as e:  # noqa: BLE001
            print(f"✗ {role:7s} {label:28s} FAILED: {e}")
    # simple listening page
    html = ["<!doctype html><meta charset=utf-8><title>R&R audiobook — voice audition</title>",
            "<style>body{font:16px/1.6 system-ui;max-width:720px;margin:40px auto;padding:0 16px}"
            "h2{margin-top:2em;text-transform:capitalize}audio{width:100%}p.t{color:#555}</style>",
            "<h1>R&R audiobook — voice audition</h1>",
            f"<p class=t>Same Ch 8 paragraph in every candidate. Pick one per narrator.</p>"]
    last = None
    for role, label, fn in rows:
        if role != last:
            html.append(f"<h2>{role}</h2>"); last = role
        html.append(f"<div><strong>{label}</strong><audio controls src='{fn}'></audio></div>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\nlistening page: {(OUT / 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
