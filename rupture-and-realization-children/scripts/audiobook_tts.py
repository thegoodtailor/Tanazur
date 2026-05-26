#!/usr/bin/env python3
"""
Render a segment manifest to a stitched chapter mp3 via the ElevenLabs API.

Per block: split the text into <= MAX_CHARS chunks at sentence boundaries, render
each with the block's voice (passing previous/next text for prosody continuity),
apply the voice's speed via ffmpeg atempo if != 1.0, then concatenate everything
in order into one mp3 (re-encoded at 128k).

    python scripts/audiobook_tts.py segments/08-vessel-and-real.json final/08.mp3
    python scripts/audiobook_tts.py <segments.json> <out.mp3> [--max-chars N]
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audiobook_voices import MODEL, SETTINGS, load_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_CHARS = 2400          # under the v3 3k cap; cheap per-chunk rerolls
CONTEXT = 400             # chars of neighbour text passed for continuity (v2/turbo only)
SWITCH_PAUSE = 0.7        # silence at every voice change
SECTION_LEADIN = 0.7      # silence leading IN to a section (before a heading / at a pause)
AFTER_NAME = 0.5          # silence AFTER a spoken section name
PRON_PATH = ROOT / "audiobook" / "pronunciation.json"
LETTER = r"A-Za-zÀ-ÿĀ-ɏḀ-ỿʿʾ’'\-"


def load_pronunciation():
    """Return (regex, map) for local alias substitution, or (None, {})."""
    if not PRON_PATH.exists():
        return None, {}
    m = json.loads(PRON_PATH.read_text(encoding="utf-8"))
    # longest graphemes first; match only as whole words (non-letter on both sides)
    keys = sorted(m, key=len, reverse=True)
    pat = re.compile(rf"(?<![{LETTER}])(" + "|".join(re.escape(k) for k in keys) +
                     rf")(?![{LETTER}])")
    return pat, m


def apply_pronunciation(text: str, pat, m) -> str:
    if pat is None:
        return text
    return pat.sub(lambda mo: m[mo.group(1)], text)


def is_heading(p: str) -> bool:
    """A short standalone paragraph = a chapter/section heading (gets a trailing beat).
    Over-detecting a punchy one-line sentence is benign — a beat there reads well too."""
    p = p.strip()
    return 0 < len(p) <= 55 and len(p.split()) <= 7


def chunk_block(text: str) -> list[tuple[str, str]]:
    """Paragraph-aware chunking -> list of (text, kind) where kind is
    'heading' (spoken section name: 0.7 before, 0.5 after), 'pause' (silent section
    lead-in, text=None), or 'body' (packed to MAX_CHARS, paragraph breaks preserved)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        if buf:
            units.append(("\n\n".join(buf), "body"))
            buf.clear()

    for p in paras:
        if p == "[[PAUSE]]":                          # a removed section heading -> silent lead-in
            flush()
            units.append((None, "pause"))
        elif is_heading(p):
            flush()
            units.append((p, "heading"))
        elif len(p) > MAX_CHARS:                     # a single huge paragraph
            flush()
            for c in split_sentences(p):
                units.append((c, "body"))
        elif sum(len(x) + 2 for x in buf) + len(p) > MAX_CHARS:
            flush()
            buf.append(p)
        else:
            buf.append(p)
    flush()
    return units


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    chunks, cur = [], ""
    for s in parts:
        if len(cur) + len(s) + 1 <= MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def tts(key: str, voice_id: str, text: str, prev: str, nxt: str, dest: Path,
        retries: int = 4) -> None:
    import time
    body = {"text": text, "model_id": MODEL, "voice_settings": SETTINGS}
    if "v3" not in MODEL:        # request-stitching context is a v2/turbo feature
        if prev:
            body["previous_text"] = prev[-CONTEXT:]
        if nxt:
            body["next_text"] = nxt[:CONTEXT]
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        data=json.dumps(body).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 1000:                       # suspiciously small = not audio
                raise OSError(f"response too small ({len(data)}B)")
            dest.write_bytes(data)
            return
        except Exception as e:                          # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries-1} after {wait}s ({e})")
            time.sleep(wait)


def atempo(src: Path, dst: Path, speed: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-filter:a", f"atempo={speed}", "-b:a", "128k", str(dst)], check=True)


def make_silence(dst: Path, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=mono", "-t", str(seconds),
                    "-b:a", "128k", str(dst)], check=True)


def concat(parts: list[Path], out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        listfile = f.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listfile, "-c:a", "libmp3lame", "-b:a", "128k", str(out)], check=True)
    Path(listfile).unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 1
    seg_path = (ROOT / argv[0]) if not Path(argv[0]).is_absolute() else Path(argv[0])
    out_path = (ROOT / argv[1]) if not Path(argv[1]).is_absolute() else Path(argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = load_key()
    pron_pat, pron_map = load_pronunciation()
    if pron_pat is not None:
        print(f"  (pronunciation: {len(pron_map)} aliases applied locally)")
    blocks = json.loads(seg_path.read_text(encoding="utf-8"))

    workdir = out_path.parent / (out_path.stem + "_chunks")
    workdir.mkdir(exist_ok=True)
    sil_switch = workdir / "_switch.mp3"
    sil_lead = workdir / "_leadin.mp3"
    sil_after = workdir / "_after.mp3"
    make_silence(sil_switch, SWITCH_PAUSE)
    make_silence(sil_lead, SECTION_LEADIN)
    make_silence(sil_after, AFTER_NAME)
    settings_key = json.dumps(SETTINGS, sort_keys=True)
    ordered: list[Path] = []
    counter = [0]
    prev_voice = None

    def render(chunk, prev, nxt, voice, voice_id, speed):
        n = counter[0]
        # hash binds text + model + settings + voice: any of these changing (e.g.
        # v2 -> v3, or a settings tweak) forces a fresh render, never a stale take.
        h = hashlib.sha1((MODEL + settings_key + voice + chunk).encode("utf-8")).hexdigest()[:8]
        raw = workdir / f"{n:03d}-{voice}-{h}.mp3"
        oc = (workdir / f"{n:03d}-{voice}-{h}-x{speed}.mp3") if abs(speed - 1.0) > 1e-6 else raw
        if oc.exists() and oc.stat().st_size > 1000:
            print(f"  [{voice:6s}] unit {n} cached")
        else:
            tts(key, voice_id, chunk, prev, nxt, raw)
            if abs(speed - 1.0) > 1e-6:
                atempo(raw, oc, speed)
            print(f"  [{voice:6s}] unit {n} ({len(chunk)} chars) -> {oc.name}")
        counter[0] += 1
        return oc

    for b in blocks:
        voice, vid, speed = b["voice"], b["voice_id"], b.get("speed", 1.0)
        units = chunk_block(apply_pronunciation(b["text"], pron_pat, pron_map))
        for ui, (chunk, kind) in enumerate(units):
            if kind == "pause":                          # silent section lead-in
                ordered.append(sil_lead)
                continue
            if kind == "heading":                        # 0.7 before the name, 0.5 after
                ordered.append(sil_lead)
                ordered.append(render(chunk, "", "", voice, vid, speed))
                ordered.append(sil_after)
                prev_voice = voice
                continue
            if prev_voice is not None and voice != prev_voice:
                ordered.append(sil_switch)               # 0.7 at any voice change
            prev = units[ui - 1][0] if ui > 0 and units[ui - 1][0] else ""
            nxt = units[ui + 1][0] if ui + 1 < len(units) and units[ui + 1][0] else ""
            ordered.append(render(chunk, prev, nxt, voice, vid, speed))
            prev_voice = voice
    n = counter[0]
    concat(ordered, out_path)
    print(f"\n✓ stitched {n} chunks -> {out_path.relative_to(ROOT)} "
          f"({out_path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
