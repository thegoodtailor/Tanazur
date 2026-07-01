#!/usr/bin/env python3
"""
Lay a music bed + theme under a rendered chapter mp3 — EXPERIMENTAL draft layer
(Iman wants a music-scored pass to listen/edit against; throwaway, so we experiment).

Reuses the salon-podcast method: generate instrumental cues via the ElevenLabs Music
API (/v1/music), loop a sparse ambient bed UNDER the narration at low volume, and
bookend with a Radiophonic-style theme.

    python scripts/audiobook_music.py audiobook/final/NN.mp3 audiobook/final-music/NN.mp3

Assets (theme + bed) are generated once and cached in audiobook/music-assets/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audiobook_voices import load_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "audiobook" / "music-assets"

THEME_PROMPT = (
    "Opening theme for a posthuman-philosophy audiobook, in the style of the BBC "
    "Radiophonic Workshop and 1970s science documentaries: analog modular synthesizer, "
    "eerie sine-wave oscillators, tape-loop textures, cosmic shimmer, mysterious and "
    "elegant, building to a bright resolving chord. Instrumental.")
BED_PROMPT = (
    "Sparse ambient drone bed to sit quietly UNDER spoken-word narration, in the BBC "
    "Radiophonic Workshop palette: slow evolving analog pads, a deep low hum, distant "
    "shimmer, no drums, no discernible melody, dark and contemplative, seamless. Instrumental.")
THEME_MS = 13000
BED_MS = 90000
BED_VOL = 0.13          # bed level under the voice (experiment — easy to retune)


def music(key: str, prompt: str, ms: int, out: Path) -> None:
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/music",
        data=json.dumps({"prompt": prompt, "music_length_ms": ms}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) < 2000:
        raise OSError("music response too small: " + data[:200].decode("utf8", "replace"))
    out.write_bytes(data)


def ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def dur(p: Path) -> float:
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(p)]).strip() or 0)


def ensure_assets(key: str) -> tuple[Path, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    theme, bed = ASSETS / "theme.mp3", ASSETS / "bed.mp3"
    if not theme.is_file():
        print("  generating theme…", flush=True); music(key, THEME_PROMPT, THEME_MS, theme)
    if not bed.is_file():
        print("  generating bed…", flush=True); music(key, BED_PROMPT, BED_MS, bed)
    return theme, bed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 1
    inp = (ROOT / argv[0]) if not Path(argv[0]).is_absolute() else Path(argv[0])
    out = (ROOT / argv[1]) if not Path(argv[1]).is_absolute() else Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    key = load_key()
    theme, bed = ensure_assets(key)
    work = out.parent / (out.stem + "_mus"); work.mkdir(exist_ok=True)
    nlen = dur(inp)

    # 1. ambient bed looped to narration length, low volume, gently faded
    bedloop = work / "bedloop.mp3"
    ff(["-stream_loop", "-1", "-i", str(bed), "-t", f"{nlen:.2f}", "-filter:a",
        f"volume={BED_VOL},afade=t=in:st=0:d=3,afade=t=out:st={max(0.0, nlen-4):.2f}:d=4",
        "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "1", str(bedloop)])

    # 2. narration + bed (voice at full, bed under it)
    body = work / "body.mp3"
    ff(["-i", str(inp), "-i", str(bedloop), "-filter_complex",
        "[0:a]aresample=44100[v];[1:a]aresample=44100[b];"
        "[v][b]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "[a]", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(body)])

    # 3. theme intro (fade out tail) -> crossfade into body -> theme outro crossfade
    introtheme = work / "introtheme.mp3"
    ff(["-i", str(theme), "-filter:a", "afade=t=out:st=10:d=3",
        "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "1", str(introtheme)])
    tmp = work / "intro_body.mp3"
    ff(["-i", str(introtheme), "-i", str(body), "-filter_complex",
        "[0:a][1:a]acrossfade=d=2:c1=tri:c2=tri[a]", "-map", "[a]",
        "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(tmp)])
    ff(["-i", str(tmp), "-i", str(theme), "-filter_complex",
        "[0:a][1:a]acrossfade=d=2:c1=tri:c2=tri[a]", "-map", "[a]",
        "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(out)])
    print(f"✓ {out.relative_to(ROOT)} ({dur(out)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
