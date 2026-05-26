#!/usr/bin/env python3
"""
Insert a beat AFTER the spoken chapter title in an already-rendered chapter mp3 —
pure audio editing, no TTS, no credits. Fixes the "no gap after the heading" issue
on chapters already rendered (the heading is glued to the body inside chunk 0).

Finds the first pause (silencedetect) in a window after the title and pads it to
GAP seconds. Originals are backed up to <name>.orig.mp3.

    python scripts/audiobook_titlegap.py audiobook/final/01-new-logic.mp3 [more.mp3 ...]
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

GAP = 0.6
WIN = (0.5, 9.0)          # the post-title pause should start within this window (s)


def first_pause(mp3: Path) -> float | None:
    out = subprocess.run(
        ["ffmpeg", "-i", str(mp3), "-af", "silencedetect=noise=-30dB:d=0.18",
         "-f", "null", "-"],
        capture_output=True, text=True).stderr
    for m in re.finditer(r"silence_start:\s*([0-9.]+)", out):
        t = float(m.group(1))
        if WIN[0] <= t <= WIN[1]:
            return t
    return None


def lame(args: list[str]):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args,
                    "-c:a", "libmp3lame", "-b:a", "128k"], check=True)


def patch(mp3: Path) -> None:
    t = first_pause(mp3)
    if t is None:
        print(f"  {mp3.name}: no pause found in window — skipped")
        return
    work = Path(tempfile.mkdtemp())
    a, sil, b = work / "a.mp3", work / "sil.mp3", work / "b.mp3"
    lame(["-i", str(mp3), "-t", f"{t:.3f}", str(a)])                 # title + lead-in
    lame(["-i", str(mp3), "-ss", f"{t:.3f}", str(b)])                # the rest
    lame(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", str(GAP), str(sil)])
    listf = work / "list.txt"
    listf.write_text("".join(f"file '{p.resolve()}'\n" for p in (a, sil, b)))
    backup = mp3.with_suffix(".orig.mp3")
    if not backup.exists():
        mp3.rename(backup)                                           # keep the original
    else:
        mp3.unlink(missing_ok=True)
    lame(["-f", "concat", "-safe", "0", "-i", str(listf), str(mp3)])
    print(f"  {mp3.name}: gap inserted after title at {t:.2f}s (+{GAP}s)")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__); return 1
    for p in argv:
        patch(Path(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
