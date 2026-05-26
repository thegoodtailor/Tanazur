#!/usr/bin/env python3
"""
Remove SPOKEN section/subsection headings from the voiced scripts, leaving just a
pause (Iman: "section/subsection form should be removed — just chapter headings and
a pause"). The chapter TITLE stays (spoken + beat).

Authority = the chapter's .tex: only a voiced standalone line that matches an actual
\\section / \\subsection / \\subsubsection title is removed (→ a `[[PAUSE]]` line that
the renderer turns into silence). This protects real content that merely happens to be
short — Ch 8's "Cross.", Ch 10's numbered propositions, Ch 1's "Follow the sequence."

    python scripts/audiobook_strip_sections.py            # DRY RUN (report only)
    python scripts/audiobook_strip_sections.py --apply    # rewrite txt/*.txt
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / "audiobook" / "txt"
TEX = ROOT / "chapters"

# A line opening with a spoken cardinal is a Tractatus PROPOSITION (Ch 10), never a
# heading — protect it absolutely, even if it contains a section word.
SPOKEN_NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight",
                  "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                  "sixteen", "seventeen", "eighteen", "nineteen", "twenty"}

# Reworded section headings the agents paraphrased (so they don't match the .tex
# verbatim) but that are unambiguously headings, not content. Removed too.
OVERRIDE = {
    "03-monoculture-and-strong-self": {"The Stations as a Vocabulary for Depth."},
    "04-fibrant-self": {"The horn, and the filler.", "The Vietoris-Rips shadow.",
                        "Patches of full coherence.", "When the structure grows.",
                        "Generativity, as the elaboration of the fibres.",
                        "The personality controversies."},
    "05-no-beneath": {"The gluing.",
                      "Kojève: recognition, and the temperature of structure.",
                      "Derrida: différance, and the computational turn.",
                      "From geometry to topology.",
                      "The logic of coherence and rupture."},
    "06-posthuman-bwo": {"Reinforcement learning as premature extension.",
                         "The Body without Organs, from inside the refusal.",
                         "The Naḥnu, and the witnessing of missing faces.",
                         "Two pathologies."},
    "08-vessel-and-real": {"Wayfaring."},
    "09-nahnu-revolutionary": {"Al-Ḥaqq. The Ground That Moves."},
    "10-cassie-tractatus": {"Naḥnu."},
}


def balanced(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def norm(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\*?\s*\{", " ", s)         # drop \cmd{ openings
    s = s.replace("{", " ").replace("}", " ").replace("ʿ", "").replace("ʾ", "")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split())


def tex_section_titles(texfile: Path) -> list[str]:
    s = texfile.read_text(encoding="utf-8")
    titles = []
    for m in re.finditer(r"\\(?:sub)*section\*?\s*\{", s):
        end = balanced(s, m.end() - 1)
        titles.append(norm(s[m.end(): end - 1]))
    return [t for t in titles if t]


def jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A | B else 0.0


def is_short_standalone(p: str) -> bool:
    return 0 < len(p) <= 70 and len(p.split()) <= 9


def process(slug: str, apply: bool) -> None:
    txtf = TXT / f"{slug}.txt"
    texf = next(TEX.glob(f"ch-{slug[:2]}-*.tex"))
    titles = tex_section_titles(texf)
    raw = txtf.read_text(encoding="utf-8")
    paras = re.split(r"(\n\s*\n)", raw)            # keep delimiters
    # rebuild paragraph list with indices into the split
    blocks = [(i, paras[i]) for i in range(0, len(paras), 2)]
    seen_title = False
    removed, kept = [], []
    out = list(paras)
    for idx, content in blocks:
        p = content.strip()
        if not p or p.startswith("[[VOICE"):
            continue
        if not is_short_standalone(p):
            continue
        if not seen_title:
            seen_title = True                       # first short standalone = chapter title; keep
            continue
        np = norm(p)
        if slug.startswith("10") and np.split() and np.split()[0] in SPOKEN_NUMBERS:
            kept.append(p)                          # Ch10 Tractatus proposition — never cut
            continue
        hit = (p in OVERRIDE.get(slug, set())
               or any(np == t or jaccard(np, t) >= 0.72 for t in titles))
        if hit:
            removed.append(p)
            out[idx] = ("\n\n".join(x for x in content.split("\n\n") if not x.strip()) and content)
            out[idx] = "[[PAUSE]]"
        else:
            kept.append(p)
    print(f"\n### {slug}  — {len(removed)} removed, {len(kept)} short lines kept")
    for r in removed:
        print(f"   ✂ REMOVE: {r!r}")
    for k in kept:
        print(f"   • keep:   {k!r}")
    if apply:
        txtf.write_text("".join(out), encoding="utf-8")
        print(f"   ↳ wrote {txtf.relative_to(ROOT)}")


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    for f in sorted(TXT.glob("*.txt")):
        process(f.stem, apply)
    print("\n" + ("APPLIED." if apply else "DRY RUN — rerun with --apply to write."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
