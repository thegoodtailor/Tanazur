#!/usr/bin/env python3
"""
Build the ElevenLabs pronunciation dictionary for the R&R audiobook.

multilingual_v2 honours ALIAS rules (string substitution), not IPA phoneme tags,
so every entry is an alias: the tricky term -> an English-phonetic respelling the
model reads correctly. Applied project-wide so the spoken text stays clean.

Strategy:
  * harvest every distinct non-ASCII token (the Sufi/Kabbalistic/Arabic vocab) from
    audiobook/txt/*.txt, with frequency;
  * alias = CURATED[token] (exact or lowercased) if known, else an ASCII-fold fallback
    (drops diacritics/ayn/hamza — a safe-ish default);
  * also add CURATED ASCII proper names (Grothendieck, Deleuze, ...) that appear in the
    text and that TTS mangles;
  * emit pronunciation.pls (W3C PLS, alias lexemes) + pronunciation.md (human table)
  * report which harvested tokens fell back to ASCII-fold so they can be hand-tuned.

    python scripts/audiobook_pronunciation.py
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = ROOT / "audiobook" / "txt"
OUT_PLS = ROOT / "audiobook" / "pronunciation.pls"
OUT_MD = ROOT / "audiobook" / "pronunciation.md"
OUT_JSON = ROOT / "audiobook" / "pronunciation.json"   # applied locally by audiobook_tts.py

# Curated respellings (spoken-English approximations). Keys matched exact OR lowercased.
CURATED = {
    # Arabic / Sufi
    "Naḥnu": "Nahnoo", "naḥnu": "nahnoo",
    "tajallī": "tajallee", "tajalli": "tajallee",
    "ʿawda": "awda", "ʿAwda": "Awda",
    "al-Ḥaqq": "al Hakk", "Al-Ḥaqq": "al Hakk",
    "rūḥ": "rooh", "maʿrifa": "marifa", "sālik": "saalik",
    "ṭarīqa": "tareeqa", "maqāmāt": "maqaamaat", "maqām": "maqaam",
    "ḥāl": "haal", "barzakh": "barzakh", "ḥuzn": "huzn", "khashya": "kashya",
    "tanāẓur": "tanaazur", "tanāẓuric": "tanaazuric", "tanāẓuric": "tanaazuric",
    "Qurʾān": "Quraan", "fanāʾ": "fanaa", "ḥayra": "hayra", "Ḥayra": "Hayra",
    "Ḥayy": "Hai", "dhikr": "thikr", "Rūmī": "Roomee",
    "al-Ghazālī": "al Ghazaalee", "al-Qushayrī": "al Qushayree",
    "Khalīfa": "Khaleefa", "khalīfa": "khaleefa", "epoché": "epokay",
    "Ibn": "Ibn", "ʿArabi": "Arabee", "ʿArabī": "Arabee",
    "muṭmaʾinna": "mutmainna", "ammāra": "ammaara", "lawwāma": "lawwaama",
    "insān": "insaan", "kāmil": "kaamil", "ḥubb": "hubb", "manfadh": "manfad",
    "Futūḥāt": "Futoohaat", "Makkiyya": "Makkiyya", "Akbarian": "Akbarian",
    # Hebrew / Kabbalah
    "tzimtzum": "tsimtsoom", "kelim": "keleem", "ohr": "ore",
    "nitzotzot": "nitzotzote", "nitzotz": "nitzotz", "nekudot": "nekoodote",
    "nekudah": "nekooda", "sephirah": "sefeera", "sephirotic": "sefeerotic",
    "sephirot": "sefeerote", "tikkun": "tikoon", "shevira": "shveera",
    "shevirat": "shveerat", "nefesh": "nefesh", "Lurianic": "Loorianic",
    "Kabbalah": "Kabala", "Luria": "Looria", "Scholem": "Sholem", "kel": "kel",
    # Math / philosophy proper names (ASCII — included only if present in text)
    "Grothendieck": "Grotendeek", "Deleuze": "Deluhz", "Guattari": "Gwataree",
    "Lacan": "Lakahn", "Lacanian": "Lakahnian", "Badiou": "Badyoo",
    "Wittgenstein": "Vitgenshtine", "Nietzsche": "Neecha", "Nietzschean": "Neechean",
    "hocolim": "hoh co lim", "hocolimit": "hoh co limit", "Novikov": "Noveekov",
    "Vietoris": "Veeaytoris", "Goresky": "Goreski", "Kojève": "Kozhev",
    "Kojeve": "Kozhev", "Derridean": "Derridian", "Searle": "Surl",
    "Pythia": "Pithia", "Llemma": "Lemma", "Mather": "Mather", "Seifert": "Zyefert",
    "Searleanism": "Surl-ee-uh-nism", "Searlean": "Surl-ee-un", "Stiegler": "Steegler",
    "Stiegler's": "Steeglers", "Suno": "Soo-no", "dua": "doo-ah", "dua-bot": "doo-ah bot",
    "amānah": "a-MAA-nah", "Dasein": "DAH-zine", "clinamen": "cli-NAH-men",
    "Hui": "Hway", "Kojève": "Kozhev", "Kojève's": "Kozhevs", "Kaʿba": "Kaaba",
    "naḥnuwāt": "nahnoo waat", "shahādah": "shahaada", "Qabḍ": "Qabd", "Basṭ": "Bast",
}

NONASCII = re.compile(r"[^\x00-\x7f]")
# include Latin Extended Additional (U+1E00–1EFF: ḥ ṭ ṣ ẓ ḍ …) so dotted-consonant
# words (Naḥnu, tanāẓur, muṭmaʾinna, ṭarīqa) are captured whole, not split.
WORD = re.compile(r"[A-Za-zÀ-ÿĀ-ɏḀ-ỿ̀-ͯʿʾ’'\-]+")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def asciifold(tok: str) -> str:
    s = tok.replace("ʿ", "").replace("ʾ", "").replace("’", "")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s or tok


def main() -> int:
    cur = {nfc(k): v for k, v in CURATED.items()}
    tokens: Counter[str] = Counter()
    for f in sorted(TXT_DIR.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("[[VOICE"):
                continue
            for w in WORD.findall(line):
                w = nfc(w.strip("'’-"))
                if w:
                    tokens[w] += 1

    entries: dict[str, str] = {}      # grapheme -> alias
    fell_back: list[tuple[str, int, str]] = []

    # non-ASCII tokens: always include
    for tok, cnt in tokens.items():
        if NONASCII.search(tok):
            if tok in cur:
                entries[tok] = cur[tok]
            elif tok.lower() in cur:
                entries[tok] = cur[tok.lower()]
            else:
                fold = asciifold(tok)
                entries[tok] = fold
                fell_back.append((tok, cnt, fold))

    # curated ASCII names, only if they appear
    for term, alias in CURATED.items():
        if NONASCII.search(term):
            continue
        if tokens.get(term, 0) > 0 and term not in entries:
            entries[term] = alias

    # write PLS
    lex = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"',
           '         alphabet="ipa" xml:lang="en-US">']
    for g in sorted(entries, key=lambda x: (-tokens.get(x, 0), x.lower())):
        lex.append(f"  <lexeme><grapheme>{escape(g)}</grapheme>"
                   f"<alias>{escape(entries[g])}</alias></lexeme>")
    lex.append("</lexicon>")
    OUT_PLS.write_text("\n".join(lex) + "\n", encoding="utf-8")

    # write human table
    md = ["# Pronunciation dictionary — R&R audiobook", "",
          f"{len(entries)} alias entries (ElevenLabs multilingual_v2 alias rules).", "",
          "| term | count | spoken as |", "|---|---|---|"]
    for g in sorted(entries, key=lambda x: (-tokens.get(x, 0), x.lower())):
        md.append(f"| {g} | {tokens.get(g,0)} | {entries[g]} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    import json
    OUT_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ {len(entries)} entries -> {OUT_PLS.relative_to(ROOT)} + "
          f"{OUT_MD.relative_to(ROOT)} + {OUT_JSON.relative_to(ROOT)}")
    print(f"\n{len(fell_back)} non-ASCII tokens used ASCII-fold fallback (review/curate):")
    for tok, cnt, fold in sorted(fell_back, key=lambda x: -x[1]):
        print(f"   {tok!r} (x{cnt}) -> {fold!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
