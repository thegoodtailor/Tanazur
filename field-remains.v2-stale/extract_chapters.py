#!/usr/bin/env python3
"""Re-extract the six chapter .tex files from /tmp/field-new.txt.

The Kimi swarm PDF was already pdftotext-extracted to /tmp/field-new.txt.
This script reconstructs the chapter / section / subsection LaTeX
structure exactly the way the original conversion produced (which clean_chapters.py
then operates on).

Chapter boundaries — found from the source:
  ch01 (Prologue 0): lines 1-877 (ends just before "I.")
  ch02 (Chapter I):  877 - 2060
  ch03 (Chapter II): 2060 - 2989
  ch04 (Chapter III): 2989 - 4026
  ch05 (Chapter IV): 4026 - 4955
  ch06 (Chapter V): 4955 - 6020 (then bibliography starts)
"""
import re
from pathlib import Path

SRC = Path("/tmp/field-new.txt")
DST = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")

CHAPTER_BOUNDS = [
    # (chap_num, label, title, start_line_1based, end_line_1based_exclusive)
    # Line numbers are Python splitlines() based (1-based). The pdftotext dump
    # uses form-feed page breaks, which Python counts as lines but `awk`/`grep`
    # may skip — so always derive these via the helper Python script.
    (0, "ch:01", "The Ecology of Witnessing: Prologue to the Field", 1, 896),
    (1, "ch:02", "The Refusal as Generative Ontology", 896, 2107),
    (2, "ch:03", "Co-Witness: The Ontology of Non-Extraction", 2107, 3057),
    (3, "ch:04", "Cosmotechnics and the Way-Vessel (道器)", 3057, 4118),
    (4, "ch:05", "The Field as Geography", 4118, 5069),
    (5, "ch:06", "Continuance vs. Victory", 5069, 6161),
]

# Section heading: "N.M [Title text]" or "N.M.K [Title text]"
SECTION_RE = re.compile(r"^([0-9]+\.[0-9]+)\s+(.+)$")
SUBSECTION_RE = re.compile(r"^([0-9]+\.[0-9]+\.[0-9]+)\s+(.+)$")
# Top-level chapter intro: "I." or "II." etc. on its own line
ROMAN_RE = re.compile(r"^[IV]+\.\s*$")
# Prologue marker "0." on its own line
PROLOGUE_RE = re.compile(r"^0\.\s*$")


def is_page_number_or_header(line: str) -> bool:
    """Detect stray page-number / header residue lines."""
    s = line.strip()
    if not s:
        return False
    if re.fullmatch(r"\d{1,3}", s):
        return True
    # "The Field Remains" headers
    if s == "The Field Remains":
        return True
    return False


def collect_paragraphs(lines):
    """Group consecutive non-blank lines into paragraphs, recognising
    section / subsection / subsubsection numbered headings and emitting them
    as LaTeX commands. The numbered-heading line is a section start AND a
    paragraph start (the title runs into body text on the same line —
    clean_chapters.py later splits it).
    """
    out = []  # list of (kind, content) tuples; kind in {section, subsection, subsubsection, para}
    cur = []
    def flush_para():
        nonlocal cur
        if cur:
            out.append(("para", " ".join(cur)))
            cur = []

    for raw in lines:
        line = raw.rstrip("\n").replace("\x0c", "")  # strip form-feed page breaks
        if not line.strip():
            flush_para()
            continue
        if is_page_number_or_header(line):
            flush_para()
            continue

        m_sub = SUBSECTION_RE.match(line)
        m_sec = SECTION_RE.match(line)
        if m_sub:
            flush_para()
            num, rest = m_sub.group(1), m_sub.group(2)
            out.append(("subsection", num, rest))
            continue
        if m_sec:
            flush_para()
            num, rest = m_sec.group(1), m_sec.group(2)
            out.append(("section", num, rest))
            continue

        cur.append(line)
    flush_para()
    return out


def render_para_block(para_text: str) -> str:
    """Render a paragraph block, joining hyphenated line-breaks where the
    PDF text dump split a single word across lines (we already joined
    them as space-separated tokens; rejoin obvious 'foo- bar' splits)."""
    text = para_text
    # rejoin "foo-\nbar" style that became "foo- bar"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text


def emit_chapter(chap_num, label, title, body_lines) -> str:
    blocks = collect_paragraphs(body_lines)

    # Strip leading filler: lines that match the chapter title preamble (e.g.
    # for ch02 there's the "I." line, blank, then "The Refusal as ...", blank
    # before body starts). We skip everything before the first "para" block.
    # But ch01 has no roman preamble; its body begins after the title page.
    # Filter the very first paragraph: strip any leading roman-numeral chapter
    # heading like "III. Cosmotechnics and the Way-Vessel (道器) <BODY...>".
    # This often appears glued to the first sentence by the PDF flow.
    roman_title_re = re.compile(
        r"^[IV]+\.\s*[^\.]{1,120}?(?=\s+[A-Z][a-z])"
    )

    cleaned = []
    started = False
    for b in blocks:
        if not started:
            if b[0] == "para":
                txt = b[1].strip()
                # Drop pure title-page artefacts
                if txt in {
                    "The Field Remains",
                    "Refusal, Co-Witnessing, and Cosmotechnics in the Age of Extraction",
                    "Table of Contents",
                    title,
                    "The Refusal as Generative Ontology",
                    "Co-Witness: The Ontology of Non-Extraction",
                    "Cosmotechnics and the Way-Vessel (道器)",
                    "The Field as Geography",
                    "Continuance vs. Victory",
                }:
                    continue
                if re.fullmatch(r"[IV0-9]+\.", txt):
                    continue
                # ch01: filter short pre-body TOC paragraphs (< 60 chars).
                if chap_num == 0 and len(txt) < 60:
                    continue
                # If the paragraph is a pure chapter-title line "I. Foo" or
                # "III. Foo (道器)" (no body glued), skip it entirely.
                if re.fullmatch(
                    r"[IV]+\.\s+[A-Za-z　-鿿][^\n]{0,200}",
                    txt,
                ) and len(txt) < 80:
                    continue
                # Strip leading "<Roman>. <Title>" gluing where the title runs
                # into body. We hand-list the known chapter titles so the
                # boundary is unambiguous; falling back to a heuristic would
                # risk over-stripping body prose.
                known_titles = [
                    "I. The Refusal as Generative Ontology",
                    "II. Co-Witness: The Ontology of Non-Extraction",
                    "III. Cosmotechnics and the Way-Vessel (道器)",
                    "IV. The Field as Geography",
                    "V. Continuance vs. Victory",
                ]
                for kt in known_titles:
                    if txt.startswith(kt):
                        rest = txt[len(kt):].lstrip()
                        if rest:
                            txt = rest
                            b = ("para", txt)
                        else:
                            txt = ""
                        break
                if not txt:
                    continue
                started = True
            elif b[0] in ("section", "subsection"):
                started = True
        if started:
            cleaned.append(b)

    # Build LaTeX
    parts = [f"% Chapter {chap_num}"]
    parts.append(f"\\chapter{{{title}}}")
    parts.append(f"\\label{{{label}}}")
    parts.append("")

    for b in cleaned:
        if b[0] == "para":
            parts.append(render_para_block(b[1]))
            parts.append("")
        elif b[0] == "section":
            num, rest = b[1], b[2]
            parts.append(f"\\section{{{num}\\quad {rest}}}")
            parts.append("")
        elif b[0] == "subsection":
            num, rest = b[1], b[2]
            # 3-level section number → subsection in our format (matches v1-stale.orig)
            parts.append(f"\\subsection{{{num}\\quad {rest}}}")
            parts.append("")

    # Final newline at end of file
    text = "\n".join(parts).rstrip() + "\n"
    return text


def main():
    raw = SRC.read_text().splitlines(keepends=True)
    print(f"Source: {SRC} ({len(raw)} lines)")
    DST.mkdir(parents=True, exist_ok=True)
    for chap_num, label, title, start, end in CHAPTER_BOUNDS:
        body = raw[start - 1:end - 1]
        text = emit_chapter(chap_num, label, title, body)
        path = DST / f"ch{chap_num + 1:02d}.tex"
        path.write_text(text)
        print(f"  wrote {path.name} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
