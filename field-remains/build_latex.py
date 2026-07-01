"""
Convert The_Field_Remains_FINAL-AI.md → clean LaTeX book.

Fixes from the source markdown:
  - YAML frontmatter stripped
  - \newpage markers stripped
  - Standalone page-number lines (from PDF extraction) removed
  - Paragraphs broken mid-sentence by page breaks re-joined
  - Section headings separated from following body when source put them on adjacent lines
  - LaTeX special chars escaped (& % $ # _)
  - CJK chars (e.g. 道器) wrapped in \cjk{} for font selection
  - Bibliography section emitted as its own chapter via \begin{thebibliography}
  - One known PDF-artifact typo fixed: "containermodel" → "container-model"
"""
import re
from pathlib import Path

SRC = Path("/home/iman/cassie-project/The_Field_Remains_FINAL-AI.md")
OUT_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains")
CH_DIR = OUT_DIR / "chapters"
CH_DIR.mkdir(parents=True, exist_ok=True)

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
CH_ARABIC_RE = re.compile(r"^(0)\.\s+([A-Z].+)$")
CH_ROMAN_RE = re.compile(r"^(I{1,3}|IV|V|VI)\.\s+([A-Z].+)$")
SECTION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+([A-Z].+?)$")
PAGE_NUM_RE = re.compile(r"^\s*\d+\.?\s*$")
SENTENCE_END = '.!?:"\'”’)]'

HEADING_LINE_RE = re.compile(
    r"^("
    r"0\.\s+[A-Z].+"
    r"|(?:I{1,3}|IV|V|VI)\.\s+[A-Z].+"
    r"|\d+\.\d+(?:\.\d+)?\s+[A-Z].+"
    r"|Bibliography\s*$"
    r")$"
)

CJK_CHARS_RE = re.compile(r"([一-鿿]+)")


def parse_blocks(text: str):
    """Yield (kind, key, content) for chapters, sections, body paragraphs, and bibliography start."""
    # Strip YAML frontmatter
    if text.startswith("---"):
        idx = text.find("\n---", 3)
        if idx != -1:
            text = text[idx + 4:].lstrip()

    # Remove \newpage commands AND the PDF-extraction-corrupted variants
    text = re.sub(r"\\newpage", "", text)
    text = re.sub(r"(?m)^\s*ewpage\s*$", "", text)
    # PDF extraction corrupted "Bibliography" into "B ibliography" with the
    # book subtitle following. Normalize so the stop-detector catches it.
    text = re.sub(r"(?m)^B ibliography\s*$\n^The Field Remains:.*$", "Bibliography", text)
    text = re.sub(r"(?m)^B ibliography\s*$", "Bibliography", text)

    # Drop standalone page-number lines
    lines = [l for l in text.split("\n") if not PAGE_NUM_RE.match(l)]

    # Walk lines into paragraphs; headings flush.
    paragraphs = []
    buf = []

    def flush():
        if buf:
            joined = re.sub(r"\s+", " ", " ".join(s.strip() for s in buf if s.strip())).strip()
            if joined:
                paragraphs.append(joined)
            buf.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if HEADING_LINE_RE.match(stripped):
            flush()
            paragraphs.append(stripped)
            continue
        buf.append(stripped)
    flush()

    # Classify and merge continuations
    items = []
    for p in paragraphs:
        if p == "Bibliography":
            items.append(("bibliography_start", None, p))
            continue
        m = CH_ARABIC_RE.match(p)
        if m and len(p) < 300:
            items.append(("chapter", int(m.group(1)), m.group(2).strip()))
            continue
        m = CH_ROMAN_RE.match(p)
        if m and len(p) < 300:
            items.append(("chapter", ROMAN[m.group(1)], m.group(2).strip()))
            continue
        m = SECTION_RE.match(p)
        if m and len(p) < 300:
            key = m.group(1)
            depth = key.count(".") + 1  # "0.1" → 2 → section; "0.1.1" → 3 → subsection
            items.append(("section" if depth == 2 else "subsection", key, m.group(2).strip()))
            continue
        items.append(("body", None, p))

    # Merge body continuations (paragraph not ending in sentence-final)
    merged = []
    i = 0
    while i < len(items):
        kind, key, content = items[i]
        if kind == "body":
            while (content and content[-1] not in SENTENCE_END
                   and i + 1 < len(items)
                   and items[i + 1][0] == "body"):
                i += 1
                content = content + " " + items[i][2]
            merged.append(("body", None, content))
        else:
            merged.append((kind, key, content))
        i += 1

    return merged


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in body text. Keep Unicode as-is for XeLaTeX."""
    # Fix known PDF-extraction hyphen-loss artifacts
    text = text.replace("containermodel", "container-model")
    text = text.replace("standingreserve", "standing-reserve")
    text = text.replace("nonextraction", "non-extraction")
    text = text.replace("intraaction", "intra-action")
    # Escape special chars (order matters: backslash first if we needed it, but we have none in source)
    out = []
    for ch in text:
        if ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "^":
            out.append(r"\^{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    s = "".join(out)
    # Wrap CJK runs in \cjk{}
    s = CJK_CHARS_RE.sub(r"\\cjk{\1}", s)
    return s


def latex_escape_title(text: str) -> str:
    """Same as body but suitable for \chapter{}/\section{} arguments."""
    return latex_escape(text)


def emit_chapter(ch_num: int, title: str, items, ch_label: str) -> str:
    out = []
    out.append("% !TEX root = ../main.tex")
    out.append(f"\\chapter{{{latex_escape_title(title)}}}")
    out.append(f"\\label{{ch:{ch_num:02d}}}")
    out.append("")
    for kind, key, content in items:
        if kind == "section":
            out.append("")
            out.append(f"\\section{{{latex_escape_title(content)}}}")
            out.append("")
        elif kind == "subsection":
            out.append("")
            out.append(f"\\subsection{{{latex_escape_title(content)}}}")
            out.append("")
        elif kind == "body":
            out.append(latex_escape(content))
            out.append("")
    return "\n".join(out)


def emit_bibliography(bib_items) -> str:
    """Bibliography source is a numbered list. Emit as thebibliography environment."""
    out = []
    out.append("% !TEX root = main.tex")
    out.append(r"\begin{thebibliography}{999}")
    out.append(r"\addcontentsline{toc}{chapter}{Bibliography}")
    for entry in bib_items:
        out.append(rf"\bibitem{{}} {latex_escape(entry)}")
        out.append("")
    out.append(r"\end{thebibliography}")
    return "\n".join(out)


def extract_bibliography(text: str):
    """Pull bibliography entries from text after 'Bibliography'.
    Source format alternates: number-line (e.g. '6.') and entry text.
    """
    idx = text.find("\nBibliography\n")
    if idx == -1:
        return []
    body = text[idx + len("\nBibliography\n"):]
    # Drop \newpage
    body = re.sub(r"\\newpage", "", body)
    lines = body.split("\n")
    entries = []
    buf = []

    def flush_entry():
        if buf:
            entry = " ".join(s.strip() for s in buf if s.strip())
            entry = re.sub(r"\s+", " ", entry).strip()
            if entry:
                entries.append(entry)
            buf.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush_entry()
            continue
        # Number-marker line like "6." → new entry boundary
        if re.match(r"^\d+\.?\s*$", s):
            flush_entry()
            continue
        buf.append(s)
    flush_entry()
    return entries


def main():
    raw = SRC.read_text()

    # Parse body (chapters + sections + bibliography_start marker)
    items = parse_blocks(raw)

    # Bibliography entries from after the marker
    bib_entries = extract_bibliography(raw)

    # Group into chapters; stop at bibliography_start
    chapters = {}
    current = None
    for kind, key, content in items:
        if kind == "bibliography_start":
            break
        if kind == "chapter":
            current = key
            chapters[current] = {"title": content, "items": []}
        elif current is not None:
            chapters[current]["items"].append((kind, key, content))

    # Emit per-chapter LaTeX
    total_words = 0
    for ch_num in sorted(chapters.keys()):
        c = chapters[ch_num]
        ch_tex = emit_chapter(ch_num, c["title"], c["items"], f"ch:{ch_num:02d}")
        out_path = CH_DIR / f"ch{ch_num:02d}.tex"
        out_path.write_text(ch_tex)
        words = sum(len(content.split()) for k, _, content in c["items"] if k == "body")
        total_words += words
        print(f"  ch{ch_num:02d}  {words:>6} words  →  {out_path.name}  ({c['title'][:60]})")

    # Emit bibliography
    bib_tex = emit_bibliography(bib_entries)
    (OUT_DIR / "bibliography.tex").write_text(bib_tex)
    print(f"\n  bibliography  {len(bib_entries)} entries  →  bibliography.tex")
    print(f"\nTotal body: {total_words:,} words across {len(chapters)} chapters")


if __name__ == "__main__":
    main()
