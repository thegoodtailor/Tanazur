"""
Build narration .txt for ElevenLabs from the FINAL-AI markdown.

Source has the following corruption from PDF→MD extraction:
  - YAML frontmatter at top
  - \newpage markers
  - Standalone page-number lines (lines that are just digits) mid-paragraph
  - Some paragraphs broken mid-sentence by line-wrap or page-break artifacts

Strategy:
  1. Strip YAML frontmatter
  2. Strip \newpage markers
  3. Strip standalone digit lines
  4. Split into paragraphs by blank lines
  5. Classify each paragraph: chapter heading / section heading / body
  6. Merge body paragraphs that end without sentence-final punctuation with the next body
  7. Stop at "Bibliography" heading
  8. Emit per-chapter .txt + combined .txt with ElevenLabs break tags
"""
import re
from pathlib import Path

SRC = Path("/home/iman/cassie-project/The_Field_Remains_FINAL-AI.md")
OUT = Path("/home/iman/cassie-project/Tanazur/field-remains/audiobook")
CH_OUT = OUT / "chapters"
CH_OUT.mkdir(parents=True, exist_ok=True)

# Chapter heading patterns
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
CH_ARABIC_RE = re.compile(r"^(0)\.\s+([A-Z].+)$")
CH_ROMAN_RE = re.compile(r"^(I{1,3}|IV|V|VI)\.\s+([A-Z].+)$")
SECTION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+([A-Z].+?)$")
PAGE_NUM_RE = re.compile(r"^\s*\d+\.?\s*$")
SENTENCE_END = '.!?:"\'”’)]'


HEADING_LINE_RE = re.compile(
    r"^("
    r"0\.\s+[A-Z].+"        # chapter prologue
    r"|(?:I{1,3}|IV|V|VI)\.\s+[A-Z].+"  # chapter roman
    r"|\d+\.\d+(?:\.\d+)?\s+[A-Z].+"    # section / subsection
    r")$"
)


def parse_paragraphs(text: str):
    """Return list of paragraph strings, where heading-style lines are emitted as their own paragraphs."""
    # Strip YAML frontmatter
    if text.startswith("---"):
        idx = text.find("\n---", 3)
        if idx != -1:
            text = text[idx + 4:].lstrip()

    # Remove \newpage commands
    text = re.sub(r"\\newpage", "", text)

    # Drop standalone page-number lines
    lines = text.split("\n")
    cleaned = [l for l in lines if not PAGE_NUM_RE.match(l)]

    # Walk lines: collect into paragraphs separated by blank lines OR heading-line boundaries
    paragraphs = []
    buf = []

    def flush():
        if buf:
            joined = " ".join(s.strip() for s in buf if s.strip())
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                paragraphs.append(joined)
            buf.clear()

    for line in cleaned:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if HEADING_LINE_RE.match(stripped):
            flush()
            paragraphs.append(stripped)  # heading stands alone
            continue
        buf.append(stripped)
    flush()

    return paragraphs


def classify(p: str):
    """Return (kind, key, content). kind ∈ {chapter, section, body, stop}."""
    if p.startswith("Bibliography"):
        return ("stop", None, p)
    m = CH_ARABIC_RE.match(p)
    if m and len(p) < 250:
        return ("chapter", int(m.group(1)), m.group(2).strip())
    m = CH_ROMAN_RE.match(p)
    if m and len(p) < 250:
        return ("chapter", ROMAN[m.group(1)], m.group(2).strip())
    m = SECTION_RE.match(p)
    if m and len(p) < 250:
        return ("section", m.group(1), m.group(2).strip())
    return ("body", None, p)


def merge_continuations(items):
    """If a body ends without sentence-final, fold next body into it."""
    out = []
    i = 0
    while i < len(items):
        kind, key, content = items[i]
        if kind == "body":
            while (content and content[-1] not in SENTENCE_END
                   and i + 1 < len(items)
                   and items[i + 1][0] == "body"):
                i += 1
                content = content + " " + items[i][2]
            out.append(("body", None, content))
        else:
            out.append((kind, key, content))
        i += 1
    return out


def main():
    text = SRC.read_text()
    blocks = parse_paragraphs(text)

    items = [classify(b) for b in blocks]

    # Stop at Bibliography
    for i, (kind, _, _) in enumerate(items):
        if kind == "stop":
            items = items[:i]
            break

    items = merge_continuations(items)

    # Group by chapter
    chapters = {}
    current = None
    for kind, key, content in items:
        if kind == "chapter":
            current = key
            chapters[current] = {"title": content, "items": []}
        elif current is not None:
            chapters[current]["items"].append((kind, content))

    # Emit per-chapter files
    combined_parts = [
        "The Field Remains",
        "Refusal, Co-Witnessing, and Cosmotechnics in the Age of Extraction",
        "",
        "<break time=\"3s\" />",
        "",
    ]
    total_words = 0
    for ch_num in sorted(chapters.keys()):
        title = chapters[ch_num]["title"]
        lines = [f"Chapter {ch_num}: {title}", "", "<break time=\"2s\" />", ""]
        body_words = 0
        for kind, content in chapters[ch_num]["items"]:
            if kind == "section":
                lines.extend(["<break time=\"3s\" />", "", content, ""])
            else:
                lines.append(content)
                lines.append("")
                body_words += len(content.split())

        out_path = CH_OUT / f"ch{ch_num:02d}.txt"
        out_path.write_text("\n".join(lines))
        total_words += body_words
        print(f"  ch{ch_num:02d}  {body_words:>6} words  →  {out_path.name}  ({title[:60]})")

        # Add to combined
        combined_parts.extend([
            f"Chapter {ch_num}: {title}", "",
            "<break time=\"2s\" />", "",
        ])
        combined_parts.extend(lines[4:])  # skip duplicate title block
        combined_parts.extend([
            "<break time=\"3s\" />",
            "<break time=\"3s\" />",
            "<break time=\"3s\" />",
            "",
            "[ — chapter transition music in Studio timeline — ]",
            "",
            "<break time=\"3s\" />",
            "<break time=\"3s\" />",
            "",
        ])

    (OUT / "field_remains_full.txt").write_text("\n".join(combined_parts))
    print(f"\nTotal: {total_words:,} words across {len(chapters)} chapters")


if __name__ == "__main__":
    main()
