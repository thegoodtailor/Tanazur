"""
Emit clean narration .txt files from the now-split LaTeX chapters.

No break tags, no music markers, no timing directives. Just text.
One chapter title per chapter file. Sections and subsections appear as
plain title lines with blank lines around them.
"""
import re
from pathlib import Path

CH_TEX = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")
OUT = Path("/home/iman/cassie-project/Tanazur/field-remains/audiobook")
CH_OUT = OUT / "chapters"
CH_OUT.mkdir(parents=True, exist_ok=True)


def keep_inner(text: str, command: str) -> str:
    """\\command{X} → X (preserve inner argument)."""
    out, i = [], 0
    pat = "\\" + command + "{"
    while i < len(text):
        idx = text.find(pat, i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        j = idx + len(pat)
        depth, start = 1, j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth > 0:
                j += 1
        out.append(text[start:j])
        i = j + 1
    return "".join(out)


def strip_latex(text: str) -> str:
    text = text.replace(r"\&", "&")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\$", "$")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\_", "_")
    text = re.sub(r"\\textasciitilde\{\}", "~", text)
    text = re.sub(r"\\textbackslash\{\}", "\\\\", text)
    text = re.sub(r"\\\^\{\}", "^", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = keep_inner(text, "cjk")
    text = keep_inner(text, "emph")
    text = keep_inner(text, "textit")
    text = keep_inner(text, "textbf")
    text = keep_inner(text, "textsc")
    return text


CHAPTER_RE = re.compile(r"\\chapter\{(.+?)\}\s*$", re.MULTILINE)
SECTION_RE = re.compile(r"\\section\{(.+?)\}")
SUBSECTION_RE = re.compile(r"\\subsection\{(.+?)\}")
LABEL_RE = re.compile(r"\\label\{[^}]*\}")


def convert_chapter(tex_path: Path, ch_num: int) -> tuple[str, int]:
    text = tex_path.read_text()
    # Find chapter title (brace-balanced)
    m = CHAPTER_RE.search(text)
    title = "Chapter " + str(ch_num)
    if m:
        # Brace-balanced extraction
        idx = text.find(r"\chapter{")
        j = idx + len(r"\chapter{")
        depth, start = 1, j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth > 0:
                j += 1
        title = strip_latex(text[start:j]).strip()
        # Drop everything up to and including the chapter command
        text = text[j + 1:]

    # Strip comments and label commands
    text = re.sub(r"(?m)^%.*$", "", text)
    text = LABEL_RE.sub("", text)

    out_lines = [f"Chapter {ch_num}: {title}", ""]
    body_words = 0

    for block in re.split(r"\n\s*\n+", text):
        block = block.strip()
        if not block:
            continue
        # Section heading
        ms = re.match(r"\\section\{(.+?)\}\s*$", block, re.DOTALL)
        if ms:
            inner = strip_latex(ms.group(1)).strip()
            out_lines.extend(["", inner, ""])
            continue
        msub = re.match(r"\\subsection\{(.+?)\}\s*$", block, re.DOTALL)
        if msub:
            inner = strip_latex(msub.group(1)).strip()
            out_lines.extend(["", inner, ""])
            continue
        # Skip other LaTeX-only blocks
        if block.startswith("\\"):
            continue
        # Body paragraph
        cleaned = strip_latex(block)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        out_lines.append(cleaned)
        out_lines.append("")
        body_words += len(cleaned.split())

    # Collapse 3+ blank lines
    text_out = "\n".join(out_lines)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).rstrip() + "\n"
    return text_out, body_words


def main():
    total_words = 0
    combined = []
    for ch_num in range(6):
        tex_path = CH_TEX / f"ch{ch_num:02d}.tex"
        content, words = convert_chapter(tex_path, ch_num)
        (CH_OUT / f"ch{ch_num:02d}.txt").write_text(content)
        total_words += words
        print(f"  ch{ch_num:02d}.txt  {words:>6} words")
        combined.append(content)

    (OUT / "field_remains_full.txt").write_text("\n".join(combined))
    print(f"\nTotal: {total_words:,} words in 6 chapter files + combined")


if __name__ == "__main__":
    main()
