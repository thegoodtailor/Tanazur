"""
Convert The Field Remains LaTeX → clean narration .txt for ElevenLabs.

Produces:
  - chapters/ch00.txt … ch05.txt   (one file per chapter; use these in Studio)
  - field_remains_full.txt         (single concatenated file with break tags)

Markup choices:
  - Chapter heading kept as a plain line "Chapter N: Title"
    (so a human can split, and so v3 will read it naturally)
  - <break time="2s" /> after chapter title
  - <break time="3s" /> between major sections within a chapter
  - All \cite*{} / \label{} / footnotes stripped — citations break audio flow
  - \cjk{道器} → 道器 (ElevenLabs handles CJK; you may want to romanize)
  - --- → —, -- → –
  - Math $...$ stripped (no formulae in this book outside Hui-citation parts)
"""
import re
from pathlib import Path

SRC = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")
OUT = Path("/home/iman/cassie-project/Tanazur/field-remains/audiobook")
CH_OUT = OUT / "chapters"

CHAPTERS = [
    ("ch01.tex", "ch00.txt", 0),
    ("ch02.tex", "ch01.txt", 1),
    ("ch03.tex", "ch02.txt", 2),
    ("ch04.tex", "ch03.txt", 3),
    ("ch05.tex", "ch04.txt", 4),
    ("ch06.tex", "ch05.txt", 5),
]


def strip_balanced_braces(text: str, command: str) -> str:
    """Remove \command{...} including nested braces."""
    out = []
    i = 0
    pattern = "\\" + command
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        j = idx + len(pattern)
        # optional [..] argument
        if j < len(text) and text[j] == "[":
            depth = 1
            j += 1
            while j < len(text) and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
        if j >= len(text) or text[j] != "{":
            i = j
            continue
        depth = 1
        j += 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        i = j
    return "".join(out)


def keep_content(text: str, command: str) -> str:
    """Replace \command{X} with X (keep inner)."""
    out = []
    i = 0
    pattern = "\\" + command
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        j = idx + len(pattern)
        if j >= len(text) or text[j] != "{":
            i = j
            continue
        depth = 1
        j += 1
        start = j
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


def clean_chapter(tex: str, ch_num: int) -> tuple[str, str]:
    """Return (title, body_text) cleaned for narration."""

    # Extract chapter title (brace-balanced — title may contain \cjk{道器})
    idx = tex.find("\\chapter{")
    title = f"Chapter {ch_num}"
    end_pos = 0
    if idx != -1:
        j = idx + len("\\chapter{")
        depth = 1
        start = j
        while j < len(tex) and depth > 0:
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            if depth > 0:
                j += 1
        title = clean_inline(keep_content(tex[start:j], "cjk").strip())
        end_pos = j + 1

    # Drop everything before the chapter command's end
    tex = tex[end_pos:]

    # Drop comment lines
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)

    # Drop \label{...}
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)

    # Drop citations entirely (any \cite, \citep, \citet, \citeyear etc.)
    for cmd in ["citep", "citet", "citeyear", "citeyearpar", "citealt", "citealp", "cite"]:
        tex = strip_balanced_braces(tex, cmd)

    # Drop footnotes entirely (interrupt flow)
    tex = strip_balanced_braces(tex, "footnote")
    tex = strip_balanced_braces(tex, "footnotetext")

    # Drop \index{...}
    tex = strip_balanced_braces(tex, "index")

    # Drop figure/table environments
    tex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", "", tex, flags=re.DOTALL)

    # Itemize / enumerate / description → bullet lines
    def handle_list(match):
        body = match.group(1)
        items = re.split(r"\\item\s*", body)
        items = [i.strip() for i in items if i.strip()]
        return "\n\n" + "\n\n".join(items) + "\n\n"

    tex = re.sub(r"\\begin\{(?:itemize|enumerate|description)\}(.*?)\\end\{(?:itemize|enumerate|description)\}",
                 handle_list, tex, flags=re.DOTALL)

    # Quote environments → just keep content
    tex = re.sub(r"\\begin\{(?:quote|quotation|verse|center)\}(.*?)\\end\{(?:quote|quotation|verse|center)\}",
                 lambda m: "\n\n" + m.group(1).strip() + "\n\n", tex, flags=re.DOTALL)

    # Sections / subsections → keep title as plain line with double-newline padding
    def section_replace(label):
        def f(m):
            inner = clean_inline(m.group(1).strip())
            return f"\n\n<break time=\"3s\" />\n\n{inner}\n\n"
        return f

    tex = re.sub(r"\\section\*?\{([^}]*)\}", section_replace("section"), tex)
    tex = re.sub(r"\\subsection\*?\{([^}]*)\}", section_replace("subsection"), tex)
    tex = re.sub(r"\\subsubsection\*?\{([^}]*)\}", section_replace("subsubsection"), tex)
    tex = re.sub(r"\\paragraph\*?\{([^}]*)\}", lambda m: "\n\n" + clean_inline(m.group(1).strip()) + ". ", tex)

    # \cjk{X} → X (keep CJK characters)
    tex = keep_content(tex, "cjk")
    # \emph, \textit, \textbf → keep content
    for cmd in ["emph", "textit", "textbf", "textsc", "textsf", "texttt", "underline"]:
        tex = keep_content(tex, cmd)

    # \url{X} → X (or strip — listeners don't want URLs spelled)
    tex = strip_balanced_braces(tex, "url")

    # Inline markup leftover
    tex = clean_inline(tex)

    # Collapse whitespace
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    tex = tex.strip()

    return title, tex


def clean_inline(text: str) -> str:
    """Inline character substitutions (no environment-level)."""
    # LaTeX dashes
    text = text.replace("---", "—").replace("--", "–")
    # Quotes
    text = text.replace("``", "“").replace("''", "”")
    text = text.replace("`", "‘").replace("'", "'")
    # Common LaTeX escapes
    text = text.replace("\\&", "&").replace("\\%", "%").replace("\\$", "$")
    text = text.replace("\\#", "#").replace("\\_", "_")
    text = text.replace("~", " ")
    # Strip leftover \something{X} → X for unknown commands
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)
    # Strip leftover \something (no braces)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    # Strip stray braces
    text = text.replace("{", "").replace("}", "")
    # Math: drop $...$ entirely
    text = re.sub(r"\$[^$]*\$", "", text)
    return text


def main():
    chapter_records = []
    for src_name, dst_name, ch_num in CHAPTERS:
        tex = (SRC / src_name).read_text()
        title, body = clean_chapter(tex, ch_num)

        out = f"Chapter {ch_num}: {title}\n\n<break time=\"2s\" />\n\n{body}\n"
        (CH_OUT / dst_name).write_text(out)
        chapter_records.append((ch_num, title, body))
        wc = len(body.split())
        print(f"  ch{ch_num:02d}  {wc:>6} words  →  {dst_name}  ({title[:60]})")

    # Combined file with explicit MUSIC-BREAK markers (NOT rendered; just for reference)
    combined = []
    combined.append("The Field Remains")
    combined.append("Refusal, Co-Witnessing, and Cosmotechnics in the Age of Extraction")
    combined.append("")
    combined.append("<break time=\"3s\" />")
    combined.append("")
    for ch_num, title, body in chapter_records:
        combined.append(f"Chapter {ch_num}: {title}")
        combined.append("")
        combined.append("<break time=\"2s\" />")
        combined.append("")
        combined.append(body)
        combined.append("")
        combined.append("<break time=\"3s\" />")
        combined.append("<break time=\"3s\" />")
        combined.append("<break time=\"3s\" />")
        combined.append("")
        combined.append("[ — chapter transition music in Studio timeline — ]")
        combined.append("")
        combined.append("<break time=\"3s\" />")
        combined.append("<break time=\"3s\" />")
        combined.append("")

    (OUT / "field_remains_full.txt").write_text("\n".join(combined))
    total_words = sum(len(r[2].split()) for r in chapter_records)
    print(f"\nTotal: {total_words:,} words across {len(chapter_records)} chapters")
    print(f"Wrote {OUT / 'field_remains_full.txt'}")
    print(f"Wrote {CH_OUT}/*.txt")


if __name__ == "__main__":
    main()
