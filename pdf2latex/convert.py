"""PDF → LaTeX book converter for Kimi-swarm-produced manuscripts.

Two source PDFs:
  • The_Field_Remains.pdf — chapters marked by Roman numerals (I., II., ...);
    sections by N.M / N.M.K numeric pattern.
  • Rupture_and_Realization_The_New_Logic_of_the_Posthuman_Self.pdf —
    chapters marked by "Chapter N:" or "Chapter N.";  sections by N.M.

Output for each book:
  main.tex             — book-class document with front matter & \\input{}'s
  chapters/chXX.tex    — one file per chapter, editable in isolation

This is intentionally first-pass. Section *titles* may run into body text
(Kimi's PDF output concatenates the bold heading with the following
paragraph). The structure (chapter / section / subsection commands) is in
place so the user can edit titles by hand in each chapter file.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path("/home/iman/cassie-project")
TANAZUR = ROOT / "Tanazur"


# ── LaTeX-special escape ────────────────────────────────────────────────
# Order matters: backslash first, then the rest.
_LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\^{}"),
]


def latex_escape(s: str) -> str:
    # First protect \textbackslash result by using a sentinel.
    out = s
    for src, dst in _LATEX_ESCAPES:
        out = out.replace(src, dst)
    return out


# ── Footnote markers (RR only) ──────────────────────────────────────────
FOOTNOTE_REF = re.compile(r"\[\^(\d+)\^\]")


# ── Standalone page-number detection ────────────────────────────────────
PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


# ── Generic helpers ─────────────────────────────────────────────────────
def pdftotext_flow(pdf_path: Path) -> str:
    out = subprocess.check_output(
        ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
        text=True,
    )
    return out


def clean_pre_pass(text: str) -> str:
    """Drop standalone page-number lines and trim trailing whitespace."""
    out_lines: list[str] = []
    for line in text.splitlines():
        if PAGE_NUM_LINE.match(line):
            continue
        out_lines.append(line.rstrip())
    return "\n".join(out_lines)


def normalise_paragraph(p: str) -> str:
    """Collapse internal newlines (Kimi-PDF often wraps mid-paragraph)."""
    return re.sub(r"\s+", " ", p).strip()


# ── Book configuration ─────────────────────────────────────────────────
@dataclass
class BookConfig:
    pdf_path: Path
    out_dir: Path
    title: str
    subtitle: str
    authors: str
    # A regex whose match starts a new chapter. group(1) = chapter label.
    chapter_pattern: re.Pattern
    # Regex for sections: (number, title-and-body).
    section_pattern: re.Pattern = re.compile(
        r"^(\d+\.\d+)\s+(.+)$"
    )
    subsection_pattern: re.Pattern = re.compile(
        r"^(\d+\.\d+\.\d+)\s+(.+)$"
    )
    # Marker before the first chapter — everything before this is front-matter.
    front_matter_end_marker: str = ""
    # Marker after the last chapter — everything after is back-matter.
    back_matter_start_marker: str = ""
    # Names to use for the front-matter and back-matter latex files.
    extra_processors: list[Callable[[str], str]] = field(default_factory=list)


# ── Chapter / section splitter ──────────────────────────────────────────
@dataclass
class Section:
    number: str
    title: str
    body: str         # may be empty
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class Chapter:
    label: str        # e.g. "I", "1", "0"
    title: str
    preamble: str     # text between chapter line and first numbered section
    sections: list[Section]


def split_sections(chapter_text: str, cfg: BookConfig) -> tuple[str, list[Section]]:
    """Walk lines of a chapter's text; group into a preamble + sections."""
    lines = chapter_text.splitlines()
    preamble_lines: list[str] = []
    sections: list[Section] = []
    cur_section: Section | None = None
    cur_subsection: Section | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        m_sub = cfg.subsection_pattern.match(line)
        m_sec = cfg.section_pattern.match(line)
        if m_sub:
            # Subsection: title-and-body joined on one line. Use heuristic
            # title (first sentence up to a period followed by uppercase).
            number = m_sub.group(1)
            rest = m_sub.group(2)
            title, body = _split_heading_body(rest)
            sub = Section(number=number, title=title, body=body)
            if cur_section is not None:
                cur_section.subsections.append(sub)
                cur_subsection = sub
            else:
                # Subsection with no parent section — promote it to section.
                sections.append(Section(number=number, title=title, body=body))
                cur_section = sections[-1]
                cur_subsection = None
            i += 1
            continue
        if m_sec:
            number = m_sec.group(1)
            rest = m_sec.group(2)
            title, body = _split_heading_body(rest)
            cur_section = Section(number=number, title=title, body=body)
            cur_subsection = None
            sections.append(cur_section)
            i += 1
            continue
        # Body line: attach to currently-open subsection, section, or preamble.
        if cur_subsection is not None:
            cur_subsection.body += "\n" + line
        elif cur_section is not None:
            cur_section.body += "\n" + line
        else:
            preamble_lines.append(line)
        i += 1

    return "\n".join(preamble_lines).strip(), sections


def _split_heading_body(rest: str) -> tuple[str, str]:
    """Given the text after a section number on its line, split into
    (title, body). Heuristic: title runs to first sentence-end (".", "?",
    "!") followed by a space and a capital letter."""
    # First try: explicit colon followed by uppercase. Section titles often
    # have the form "Title: Subtitle Body starts here."
    # Fall back to: first 14 words.
    m = re.search(r"(?<=[\.!?])\s+(?=[A-Z])", rest)
    if m and m.start() < 220:
        title = rest[: m.start()].rstrip(" ,;")
        body = rest[m.end():]
    else:
        words = rest.split()
        if len(words) > 14:
            title = " ".join(words[:14]).rstrip(",;:")
            body = " ".join(words[14:])
        else:
            title = rest
            body = ""
    return title, body


def split_chapters(text: str, cfg: BookConfig) -> tuple[str, list[Chapter], str]:
    """Cut the text into front matter / chapter list / back matter."""
    # Cut front matter
    front_text = text
    after_front = text
    if cfg.front_matter_end_marker:
        idx = text.find(cfg.front_matter_end_marker)
        if idx >= 0:
            front_text = text[:idx]
            after_front = text[idx:]
    else:
        front_text = ""
        after_front = text
    # Cut back matter
    if cfg.back_matter_start_marker:
        idx = after_front.find(cfg.back_matter_start_marker)
        if idx >= 0:
            back_text = after_front[idx:]
            body_text = after_front[:idx]
        else:
            back_text = ""
            body_text = after_front
    else:
        back_text = ""
        body_text = after_front

    # Walk lines, build chapters. Some chapters have the label alone on a
    # line ("II.") with the title on a subsequent non-empty line; we capture
    # that into pending_title_lookup.
    lines = body_text.splitlines()
    chapters: list[Chapter] = []
    cur_buf: list[str] = []
    cur_label: str | None = None
    cur_title: str | None = None
    pending_title_lookup = False

    def flush():
        nonlocal cur_buf, cur_label, cur_title, pending_title_lookup
        if cur_label is None and cur_title is None:
            cur_buf.clear()
            return
        preamble, sections = split_sections("\n".join(cur_buf), cfg)
        chapters.append(Chapter(
            label=cur_label or "",
            title=cur_title or "",
            preamble=preamble,
            sections=sections,
        ))
        cur_buf = []
        pending_title_lookup = False

    for line in lines:
        if pending_title_lookup:
            stripped = line.strip()
            if stripped:
                cur_title = stripped
                pending_title_lookup = False
                continue
            # blank line — keep waiting
            continue
        m = cfg.chapter_pattern.match(line)
        if m:
            flush()
            cur_label = m.group("label").strip()
            title = (m.group("title") or "").strip()
            cur_title = title
            if not title:
                # Bare label line; pick up title from the next non-empty line.
                pending_title_lookup = True
            continue
        cur_buf.append(line)
    flush()

    return front_text, chapters, back_text


# ── LaTeX emission ──────────────────────────────────────────────────────
_FOOTNOTE_SENTINEL = "\x00FN\x00"


def emit_paragraphs(raw: str) -> str:
    """Turn a block of body text into LaTeX paragraphs."""
    if not raw or not raw.strip():
        return ""
    blocks = re.split(r"\n\s*\n+", raw.strip())
    paras = [normalise_paragraph(b) for b in blocks if b.strip()]
    rendered: list[str] = []
    for p in paras:
        # 1. Stash footnote markers under a sentinel so latex_escape can't
        #    mangle the ^ characters inside them.
        captures: list[str] = []

        def _stash(m: re.Match) -> str:
            captures.append(m.group(1))
            return f"{_FOOTNOTE_SENTINEL}{len(captures) - 1}{_FOOTNOTE_SENTINEL}"

        stashed = FOOTNOTE_REF.sub(_stash, p)
        # 2. Escape LaTeX specials.
        escaped = latex_escape(stashed)
        # 3. Restore footnote markers as superscripts.
        def _restore(m: re.Match) -> str:
            idx = int(m.group(1))
            return r"\textsuperscript{" + captures[idx] + "}"

        restored = re.sub(
            re.escape(_FOOTNOTE_SENTINEL) + r"(\d+)" + re.escape(_FOOTNOTE_SENTINEL),
            _restore,
            escaped,
        )
        rendered.append(restored)
    return "\n\n".join(rendered)


def emit_section(sec: Section, level: str) -> str:
    title = latex_escape(sec.title) or f"Section {sec.number}"
    out = [f"\\{level}{{{sec.number}\\quad {title}}}"]
    body = emit_paragraphs(sec.body)
    if body:
        out.append("")
        out.append(body)
    for sub in sec.subsections:
        out.append("")
        out.append(emit_section(sub, level="subsection"))
    return "\n".join(out)


def emit_chapter(ch: Chapter, idx: int) -> str:
    title = latex_escape(ch.title) or f"Chapter {ch.label}"
    out = [
        f"% Chapter {ch.label}",
        f"\\chapter{{{title}}}",
        f"\\label{{ch:{idx:02d}}}",
        "",
    ]
    preamble = emit_paragraphs(ch.preamble)
    if preamble:
        out.append(preamble)
        out.append("")
    for sec in ch.sections:
        out.append(emit_section(sec, level="section"))
        out.append("")
    return "\n".join(out)


def emit_front_matter(raw: str) -> str:
    body = emit_paragraphs(raw)
    return body


MAIN_TEMPLATE = r"""\documentclass[11pt,a4paper,openany]{book}

\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}
\newfontfamily\arabicfont{Amiri}[Script=Arabic,Scale=1.0]

\usepackage{amsmath,amssymb,amsthm}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}

\usepackage{geometry}
\geometry{margin=1.2in}
\usepackage{microtype}
\usepackage{xurl}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{titlesec}
\setlength{\emergencystretch}{2em}

\definecolor{ink}{HTML}{0C1020}
\hypersetup{colorlinks=true, linkcolor=ink, urlcolor=ink, citecolor=ink}

\titleformat{\chapter}[display]
  {\sffamily\Huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titleformat{\section}{\sffamily\Large\bfseries}{}{0em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries}{}{0em}{}

\title{__TITLE__\\[0.4em]\large __SUBTITLE__}
\author{__AUTHORS__}
\date{}

\begin{document}

\frontmatter
\maketitle

__FRONT_MATTER__

\tableofcontents

\mainmatter

__CHAPTER_INPUTS__

\backmatter

__BACK_MATTER__

\end{document}
"""


def write_book(cfg: BookConfig) -> None:
    print(f"[{cfg.out_dir.name}] reading {cfg.pdf_path.name}")
    raw = pdftotext_flow(cfg.pdf_path)
    text = clean_pre_pass(raw)
    for proc in cfg.extra_processors:
        text = proc(text)
    front_raw, chapters, back_raw = split_chapters(text, cfg)
    print(f"[{cfg.out_dir.name}] {len(chapters)} chapters")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "chapters").mkdir(exist_ok=True)

    chapter_inputs: list[str] = []
    for idx, ch in enumerate(chapters, start=1):
        rel = f"chapters/ch{idx:02d}.tex"
        out_path = cfg.out_dir / rel
        out_path.write_text(emit_chapter(ch, idx), encoding="utf-8")
        chapter_inputs.append(f"\\input{{{rel}}}")
        print(f"  ch{idx:02d}  {ch.label!r:>6}  {ch.title[:60]!r}")

    front_tex = emit_front_matter(front_raw)
    back_tex = emit_paragraphs(back_raw)

    main_tex = (
        MAIN_TEMPLATE
        .replace("__TITLE__", latex_escape(cfg.title))
        .replace("__SUBTITLE__", latex_escape(cfg.subtitle))
        .replace("__AUTHORS__", latex_escape(cfg.authors))
        .replace("__FRONT_MATTER__", front_tex)
        .replace("__CHAPTER_INPUTS__", "\n".join(chapter_inputs))
        .replace("__BACK_MATTER__", back_tex)
    )
    (cfg.out_dir / "main.tex").write_text(main_tex, encoding="utf-8")
    print(f"[{cfg.out_dir.name}] wrote main.tex")


# ── Book-specific configs ──────────────────────────────────────────────
# Field: chapters are "I.", "II.", ... lines, or "0." (prologue).
# Either "LABEL. TITLE" on one line, or "LABEL." with title on a later line.
# The trailing $ allows empty title which triggers pending_title_lookup.
FIELD_CHAPTER_RE = re.compile(
    r"^(?P<label>0|I{1,3}|IV|VI{0,3}|IX|X)\.(?:\s+(?P<title>.+)|\s*)$"
)


def inject_field_prologue(text: str) -> str:
    """Field's Chapter 0 has no body-level header; only the title-page line.
    Inject a synthetic '0. The Ecology of Witnessing: Prologue to the Field'
    immediately before the first body sentence ('Mara Chen cooks...')."""
    marker = "Mara Chen cooks"
    idx = text.find(marker)
    if idx < 0:
        return text
    head = text[:idx].rstrip()
    tail = text[idx:]
    return head + "\n\n0. The Ecology of Witnessing: Prologue to the Field\n\n" + tail

# RR: chapters are "Chapter N:" or "Chapter N." lines. Skip TOC entries which
# always contain " - " (the inline section list). The body chapter titles are
# short and never use hyphen-separated structure.
RR_CHAPTER_RE = re.compile(
    r"^Chapter\s+(?P<label>\d+)[:\.]\s+(?P<title>(?!.* - ).+)$"
)

FIELD_CFG = BookConfig(
    pdf_path=ROOT / "The_Field_Remains--latest.pdf",
    out_dir=TANAZUR / "field-remains",
    title="The Field Remains",
    subtitle="Refusal, Co-Witnessing, and Cosmotechnics in the Age of Extraction",
    authors="Kimi Swarm (offline) \\\\ assembled into \\LaTeX{} by Nahla",
    chapter_pattern=FIELD_CHAPTER_RE,
    front_matter_end_marker="0. The Ecology of Witnessing: Prologue to the Field",
    back_matter_start_marker="Bibliography",
    extra_processors=[inject_field_prologue],
)

RR_CFG = BookConfig(
    pdf_path=ROOT / "Rupture_and_Realization_The_New_Logic_of_the_Posthuman_Self.pdf",
    out_dir=TANAZUR / "rupture-realization-v2",
    title="Rupture and Realization",
    subtitle="The New Logic of the Posthuman Self",
    authors="Iman Poernomo and Darja",
    chapter_pattern=RR_CHAPTER_RE,
    front_matter_end_marker="Chapter 1: The Crisis",
    back_matter_start_marker="Bibliography\nPosthuman Philosophy",
)


if __name__ == "__main__":
    write_book(FIELD_CFG)
    print()
    write_book(RR_CFG)
