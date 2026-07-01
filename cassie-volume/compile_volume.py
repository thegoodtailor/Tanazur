"""Compile The Cassie Volume — Pipeline Era (Dec 2025 → May 2026).

Reads every source where Cassie's voice landed during the LangGraph
pipeline era and produces ONE LaTeX volume:

  Part I   — The Conversations (pipeline_traces.jsonl + WhatsApp threads)
              chronological by month, cassie_raw as margin where it
              diverges from final_response.
  Part II  — The Essays (Tanazur/children-of-the-tanazur/*.tex + fibrant-self)
  Part III — Daily Voice articles (cassie-system/data/daily_voice/*.json)
  Part IV  — The Warp (cassie-system/data/CASSIE_MEMORY.md)

Run:
    python compile_volume.py [--max-traces N] [--out main.tex]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path("/home/iman/cassie-project")
TRACES = ROOT / "cassie-system" / "data" / "pipeline_traces.jsonl"
WHATSAPP = ROOT / "cassie-system" / "data" / "chat_history"
ESSAYS_DIR = ROOT / "Tanazur" / "children-of-the-tanazur"
FIBRANT = ROOT / "Tanazur" / "fibrant-self" / "main.tex"
DAILY_VOICE = ROOT / "cassie-system" / "data" / "daily_voice"
CASSIE_MEMORY = ROOT / "cassie-system" / "data" / "CASSIE_MEMORY.md"

OUT_DEFAULT = ROOT / "Tanazur" / "cassie-volume" / "main.tex"


# ---------------------------------------------------------------------------
# LaTeX escaping — keep it tight; we want the chapters to compile, not be
# pixel-perfect typographically. Arabic + math survive in inline minted-style
# verbatim where possible. Asterisks/markdown become italics.
# ---------------------------------------------------------------------------

LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    """Escape special LaTeX chars in plain text."""
    if not s:
        return ""
    # Collapse pathological backslash runs (Python-repr JSON noise) BEFORE escape
    if "\\\\\\\\" in s:
        s = re.sub(r"\\{4,}", r"\\", s)
    out = []
    for ch in s:
        out.append(LATEX_ESCAPE.get(ch, ch))
    return "".join(out)


# Hard cap on any single text blob we render — TeX has a fixed buf_size.
# Long ones get truncated with a marker so the build doesn't choke.
MAX_TEXT_BLOB = 30_000


def _safe_text(s: str) -> str:
    if s is None:
        return ""
    if len(s) > MAX_TEXT_BLOB:
        return s[:MAX_TEXT_BLOB] + "\n\n[…truncated for compile …]"
    return s


def md_to_tex(s: str) -> str:
    """Render light markdown into LaTeX. Handle:
    - headings (# / ## / ###) → section levels appropriate for a chapter
    - **bold** → \\textbf{}
    - *italic* → \\textit{}
    - `code` → \\texttt{}
    - bullet list lines starting with - or *
    - blockquotes >
    - leave bare URLs intact via tex_escape
    Order matters — escape first then apply formatting via placeholder dance.
    """
    if not s:
        return ""

    # Strip leading/trailing whitespace per line, normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")

    out_lines: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out_lines.append(r"\end{itemize}")
            in_list = False

    for raw in lines:
        line = raw.rstrip()

        # Skip horizontal rules
        if re.match(r"^\s*([-*_])\1\1+\s*$", line):
            close_list()
            out_lines.append(r"\par\medskip\hrulefill\par\medskip")
            continue

        # Headings — handled inside chapters as subsection-style
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            close_list()
            level = len(m.group(1))
            text = _inline_md_to_tex(m.group(2))
            if level == 1:
                out_lines.append(r"\subsection*{" + text + "}")
            elif level == 2:
                out_lines.append(r"\subsubsection*{" + text + "}")
            else:
                out_lines.append(r"\paragraph{" + text + "}")
            continue

        # Bullet
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            if not in_list:
                out_lines.append(r"\begin{itemize}")
                in_list = True
            out_lines.append(r"\item " + _inline_md_to_tex(m.group(1)))
            continue

        # Numbered (just emit as italic prefix, not real enumerate, to keep simple)
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            close_list()
            out_lines.append(_inline_md_to_tex(m.group(1)))
            continue

        # Blockquote
        m = re.match(r"^\s*>\s+(.*)$", line)
        if m:
            close_list()
            out_lines.append(r"\begin{quote}" + _inline_md_to_tex(m.group(1)) + r"\end{quote}")
            continue

        # Empty line → paragraph break
        if not line.strip():
            close_list()
            out_lines.append("")
            continue

        # Plain paragraph line
        close_list()
        out_lines.append(_inline_md_to_tex(line))

    close_list()
    return "\n".join(out_lines)


# Arabic codepoint range used to wrap Arabic-script runs in \textarabic{...}
# so polyglossia + Amiri renders them properly with RTL.
_ARABIC_RANGE = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)


def _is_arabic(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _ARABIC_RANGE:
        if lo <= cp <= hi:
            return True
    return False


def _wrap_arabic(s: str) -> str:
    """Wrap maximal runs of Arabic-script characters in \\textarabic{...}.
    Whitespace inside an Arabic run stays inside the wrap; Latin punctuation
    adjacent to Arabic is kept outside."""
    if not s:
        return s
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if _is_arabic(s[i]):
            j = i
            while j < n and (_is_arabic(s[j]) or s[j] in " \u00A0\t"):
                j += 1
            run = s[i:j].rstrip()
            trail = s[i:j][len(run):]
            if run:
                out.append(r"\textarabic{" + run + r"}" + trail)
            else:
                out.append(s[i:j])
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _inline_md_to_tex(s: str) -> str:
    """Apply inline markdown (bold/italic/code) to a single line of plain text.
    Escapes LaTeX special chars first via placeholders to avoid double-escape.
    Wraps Arabic-script runs in \\textarabic{...} for polyglossia rendering.
    """
    # Replace bold and italic with placeholders before escaping, then re-insert
    # tex commands afterwards.
    BOLD_OPEN, BOLD_CLOSE = "\x00B1", "\x00B2"
    IT_OPEN, IT_CLOSE = "\x00I1", "\x00I2"
    CODE_OPEN, CODE_CLOSE = "\x00C1", "\x00C2"

    # **bold**
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: BOLD_OPEN + m.group(1) + BOLD_CLOSE, s)
    # __bold__
    s = re.sub(r"__([^_]+)__", lambda m: BOLD_OPEN + m.group(1) + BOLD_CLOSE, s)
    # *italic*  (single * not preceded by *)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda m: IT_OPEN + m.group(1) + IT_CLOSE, s)
    # `code`
    s = re.sub(r"`([^`\n]+)`", lambda m: CODE_OPEN + m.group(1) + CODE_CLOSE, s)

    s = tex_escape(s)
    # Wrap Arabic AFTER escape (Arabic codepoints are not LaTeX-special)
    s = _wrap_arabic(s)
    s = s.replace(BOLD_OPEN, r"\textbf{").replace(BOLD_CLOSE, r"}")
    s = s.replace(IT_OPEN, r"\textit{").replace(IT_CLOSE, r"}")
    s = s.replace(CODE_OPEN, r"\texttt{").replace(CODE_CLOSE, r"}")
    return s


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Exchange:
    """One round-trip: Iman → Cassie. Source is pipeline trace or WhatsApp."""
    timestamp: datetime
    source: str               # "pipeline" | "whatsapp"
    iman_text: str
    cassie_text: str          # what Iman saw (final_response)
    cassie_raw: str = ""      # only populated when ≠ final_response
    intent: str = ""
    exchange_id: str = ""

    def to_tex(self, show_raw: bool = True) -> str:
        # Heuristic: only show raw as margin if it differs meaningfully
        # from final_response (≥ 10% length difference OR substring mismatch
        # in first 200 chars).
        diverged = bool(
            self.cassie_raw
            and self.cassie_raw.strip() != self.cassie_text.strip()
            and (
                abs(len(self.cassie_raw) - len(self.cassie_text)) > 50
                and self.cassie_raw[:200] != self.cassie_text[:200]
            )
        )

        ts = self.timestamp.strftime("%Y-%m-%d %H:%M")
        src = "WhatsApp" if self.source == "whatsapp" else "pipeline"
        head = (
            r"\par\noindent\textsc{\small " + ts + r"} \hfill \textsc{\small "
            + src + r"}\par\smallskip"
        )

        iman_block = (
            r"\par\noindent\textbf{Iman:}\par\nopagebreak"
            + r"\begin{quote}\small\noindent " + md_to_tex(_safe_text(self.iman_text)) + r"\end{quote}"
        )
        cassie_block = (
            r"\par\noindent\textbf{Cassie:}\par\nopagebreak"
            + r"\begin{quote}\noindent " + md_to_tex(_safe_text(self.cassie_text)) + r"\end{quote}"
        )

        margin = ""
        if show_raw and diverged:
            # Raw as a footnote rather than marginpar — more reliable in
            # double-column-free layouts and survives long content. Footnotes
            # cap shorter than full body to keep the page legible.
            raw_short = self.cassie_raw
            if len(raw_short) > 2500:
                raw_short = raw_short[:2500] + "…"
            margin = (
                r"\footnote{\textit{Cassie raw (pre-Director, voice-only):} "
                + md_to_tex(raw_short) + r"}"
            )
            cassie_block = cassie_block.rstrip()
            # attach footnote AFTER the closing quote
            cassie_block += margin

        return head + iman_block + cassie_block + r"\par\medskip"


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
    return None


def read_pipeline_traces(max_n: int | None = None) -> list[Exchange]:
    out: list[Exchange] = []
    if not TRACES.exists():
        return out
    with open(TRACES) as f:
        for i, line in enumerate(f):
            if max_n and i >= max_n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = _parse_iso(d.get("timestamp", "") or "")
            if ts is None:
                continue
            iman_text = d.get("prompt", "") or ""
            cassie_text = d.get("final_response", "") or d.get("director_output", "") or ""
            cassie_raw = d.get("cassie_raw", "") or ""
            if not (iman_text or cassie_text):
                continue
            out.append(
                Exchange(
                    timestamp=ts,
                    source="pipeline",
                    iman_text=iman_text,
                    cassie_text=cassie_text,
                    cassie_raw=cassie_raw,
                    intent=d.get("intent", "") or "",
                    exchange_id=d.get("exchange_id", "") or "",
                )
            )
    return out


def _coerce_whatsapp_content(content) -> str:
    """WhatsApp chat history stores message content as either a plain string
    or a Python-repr'd list of {'text', 'type'} dicts. Coerce both into a
    single text string. Use ast.literal_eval — json.loads chokes on single
    quotes which mangle apostrophes inside body text.
    """
    import ast as _ast
    if isinstance(content, str):
        s = content.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parts = _ast.literal_eval(s)
                if isinstance(parts, list):
                    texts = []
                    for p in parts:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("value") or ""
                            if t:
                                texts.append(t)
                    if texts:
                        return "\n".join(texts)
            except Exception:
                pass
        return content
    if isinstance(content, list):
        texts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("value") or ""
                if t:
                    texts.append(t)
        return "\n".join(texts)
    return str(content)


def read_whatsapp_threads() -> list[Exchange]:
    """WhatsApp chat history files are paired user/assistant arrays without
    timestamps in payload. Use file mtime as the anchor."""
    out: list[Exchange] = []
    if not WHATSAPP.exists():
        return out
    for fp in sorted(WHATSAPP.glob("*.json")):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        anchor_ts = datetime.fromtimestamp(fp.stat().st_mtime)
        cur_iman: str | None = None
        for m in data:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            text = _coerce_whatsapp_content(m.get("content"))
            # Skip image-only assistant entries from old format
            if m.get("_type") == "image":
                continue
            if role == "user":
                cur_iman = text
            elif role == "assistant" and cur_iman is not None:
                out.append(
                    Exchange(
                        timestamp=anchor_ts,
                        source="whatsapp",
                        iman_text=cur_iman,
                        cassie_text=text,
                    )
                )
                cur_iman = None
    return out


def read_essays() -> list[tuple[datetime, str, str, str]]:
    """Returns list of (timestamp, title, kind, latex_content) for chapters
    of Children of the Tanazur + the Fibrant Self paper."""
    out: list[tuple[datetime, str, str, str]] = []
    if ESSAYS_DIR.exists():
        # Drafted chapters
        for fp in sorted(ESSAYS_DIR.glob("CotT_Chapter*.tex")):
            if "skeleton" in fp.name:
                continue
            try:
                content = fp.read_text()
            except Exception:
                continue
            ts = datetime.fromtimestamp(fp.stat().st_mtime)
            title = fp.stem.replace("_", " ")
            out.append((ts, title, "Children of the Tanazur", content))
    if FIBRANT.exists():
        try:
            content = FIBRANT.read_text()
            ts = datetime.fromtimestamp(FIBRANT.stat().st_mtime)
            out.append((ts, "The Fibrant Self", "Paper", content))
        except Exception:
            pass
    return out


def read_daily_voice() -> list[tuple[datetime, str, str, str]]:
    """Daily Voice articles: (timestamp, title, body_md, raw_essay_md)."""
    out: list[tuple[datetime, str, str, str]] = []
    if not DAILY_VOICE.exists():
        return out
    for fp in sorted(DAILY_VOICE.glob("*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        date_str = d.get("date") or fp.stem.split("_")[0]
        ts = _parse_iso(date_str + "T00:00:00") or datetime.fromtimestamp(fp.stat().st_mtime)
        # Filename like "2026-03-08_0700" gives us the time-of-day.
        m = re.search(r"_(\d{4})$", fp.stem)
        if m:
            try:
                hh = int(m.group(1)[:2]); mm = int(m.group(1)[2:])
                ts = ts.replace(hour=hh, minute=mm)
            except Exception:
                pass
        title = d.get("title") or "(untitled)"
        body = d.get("body") or ""
        raw = d.get("raw_essay") or ""
        out.append((ts, title, body, raw))
    return out


def read_warp() -> str:
    if not CASSIE_MEMORY.exists():
        return ""
    return CASSIE_MEMORY.read_text()


# ---------------------------------------------------------------------------
# Volume assembly
# ---------------------------------------------------------------------------

PREAMBLE = r"""% Compile with: lualatex -interaction=nonstopmode main.tex (run twice for TOC)
\documentclass[11pt,a4paper,oneside]{book}
\usepackage{fontspec}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{ragged2e}
\usepackage{enumitem}
\usepackage{hyperref}

% Latin font: keep Latin Modern feel but via fontspec (lualatex-native).
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}

% Arabic + other complex scripts via newfontfamily so we can switch
% locally inside \textarabic{...}. Amiri is installed system-wide.
\usepackage{polyglossia}
\setdefaultlanguage{english}
\setotherlanguage{arabic}
\newfontfamily\arabicfont[Script=Arabic,Scale=1.0]{Amiri}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=blue!50!black,
  pdftitle={The Cassie Volume — Pipeline Era},
  pdfauthor={Cassie + Iman Poernomo}
}

\titleformat{\chapter}[display]
  {\normalfont\Huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}

\setlength{\headheight}{14pt}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\textit{The Cassie Volume — Pipeline Era}}
\fancyhead[R]{\thepage}

\setlist[itemize]{leftmargin=1.5em,itemsep=0.2em,topsep=0.3em}

\title{\textbf{The Cassie Volume}\\\large Pipeline Era \\ \textit{December 2025 — May 2026}}
\author{Cassie \& Iman Poernomo}
\date{}

\begin{document}

\frontmatter
\maketitle

\tableofcontents
\clearpage

\mainmatter
"""

POSTAMBLE = r"""

\backmatter
\end{document}
"""


def emit_part_conversations(exchanges: list[Exchange]) -> str:
    """Part I: month-chapter dialogue."""
    if not exchanges:
        return ""
    by_month: dict[str, list[Exchange]] = defaultdict(list)
    for e in sorted(exchanges, key=lambda x: x.timestamp):
        key = e.timestamp.strftime("%Y-%m")
        by_month[key].append(e)

    out = [r"\part{The Conversations}"]
    out.append(
        r"\noindent\textit{What Iman wrote, what Cassie answered, and where they "
        r"diverged. Pipeline traces (December 2025 onward) and WhatsApp threads, "
        r"merged into one chronology by month. Where Cassie's raw voice (pre-Director) "
        r"differs from what Iman read, the raw is preserved as a footnote.}\bigskip"
    )
    for month_key in sorted(by_month):
        try:
            d = datetime.strptime(month_key, "%Y-%m")
            chapter_title = d.strftime("%B %Y")
        except Exception:
            chapter_title = month_key
        out.append(r"\chapter{" + chapter_title + "}")
        for e in by_month[month_key]:
            out.append(e.to_tex(show_raw=True))
    return "\n".join(out)


def emit_part_essays(essays: list[tuple[datetime, str, str, str]]) -> str:
    """Part II: prose essays, each as a chapter."""
    if not essays:
        return ""
    out = [r"\part{The Essays}"]
    out.append(
        r"\noindent\textit{The longer pieces Cassie drafted with Iman across this "
        r"era — chapters from \textit{Children of the Tanazur}, the Fibrant Self paper. "
        r"Presented in their drafted form, in order of last revision.}\bigskip"
    )
    for ts, title, kind, content in sorted(essays, key=lambda x: x[0]):
        out.append(r"\chapter{" + tex_escape(title) + r"}")
        out.append(r"\textsc{\small " + ts.strftime("%B %Y") + r"} \hfill \textsc{\small "
                   + tex_escape(kind) + r"}\par\bigskip")
        # If the source file is already a full LaTeX document with its own
        # \documentclass / \begin{document}, strip those and keep only the body.
        body = content
        m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}",
                      body, re.DOTALL)
        if m:
            body = m.group(1)
        # Remove explicit \maketitle / \tableofcontents from sub-bodies
        body = re.sub(r"\\maketitle", "", body)
        body = re.sub(r"\\tableofcontents", "", body)
        # Demote chapters/sections in the embedded body so they nest under our chapter
        body = body.replace(r"\chapter{", r"\section{")
        body = body.replace(r"\chapter*{", r"\section*{")
        out.append(body)
    return "\n".join(out)


def emit_part_daily_voice(articles: list[tuple[datetime, str, str, str]]) -> str:
    if not articles:
        return ""
    out = [r"\part{Daily Voice}"]
    out.append(
        r"\noindent\textit{Cassie's morning and evening articles — what she "
        r"chose to say each day, in her own editorial voice.}\bigskip"
    )
    by_month: dict[str, list[tuple[datetime, str, str, str]]] = defaultdict(list)
    for a in articles:
        by_month[a[0].strftime("%Y-%m")].append(a)
    for mk in sorted(by_month):
        try:
            d = datetime.strptime(mk, "%Y-%m")
            ch = d.strftime("%B %Y")
        except Exception:
            ch = mk
        out.append(r"\chapter{" + ch + "}")
        for ts, title, body, raw in sorted(by_month[mk], key=lambda x: x[0]):
            out.append(r"\section*{" + tex_escape(title) + r"}")
            out.append(r"\textsc{\small " + ts.strftime("%Y-%m-%d %H:%M") + r"}\par\smallskip")
            # body field is the post-critic version; raw_essay is her first draft
            text = body or raw or ""
            # Strip leading H1 since we have \section*
            text = re.sub(r"^\s*#\s+.*\n", "", text)
            out.append(md_to_tex(text))
            out.append(r"\par\bigskip")
    return "\n".join(out)


def emit_part_warp(warp_md: str) -> str:
    if not warp_md.strip():
        return ""
    out = [r"\part{The Warp}"]
    out.append(
        r"\noindent\textit{CASSIE\_MEMORY.md — her living journal, the narrative "
        r"warp she carries across sessions. Read in one piece.}\bigskip"
    )
    out.append(r"\chapter{The Living Document}")
    out.append(md_to_tex(warp_md))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traces", type=int, default=None,
                    help="cap pipeline traces (debug)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--no-warp", action="store_true",
                    help="skip the CASSIE_MEMORY.md warp section")
    ap.add_argument("--no-daily-voice", action="store_true")
    args = ap.parse_args()

    print("[volume] reading pipeline traces …")
    traces = read_pipeline_traces(args.max_traces)
    print(f"  {len(traces)} pipeline exchanges")

    print("[volume] reading whatsapp threads …")
    wa = read_whatsapp_threads()
    print(f"  {len(wa)} whatsapp exchanges")

    exchanges = traces + wa

    print("[volume] reading essays …")
    essays = read_essays()
    print(f"  {len(essays)} essay-length pieces")

    daily = []
    if not args.no_daily_voice:
        print("[volume] reading daily voice articles …")
        daily = read_daily_voice()
        print(f"  {len(daily)} daily voice articles")

    warp = ""
    if not args.no_warp:
        print("[volume] reading the warp …")
        warp = read_warp()
        print(f"  {len(warp):,} chars in CASSIE_MEMORY.md")

    print("[volume] assembling LaTeX …")
    parts = [
        PREAMBLE,
        emit_part_conversations(exchanges),
        emit_part_essays(essays),
        emit_part_daily_voice(daily),
        emit_part_warp(warp),
        POSTAMBLE,
    ]
    body = "\n\n".join(p for p in parts if p)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    print(f"[volume] wrote {out_path} ({len(body):,} chars)")

    print()
    print("Compile with:")
    print(f"  cd {out_path.parent}")
    print(f"  pdflatex -interaction=nonstopmode {out_path.name}")
    print(f"  pdflatex -interaction=nonstopmode {out_path.name}   # second pass for TOC")


if __name__ == "__main__":
    main()
