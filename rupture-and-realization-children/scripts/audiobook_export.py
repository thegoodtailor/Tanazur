#!/usr/bin/env python3
"""
Audiobook mechanical pre-pass.

Turns the LaTeX chapter sources of *Rupture and Realization: Children of the
Tanazur* into clean raw text drafts for the per-chapter VOICING agents.

This script is SYNTACTIC ONLY (markup stripping, macro/diacritic expansion,
whitespace) — exactly the format work the project's Retrieval Imperative permits
regex for. It makes NO semantic judgments:

  * Mathematics is NOT verbalised. Every formula is preserved verbatim inside a
    clear marker so the voicing agent can render a prose isomorph and none get
    silently dropped:
        inline  $x$        ->  «x»
        display \\[ x \\]    ->  a line  ⟦⟦ x ⟧⟧
        align*/equation    ->  ⟦⟦ ... ⟧⟧
  * definition/theorem/proof/... environments keep a spoken label + their body.
  * \\tractprop{1.1}{...}  (Ch 10) -> a line  ⟦PROP 1.1⟧ ...   — the spoken
    numbering ("one point one") is applied by Cassie's dedicated Ch 10 pass.
  * tikzpicture/figure/table/landscape -> dropped, replaced by ⟦DIAGRAM OMITTED⟧
    so the agent can write an audio bridge if the visual was load-bearing.
  * substantive footnotes -> ⟦FOOTNOTE: ...⟧ for the agent to inline or cut;
    citation-only footnotes -> dropped.

Output: audiobook/raw/NN-slug.txt  (a DRAFT, not the deliverable).

Usage:
    python scripts/audiobook_export.py            # all chapters
    python scripts/audiobook_export.py ch-08      # one chapter (substring match)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../rupture-and-realization-children
CH_DIR = ROOT / "chapters"
OUT_DIR = ROOT / "audiobook" / "raw"

# ---------------------------------------------------------------------------
# Custom volume macros (from main.tex preamble). Text-context expansions only;
# inside math markers the raw LaTeX is kept for the agent to read.
# ---------------------------------------------------------------------------
MACROS = {
    r"\OHTT": "OHTT",
    r"\DOHTT": "D-OHTT",
    r"\hocolim": "homotopy colimit",
    r"\Nahnu": "Nahnu",
    r"\coh": "coh",
    r"\gap": "gap",
}

# Spoken labels for math/structural environments (body kept).
ENV_LABELS = {
    "definition": "Definition",
    "theorem": "Theorem",
    "proposition": "Proposition",
    "lemma": "Lemma",
    "corollary": "Corollary",
    "principle": "Principle",
    "remark": "Remark",
    "proof": "Proof",
}
# Environments dropped wholesale (visual apparatus).
ENV_DROP = {"tikzpicture", "figure", "table", "tabular", "landscape", "center"}
# Environments whose body is kept as plain prose, no label.
ENV_PLAIN = {"quote", "quotation", "itemize", "enumerate"}

# LaTeX accent command -> combining codepoint (applied to the wrapped char).
COMBINING = {
    "d": "̣",   # dot below     \d{h} -> ḥ
    "=": "̄",   # macron        \={a} -> ā
    "'": "́",   # acute         \'{e} -> é
    "`": "̀",   # grave
    "^": "̂",   # circumflex
    '"': "̈",   # diaeresis
    "~": "̃",   # tilde         \~{n} -> ñ
    "c": "̧",   # cedilla       \c{c} -> ç
    "v": "̌",   # caron         \v{z} -> ž
    ".": "̇",   # dot above
    "u": "̆",   # breve
    "H": "̋",   # double acute
}

PLACEHOLDER = "\x00MATH{}\x00"   # protect math from later text transforms


def _balanced(s: str, start: int) -> int:
    """Index just past the '}' matching the '{' at s[start]."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(s)


def _strip_wrappers(s: str, commands: tuple[str, ...]) -> str:
    """\\cmd{inner} -> inner, for formatting wrappers, brace-balanced + nested."""
    pattern = re.compile(r"\\(" + "|".join(commands) + r")\s*\{")
    while True:
        m = pattern.search(s)
        if not m:
            return s
        open_brace = m.end() - 1
        close = _balanced(s, open_brace)
        inner = s[open_brace + 1: close - 1]
        s = s[: m.start()] + inner + s[close:]


def _drop_cmd_arg(s: str, commands: tuple[str, ...], keep_last: bool = False) -> str:
    """Remove \\cmd{...}{...}; if keep_last, keep the final argument's content."""
    pattern = re.compile(r"\\(" + "|".join(commands) + r")\s*(\[[^\]]*\])?\s*\{")
    while True:
        m = pattern.search(s)
        if not m:
            return s
        first = m.end() - 1
        close = _balanced(s, first)
        replacement = ""
        # consume any further consecutive {..} args, remembering the last
        last_inner = s[first + 1: close - 1]
        j = close
        while j < len(s) and s[j] == "{":
            nxt = _balanced(s, j)
            last_inner = s[j + 1: nxt - 1]
            j = nxt
        if keep_last:
            replacement = last_inner
        s = s[: m.start()] + replacement + s[j:]


def _apply_accents(s: str) -> str:
    """\\d{h} / \\='{a} / \\c{c} ... -> precomposed Unicode where possible."""
    # \cmd{x}  and  \cmd x  (single token, incl. \i -> dotless i base)
    def repl(m: re.Match) -> str:
        cmd, ch = m.group(1), m.group(2)
        comb = COMBINING.get(cmd)
        if comb is None:
            return m.group(0)
        if ch in (r"\i", r"\j"):
            ch = "i" if ch == r"\i" else "j"
        ch = ch.lstrip("\\")[:1] or " "
        return unicodedata.normalize("NFC", ch + comb)

    s = re.sub(r"\\([dDcv=`'^\"~.uH])\s*\{\s*(\\?[a-zA-Z])\s*\}", repl, s)
    s = re.sub(r"\\([dDcv=`'^\"~.uH])\s+(\\?[a-zA-Z])", repl, s)
    s = s.replace(r"\i", "i").replace(r"\j", "j").replace(r"\ss", "ß")
    return s


def _protect_math(s: str, store: list[str]) -> str:
    """Pull every formula out into `store`, leaving a placeholder behind."""
    def stash(latex: str, display: bool) -> str:
        latex = re.sub(r"\s+", " ", latex.strip())
        rendered = f"⟦⟦ {latex} ⟧⟧" if display else f"«{latex}»"
        store.append(rendered)
        return PLACEHOLDER.format(len(store) - 1)

    # display \[ ... \]
    s = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n" + stash(m.group(1), True) + "\n",
               s, flags=re.DOTALL)
    # equation / align(*) / multline / gather environments
    s = re.sub(r"\\begin\{(equation\*?|align\*?|multline\*?|gather\*?)\}(.+?)\\end\{\1\}",
               lambda m: "\n" + stash(m.group(2), True) + "\n", s, flags=re.DOTALL)
    # inline $ ... $
    s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: stash(m.group(1), False),
               s, flags=re.DOTALL)
    return s


def _handle_footnotes(s: str) -> str:
    """citation-only footnotes -> dropped; substantive -> ⟦FOOTNOTE: ...⟧."""
    while True:
        m = re.search(r"\\footnote\s*\{", s)
        if not m:
            return s
        open_brace = m.end() - 1
        close = _balanced(s, open_brace)
        body = s[open_brace + 1: close - 1]
        prose = _drop_cmd_arg(body, ("cite", "citep", "citet"))
        prose = re.sub(r"[\s~,.;:]+", "", prose)
        replacement = ""
        if len(prose) > 8:                       # has real prose beyond citations
            clean = _drop_cmd_arg(body, ("cite", "citep", "citet")).strip()
            clean = re.sub(r"\s+", " ", clean)
            replacement = f" ⟦FOOTNOTE: {clean}⟧"
        s = s[: m.start()] + replacement + s[close:]


def _handle_environments(s: str) -> str:
    """Labelled math envs -> 'Label.' + body; drop visual envs; flatten the rest."""
    # Visual apparatus -> a marker so the agent can bridge if it was load-bearing.
    # Order matters: outer wrappers (landscape) first so a nested figure/tikz is
    # consumed once, not double-counted.
    for env in ("landscape", "figure", "table", "tabular", "tikzpicture"):
        s = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
                   "\n⟦DIAGRAM OMITTED⟧\n", s, flags=re.DOTALL)
    # center: keep the inner text (could be a centered pull-quote), drop the wrapper.
    s = re.sub(r"\\(begin|end)\{center\*?\}", "\n", s)
    for env, label in ENV_LABELS.items():
        # \begin{env}[Optional Title] body \end{env}
        def repl(m: re.Match, label=label) -> str:
            title = (m.group(1) or "").strip()
            head = f"{label} ({title})." if title else f"{label}."
            return f"\n{head} {m.group(2).strip()}\n"
        s = re.sub(r"\\begin\{" + env + r"\}(?:\[(.*?)\])?(.+?)\\end\{" + env + r"\}",
                   repl, s, flags=re.DOTALL)
    for env in ENV_PLAIN:
        s = re.sub(r"\\(begin|end)\{" + env + r"\}", "\n", s)
    s = s.replace(r"\item", "\n")
    return s


def _handle_tractprop(s: str) -> str:
    """\\tractprop{1.1}{text} -> line  ⟦PROP 1.1⟧ text  (spoken numbering: Ch10 pass)."""
    out = []
    while True:
        m = re.search(r"\\tractprop\s*\{", s)
        if not m:
            out.append(s)
            break
        out.append(s[: m.start()])
        num_open = m.end() - 1
        num_close = _balanced(s, num_open)
        num = s[num_open + 1: num_close - 1].strip().rstrip(".")
        txt_open = num_close
        while txt_open < len(s) and s[txt_open] != "{":
            txt_open += 1
        txt_close = _balanced(s, txt_open)
        text = s[txt_open + 1: txt_close - 1].strip()
        out.append(f"\n⟦PROP {num}⟧ {text}\n")
        s = s[txt_close:]
    return "".join(out)


def _headings(s: str) -> str:
    s = _drop_cmd_arg(s, ("chapter",), keep_last=True)
    for cmd in ("section", "subsection", "subsubsection", "paragraph"):
        s = re.sub(r"\\" + cmd + r"\*?\s*(\[[^\]]*\])?\s*\{",
                   lambda m: "\n\n", s)  # opening; closing brace cleaned below
    # the above leaves the title text followed by a stray '}'; convert \cmd{Title}
    # properly with a balanced pass instead:
    return s


def _sections_balanced(s: str) -> str:
    for cmd in ("chapter", "section", "subsection", "subsubsection", "paragraph"):
        pattern = re.compile(r"\\" + cmd + r"\*?\s*(\[[^\]]*\])?\s*\{")
        while True:
            m = pattern.search(s)
            if not m:
                break
            open_brace = m.end() - 1
            close = _balanced(s, open_brace)
            title = s[open_brace + 1: close - 1].strip()
            s = s[: m.start()] + f"\n\n{title}\n\n" + s[close:]
    return s


def clean_chapter(text: str) -> str:
    s = text
    # 1. comments
    s = re.sub(r"(?<!\\)%.*", "", s)
    # 2. protect math (before any text transform touches $ \ { })
    math_store: list[str] = []
    s = _protect_math(s, math_store)
    # 3. structural environments
    s = _handle_environments(s)
    # 4. tractatus props (Ch 10)
    s = _handle_tractprop(s)
    # 5. footnotes (detect citation-only vs substantive)
    s = _handle_footnotes(s)
    # 6. headings -> plain title lines (balanced)
    s = _sections_balanced(s)
    # 7. drop pure-reference commands
    s = _drop_cmd_arg(s, ("cite", "citep", "citet", "label", "ref", "eqref",
                          "index", "hypertarget", "hyperlink", "providecommand",
                          "vspace", "hspace"))
    s = _drop_cmd_arg(s, ("hypertarget", "hyperlink"), keep_last=True)  # safety
    # 8. unwrap formatting commands (keep inner text)
    s = _strip_wrappers(s, ("emph", "textit", "textbf", "textsf", "texttt",
                            "text", "textsc", "underline", "uline"))
    # 9. custom macros (text context only — math is stashed)
    for mac, rep in MACROS.items():
        s = s.replace(mac, rep)
    # 10. accents / diacritics
    s = _apply_accents(s)
    # 11. spacing & leftover control sequences
    s = s.replace("\\\\", "\n").replace("~", " ")
    s = re.sub(r"\\[,;:!> ]", " ", s)              # \, \; \! \> thin spaces
    s = re.sub(r"\\(quad|qquad|medskip|bigskip|smallskip|noindent|par|centering"
               r"|raggedright|arraybackslash|newpage|clearpage)\b", " ", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(r"\&", "and").replace("&", " ")
    s = re.sub(r"\\[a-zA-Z]+", "", s)              # any surviving bare command
    # 11b. LaTeX typography -> Unicode (text only; math is still stashed).
    s = s.replace("``", "“").replace("''", "”").replace("`", "‘")
    s = re.sub(r"-{3}", "—", s)                    # --- em dash (a beat, per BRIEF §4)
    s = re.sub(r"-{2}", "–", s)                    # --  en dash
    s = re.sub(r"[ \t]+([.,;:!?])", r"\1", s)      # drop space left by removed \cite
    # 12. restore math
    for i, frag in enumerate(math_store):
        s = s.replace(PLACEHOLDER.format(i), frag)
    # 13. whitespace / paragraph normalisation
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    lines = [ln.strip() for ln in s.split("\n")]
    # rebuild paragraphs: blank line = boundary
    out, buf = [], []
    for ln in lines:
        if ln == "":
            if buf:
                out.append(" ".join(buf)); buf = []
        elif ln.startswith("⟦") or ln.startswith("⟦⟦"):
            if buf:
                out.append(" ".join(buf)); buf = []
            out.append(ln)
        else:
            buf.append(ln)
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(p for p in out if p.strip()).strip() + "\n"


def main(argv: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chapters = sorted(CH_DIR.glob("ch-*.tex"))
    if argv:
        chapters = [c for c in chapters if argv[0] in c.name]
    if not chapters:
        print("no matching chapters", file=sys.stderr)
        return 1
    for ch in chapters:
        raw = ch.read_text(encoding="utf-8")
        cleaned = clean_chapter(raw)
        slug = ch.stem.replace("ch-", "")
        out = OUT_DIR / f"{ch.stem[3:]}.txt"
        out.write_text(cleaned, encoding="utf-8")
        n_math = cleaned.count("«") + cleaned.count("⟦⟦")
        n_fn = cleaned.count("⟦FOOTNOTE")
        n_prop = cleaned.count("⟦PROP")
        n_diag = cleaned.count("⟦DIAGRAM")
        print(f"{ch.name:42s} -> {out.relative_to(ROOT)}  "
              f"({len(cleaned):>6d} chars, math:{n_math} prop:{n_prop} "
              f"fn:{n_fn} diag:{n_diag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
