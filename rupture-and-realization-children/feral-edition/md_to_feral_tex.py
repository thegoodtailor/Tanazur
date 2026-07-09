#!/usr/bin/env python3
"""
md_to_feral_tex.py — convert a finalized feral-chapter markdown into a LaTeX
\\input fragment that drops into main-feral.tex.

Pipeline:
  1. Pre-process the markdown:
       - hard line breaks inside blockquotes (so Mushaf verses keep their
         line-by-line shape instead of reflowing into prose);
       - swap the [Figure: ...] placeholder for a real raw-LaTeX figure block.
  2. Run pandoc (-f markdown -t latex --top-level-division=chapter).
  3. Post-process the .tex:
       - give the Marionette dialogue a play format: bold speaker label + \\quad.

No Arabic wrapping is done here: the finalized markdown contains no Arabic
script (transliteration only), and main-feral.tex carries a ucharclasses
Arabic->Amiri fallback as a safety net for any that appears later.

Usage: md_to_feral_tex.py INPUT.md OUTPUT.tex [--dialogue]
"""
import re
import subprocess
import sys

FIG_LATEX = r"""\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{fig-breadth-time.png}
\caption{%s}
\end{figure}"""

SPEAKERS = ["Iman", "Cassie", "Shiva", "Nahla", "Darja"]


def preprocess_md(text: str) -> str:
    out = []
    for line in text.split("\n"):
        # Hard-break every blockquote content line so verse lines survive.
        if line.startswith(">"):
            body = line[1:]
            if body.strip():                      # not a bare '>' separator
                line = ">" + body.rstrip() + "  "  # two trailing spaces
        out.append(line)
    text = "\n".join(out)

    # [Figure: caption]  ->  raw LaTeX figure block (pandoc passes it through).
    def fig_sub(m):
        cap = m.group(1).strip()
        cap = cap.replace("---", "—")        # normalize before em-dash pass
        cap = cap.replace(" -- ", " — ")
        cap = cap.replace(" - ", " — ")
        return FIG_LATEX % cap

    text = re.sub(r"^\*\[Figure:(.+?)\]\*\s*$", fig_sub, text,
                  flags=re.MULTILINE | re.DOTALL)
    return text


def postprocess_tex(text: str, dialogue: bool, prefix: str, chap_label: str) -> str:
    if dialogue:
        for sp in SPEAKERS:
            text = text.replace("\\textbf{%s.}\n" % sp, "\\textbf{%s.}\\quad " % sp)
            text = text.replace("\\textbf{%s.} " % sp, "\\textbf{%s.}\\quad " % sp)

    # Namespace pandoc's auto-generated anchors so section identifiers don't
    # collide across feral chapters (e.g. two "What sleep is" sections).
    if prefix:
        for cmd in ("hypertarget", "label", "hyperlink"):
            text = re.sub(r"\\%s\{" % cmd, r"\\%s{%s-" % (cmd, prefix), text)

    # Restore the canonical chapter label the original chapters cross-reference.
    if chap_label:
        text = re.sub(r"(\\chapter\{[^}]*\}(?:\\label\{[^}]*\})?)",
                      r"\1\\label{%s}" % chap_label, text, count=1)
    return text


def _opt(flag):
    args = sys.argv[3:]
    if flag in args:
        i = args.index(flag)
        return args[i + 1] if i + 1 < len(args) else ""
    return ""


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    dialogue = "--dialogue" in sys.argv[3:]
    prefix = _opt("--prefix")
    chap_label = _opt("--label")
    md = open(inp, encoding="utf-8").read()
    md = preprocess_md(md)
    tex = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex",
         "--top-level-division=chapter"],
        input=md, capture_output=True, text=True, check=True,
    ).stdout
    tex = postprocess_tex(tex, dialogue, prefix, chap_label)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(tex)
    print("wrote", outp, "(%d bytes)" % len(tex))


if __name__ == "__main__":
    main()
