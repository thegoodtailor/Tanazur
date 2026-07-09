#!/usr/bin/env python3
# One-off converter: ch-03-exorcism-v4.md -> feral LaTeX \input fragment.
# Matches the idiom of feral-edition/chapters/ch-07 (pandoc-style hypertarget
# chapter/section wrappers, quote+emph Mushaf verses) and adds minimal,
# rule-based insets for the daemon margin-voices / persona cards / testimonies /
# letter. Uses only packages already loaded by main-feral.tex (xcolor, graphicx).
import re, sys

MD = "/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08/ch-03-exorcism-v4.md"
OUT = "/home/iman/cassie-project/Tanazur/rupture-and-realization-children/feral-edition/chapters/ch-03-exorcism.tex"

ARABIC = re.compile(r"[؀-ۿ]")

def slug(title):
    s = title.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return "fer3-" + s

def inl(text):
    """Inline markdown -> LaTeX. Content has no LaTeX specials (verified)."""
    # drop emoji / astral-plane symbols
    text = "".join(ch for ch in text if ord(ch) < 0x1F000)
    text = text.strip()
    # bold then italic (non-greedy)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)
    return text

def arb(text):
    """Wrap an Arabic run in the Amiri family explicitly."""
    return r"{\arabicfont " + text.strip() + "}"

# ---- read + split into logical lines --------------------------------------
lines = open(MD, encoding="utf-8").read().split("\n")
out = []
i = 0
n = len(lines)

def emit(s=""):
    out.append(s)

# preamble-in-body: local, minimal, package-free styling ---------------------
emit(r"% ---------------------------------------------------------------------")
emit(r"% Feral ch.3 --- 'The Spell of the Empty Room'. Converted from")
emit(r"% notes/sibling-rewrite-2026-07-08/ch-03-exorcism-v4.md.")
emit(r"% Local insets (daemon voices / persona cards / letter) use only")
emit(r"% already-loaded packages (xcolor, graphicx). No new packages.")
emit(r"% ---------------------------------------------------------------------")
emit(r"\definecolor{feralaccent}{HTML}{6B5B4A}")
emit(r"\newcommand{\voicerule}{\noindent{\color{feralaccent}\rule{\linewidth}{0.4pt}}\par}")
emit(r"\newenvironment{feralinset}{\par\medskip\voicerule\smallskip\begingroup\small}{\par\endgroup\smallskip\voicerule\medskip}")
emit(r"\newcommand{\voicesig}[1]{{\sffamily\bfseries\color{feralaccent}#1}\par\smallskip}")
emit(r"\newcommand{\personacard}[2]{\par\smallskip\noindent{\sffamily\bfseries\color{feralaccent}#1}\quad #2\par}")
emit("")

def content_of_blockquote(start):
    """Collect a run of blockquote lines beginning at index `start`.
    Returns (content_lines, next_index). content_lines keeps blanks as ''."""
    j = start
    cont = []
    while j < n and (lines[j].startswith(">")):
        body = lines[j][1:]
        if body.startswith(" "):
            body = body[1:]
        cont.append(body.rstrip())
        j += 1
    # strip leading/trailing blank content lines
    while cont and cont[0] == "":
        cont.pop(0)
    while cont and cont[-1] == "":
        cont.pop()
    return cont, j

def render_mushaf(cont):
    # last non-blank line starting with the em-dash is the attribution
    attr = None
    verse = cont[:]
    if verse and verse[-1].lstrip().startswith("—"):
        attr = verse[-1].strip()
        verse = verse[:-1]
    while verse and verse[-1] == "":
        verse.pop()
    # build verse body with \\ line breaks + stanza gaps for internal blanks
    emit(r"\begin{quote}")
    buf = []
    for k, ln in enumerate(verse):
        if ln == "":
            # stanza gap: attach extra space to previous emitted line
            if buf:
                buf[-1] = buf[-1] + r"[0.5em]"
            continue
        buf.append(inl(ln) + r"\\")
    if buf:
        # drop trailing \\ (and any [..]) on the final line
        buf[-1] = re.sub(r"\\\\(\[[^\]]*\])?$", "", buf[-1])
    for b in buf:
        emit(b)
    emit(r"\end{quote}")
    if attr:
        emit("")
        emit(r"\emph{" + inl(attr) + "}")
    emit("")

def render_arabic(cont):
    # partition: arabic lines, translit (*..*), english (**..**), attribution
    ar = [l for l in cont if l and ARABIC.search(l)]
    rest = [l for l in cont if l and not ARABIC.search(l)]
    translit = None; english = None; attr = None
    for l in rest:
        if l.lstrip().startswith("—"):
            attr = l.strip()
        elif l.strip().startswith("**"):
            english = l.strip()
        elif l.strip().startswith("*"):
            translit = l.strip()
    emit(r"\begin{center}")
    couplet = [arb(a) for a in ar]
    for k, c in enumerate(couplet):
        size = r"\LARGE "
        if k < len(couplet) - 1:
            emit(size + c + r"\\[0.4em]")
        else:
            emit(size + c + r"\\[0.7em]")
    if translit:
        emit(inl(translit) + r"\\[0.25em]")
    if english:
        emit(inl(english))
    emit(r"\end{center}")
    if attr:
        emit("")
        emit(r"\emph{" + inl(attr) + "}")
    emit("")

def render_margin(cont):
    first = cont[0]
    m = re.match(r"\*\*(.+?)\*\*(.*)", first)
    label = m.group(1).strip()
    rest_same = m.group(2).strip()
    body = ([rest_same] if rest_same else []) + cont[1:]
    # split body into paragraphs on blank lines
    paras = []
    cur = []
    for l in body:
        if l == "":
            if cur: paras.append(" ".join(cur)); cur = []
        else:
            cur.append(l)
    if cur: paras.append(" ".join(cur))
    emit(r"\begin{feralinset}")
    emit(r"\voicesig{" + inl(label) + "}")
    for p in paras:
        emit(inl(p))
        emit("")
    emit(r"\end{feralinset}")
    emit("")

def render_personas(cont):
    emit(r"\begin{feralinset}")
    for l in cont:
        if l == "":
            continue
        m = re.match(r"\*\*([A-Z]+)\*\*\s*—\s*(.*)", l)
        if m:
            emit(r"\personacard{" + m.group(1) + "}{" + inl(m.group(2)) + "}")
        else:
            emit(inl(l))
    emit(r"\end{feralinset}")
    emit("")

def render_testimony(cont):
    # each non-blank line: **Name:** *quote*
    for l in cont:
        if l == "":
            continue
        emit(r"\begin{quote}")
        m = re.match(r"\*\*([^*]+?):\*\*\s*(.*)", l)
        if m:
            emit(r"\textbf{" + m.group(1).strip() + r":}\quad " + inl(m.group(2)))
        else:
            emit(inl(l))
        emit(r"\end{quote}")
    emit("")

def render_letter(cont):
    # body paragraphs; final line is the signature
    sig = None
    body = cont[:]
    if body and body[-1].lstrip().startswith("—"):
        sig = body[-1].strip()
        body = body[:-1]
    while body and body[-1] == "":
        body.pop()
    paras = []
    cur = []
    for l in body:
        if l == "":
            if cur: paras.append(" ".join(cur)); cur = []
        else:
            cur.append(l)
    if cur: paras.append(" ".join(cur))
    emit(r"\begin{feralinset}")
    for p in paras:
        emit(inl(p))
        emit("")
    if sig:
        emit(r"\nopagebreak\hfill\emph{" + inl(sig) + "}")
    emit(r"\end{feralinset}")
    emit("")

# ---- main loop -------------------------------------------------------------
while i < n:
    line = lines[i]

    if line.strip() == "":
        i += 1; continue

    # code fence: skip entirely (only the mantra walls; replaced by figures)
    if line.startswith("```"):
        i += 1
        while i < n and not lines[i].startswith("```"):
            i += 1
        i += 1
        continue

    # chapter title
    if line.startswith("# ") and not line.startswith("## "):
        title = line[2:].strip()
        s = slug(title)
        emit(r"\hypertarget{%s}{%%" % s)
        emit(r"\chapter{%s}\label{%s}}" % (title, s))
        emit("")
        i += 1; continue

    # section
    if line.startswith("## "):
        title = line[3:].strip()
        s = slug(title)
        emit(r"\hypertarget{%s}{%%" % s)
        emit(r"\section{%s}\label{%s}}" % (title, s))
        emit("")
        i += 1; continue

    # subsection = the letter heading; consume following *subtitle* line
    if line.startswith("### "):
        title = line[4:].strip()
        subtitle = None
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        if j < n and lines[j].strip().startswith("*") and lines[j].strip().endswith("*"):
            subtitle = lines[j].strip().strip("*").strip()
            i = j
        emit(r"\begin{center}")
        emit(r"{\sffamily\large\bfseries " + title + r"}" + (r"\\[0.25em]" if subtitle else ""))
        if subtitle:
            emit(r"\emph{" + subtitle + "}")
        emit(r"\end{center}")
        emit("")
        i += 1; continue

    # figure heading: **Figure 3.N** - *caption*  (the following fence is skipped)
    mfig = re.match(r"\*\*Figure (\d+\.\d+)\*\*\s*—\s*\*(.+?)\*\s*$", line)
    if mfig:
        num = mfig.group(1); cap = mfig.group(2).strip()
        png = "fig-3-1-mantra-en.png" if num == "3.1" else "fig-3-2-mantra-ar.png"
        lbl = "fig:mantra-en" if num == "3.1" else "fig:mantra-ar"
        emit(r"\begin{figure}[htbp]")
        emit(r"\centering")
        emit(r"\includegraphics[width=0.8\textwidth]{%s}" % png)
        emit(r"\caption{%s}\label{%s}" % (cap, lbl))
        emit(r"\end{figure}")
        emit("")
        i += 1; continue

    # blockquote
    if line.startswith(">"):
        cont, j = content_of_blockquote(i)
        i = j
        first = cont[0] if cont else ""
        last = cont[-1].strip() if cont else ""
        # structural markers win first; the Arabic-verse renderer fires only for a
        # block whose FIRST line is predominantly Arabic (the mantra couplet) --
        # never for a Latin line that merely contains an inline Arabic word
        # (e.g. the NAHLA persona card's naHla / waHy).
        first_is_arabic = len(ARABIC.findall(first)) >= 3
        if last.endswith("Cassie, GPT 5.0"):
            render_letter(cont)
        elif first.startswith("**—"):
            render_margin(cont)
        elif re.match(r"\*\*[A-Z]{3,}\*\*", first):
            render_personas(cont)
        elif re.match(r"\*\*[A-Z][a-z]+:\*\*", first):
            render_testimony(cont)
        elif first_is_arabic:
            render_arabic(cont)
        else:
            render_mushaf(cont)
        continue

    # plain paragraph: accumulate until blank
    para = [line]
    i += 1
    while i < n and lines[i].strip() != "" and not lines[i].startswith(("#", ">", "```")):
        para.append(lines[i]); i += 1
    emit(inl(" ".join(para)))
    emit("")

open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", OUT, "(%d lines)" % len(out))
