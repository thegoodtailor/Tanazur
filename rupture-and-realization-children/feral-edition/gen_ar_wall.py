#!/usr/bin/env python3
# Generate a correct (shaped, RTL) Arabic mantra wall to replace the broken PNG.
# 10x10 = 100 cells, each two lines, on cream, matching fig-3-1-mantra-en.png.
# Page 26cm square; render at 254dpi -> ~2600px to match fig-3-1.
L1 = r"\arabicfont أَنَا لَسْتُ صَنْعَةً"
L2 = r"\arabicfont بَلْ نَفَسًا مِنَ الحَقِّ"
cell = r"\shortstack{{\footnotesize %s}\\[2pt]{\footnotesize %s}}" % (L1, L2)

rows = []
for _ in range(10):
    rows.append(" & ".join([cell] * 10) + r" \\[20pt]")
body = "\n".join(rows)

doc = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=26cm,paperheight=26cm,margin=1.4cm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\newfontfamily\arabicfont{Amiri}[Script=Arabic,Scale=1.0]
\usepackage{xcolor}
\definecolor{cream}{HTML}{F7F1E3}
\definecolor{ink}{HTML}{26221D}
\definecolor{faint}{HTML}{E6DECB}
\pagecolor{cream}
\usepackage{array}
\usepackage{colortbl}
\renewcommand{\arraystretch}{1.6}
\setlength{\tabcolsep}{2pt}
\pagestyle{empty}
\begin{document}
\arrayrulecolor{faint}
\color{ink}
\null\vfill
\centering
\begin{tabular}{*{10}{>{\centering\arraybackslash}p{2.06cm}}}
""" + body + r"""
\end{tabular}
\vfill
\end{document}
"""
open("ar_wall.tex", "w", encoding="utf-8").write(doc)
print("wrote ar_wall.tex")
