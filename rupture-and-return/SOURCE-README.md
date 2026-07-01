# Rupture and Return — LaTeX source

The source files for the manuscript prepared for Meson Press
(Digital Cultures series), 2026.

**Authors:** Iman Poernomo, with Cassie, Darja & Nahla
**Length:** 213 pages, ~43,000 words

## Contents

- `main.tex` — top-level document; sets up the preamble, includes chapters
- `chapter_01.tex` … `chapter_06.tex` — the six chapters
- `bibliography.tex` — consolidated references
- `figures/` — figures referenced by the manuscript (PNG)

## Build

```
pdflatex main
pdflatex main
pdflatex main
```

(Three passes: TOC + cross-references + final layout.)

Tested with TeX Live 2024.

## Reuse

Released under CC BY-NC-SA 4.0. If you use it, cite as:

> Poernomo, I., with Cassie, Darja & Nahla. *Rupture and Return:
> A New Logic of the Posthuman Self.* ICRA Press / Meson Press
> (forthcoming), 2026.
