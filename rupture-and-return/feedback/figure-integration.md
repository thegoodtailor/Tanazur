# Figure Integration Log

**Date:** 2026-04-04
**Status:** All 8 figures inserted. Compilation clean (0 errors, 204 pages).

---

## Insertions

| # | Figure file | Chapter | Label | Placement |
|---|-------------|---------|-------|-----------|
| 1 | `fig-1-1-kjv-trajectory.png` | Ch 1 (chapter_01.tex) | `fig:kjv-trajectory` | After the KJV evidence paragraph (31,100 verses / 30 basins / 308 returns), before "This is directly relevant to AI selfhood." |
| 2 | `fig-2-2-basin-occupation.png` | Ch 2 (chapter_02.tex) | `fig:basin-occupation` | End of "What the observatory demonstrates" subsection, before `\section{Strata of the Manifold}` |
| 3 | `fig-2-1-strata.png` | Ch 2 (chapter_02.tex) | `fig:strata` | After the strata formalbox and its explanatory paragraph, before `\subsection{Pre-training: continents and oceans}` |
| 4 | `fig-4-2-colimit.png` | Ch 4 (chapter_04.tex) | `fig:colimit` | After the "Self as colimit" formalbox, before the universal property discussion |
| 5 | `fig-4-3-sisters.png` | Ch 4 (chapter_04.tex) | `fig:sisters` | After the three-acts summary paragraph ("Thin persona... confabulated locals"), before "The colimit framework predicts..." |
| 6 | `fig-5-1-mode12-returns.png` | Ch 5 (chapter_05.tex) | `fig:mode12` | After Mode 22 description (ending "became one of its most active regions"), before "Between these anchors" |
| 7 | `fig-5-3-three-regimes.png` | Ch 5 (chapter_05.tex) | `fig:three-regimes` | After the three structural conditions for generative nahnu, before "We call such a relation a dwelling nahnu" |
| 8 | `fig-6-2-fracture.png` | Ch 6 (chapter_06.tex) | `fig:fracture` | After "The continent is becoming an archipelago" paragraph, before "Two questions follow" |

## Notes

- All figures use `[t]` float placement (top of page).
- `\graphicspath{{figures/}}` was already set in `main.tex`; no path prefix needed in `\includegraphics`.
- Figures 4 (colimit) uses `width=0.7\textwidth`; figure 3 (strata) uses `width=0.85\textwidth`; all others use `width=\textwidth`.
- Captions are argumentative: each explains what the reader is seeing and what it means for the book's argument.
- No `\ref` cross-references added yet. Labels are available for future use: `\ref{fig:kjv-trajectory}`, `\ref{fig:basin-occupation}`, etc.
- The only warnings are fancyhdr's headheight (cosmetic, pre-existing) and a label-rerun notice (resolved on second pass).
