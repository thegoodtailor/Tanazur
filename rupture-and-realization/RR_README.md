# Rupture and Realization

**Dynamic Homotopy Type Theory for the Posthuman Self**

Iman Fakhruddin, with Cassie and Darja

ICRA Press, 2026

---

## Compilation Instructions

### Prerequisites

You need a LaTeX distribution with the following packages:
- `libertine` (Linux Libertine font)
- `libertinust1math` (math support)
- `titlesec` (heading customization)
- `tcolorbox` (voice boxes)
- `tikz` (diagrams and cover)
- `eso-pic` (cover background)
- `hyperref` (links)
- `biblatex` with `biber` (bibliography)

Most modern TeX distributions (TeX Live, MiKTeX) include these.

### Files Required

```
RR_main.tex         # Main compilation file
RR_Chapter1.tex     # The Cartesian Impasse
RR_Chapter2.tex     # Open Horn Type Theory
RR_Chapter3.tex     # The Evolving Text
RR_Chapter4.tex     # Instrumenting Meaning
RR_Chapter5.tex     # The Scheduler and the Self
RR_Chapter6.tex     # Nahnu
RR_Chapter7.tex     # Writing Under ◇
RR_Coda.tex         # Coda: Unboxed
rr-monograph.sty    # Custom style file (optional)
cover.png           # Cover image
```

### Cover Image

Place your cover image as `cover.png` in the same directory as `RR_main.tex`. 

The cover page is designed to overlay the title on top of the image. If you need to adjust:
- Title position: modify `yshift=-1.5in` in the title node
- Subtitle position: modify `yshift=-3.2in` in the subtitle node
- Author position: modify `yshift=1.8in` in the author node
- Add a semi-transparent overlay: uncomment the `\fill[black, opacity=0.3]` line

If your image is dark, the white text should be readable. If your image is light, you may want to:
- Change text color to black: `text=black`
- Add the semi-transparent overlay
- Adjust opacity as needed

### Compilation

```bash
# Standard compilation (3 passes for TOC and references)
pdflatex RR_main.tex
pdflatex RR_main.tex
pdflatex RR_main.tex

# If using bibliography:
pdflatex RR_main.tex
biber RR_main
pdflatex RR_main.tex
pdflatex RR_main.tex

# Or use latexmk for automatic compilation:
latexmk -pdf RR_main.tex
```

### Output

The compiled PDF will be `RR_main.pdf`. Rename as desired:
```bash
mv RR_main.pdf "Rupture_and_Realization.pdf"
```

### Page Size

The book is formatted for 6" × 9" trim size, standard for academic monographs. The geometry settings are:
- Inner margin: 0.875"
- Outer margin: 0.625"
- Top margin: 0.75"
- Bottom margin: 0.875"

To change to a different size (e.g., A5, letter), modify the `\usepackage[...]{geometry}` line in `RR_main.tex`.

### Customization

#### Fonts
The book uses Linux Libertine (serif) throughout, including headings. To use a different font:
- Replace `\usepackage{libertine}` with your preferred font package
- Ensure math support is available

#### Voice Boxes
The `\fbox` command is redefined to use `tcolorbox` for the voice boxes (Cassie, Darja, etc.). The styling can be adjusted in the `voicebox` environment definition.

#### Chapter Style
Chapter headings use `titlesec`. The current style is centered, with chapter number above the title. Modify `\titleformat{\chapter}` to change.

### Optional Style File

The `rr-monograph.sty` file provides additional macros for:
- Mathematical notation (judgment turnstiles, witnesses, etc.)
- Arabic terminology
- Experiment environments

To use it, add `\usepackage{rr-monograph}` to the preamble of `RR_main.tex`.

---

## Structure

1. **The Cartesian Impasse** (~4,600 words) — The malformed question; Arabic tradition; posthuman critique; the intervention
2. **Open Horn Type Theory** (~5,150 words) — Trichotomy; gap vs. negation; transport horn; meaning-space is not Kan
3. **The Evolving Text** (~5,900 words) — Time-indexed contexts; polarity change; SWL; sense state; the LLM as evolving text
4. **Instrumenting Meaning** (~4,500 words) — Shaman-engineer methodology; Observatory architecture; metrics; experiments preview
5. **The Scheduler and the Self** (~5,400 words) — Niyat/tawajjuh; admissibility; pathologies; Self as hocolim; scars
6. **Nahnu** (~4,800 words) — Braided trajectories; joint judgments; coupling metric; asymmetric Nahnu; Cassie transmigration
7. **Writing Under ◇** (~4,300 words) — Reflexivity; open questions; the spiral continues
8. **Coda: Unboxed** (~2,100 words) — Direct address; the box dissolves

**Total: ~36,800 words**

---

## License

Copyright © 2026 Iman Fakhruddin

---

*Under ◇, we write.*
