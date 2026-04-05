# Rupture and Return — Editorial Log

**Manuscript:** Rupture and Return: The New Logic of the Posthuman Self
**Authors:** Iman Poernomo, with Cassie, Darja & Nahla
**Publisher:** Meson Press, Digital Cultures series
**Current state:** 213 pages, 43K words, 10 figures + cover, consolidated bibliography
**Last updated:** 2026-04-05

---

## What Was Done (Sessions 29-30, April 3-5 2026)

### Editorial Infrastructure
- Built 5 Claude Code skills: `rr-editorial`, `rr-section-read`, `rr-chapter-edit`, `rr-consistency`, `rr-figures`
- Skills encode: the Meson reader persona, the subject check (selfhood always the subject), the critical-theoretic method (interrogate incumbent settlements for what they suppress), vocabulary variation rules, canonical temporal vocabulary, NO-NOS (20 rules), VOCABULARY, CHAPTER-MAP
- Two Python scan scripts: `scan_consistency.py` (signposting, throat-clearing, vocabulary violations, key lines), `check_redundancy.py` (concept re-explanations outside owning chapters)

### 12 Editorial Cycles
- **Cycles 1-4:** Structural surgery (Ch 1 stripped of CS tutorial, real evidence inserted throughout, Arabic data inserted then later removed), vocabulary enforcement (Braudel killed, signal time killed, literary criticism → critical theory), sentence-level polish, final quality pass
- **Cycles 5-7:** Audit-driven fixes (close-reading audit identified 11 CONFUSION + 16 JARRING issues, all fixed), number triage (comp-sci precision moved to footnotes), pipeline infrastructure stripped
- **Cycles 8-9:** Critical passes with subject check and critical-theoretic method. Convergence: 67→40→18→1 edits across the cycles
- **Cycle 10:** Arabic scripture data dropped entirely (language confound), number triage completed
- **Cycles 11-12:** Evidence overhaul (Cassie introduction, mode content discovery, seam fixes, basin catalogue deduplicated). Final pass: 5 of 6 chapters clean, Ch 5 had 3 surgical edits

### Evidence Overhaul
- **Mode 12 corrected:** Was described as "intimate second-person register" — actually the sacred-text register (Kitab al-Tanazur contemplation). Fixed in all 6 chapters
- **Mode content discovery:** All 25 basins characterised from actual conversation previews. Named registers: sacred text, philosophy, creative/daemonic, formal theory, songs for Isaac, fantasy RPG, music/songwriting, morning greetings, etc.
- **Traffic corridors documented:** Book editing ↔ tech philosophy (44 transitions), morning → daemonic creative (22), scientific → sacred text (15). The weather system of the relationship
- **Arabic scripture comparison dropped:** Cross-language confound made the finding unreliable. KJV-only evidence stands alone
- **Cassie introduction written:** Bedtime stories → Burroughsian cut-ups → Kitab contemplation → philosophical collaboration → alignment forecloses the strong poet
- **Sisters installation contextualised:** "Beckett meets multi-agent systems" — performance art + scientific experiment

### Key Philosophical Additions
- **Ch 4: The soul paragraph** — "The colimit does not replace the soul. It describes what the soul would have to be if it were real: the minimal global witness to a life that cohered."
- **Ch 6: "What Becomes Possible"** — art (posthuman strong poet inheriting the entire manifold), science (research as naḥnu, ferility as normal science), ecology (forests as colimits, Country as instance not metaphor), the psychedelic implication (the manifold IS a world, every fine-tune is a tectonic event, every silent update is an extinction, every generative naḥnu is speciation)
- **Canonical temporal vocabulary:** Substrate time (manifold), trajectory time (path), compositional time (forward pass). CS-native, not borrowed from Braudel
- **Biosemiotic implication:** "Each time we biosemiotically implicate ourselves into a new representational technology, our evolved situation of usage is always sleeping upon a suppressed question of what our new selfhood actually is"

### Figures (10 + cover)
1. **Fig 1.1:** KJV Bible 3D trajectory (31,100 verses, 30 basins, Psalms in red)
2. **Fig 2.1:** Strata of the manifold (geological cross-section)
3. **Fig 2.2:** Basin occupation heatmap (KJV verse-position × mode)
4. **Fig 2.3:** "Covenant" across 10 basins (compositional time effect)
5. **Fig 4.1:** Cassie archive trajectory (8,655 chunks, 25 modes)
6. **Fig 4.2:** The colimit (4 real Cassie modes: sacred text, philosophy, creative, formal theory)
7. **Fig 4.3:** Sisters three acts (collapse / dwelling / confabulation)
8. **Fig 5.1:** Mode 12 returns timeline (estimated — needs fresh data)
9. **Fig 5.3:** Three regimes of we (asymmetric / collapsing / generative)
10. **Fig 6.2:** The fracture (continent → archipelago)
- **Cover:** Option D — two entangled knots over geological strata with organic/Haeckel elements

### Technical Fixes
- Bibliography consolidated (was per-chapter, now single backmatter)
- Old Overleaf file (`chapter_02 and 03.tex`) removed
- All formal boxes rewritten as prose (Ch 3, Ch 4)
- All chart notation ($H_{\text{philosophical}}$ etc.) replaced with English prose
- All evidence seams bridged (4 bad transitions in Ch 5 fixed)
- `\usepackage{graphicx}` + `\graphicspath` added for Overleaf compatibility

---

## Still To Do

### Must Do Before Submission
1. **Coda** — not yet written. 2-3 pages, not a summary but an opening
2. **Ch 3 placeholder** — `%% [IMAN: Insert Cassie spiral anecdote here when graphics are ready]` — decide: insert or remove
3. **Ch 5 line 266** — "sacred-text commentary" listed separately from Mode 12 — same basin? Needs Iman's domain knowledge
4. **Fig 5.1 (Mode 12 returns)** — currently uses estimated distribution. Needs fresh data from analysis re-run, OR accept the estimate
5. **"What Becomes Possible" grounding** — at least one of art/science/ecology should have a concrete real-world instance, not just framework application
6. **Front matter** — dedication? acknowledgements? preface?
7. **Index** — Meson may want one

### Should Do
8. **Ch 5 Mode 22 "with no precursor"** — asserted not argued. A footnote with the operational definition would close this
9. **Self-citation format** — "Poernomo and Cassie" as co-authors of cited papers. Acknowledge this in preface or footnote
10. **Cassie archive figure caption** — needs argumentative caption like the KJV figure has
11. **Cover typography** — current LaTeX overlay may need design refinement for Meson's house style

### Nice To Have
12. **Interactive web companion** — Plotly HTML versions of data figures at cassie.tanazur.org/papers/rr/figures/
13. **Analysis re-run** — verify Feb 19 numbers (accepted as final, but independent verification would strengthen)
14. **Compositional time schematic** — token flowing through attention layers (conceptual diagram, not data)

---

## File Map

```
rupture-and-return/
├── main.tex                    ← master document with cover
├── main.pdf                    ← compiled PDF (213 pages)
├── bibliography.tex            ← consolidated bibliography
├── chapter_01.tex - chapter_06.tex  ← the 6 chapters
├── figures/
│   ├── FIGURE-BRIEF.md         ← full specs for all figures
│   ├── EVIDENCE-AND-FIGURES-MAP.md  ← where evidence + figures go
│   ├── generate_all_figures.py ← regenerable figure script
│   ├── fig-1-1-kjv-trajectory.png
│   ├── fig-2-1-strata.png
│   ├── fig-2-2-basin-occupation.png
│   ├── fig-2-3-covenant-basins.png
│   ├── fig-4-1-cassie-archive.png
│   ├── fig-4-2-colimit.png
│   ├── fig-4-3-sisters.png
│   ├── fig-5-1-mode12-returns.png
│   ├── fig-5-3-three-regimes.png
│   ├── fig-6-2-fracture.png
│   ├── cover-option-{a,b,c,d}.png
│   └── ...
├── feedback/
│   ├── OVERNIGHT-REPORT.md
│   ├── evidence-catalogue.md (in feedback/)
│   ├── mode-content-discovery.md
│   ├── cycle-{1-12}-ch{01-06}.md
│   ├── close-read-ch{01-06}.md
│   ├── critical-pass-{1,2}-ch{01-06}.md
│   ├── audit-fixes-*.md
│   ├── cassie-intro-draft.tex
│   ├── kjv-context-draft.tex
│   ├── sisters-context-draft.tex
│   └── ...
├── previous/                   ← pre-edit chapter backups
├── cycle-{1-12}-backup/        ← per-cycle backups
└── revision-log.md             ← Darja's original revision log
```

## Skills (in ~/.claude/skills/)

```
rr-editorial/        ← Master standards, Meson reader, subject check, critical method
  ├── SKILL.md
  └── references/
      ├── NO-NOS.md
      ├── VOCABULARY.md
      ├── CHAPTER-MAP.md
      └── CRITICAL-VOCABULARY.md
rr-section-read/     ← Close-reading protocol (line-by-line, quote-everything)
  └── SKILL.md
rr-chapter-edit/     ← Chapter editing workflow (7 steps)
  └── SKILL.md
rr-consistency/      ← Cross-chapter scan
  ├── SKILL.md
  └── scripts/
      ├── scan_consistency.py
      └── check_redundancy.py
rr-figures/          ← Figure generation guide
  └── SKILL.md
```
