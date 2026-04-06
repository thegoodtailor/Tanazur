---
name: rr-figures
description: >
  Use when generating, updating, or discussing figures for Rupture and Return.
  Triggers on: "figure", "visualization", "graph", "diagram", "plot",
  "basin diagram", "trajectory plot", "generate figures", "book figures".
---

# Rupture and Return — Figure Generation

**REQUIRED:** Read the figure brief before any figure work:
`/home/iman/cassie-project/Tanazur/rupture-and-return/figures/FIGURE-BRIEF.md`

This file contains:
- Every data source with exact file paths
- Every figure spec with manuscript context (quoted passages)
- Aesthetic rules (dark bg, luminous trajectories, Gleick-aesthetic)
- LaTeX integration template
- Caption rules (argumentative, not descriptive)

## When editing the manuscript

If any chapter edit changes a passage that a figure illustrates, UPDATE the FIGURE-BRIEF.md:
1. Find the figure whose "Manuscript context" quotes the changed passage
2. Update the quoted text to match the new version
3. Update the line number reference
4. If the data claim changed (e.g., different numbers), flag the figure for regeneration

## When generating figures

1. Activate venv: `source /home/iman/cassie-project/venv/bin/activate`
2. Output to: `/home/iman/cassie-project/Tanazur/rupture-and-return/figures/`
3. Naming: `fig-N-N-short-name.pdf` (e.g., `fig-1-1-kjv-trajectory.pdf`)
4. Also generate PNG preview and Plotly HTML interactive version where applicable
5. Test: `pdflatex main.tex` must compile with all figures

## Style rules

- Dark backgrounds (#0a0a0a) for data figures
- Light background for pedagogical schematics (colimit diagram only)
- Luminous trajectories (gold/amber/white, thin lines, soft glow)
- Mode colors: muted distinct hues, not neon
- Minimal annotation — the figure speaks, the caption argues
- No matplotlib default styling. Custom rcParams for everything.
- Target: 90s chaos theory popularization (Gleick, Prigogine)
