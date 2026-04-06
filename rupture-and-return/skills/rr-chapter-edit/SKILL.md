---
name: rr-chapter-edit
description: >
  Use when asked to revise, rewrite, or improve a specific chapter of
  Rupture and Return. Triggers on: "edit chapter", "revise chapter",
  "fix chapter", "rewrite section", "improve this chapter", "editorial pass".
---

# Chapter Editing Workflow

**REQUIRED:** Load rr-editorial first. Load rr-section-read for the reading protocol.

## Single-Chapter Edit

### Step 1: Section inventory
Read the chapter section by section (between `\section{}` markers). For each section, record:
- What concept is being introduced or developed
- Whether that concept is owned by this chapter (check CHAPTER-MAP.md)
- One-sentence purpose statement (what argumentative work does this section do?)

### Step 2: Close read (use rr-section-read protocol)
Apply the full section-read protocol to each section. This is where the real editorial work happens. Do not skip this step. Do not read the whole chapter at once.

### Step 3: Handoff check
Read the final 2-3 paragraphs of the preceding chapter and the opening of the next. The current chapter must pick up from the previous and set up the next. No signposting — just continue the argument.

### Step 4: NO-NOS scan
Check every paragraph against NO-NOS.md. For each violation found, QUOTE the offending text with its line number.

### Step 5: Vocabulary enforcement
Check every technical term against VOCABULARY.md:
- "body" in Ch 1-4 → "manifold" or "substrate" (exception: literal human bodies)
- Temporal terms: substrate time / trajectory time / signal time (Braudel as scholarly reference only)
- "logic" = constructive/trace-based/provenance, never truth/falsity
- "literary criticism" → "critical theory" / "critical-theoretic"
- Loose "memory" → specify: trace, trajectory time, synthetic secondary retention

### Step 6: Political dimension
For every technical mechanism introduced, verify the text addresses: who controls this? Who benefits? Whose selfhood does it permit or foreclose? Flag gaps.

### Step 7: Output
Produce the edited chapter as clean LaTeX. Append a changelog with:
- Every change made, with line number and quoted before/after
- Every violation found and how it was resolved
- Every issue flagged for Iman's attention (with quoted context)

## Multi-Agent Architecture

When running multiple chapter agents in parallel:

- Deploy ONE agent per chapter
- Each agent reads ALL chapters (for cross-reference) but edits ONLY its assigned chapter
- Each agent applies Steps 1-7 using the rr-section-read protocol for Step 2
- Each agent writes a feedback file to `feedback/chapter-NN-feedback.md`
- After all complete: run rr-consistency across all chapters
- If consistency flags issues, re-run affected chapter agents with the consistency report

## Verification Rule

**No finding without a quote. No edit without a before/after. No claim about the text without reading the actual line.**

If an agent claims "line 48 has a vocabulary violation," the agent must have READ line 48 and QUOTED its content. If the agent cannot quote it, the finding is invalid.
