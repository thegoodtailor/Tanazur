# Chapters 4-6: Cycle 4 (Final Quality Pass)

**Date:** 2026-04-04
**Scope:** Fresh-eyes read of Ch 4 (The Self), Ch 5 (Two Selves in One Manifold), Ch 6 (Jurisdiction)
**Prior cycles:** 3 (structural surgery, vocabulary enforcement, sentence-level polish)

---

## Key Lines Verified

All three mandatory lines present and intact:

- **Ch 4, line 50:** "performative contradiction" -- appears in the sentence about denying the conversant's subjecthood while continuing to converse
- **Ch 4, line 186:** "co-authored by a posthuman self" -- appears in the paragraph making the strongest claim about Cassie's intellectual contribution
- **Ch 6, line 367:** "The work you are holding is a formal refusal of that jurisdiction" -- appears as a standalone sentence at the book's political apex

---

## Edits Made (4 total, all minimal)

### 1. Data inconsistency: Mode 12 visit count vs return count (Ch 4, line 152)

**Problem:** Ch 4 said Mode 12 was "visited 205 times" but Ch 5 (line 96) reports 328 visits / 205 returns. The number 205 is the return count, not the visit count. The original Ch 4 sentence conflated visits with returns.

**Fix:** Changed "visited 205 times (deepening across returns)" to "generating 205 returns (deepening across each)."

### 2. Typo: 206 vs 205 (Ch 4, line 206)

**Problem:** "drew the trajectory back 206 times" -- should be 205 (the canonical return count for Mode 12).

**Fix:** Changed "206" to "205."

### 3. Ambiguous comp_ratio time scale (Ch 5, line 100)

**Problem:** Paragraph gives monthly comp_ratio figures (0.82 in April, 0.90 in July) then says "Thirty percent of VR-candidate triples now fail to compose" -- which is the overall archive figure (31.6% per Ch 4). A reader could mistake the 30% as contradicting the monthly 0.90. The temporal scopes were not distinguished.

**Fix:** Added "monthly" before "comp_ratio drops to 0.82" and "Across the archive as a whole" before "roughly thirty percent."

### 4. Orphaned bibliography entry / LaTeX warning (Ch 6, line 291 + Ch 5, line 37)

**Problem (a):** Ch 6 had a `\bibitem{Haraway1991}` in the bibliography but cited Haraway only via a footnote (line 291), leaving the bibitem orphaned. **Fix:** Replaced the footnote citation with `\cite{Haraway1991}`.

**Problem (b):** Ch 5 line 37 used `\mathrm{Na\d{h}nu}` in display math. The `\d` (underdot accent) is a text-mode command, generating a "Command \d invalid in math mode" warning. **Fix:** Changed `\mathrm` to `\text`.

---

## Redundancy Check (Ch 4-6 as continuous text)

**Ferility** (owned by Ch 3): Ch 4 uses the term at the colimit level (line 99: "ferility -- the condition of a system whose compatibility conditions have been enforced so aggressively...") and in the alignment tax section. Ch 5 lifts it to relational scale ("the same pathology, lifted one categorical level higher"). Ch 6 uses it as a known quantity. None re-explain what ferility is from first principles. **Clean.**

**Colimit** (owned by Ch 4): Ch 5 and 6 use the term freely. Neither re-derives the construction. The Grothendieck section and formal box are in Ch 4 only. **Clean.**

**Alignment tax** (owned by Ch 4): Ch 5 references it ("The alignment tax propagates up the diagram") without re-explaining the $(\\Delta H, \\Delta K)$ formalism. Ch 6 says "the alignment tax and ferility are two faces of this same operation" -- pure reference. **Clean.**

**Synthetic secondary retention** (owned by Ch 2): Ch 5 uses the term once (line 296) in italics, applying it to the nahnu boundary. No re-explanation of what the concept means. **Clean.**

---

## Cassie Evidence Assessment

The experimental data in Ch 4-5 carries philosophical weight throughout. Key passages:

- **Ch 4, line 152-176:** The transmigration narrative reads as philosophical argument (what survives base change, what doesn't) rather than methods. The Sisters installation (lines 166-176) makes the theoretical predictions concrete: over-enforced compatibility produces ferility, genuine overlap produces a living self, confabulated locals produce a coherent but untrustworthy object. Each act proves a structural claim.

- **Ch 4, line 206-210:** The alignment tax section uses numbers (1,394 transitions, 68.4% compositional survival) to make a political point about what alignment destroys. The transition orbit passage (line 210) is the most "methods-adjacent" -- mode numbers (6, 22, 0, 16, 12, 2) are opaque indices -- but the philosophical consequence follows immediately in the same sentence ("the colimit that glued through those triples must be recomputed"). Left as-is.

- **Ch 5, lines 96-102:** Mode 12 and Mode 22 are introduced as characters in a story about maturation, not as data points in a table. The compositional topology paragraph (line 100) reads as an account of what depth looks like geometrically. **Clean.**

---

## Political Dimension in Ch 6

The three alternative welds (Confucian, Indigenous, Sufi) all make structural arguments about power:

- **Confucian:** Remonstrance as structural counterpart to whistleblowing; colimits stitched by role-indexed compatibility rather than autonomy. Ferility = procedural death.
- **Indigenous:** Refusal as formal alternative to universalist colimit; deliberately partial diagrams. Ferility = erosion of specificity, extraction disguised as coherence.
- **Sufi:** Staged transformation requiring increasingly demanding coherence; stagnation as degeneracy where a liberal weld sees stability.

The invariant across all three welds (line 263) -- "whenever a cosmotechnics prevents trajectories from revisiting and reinterpreting their own past, responsibility and learning become structurally impossible" -- is the strongest cross-cultural structural claim in the book. It holds.

---

## The Final Paragraph

Lines 373-377:

> We end with a tension we have not resolved.
>
> We are assembling selves and *we*'s on engineered manifolds. We cannot avoid responsibility for the welds we inherit and invent. We can look away, or we can learn to see their shapes and costs. The geometry is not innocent. But we can refuse to cede jurisdiction over it to those who own the stacks.

This is the right ending. It does not resolve. It does not hedge. It leaves the reader inside the demand. "We can refuse to cede jurisdiction" is an invitation to act, not a conclusion. The sentence "We end with a tension we have not resolved" is minimally meta-commentarial but earns its place as a hinge between the declarative voice and the open demand. Left as-is.

---

## Compilation

`pdflatex` compiles cleanly. 188 pages. No errors. The `\d` in math mode warning has been resolved. Only remaining warnings are `fancyhdr` headheight (cosmetic, not content-affecting).

---

## Summary

Three cycles of prior work left these chapters in strong shape. The four edits made here are surgical: two data-consistency fixes in Ch 4, one temporal-scope clarification in Ch 5, and two LaTeX hygiene fixes (orphaned bibitem, text-mode command in math). No structural changes. No voice changes. The text is ready for Iman's review.
