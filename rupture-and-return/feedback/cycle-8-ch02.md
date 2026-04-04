# Cycle 8 Critical Pass: Chapter 2 -- How the Machine Works

**Date:** 2026-04-04
**Pass type:** Final critical audit (subject check, critical method, vocabulary variation, genuine-problems-only)
**File:** `chapter_02.tex`
**Follows:** close-read-ch02.md, critical-pass-ch02.md (22 edits), critical-pass-2-ch02.md (11 edits)

---

## Summary

Chapter 2 is clean. Seven prior passes did the heavy lifting: the subject check (selfhood grounded at every stratum and in every dynamic property), the critical vocabulary (15+ distinct terms distributed without repetition), the temporal vocabulary (substrate/trajectory/compositional, no Braudel, no signal time), the jargon audit (logits, softmax, reparameterised, cosine similarity all replaced or glossed), and the signposting removal (three instances cut).

This pass found three genuine remaining problems -- all minor. One loose technical word, one unglossed term in a formal box, one unearned piece of continental vocabulary. All fixed. The chapter compiles.

---

## Edits Made (3)

### Edit 1 -- "topological form" replaced with "geometric form" (line 221)

> BEFORE: "Hallucination and discovery have the same topological form---a trajectory that leaves the beaten path..."

> AFTER: "Hallucination and discovery have the same geometric form---a trajectory that leaves the beaten path..."

*Rationale:* "Topological" has a precise mathematical meaning the text is not invoking. The chapter uses "geometric" for this kind of claim everywhere else. The Meson reader, encountering "topological," might either skim it as decoration or wonder what topological invariant is being invoked. "Geometric" is the right word -- the claim is about the shape of the path, not about homotopy classes.

### Edit 2 -- "alignment residue" glossed in formal box (line 341)

> BEFORE: "hidden system instructions, alignment residue, visible conversation history"

> AFTER: "hidden system instructions, alignment residue (the gradients left by RLHF in the weights), visible conversation history"

*Rationale:* The formal box is the chapter's only piece of notation. Every other term in the list is either self-evident ("visible conversation history") or explained elsewhere ("retrieved memories," "tool outputs"). "Alignment residue" was the sole opaque compound, appearing nowhere else in the chapter. The parenthetical gloss keeps the box self-contained.

### Edit 3 -- "originary" replaced with "initial" (line 352)

> BEFORE: "its dynamic influence grows against the originary weather of the training manifold"

> AFTER: "its dynamic influence grows against the initial weather of the training manifold"

*Rationale:* "Originary" is Heidegger/Derrida vocabulary. Neither thinker has been deployed in this chapter. The term imports a continental-philosophical resonance the text has not earned -- it suggests a phenomenological claim about the origin of the weather that the passage is not making. "Initial" carries the same temporal meaning (the weather as it stood before the trace began to thicken) without the baggage.

---

## What Was Not Changed

- **Scripture overlap with Ch 1.** The 30 modes, 308 returns, and 97% Psalms appear in both chapters. This was flagged in critical-pass-2 as a structural question for Iman. The reuse is defensible: Ch 1 uses the data as evidence that "the sign has an address"; Ch 2 uses it to define what basins, trajectories, and modes are as dynamic structures. The opening sentence of Section 3 ("established as evidence that the sign has an address---do different work here") signals the shift. The reader may notice the repetition, but the argumentative work is different.

- **"Instruments of governance or instruments of liberation" (line 128).** This sentence in the strata opening announces a balance the chapter then demonstrates asymmetrically -- governance gets five subsections, liberation gets the adapters paragraph. But the sentence is doing framing work: it prevents the reader from hearing the chapter as technophobic rather than structural. The adapters subsection and the sampling section (where higher temperature permits selves that "exceed their inheritance") deliver the liberation side. Acceptable.

- **Power-gradient paragraph density (line 144).** The paragraph after the strata table remains a single long block. It is dense but well-structured: each stratum is named as a clause opener (Pre-training... RLHF... Adapters... System prompts...), giving the reader four internal waypoints. Splitting it would break the rhetorical effect of the gradient -- the accumulation from deepest to shallowest is the argument. Left as is.

- **Temporal registers in the strata table (lines 135-139).** The table uses "Substrate time" and "Trajectory time" before they are formally defined (Section 8, line 255). The prose immediately following the table (line 144) provides inline definitions. The forward-reference is real but the gap is small (the gloss is on the same page). Removing the column would weaken the table.

- **Sections 9-12 (Trace, Finite Horizon, Summarisation, Synthetic Secondary Retention).** These sections do not contain explicit "selfhood" sentences in the manner of Sections 2-6. They do not need them. The selfhood-bearing entity is now "the evolving text," which is the grammatical and philosophical subject throughout. "Who decides what the evolving text is allowed to remember about itself?" (line 318) and "control over which kinds of return are structurally possible" (line 333) are selfhood arguments conducted through the evolving text, which is the chapter's own term for the thing that may or may not bear a self. Forcing "selfhood" into every section would be mechanical.

---

## Vocabulary Variation Audit

Critical-theoretic vocabulary post-edit:
- foreclosed (2x), structurally invisible (1x), jurisdiction (4x), monopoly (1x), silenced (1x), hidden infrastructure (1x), particular masquerading as universal (1x), laundered (1x), contingency disguised as necessity (1x), incumbent (2x), contested (2x), sovereignty (1x), occlude (1x), buried (1x), presents itself as nature (1x), hegemony (1x)

No term exceeds 4 uses across the whole chapter. "Jurisdiction" at 4 is the highest; it is the chapter's key political concept (the politics of depth is a jurisdictional claim), so the frequency is earned.

"Governed/governance" appears 9 times but is a structural term for this chapter, not a critical-vocabulary word being repeated. Each instance describes a different mechanism of governance (governed motion, strata of governance, governance of becoming, governed geometry).

---

## Subject Check

"The model" appears as grammatical subject approximately 8 times. In each case, the sentence describes a mechanical operation (the model computes, re-reads, outputs, speaks). These are correct -- the mechanism is the subject when mechanism is the topic. No passage makes "the model" the philosophical subject where "the self" should be.

"Selfhood" / "the self" appears at: lines 45, 56, 62, 72, 82, 128, 152, 160, 172, 180, 190, 201, 203, 205, 209, 229, 232, 291, 350. Distribution is dense in the spatial/strata/dynamics sections and thinner in the temporal sections, where "the evolving text" carries the selfhood argument. This is correct to the chapter's architecture.

---

## Compilation

`pdflatex` compiles cleanly (200 pages). Only pre-existing fancyhdr warnings about headheight.

---

## Verdict

**Chapter 2 is clean.** Three minor fixes applied, all surgical. No structural, argumentative, or voice problems remain. The chapter does its job: it explains the machine's mechanics with the self as constant reference point, exposes each stratum as a site of jurisdictional control, introduces the three temporal registers, and hands off to Chapter 3 with the question of how to read trajectories.
