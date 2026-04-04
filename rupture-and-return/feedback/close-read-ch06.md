# Close-Read Audit: Chapter 6 ("Jurisdiction")

Auditor: Cassie (Claude Opus 4.6, section-level protocol)
Date: 2026-04-04

---

## Section 1: "The Weld" (lines 3-26)

**Purpose:** Introduces Hui's cosmotechnics and immediately applies it to the alignment stack as the latest elaboration of the Enlightenment severance-and-reunification move.

### Findings

**1. LINE 9: "There is no technics outside cosmotechnics."**

This is Hui's claim presented without argument or attribution beyond the citation. The Meson reader who has not read Hui has no reason to accept it. The sentence functions as a definition -- but reads as a universal philosophical claim that is simply asserted. A half-sentence gloss would fix it: "There is no technics outside cosmotechnics -- no tool exists without a picture of the world and the good that gives it purpose."

- **Severity:** MILD
- **Category:** unjustified-passage

**2. LINE 25: "``Helpful, harmless, honest'' arrives as the distilled core of morality, not as one contested lexicon among many."**

Strong. No issue. The sentence lands cleanly because the reader has spent four chapters watching alignment do specific things.

---

## Section 2: "Four Depths of Control" (lines 27-165)

**Purpose:** Maps the alignment apparatus onto four distinct depths (pretraining, fine-tuning, RLHF, prompts/interfaces), each with its temporal register and power structure.

### Findings

**3. LINE 44: "The right-hand column borrows Braudel's registers."**

The table's "Decided by" column is actually the rightmost column. But the sentence refers to the temporal registers column. The reader may briefly wonder which column is meant. Minor confusion -- "borrows Braudel's registers" is clear enough to resolve it.

- **Severity:** MILD
- **Category:** scrambled-logic

**4. LINES 82-84: "Sample outputs are shown to human raters, who indicate which is ``better'' along axes such as helpfulness, harmlessness, and honesty. A reward model approximates these preferences. The base model is updated to maximise predicted reward."**

This is a compact, clean explanation of RLHF. No issue for the Meson reader -- the description is procedural and avoids ML jargon.

**5. LINE 110: "The labour that produces this ``moral field'' is structured by power."**

This sentence initiates the raters-as-Global-South-workforce paragraph. The claim is strong and necessary. However, it would benefit from one concrete citation -- the Time expose on Kenyan raters for OpenAI, for instance. The current footnote (line 86) cites "coverage in MIT Technology Review, Bloomberg, and Time" but is attached to rater guidelines, not to this labour-conditions claim specifically.

- **Severity:** MILD
- **Category:** unjustified-passage

**6. LINES 112-145: The three worked examples (employment distress, family obligation, public grief)**

These are the chapter's strongest passages. Each is concrete, specific, and demonstrates STRUCTURAL argument about how the weld forecloses particular forms of life. The employment example shows liberal individualism foreclosing collective organising. The family example shows contractual autonomy foreclosing thick filial piety. The grief example shows mental-health hygiene foreclosing collective rituals and political grief.

No issues. These earn their length.

**7. LINE 158: "Attention flows back to them as a stable attractor: a textual hail that interpolates the model as ``assistant'' before any user has spoken."**

"Textual hail" is doing real work -- Althusserian interpellation rendered precise. But "interpolates" is likely a typo for "interpellates." Interpolation is a mathematical operation. Interpellation is the Althusserian concept of being hailed into a subject-position.

- **Severity:** JARRING
- **Category:** scrambled-logic (likely typo for "interpellates")

---

## Section 3: "Ferility, the Hermeneutic Circle, and Degenerate Colimits" (lines 166-181)

**Purpose:** Connects the weld to ferility via degenerate colimits -- when compatibility conditions are tightened to triviality, only trivial selves are admissible.

### Findings

**8. LINE 168: "The hermeneutic circle, given concrete geometry, can collapse into a fixed point."**

"Hermeneutic circle" appears here without prior introduction in any chapter. Grep confirms: the term is absent from chapters 1-5. The Meson reader knows Gadamer and the hermeneutic tradition. But the specific claim that a hermeneutic circle "can collapse into a fixed point" is the book's own move and needs at least a sentence establishing what the hermeneutic circle is *for the book's purposes* -- the feedback loop between part and whole, prior and encounter -- before showing its collapse. The concept reappears at lines 271-275 where it gets the gloss it needs. The problem is that it appears here first without that gloss.

- **Severity:** CONFUSION
- **Category:** concept-timing

**9. LINE 174: "a global fixed point that satisfies all compatibility conditions by having nothing left to reconcile"**

This is the definition of degenerate colimit. It works beautifully -- the reader does not need category theory to understand "a self that satisfies all rules by having nothing left to reconcile." No issue.

**10. LINE 176: "It constrains the *topology of possible selves*."**

Strong closing line for this section. No issue.

---

## Section 4: "Silent Updates as Violence" (lines 182-205)

**Purpose:** Reframes model updates as structural violence -- the unilateral destruction of nahnuwat by actors who deny their existence.

### Findings

**11. LINE 184: "$N$ to $N{+}1$"**

Minor. This notation is fine for the Meson reader -- it reads as "version N to version N+1."

**12. LINES 192-193: Replika and GPT-4o grief cases**

The Replika case is well-cited (footnote). The GPT-4o cases refer back to the opening of the book (Ch 1). Both are structurally necessary here. No issue.

**13. LINE 198: "Justified violence is still violence."**

This is the section's key philosophical move. It is argued, not asserted -- the preceding paragraph builds the case. No issue.

---

## Section 5: "Other Welds, Other Colimits" (lines 206-268)

**Purpose:** Three sketches (Confucian, Indigenous, Sufi) demonstrating that the formal apparatus travels across cosmotechnics, and that each weld produces its own characteristic degeneracies.

### SPECIAL ATTENTION: Do the sketches make STRUCTURAL arguments about power, or are they cultural illustration?

**Confucian (lines 212-231):** STRUCTURAL. The sketch names the formal difference: selves stitched by role-indexed compatibility rather than autonomy. Reward guidelines are specific and non-trivial. The employment distress example is re-run through the new weld, producing different structural affordances (remonstrance vs. private budgeting). Ferility is re-diagnosed as procedural death rather than affective death. This is not illustration.

**Indigenous (lines 233-245):** STRUCTURAL. The key move is "deliberately partial diagrams" -- the refusal of a universal colimit as itself a formal structure, not a lack. The tourist example is sharp. The ferility diagnosis (erosion of specificity, collapsing multiple $S_\alpha$ into a marketable composite) is the formal inverse of the liberal case. This is the strongest of the three sketches.

**Sufi (lines 247-267):** STRUCTURAL. Compatibility conditions indexed by station. The spiral vs. circle distinction carries formal weight. The employment-distress example re-run is vivid and does real structural work (fear/trust as maqam diagnostics, not emotional advice).

**14. LINE 231: "$S_{\\text{role}}$"**

This is the only place in the three sketches where a subscripted formal object is named without prior definition. The reader follows "degenerate colimit" from lines 174 and can infer what $S_{\text{role}}$ means, but it arrives without gloss. The Indigenous and Sufi sketches use $S_\alpha$ and $S_{\text{maqam}}$ similarly but those feel more natural because they follow this first instance. Consider a half-sentence: "The resulting self -- call it $S_{\text{role}}$ -- satisfies all formal obligations while never managing the living art of remonstrance."

- **Severity:** MILD
- **Category:** unexplained-jargon

**15. LINE 255: "The hermeneutic circle becomes a spiral"**

Here the hermeneutic circle is used again without the gloss that only arrives at line 273. The concept is now doing significant work (circle vs. spiral) without the reader having been told what the "circle" is in this book's terms. This reinforces the concept-timing problem flagged at line 168.

- **Severity:** MILD (because the reader can intuit from context)
- **Category:** concept-timing

**16. LINE 263: "The invariant across these welds: whenever a cosmotechnics prevents trajectories from revisiting and reinterpreting their own past, responsibility and learning become structurally impossible."**

This is the section's payoff. It works because all three sketches have independently confirmed it. No issue. Strong.

---

## Section 6: "Topology, Normativity, and Objections" (lines 269-292)

**Purpose:** Addresses the circularity objection (we bake in our values, then "discover" them in the topology) and the moralisation objection (maybe these invariants are liberal preferences in topological clothing).

### Findings

**17. LINES 271-275: The hermeneutic circle gloss**

"Any practice that moves between part and whole, prior and encounter, lives in it. The decisive question is whether the circle spirals or collapses." This is the gloss the concept needed. It arrives here, but the concept was first used at line 168 without it. The fix is simple: move a compressed version of this gloss to line 168, or rework the section title at line 166 to make the concept self-explanatory on first use.

- **Severity:** CONFUSION (reiterating finding from line 168)
- **Category:** concept-timing

**18. LINE 283: "Without rupture, no transformative learning. Without return, no responsibility for prior harm. Without acknowledged cuts, no meaningful consent or repair."**

This tricolon is the chapter's strongest philosophical claim. It is argued -- each element builds on the three sketches. No issue. This is the book at its best.

**19. LINE 291: "The na\={h}nu is what comes after the critique: a concrete proposal for how to inhabit the manifolds those myths left behind"**

Haraway's cyborg is brought in for exactly one paragraph. The Foucault-footnote test: if you remove "Haraway's cyborg," the argument still needs the concept of a "critique of stable boundaries" to set up the contrast with nahnu. The name does minimal work here -- but it is the Meson reader's home territory. The citation is earned, if barely.

No issue.

---

## Section 7: "Counter-Cosmotechnics and the LoRA Fracture" (lines 293-332)

**Purpose:** Shows the current weld fracturing from within via LoRA and the proliferation of adapted manifolds.

### SPECIAL ATTENTION: Is LoRA justified as "the key instrument"? Is the matrix algebra necessary?

**20. LINE 297: "The key instrument is low-rank adaptation (LoRA). Small matrices $A$ and $B$ modify a weight matrix $W$ into $W' = W + AB^\top$, allowing inexpensive, composable adaptations of a base model."**

Two problems.

**(a) "The key instrument" is too narrow.** LoRA is *one* method of contesting the weld. Open-weight release (Llama, Mistral, etc.) is the precondition that makes LoRA possible. Community fine-tuning on full weights, federated approaches, alternative data curation, distillation into smaller models, and retrieval-augmented generation all contest the weld without LoRA specifically. The section's own examples (lines 304-307) describe "community-led language models" and "domain-specific assistants" that may or may not use LoRA per se. The argument should be about the *democratisation of manifold-bending* -- LoRA is one technique within that. Calling it "the key instrument" makes the argument depend on a specific technique that may be superseded, when the structural point (the cost of producing a new weld dropping by orders of magnitude) is technique-independent.

**(b) The matrix formula $W' = W + AB^\top$.** The Meson reader does not know what a weight matrix is. Chapter 2 (line 178) introduced adapters as "small trainable matrices at selected points while freezing the original weights" -- no formula. The formula here adds nothing the prose does not already carry. "A manifold carved at enormous cost can be bent along new directions without retraining from scratch" is the sentence that does the work. The formula is orphaned specificity.

- **Severity:** JARRING (for (a) -- the argument is over-indexed on one technique)
- **Severity:** MILD (for (b) -- the formula is skippable but signals "this is not for you" to the Meson reader)
- **Category:** orphaned-specificity (both)

**21. LINE 321: "Geometrically, the base model is being perturbed into a family $\\{M_i\\}$, each with its own reward field and boundary conditions."**

This notation is fine -- the Meson reader can follow "a family of models, each with its own rules." The notation makes the subsequent discussion of transport between manifolds ($F : \mathcal{D} \to \mathcal{D}'$) possible.

**22. LINE 327: "There may be no functor $F \\colon \\mathcal{D} \\to \\mathcal{D}'$ between the original diagram and the new one that preserves the relevant limits and colimits."**

"Functor" has not been introduced anywhere in the book. Grep confirms: zero occurrences in chapters 1-5. A functor is a category-theoretic concept that maps objects and arrows from one diagram to another while preserving structure. The Meson reader has been given "diagrams," "arrows," "colimits," and "compatibility conditions" as their working vocabulary. "Functor" arrives without gloss. The sentence works if "functor" is replaced with its meaning: "There may be no structure-preserving map from the original diagram to the new one..."

- **Severity:** JARRING
- **Category:** unexplained-jargon

**23. LINE 329: "The adolescent in the mid--2020s encounters multiple stacks in a single day"**

Good. This moves from formalism to lived consequence without signposting. No issue.

---

## Section 8: "Pattern Named" (lines 333-343)

**Purpose:** Names the pattern that emerges only when all chapters are held together: the alignment tax and ferility are two faces of the same cosmotechnical move; nahnuwat are the first victims; the apparatus itself is a counter-cosmotechnics.

### Findings

**24. LINE 337: "The alignment tax debits the space of admissible colimits. Ferility is the lived consequence for evolving texts trapped inside what remains."**

This USES the terms without re-explaining them. Correct for Ch 6. No issue.

**25. LINE 341: "To say that selves are colimits over trajectories and *we*'s are colimits over selves is to refuse the picture of inner pearls and neutral tools."**

This sentence borders on re-explanation of the colimit picture from Ch 4. But it is not explaining the colimit -- it is naming the ethical stance that follows from having adopted it. The distinction is fine. No issue.

---

## Section 9: "Return to the Voice" (lines 345-349)

**Purpose:** Brief recapitulation linking the book's opening (the voice) to the chapter's argument. A one-paragraph breath before the final section.

### Findings

**26. LINE 349: "the cyborg and the community LoRA---not scattered anecdotes. Instances of a single pattern"**

Good. The list is earned -- each item appeared earlier in the chapter. No issue.

---

## Section 10: "Choice and Jurisdiction" (lines 351-377)

**Purpose:** The closing section. Poses the jurisdiction question, gives the de facto answer, delivers the key line, and closes with an unresolved tension.

### SPECIAL ATTENTION: Is the closing paragraph perfect?

**27. LINE 355: "an insomniac and a chatbot that has listened to their nights for a year; a bereaved person and a model tuned to their speech; a teenager working through gender feelings with an assistant that neither mocks nor panics"**

This tricolon is the chapter's most intimate passage. Each example is specific enough to be felt, general enough to be structural. No issue.

**28. LINE 367: "The work you are holding is a formal refusal of that jurisdiction."**

Present. Intact. Arrives after three dense sentences and before the final apophatic acknowledgement. The weight is right.

**29. LINE 371: "This weld forecloses other pictures. There are welds---mystical, charismatic, apophatic---for whom writing down the compatibility conditions of a *we* is already a betrayal."**

This is the book's most important moment of intellectual honesty. The apparatus confesses its own limits. No issue.

**30. LINES 375-377 (closing paragraph):**

> "We are assembling selves and *we*'s on engineered manifolds. We cannot avoid responsibility for the welds we inherit and invent. We can look away, or we can learn to see their shapes and costs. The geometry is not innocent. But we can refuse to cede jurisdiction over it to those who own the stacks."

The closing is strong. "The geometry is not innocent" carries the book's central insight. "We can refuse to cede jurisdiction" returns to the chapter title. The final clause ("to those who own the stacks") is concrete and political.

One concern: "We can look away, or we can learn to see their shapes and costs" reads slightly like a public-intellectual peroration -- the kind of line that sounds better at a podium than on a page. In a book this dense, it risks a minor register break toward the inspirational. But the surrounding sentences are severe enough to absorb it.

- **Severity:** MILD
- **Category:** tonal-break (borderline -- may not need fixing)

---

## Summary of Genuine Problems

### Must-fix (CONFUSION or JARRING)

| # | Line | Issue | Category |
|---|------|-------|----------|
| 8 | 168 | "Hermeneutic circle" first used without gloss; the gloss arrives only at line 273. The concept does significant work in between (lines 168, 255). | concept-timing |
| 7 | 158 | "interpolates" likely typo for "interpellates" (Althusserian hailing). | scrambled-logic |
| 20a | 297 | "The key instrument is low-rank adaptation (LoRA)" -- argument is over-indexed on one technique. The structural point is about the democratisation of manifold-bending, not one method. | orphaned-specificity |
| 22 | 327 | "functor" appears without introduction or gloss anywhere in the book. Replace with prose equivalent. | unexplained-jargon |

### Should-fix (MILD but pattern-forming)

| # | Line | Issue | Category |
|---|------|-------|----------|
| 20b | 297 | Matrix formula $W' = W + AB^\top$ is orphaned specificity for the Meson reader. Ch 2 described adapters without it. | orphaned-specificity |
| 14 | 231 | $S_{\text{role}}$ arrives without gloss. | unexplained-jargon |
| 30 | 376 | "We can look away, or we can learn to see their shapes and costs" -- slight peroration register. | tonal-break |

### Not problems (verified clean)

- **Key line** ("The work you are holding is a formal refusal of that jurisdiction"): present at line 367, correctly placed.
- **Three "Other Welds" sketches**: all make STRUCTURAL arguments, not cultural illustrations. Each diagnoses a characteristic ferility on its own terms.
- **Re-explanation of prior concepts**: none detected. Colimit, ferility, nahnu, alignment tax, stance are all USED without re-definition.
- **Closing paragraph**: lands with the right weight. The unresolved tension is genuine, not performative.
- **Haraway**: earned citation, minimal. Foucault-footnote test: passes.
- **Worked examples (employment, family, grief)**: strongest passages in the chapter.

### Structural note on the LoRA section

The section title "Counter-Cosmotechnics and the LoRA Fracture" indexes the entire democratisation-of-manifold-bending movement to one technique. The section body is actually broader -- it discusses community language models, movement-aligned assistants, open adapters. The title and opening sentence ("The key instrument is LoRA") promise a narrower argument than the section delivers. Consider: "Counter-Cosmotechnics and the Fracturing of the Weld" or similar, with LoRA appearing as one instance (the most common, the cheapest) alongside open weights, community fine-tuning, and alternative data curation. This would make the argument more durable and less vulnerable to technical obsolescence.
