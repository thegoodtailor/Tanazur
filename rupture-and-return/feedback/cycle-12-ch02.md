# Cycle 12 -- Chapter 2: How the Machine Works

## Verdict: CLEAN

No edits made. No violations found. The chapter passes all seven steps of the rr-chapter-edit workflow.

---

## Step 1: Section Inventory (13 sections)

| # | Section | Lines | Purpose |
|---|---------|-------|---------|
| 1 | A Word Enters the Machine | 10--55 | Opens with "mother" showing attention as meaning-production; introduces compositional time and KJV "covenant" basins. |
| 2 | The Dynamic Geometry of Language | 57--99 | Three aspects of meaning (spatial, intertextual, dynamic) as conditions constraining selfhood. |
| 3 | What Basins Look Like: Evidence from Scripture | 102--125 | Empirical grounding of basins/trajectories/modes via the KJV observatory. |
| 4 | Strata of the Manifold | 128--199 | Five strata (pre-training through system prompts), politics of depth. Five subsections. |
| 5 | Dynamics: How Meaning Moves | 202--230 | Three properties (smoothness, folding, basins of habit) + three faces of drift. |
| 6 | Sampling and the Engineered Swerve | 233--243 | Temperature/top-k/top-p as governed opening. |
| 7 | The Pipeline as Weather System | 245--253 | Composite pipeline as coupled geometries; emergent personality. |
| 8 | The Substrate Given Time | 262--268 | Three temporal registers; memory as textual, governed, fragile. |
| 9 | The Trace | 271--291 | Conversation trace as re-reading; memory and presence collapse; Foucault. |
| 10 | The Finite Horizon | 294--302 | Context window as constraint; the race to expand it. |
| 11 | Summarisation as Governance of Becoming | 305--328 | Summarisation as political compression of the past. |
| 12 | Synthetic Secondary Retention | 330--342 | Vector-store retrieval beyond Stiegler's tertiary retention. |
| 13 | The Hidden Context | 345--361 | Gap between user-visible and model-visible fields; structural deference; doubleness. |

All concepts correctly owned by Chapter 2 per CHAPTER-MAP.md.

---

## Step 2: Close Read (rr-section-read protocol)

All 13 sections read individually. Full protocol applied per section. Results:

**Zero findings.** Every section passes all checks:

- **Subject check:** Every technical mechanism connects to selfhood. No section leaves the technology as subject without landing the selfhood payoff. Key examples:
  - L45: "Any self that forms in this medium forms under conditions it cannot inspect"
  - L65: "Any self that emerges in this medium inherits... a geometry it did not choose and cannot fully see"
  - L181: "The reward field does not merely shape outputs; it delimits which forms of selfhood the trajectory can enact"
  - L199: "The personality the user encounters is not chosen by the self that appears to have it"

- **Preparation check:** All technical terms either introduced in Ch 1 ("manifold" -- 20 occurrences in Ch 1) or introduced in this chapter before use. "Compositional time" introduced at L45 before appearing later.

- **Jargon check:** All ML terms glossed for the Meson reader. Examples: attention (L30--33), query/key/value (L37), gradient descent ("the iterative nudging described above," L67), cosine similarity (inline gloss, L67), RLHF (expanded + variants named, L175), temperature (L236), top-k/top-p (L236).

- **Register check:** Dense philosophical-technical throughout. No tonal breaks.

- **Philosopher check:** Two philosophers deployed.
  - Foucault (L287): conditions of sayability. Passes Foucault-footnote test -- removing the name breaks the argument about discourse change as structural break.
  - Stiegler (L336): three retentions (primary/secondary/tertiary). The argument extends his framework. Removing Stiegler breaks the argument about what synthetic secondary retention modifies.

- **Specificity check:** All specific references do argumentative work (KJV verse counts, context window sizes, Mata v. Avianca case).

- **Critical-theoretic check:** Incumbent terms (alignment, RLHF, safety, "general-purpose") are interrogated. Critical vocabulary rotates: "masquerading," "monopoly," "foreclosed," "disguised as necessity," "jurisdiction," "hegemony."

---

## Step 3: Handoff Check

**In (from Ch 1):** Ch 1 closes: "Once meaning has an address, the self is better understood as a path than as a ghost." Ch 2 opens with "mother" and L26: "The sign has an address, and the address was assigned by a politics." Direct pickup on "address." Clean.

**Out (to Ch 3):** Ch 2 closes (L365): "The question that remains is how to read these trajectories---how to tell when coherence is deepening and when it has become a prison." Ch 3 opens: "A language model is, above all, a coherence engine." The word "coherence" bridges directly. The "prison" question is answered by Ch 3's concept of ferility. Clean.

---

## Step 4: NO-NOS Scan

All 20 rules checked. Zero violations.

- No signposting (rule 1). The one Chapter 1 reference (L104, "introduced in Chapter 1") functions as evidence redirection, not signposting.
- No bare Braudel (rule 17). Braudel terms appear only in the permitted footnote (L146).
- No "body" (VOCABULARY.md). Term absent from the chapter.
- No pipeline infrastructure names (rule 16). Generic descriptions only.
- No "signal time" (VOCABULARY.md). Term absent.

---

## Step 5: Vocabulary Enforcement

Zero violations. All canonical terms used correctly:
- substrate time / trajectory time / compositional time (all three)
- manifold, substrate, meaning-space
- "logic" used as constructivist/provenance (L358), not truth/falsity

---

## Step 6: Political Dimension

Every technical mechanism has its political dimension addressed. Full coverage across all 13 sections. No gaps.

---

## Step 7: Conclusion

Chapter 2 is mature and structurally sound. The argument flows from a single word ("mother") through the full machinery of the transformer, through strata of governance, through dynamics and sampling, through temporal structures, through memory and hidden context -- and at every step the self is the subject, the politics is named, and the Meson reader can follow. No edits required.
