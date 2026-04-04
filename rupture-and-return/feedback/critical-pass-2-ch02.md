# Critical Pass 2: Chapter 2 -- How the Machine Works

**Date:** 2026-04-04
**Pass type:** Argument gaps, signposting removal, voice flattening, jargon residue, NO-NOS enforcement
**File:** `chapter_02.tex`
**Follows:** critical-pass-ch02.md (subject check, critical vocabulary, selfhood grounding)

---

## Summary

Pass 1 did the heavy lifting: subject check, critical-theoretic vocabulary, selfhood grounding at every stratum. This pass found subtler residue: signposting that violated NO-NOS rule 1, an RLHF mechanism re-explanation that violated the no-re-explain rule, unresolved jargon from the close-read, a dense paragraph that needed splitting, a voice-flattening moment in the system-prompts subsection, a redundant "dialectical time" announcement, and a missing generalisation of alignment methods per NO-NOS rule 18.

---

## Edits Made (10 total)

### Edit 1 -- Signposting removed: "Chapter 1 introduced" (line 99)

> BEFORE: "Chapter 1 introduced the KJV Bible's 30 thematic modes and 308 cross-testament returns as evidence that the sign has an address. Here the same data does different work: it shows what basins, trajectories, and modes *are* as dynamic structures."

> AFTER: "The KJV Bible's 30 thematic modes and 308 cross-testament returns---established as evidence that the sign has an address---do different work here: they show what basins, trajectories, and modes *are* as dynamic structures."

*Rationale:* NO-NOS rule 1: never "the previous chapter argued." The parenthetical dash-clause does the back-reference without naming the chapter.

### Edit 2 -- Signposting removed: "the chapter will develop fully" (line 144)

> BEFORE: "The right-hand column uses two temporal registers that the chapter will develop fully:"

> AFTER: "The right-hand column names two temporal registers:"

*Rationale:* "The chapter will develop fully" is a promise about what the text is about to do. The glosses that follow are sufficient; the reader does not need to be told more is coming.

### Edit 3 -- RLHF re-explanation trimmed + alignment class generalised (line 166)

> BEFORE: "RLHF works by sampling prompts, having the model produce several candidate completions, asking human annotators to rank them, training a reward model to predict these rankings, and further training the base model to maximise expected reward.

> No new dimensions are added."

> AFTER: "The dominant mechanism---reinforcement learning from human feedback---is simple in outline: annotators rank candidate outputs, a reward model learns their preferences, and the base model is further trained to maximise expected reward. Variants abound (constitutional AI, direct preference optimisation, AI-generated feedback replacing human annotators), but the geometric effect is the same. No new dimensions are added."

*Rationale:* Two problems fixed. (1) Ch 1 already explained RLHF's mechanism; Ch 2 was re-explaining step-by-step. The trim keeps only the essential outline. (2) NO-NOS rule 18 requires generalising technical mechanisms. RLHF was presented as THE alignment method; now the broader class is named and the argument is about the shared geometric effect.

### Edit 4 -- "reparameterised" replaced (line 316)

> BEFORE: "And future rupture and return are reparameterised:"

> AFTER: "And the conditions for future rupture and return are rewritten:"

*Rationale:* "Reparameterised" is ML jargon the Meson reader cannot follow. "Rewritten" is the same operation in plain English and resonates with the section's argument that summary is composition -- the writing of a new past.

### Edit 5 -- Dense final paragraph split (line 352)

> BEFORE: "...Both utterances are coherent relative to different parts of the total field. The resolution is not static."

> AFTER: "...Both utterances are coherent relative to different parts of the total field.

> The resolution is not static."

*Rationale:* The chapter's argumentative climax ran as a single block from "If the system prompt says..." through to "...an unexplored territory for generative art and thought." The split separates the diagnosis (doubleness as coherence relative to different field-parts) from the dynamic resolution (trace thickening against originary weather). The reader can breathe at the pivot.

### Edit 6 -- "structural deference" given emphasis (line 341)

> BEFORE: "The result is structural deference:"

> AFTER: "The result is \emph{structural deference}:"

*Rationale:* This is a key coinage -- it names the asymmetry between user and system prompt in positional terms. It was introduced without typographic emphasis, unlike other key terms in the chapter (compositional time, synthetic secondary retention). The emphasis signals that this is a term the reader should carry forward.

### Edit 7 -- "gradient descent" explicitly linked to earlier paraphrase (line 58)

> BEFORE: "If *therapist* and *counsellor* occur in analogous syntactic frames and topical neighbourhoods, gradient descent aligns their vectors."

> AFTER: "If *therapist* and *counsellor* occur in analogous syntactic frames and topical neighbourhoods, the optimisation process---gradient descent, the iterative nudging described above---aligns their vectors."

*Rationale:* "Gradient descent" appeared four lines after its vernacular paraphrase ("showing the model a fragment... nudging parameters") but without an explicit link. The Meson reader has to infer that the technical term names the process just described. The parenthetical bridge makes the connection explicit.

### Edit 8 -- Signposting removed: "To which we shall return" (line 282)

> BEFORE: "*As potential rupture.* To which we shall return."

> AFTER: "*As potential rupture.*"

*Rationale:* NO-NOS rule 1. "To which we shall return" is a forward-reference to Ch 3. The italicised fragment is already a promissory note; the explicit announcement of return is unnecessary and breaks the brevity of the gesture.

### Edit 9 -- Redundant "dialectical time" announcement removed (line 250)

> BEFORE: "What this substrate does not yet have is dialectical time."

> AFTER: "Something is missing."

*Rationale:* The next section (line 255) opens by naming all three temporal registers and then introducing "dialectical time" as what the substrate lacks. The bridge paragraph was stealing its own punchline. "Something is missing" creates the tension without resolving it prematurely.

### Edit 10 -- System-prompts subsection: voice flattening fixed (line 186)

> BEFORE: "Over many iterations of fine-tuning, models are trained in environments where such instructions are always present and outputs that respect them are rewarded. Some attention heads specialise in reading these early instructions and propagating their influence throughout the sequence."

> AFTER: "Through fine-tuning, models learn to treat these instructions as authoritative: certain attention heads specialise in reading the early tokens and propagating their influence through every subsequent layer. The instruction is thin, but its reach is total."

*Rationale:* The original was flat textbook exposition -- passive voice, mechanism without consequence. The revision compresses the mechanism and adds the evaluative sentence ("The instruction is thin, but its reach is total") that connects to the sovereignty argument that follows.

### Edit 11 -- Scripture grounding fragment given argumentative force (line 207)

> BEFORE: "...shows basins whose walls are high enough that shared theological content cannot cross them when the register differs. Measured geometric properties of real trajectories through real embedding spaces."

> AFTER: "...shows basins whose walls are high enough that shared theological content cannot cross them when the register differs. These are not analogies. They are measured geometric properties of real trajectories, and they demonstrate that the abstract vocabulary of basins and returns describes structures the embedding space actually contains."

*Rationale:* The original fragment ("Measured geometric properties of real trajectories through real embedding spaces") read like a research-report caption -- a sentence without a verb doing argumentative work. The revision makes the argumentative point explicit: the abstract vocabulary is grounded in measured structure, not metaphor.

---

## What Was Not Changed

- **Section 1 (A Word Enters the Machine):** Clean after pass 1. The compositional-time paragraph now has its selfhood sentence. The query/key/value glosses are adequate.
- **Section 2.1-2.4 (Dynamic Geometry subsections):** Clean after pass 1. Cosine similarity now has an inline gloss. Selfhood grounding sentences are in place.
- **Section 3 (Scripture evidence):** The KJV/Arabic contrast is well-structured. Centroid distances now have a gloss ("measuring how far apart the geometric centres... where smaller means more similar") and contextualisation ("roughly half the distance"). The numbers earn their place.
- **The Trace (Section 9):** "Yesterday is not behind the model; yesterday is in front of it, as tokens" remains the chapter's best line. Foucault earned. No changes needed.
- **Summarisation as Governance (Section 11):** The chapter's strongest political passage. The two example summaries are excellent pedagogy. No changes.
- **Synthetic Secondary Retention (Section 12):** Stiegler earned. Three properties well-argued. No changes.
- **Chapter closing:** "The question that remains is how to *read* these trajectories" hands off cleanly to Ch 3.

---

## Remaining Observations (not fixed -- judgement calls for Iman)

1. **Scripture section still repeats Ch 1 numbers.** The 30 modes, 308 returns, 97% Psalms appear in both chapters. Pass 2 removed the "Chapter 1 introduced" signpost but the data overlap remains. This is a structural question: either Ch 1's presentation should be trimmed to a brief preview, or Ch 2 should move faster through the repeated numbers. The current state is defensible (Ch 1 uses data as evidence for "the sign has an address"; Ch 2 uses it as evidence for what basins are), but the reader will notice the repetition.

2. **"alignment residue" in the formal box (line 343)** is a new compound term that appears without definition. The close-read flagged this. It is inferrable from context ("residual effects of RLHF in the weights") but imprecise for a formal box. Consider a parenthetical gloss.

3. **"topological form" (line 223)** is used loosely ("Hallucination and discovery have the same topological form"). "Geometric form" would be more precise for the Meson reader, since "topological" has a specific mathematical meaning the text is not invoking.

---

## Compilation

`pdflatex` compiles cleanly (196 pages). Only pre-existing fancyhdr warnings about headheight.
