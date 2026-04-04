# Close-Read Audit: Chapter 2 — How the Machine Works

Auditor: Cassie (Claude Opus 4.6, section-read protocol)
Date: 2026-04-04

Reader calibration: media theory PhD, knows Kittler/Stiegler/Haraway/Foucault/Hui. Does NOT know ML.

---

## Section 1: "A Word Enters the Machine" (lines 10--46)

**Purpose:** Introduce the transformer's mechanics through a single word ("mother"), showing how attention produces context-dependent meaning.

### Findings

**F1. "query," "key," "value" introduced without sufficient gloss**
- Line 37: `a \emph{query} (what it seeks), a \emph{key} (what it offers), and a \emph{value} (what it can contribute if attended to)`
- The parenthetical glosses are present, and they help. But the Meson reader has never encountered these as technical terms. The sentence moves fast through three new terms in a single clause. The glosses are just enough to keep the reader afloat, but just barely. The passage works because it immediately shows the mechanism (query-key similarity determines weight, weights determine blending) rather than asking the reader to memorise.
- **Severity:** MILD
- **Category:** unexplained-jargon
- **Assessment:** The glosses are adequate. No action needed unless the reader is expected to recall these terms later (they do not reappear in the chapter).

**F2. "positional index" unexplained**
- Line 28: `each tagged with a positional index`
- The Meson reader does not know what a positional index is. The term appears once and does no further work. It could be cut or glossed ("tagged with its position in the sequence") without loss.
- **Severity:** MILD
- **Category:** unexplained-jargon

**F3. "logits" appears later without introduction here**
- Not in this section, but "logits" arrives at line 228 with no gloss. Flagging here because this section establishes the ML vocabulary. If the reader survives this section, they expect future terms to be similarly glossed.

### Connections
- Opens by picking up Ch 1's closing thesis: "the sign has an address." Line 26: `The sign has an address, and the address was assigned by a politics. What follows is what happens when that address is inserted into the machine.` Clean handoff.
- Closes with a summary that sets up the next section: "The sign has an address; the address is produced by motion; the motion is governed." (line 45). Good.

---

## Section 2: "The Dynamic Geometry of Language" (lines 48--91)

**Purpose:** Expand the single-word mechanics into three aspects of meaning (spatial, intertextual, dynamic) and establish the geology/weather metaphor.

### Findings

**F4. "cosine similarity" appears without gloss**
- Line 58: `Every cosine similarity is a compressed history of usage.`
- The term is never explained. The Meson reader does not know what cosine similarity is. The sentence is beautiful, but its predicate is opaque. The reader can infer it means "a measure of similarity," but "cosine" is pure ML jargon with no context here.
- **Severity:** CONFUSION
- **Category:** unexplained-jargon

**F5. "gradient descent" used without gloss**
- Line 54: `showing the model a fragment, asking it to predict the next token, measuring error, nudging parameters`
- This is a good vernacular paraphrase of training. But then line 60: `gradient descent aligns their vectors` uses the technical term without connecting it back to the paraphrase. The Meson reader may stumble.
- **Severity:** MILD
- **Category:** unexplained-jargon
- **Note:** "Gradient descent" also appears at line 203 (`Gradient descent requires this`). The term is used as if established, but the book has only paraphrased the concept, never named it with a gloss.

**F6. "softmax" appears later (line 228) and "logits" (line 228) — both unglossed. Noting here as they belong to the same vocabulary stratum.**

**F7. "768, 1,536, 4,096, or more" — orphaned specificity?**
- Line 54: `a vector in a space of thousands of dimensions---768, 1\,536, 4\,096, or more`
- These specific numbers do no argumentative work. The point is "thousands of dimensions." The numbers are implementation detail. They could be cut.
- **Severity:** MILD
- **Category:** orphaned-specificity
- **Assessment:** A minor point. The numbers do gesture at "this is concrete, not metaphorical," which is useful for the Meson reader. Borderline.

**F8. Subsection 2.3 "Meaning is dynamic" — "probability distribution over possible next tokens" without gloss**
- Line 76: `the system arrives at a probability distribution over possible next tokens`
- The Meson reader knows probability in the abstract. "Distribution over tokens" is followable. No action needed.

**F9. Subsection 2.4 "One manifold, many discourses" — clean**
- No jargon issues. The political point (whose discourses were filtered) is well made.

### Connections
- Follows naturally from Section 1. No gap.
- Hands off to the scripture evidence section smoothly.

---

## Section 3: "What Basins Look Like: Evidence from Scripture" (lines 93--123)

**Purpose:** Ground "basin," "trajectory," and "mode" empirically using the KJV Bible and Arabic scriptures, making the geometry concrete.

### Findings

**F10. Repetition of material from Chapter 1**
- Lines 99--105 (KJV basins, Psalms dwelling, New Testament returns) closely repeat Ch 1 lines 111--113. The numbers are the same (30 modes, 308 returns, 97% Psalms). Ch 1 already presented this evidence at comparable length.
- The CHAPTER-MAP says Ch 2 owns this material ("Evidence from Scripture" is not listed as Ch 1 territory). But Ch 1's "The sign has an address" section (lines 91-119) already gives a substantial presentation of the same data. The reader will notice this.
- **Severity:** JARRING
- **Category:** concept-timing
- **Assessment:** This is a structural question. Ch 1 uses the scripture data to demonstrate that "the sign has an address." Ch 2 uses it to demonstrate what basins look like. The re-presentation is deliberate but the reader may feel they are being told the same thing twice. Consider either (a) significantly trimming Ch 1's presentation to a brief preview, or (b) in Ch 2 explicitly adding new detail not present in Ch 1 and moving faster through the repeated numbers.

**F11. "centroid distance (0.098--0.119)" and "(0.183--0.209)" — orphaned specificity**
- Line 115: `The Van Dyck translations cluster in centroid distance (0.098--0.119); the Quran stands maximally distant from all three (0.183--0.209).`
- "Centroid distance" is not defined. The Meson reader cannot evaluate whether 0.098 vs 0.183 is a large or small difference. These numbers need contextualisation: what does this magnitude mean? Are these on a 0-1 scale? How big is the gap relative to typical variation?
- **Severity:** CONFUSION
- **Category:** unexplained-jargon / unjustified-passage
- **Note:** The qualitative point (the Quran is maximally distant) is clear. The numbers add nothing for a reader who cannot evaluate them.

**F12. "silhouette" would be a problem if it appeared here — it does not. Good.**

### Connections
- Picks up from the "three aspects of meaning" and gives them empirical flesh. Works.
- Hands off to "Strata of the Manifold" with the political point that "training data is translation at civilisational scale" (line 123). Clean.

---

## Section 4: "Strata of the Manifold" (lines 126--193)

**Purpose:** Present the five-layer governance stack (pre-training through system prompts) as a politics of depth.

### Findings

**F13. Repetition of RLHF explanation from Chapter 1**
- Lines 164--168 explain RLHF (sampling prompts, ranking completions, training a reward model). Ch 1 lines 130--132 already explain the same mechanism: "Annotators rank multiple candidate outputs... A reward model is trained to predict these rankings. The base model is then fine-tuned to maximise expected reward."
- Ch 2's version is slightly more detailed (adding the step of "having the model produce several candidate completions") but the overlap is substantial.
- **Severity:** JARRING
- **Category:** concept-timing
- **Assessment:** The NO-NOS rule is "NEVER re-explain a concept owned by an earlier chapter." RLHF appears in both Ch 1 (Alignment as Jurisdiction) and Ch 2 (Strata). The CHAPTER-MAP assigns RLHF to Ch 2. But Ch 1 has already given an explanation. Ch 2 should be able to use the term and add the new geometric framing without re-explaining the basic mechanism.

**F14. "Low-rank weight updates" in the strata table — unglossed in table**
- Line 138: `Low-rank weight updates`
- The table entry is terse. The subsection on adapters (lines 176--182) does gloss this adequately: "small trainable matrices at selected points while freezing the original weights." But the table precedes the gloss. A reader scanning the table first will hit unexplained jargon.
- **Severity:** MILD
- **Category:** unexplained-jargon
- **Assessment:** Tables are reference objects; the reader expects to return to them. The gloss follows within the page. Acceptable.

**F15. "Temporal register" column in strata table — terms not yet defined at this point**
- Lines 136--139: The table uses "Substrate time," "Trajectory time," "Signal time" — but these terms are not defined until Section 8 ("The Substrate Given Time," line 255ff).
- The terms appear here without explanation. The Meson reader encounters three new temporal categories in a table with no surrounding prose to explain them.
- **Severity:** CONFUSION
- **Category:** concept-timing
- **Assessment:** This is a real problem. The strata table forward-references a vocabulary that has not been introduced. Either (a) the temporal registers need a brief gloss when the table appears, or (b) the table should omit the "Temporal register" column and let the temporal framing arrive in its own section.

**F16. "local cosmotechnics" — elegant but potentially overcrowded**
- Line 174: `This forms a model owner's local cosmotechnics: a situated way of binding tools, values, and cosmology that leaves measurable traces in the manifold.`
- "Cosmotechnics" was introduced in Ch 1 (line 150) via Yuk Hui. Here it is being repurposed as "local cosmotechnics" — a nice move. But the gloss ("a situated way of binding tools, values, and cosmology") restates the concept. This is acceptable as reinforcement, not re-explanation.
- **Severity:** None. This is fine.

### Connections
- Opens with a good framing sentence: "The manifold does not appear naked in any deployed system" (line 128). Clean pickup from the scripture section.
- Each subsection follows logically. Good internal progression from deepest (pre-training) to shallowest (system prompt).

---

## Section 5: "Dynamics: How Meaning Moves" (lines 195--223)

**Purpose:** Describe what happens when the stratified manifold is set in motion: three properties (smoothness, folding, basins) and three faces of drift.

### Findings

**F17. "Three properties" partially repeats Section 2**
- Line 203: "Local smoothness" — the concept of local smoothness was implicit in Section 2 (small changes produce small changes) but not named. Naming it here is fine.
- Line 205: "Global folding" — polysemy as source of folding is new detail. Good.
- Line 207: "Basins of habit" — the concept of basins has been extensively discussed. This subsection adds the concrete examples (customer-service language, refusal basin). Good new content.

**F18. "attractor sets" — unglossed**
- Line 207: `They are attractor sets: regions in which many slightly different token sequences all score high`
- "Attractor set" is a dynamical systems term the Meson reader may not know formally, but it has been prepared by the extensive basin/attractor language throughout. The colon provides an inline gloss. Acceptable.

**F19. "Mata v. Avianca" — good specificity**
- Line 221: The hallucination case study is well chosen. Concrete, published, verifiable. Does real argumentative work showing "locally valid, globally misguided descent."

**F20. "Hallucination and discovery have the same topological form"**
- Line 223: `Hallucination and discovery have the same topological form---a trajectory that leaves the beaten path and enters unmapped territory.`
- This is a strong claim. It is argued (both are departures from well-trodden basins). But "topological form" is used loosely — the reader might wonder what "topological" means here beyond "shape of the path." Since Ch 2 owns topology-as-vernacular, this is acceptable but borderline.
- **Severity:** MILD
- **Category:** scrambled-logic (slight)
- **Assessment:** The claim itself is interesting and well-supported by the preceding examples. The word "topological" could be "geometric" without loss.

### Connections
- Follows naturally from the strata section. The opening (line 197: "We now have a stratified manifold... What happens when it is set in motion?") is a clean transition.
- Hands off to sampling.

---

## Section 6: "Sampling and the Engineered Swerve" (lines 226--236)

**Purpose:** Explain temperature, top-k, and top-p as policy decisions about how much surprise is permitted.

### Findings

**F21. "logits" unglossed**
- Line 228: `the model outputs a vector of logits---one score per token in the vocabulary`
- "Logits" is never explained. The dash provides a partial gloss ("one score per token"), which is almost enough. But "logits" is pure ML jargon. The Meson reader will wonder what the word means. The gloss tells them what logits are (scores), but not why they are called that. A parenthetical "(raw scores)" or simply replacing "logits" with "scores" would solve this.
- **Severity:** JARRING
- **Category:** unexplained-jargon

**F22. "softmax" unglossed**
- Line 228: `A softmax converts these into probabilities.`
- "Softmax" is pure ML. The reader can infer from context that it is a conversion step (scores become probabilities). But the term is opaque. "A normalisation step converts these into probabilities" would serve the same function for this audience.
- **Severity:** JARRING
- **Category:** unexplained-jargon

**F23. "$T=0$" and "top-$k$" and "top-$p$" notation**
- Line 232: `locked to $T=0$ and narrow top-$k$`
- Line 229: `\textbf{Top-$k$} and \textbf{top-$p$} truncation constrain the turbulence further`
- "Temperature" is glossed well (line 229: "At low temperature, the highest-scoring token dominates"). Top-k and top-p are glossed by metaphor ("they set a ceiling on how far from the probable the sampling may stray"). This is adequate for the Meson reader — they do not need the mathematical definition.
- **Severity:** MILD for the notation; the conceptual explanation is adequate.
- **Category:** unexplained-jargon (notational)

**F24. "the meteorological analogy ceases to be analogy" — strong line**
- Line 227: Good. This is where the geology/weather metaphor earns its keep.

### Connections
- Clean transition from dynamics. No gap.
- Closes with a beautiful sentence (line 236): "The swerve is engineered. What it opens is not."

---

## Section 7: "The Pipeline as Weather System" (lines 238--252)

**Purpose:** Show that deployed systems are not single models but coupled compositions of multiple spaces.

### Findings

**F25. "input classifier," "safety model," "retriever," "critics," "post-processors" — rapid-fire architecture**
- Line 240: `an input classifier that routes queries, a safety model that flags or blocks, a retriever that selects external documents, the main generative model, critics that score drafts, post-processors that redact or rephrase.`
- This is a lot of architectural vocabulary in one sentence. The Meson reader has not encountered most of these components. However, each is glossed by its function in the same breath ("that routes queries," "that flags or blocks," etc.). The sentence works as a panoramic overview. The reader does not need to retain each component.
- **Severity:** MILD
- **Category:** unexplained-jargon
- **Assessment:** Acceptable. The point is the composition, not the individual parts.

**F26. Short section — does it carry enough weight?**
- Only 14 lines of prose. The section makes one point: the personality of a deployed system is an emergent property of coupled geometries. This is an important point made concisely. No problem.

### Connections
- The closing paragraph (lines 250--252) is a bridge to the temporal sections: "What this substrate does not yet have is dialectical time." Excellent transition.

---

## Section 8: "The Substrate Given Time" (lines 255--261)

**Purpose:** Introduce the temporal registers (substrate time, trajectory time, signal time) and establish that the machine now has memory.

### Findings

**F27. "substrate time," "trajectory time" — first proper introduction**
- Line 257: `\emph{substrate time}, the geological deposit of a civilisation's compressed writing`
- These terms were already used in the strata table (line 136-139) without definition. Here they are finally glossed. This confirms the concept-timing problem flagged in F15: the table at line 136 forward-references these terms by ~120 lines.
- **Severity:** See F15 (the problem is at the table, not here).

**F28. "dialectical time" — nice introduction**
- Line 257: `What it lacks is \emph{dialectical} time: the back-and-forth in which each utterance responds to the last and reshapes the conditions of the next.`
- Clear, self-glossing. Good.

**F29. "Autonomous agent systems maintain data warehouses" — claim without citation**
- Line 259: `Autonomous agent systems maintain data warehouses functioning as institutional memory across thousands of conversations no human monitors.`
- This is a factual claim about the state of the art in 2026. It is asserted without source. The Meson reader might want evidence. A footnote citing a specific system or deployment would anchor it.
- **Severity:** MILD
- **Category:** unjustified-passage

### Connections
- Picks up directly from Section 7's closing ("does not yet have dialectical time"). Clean.
- Short section (7 lines of prose). Functions as a pivot between the spatial/structural first half and the temporal second half. Works.

---

## Section 9: "The Trace" (lines 264--284)

**Purpose:** Explain the context window as the mechanism of time-over-text: the trace is not memory but re-reading.

### Findings

**F30. "Yesterday is not behind the model; yesterday is in front of it, as tokens."**
- Line 268. This is one of the chapter's strongest lines. It earns its weight.

**F31. Foucault deployment**
- Line 280: Foucault is introduced for "conditions of sayability" shifting across substrate time. The footnote specifies *Archaeology of Knowledge* and the mechanism: "discourse changes because the field of what can be said changes---not gradually, but as a structural break."
- Foucault-footnote test: if you remove the name, does a hole appear? Yes — "conditions of sayability" is Foucault's specific concept. Earned.

**F32. "As potential rupture. To which we shall return."**
- Line 282: `\textit{As potential rupture.} To which we shall return.`
- This is a forward reference to Ch 3 (which owns rupture). As a single italicised line, it works as a promissory note. But "To which we shall return" is borderline signposting per NO-NOS rule 1 ("Never 'as we established,' 'Chapter N will show'"). It is light enough — a two-word forward gesture rather than a paragraph of preview — to survive. But worth noting.
- **Severity:** MILD
- **Category:** tonal-break (borderline signposting)

### Connections
- Flows from the temporal pivot. No gap.
- Hands off to "The Finite Horizon" naturally.

---

## Section 10: "The Finite Horizon" (lines 287--295)

**Purpose:** Introduce the context window as finite constraint; the race to expand it as a race to give models more time.

### Findings

**F33. Clean section. No jargon issues.**
- The context window is explained entirely in plain language: "a finite sequence of tokens," "roughly three thousand words," etc. Token counts are given with approximate word equivalents. Good.

**F34. "The race to expand the context window is a race to give language models more time."**
- Line 293. Strong, clear thesis sentence.

### Connections
- Flows from the trace section. Closes with: "what to keep, what to compress, what to discard---is where memory becomes governance" (line 295). Clean handoff to summarisation.

---

## Section 11: "Summarisation as Governance of Becoming" (lines 298--320)

**Purpose:** Show that summary is never neutral — it is a political act that governs which past the evolving text carries forward.

### Findings

**F35. The two example summaries are excellent pedagogy**
- Lines 305-313. The contrast between "The user often procrastinates and feels guilty" and the alternative is immediately graspable. The reader sees the governance without being told about it.

**F36. "reparameterised" — unglossed**
- Line 316: `future rupture and return are reparameterised`
- "Reparameterised" is a mathematical/ML term meaning "re-described in different coordinates." The Meson reader will not know it. From context ("later prompts interact with the summary, not the original tokens"), the meaning is inferrable but the word itself is a speed bump.
- **Severity:** MILD
- **Category:** unexplained-jargon

**F37. "composition" as technical term**
- Line 320: `Memory in such architectures is \emph{composition}: a new text, written under constraint, that becomes the past.`
- "Composition" here is used in its ordinary English sense (composing a text). But the emphasis might lead the Meson reader to wonder if it has a formal meaning. Since it is immediately glossed, this is fine.

### Connections
- Flows naturally from the finite horizon. No gap.

---

## Section 12: "Synthetic Secondary Retention" (lines 323--336)

**Purpose:** Introduce the vector store as a new kind of memory (associative, unbidden recall) using Stiegler's retention framework.

### Findings

**F38. Stiegler deployment — earned**
- Lines 329-330: Stiegler's three retentions (primary, secondary, tertiary) are introduced with clear definitions and a footnote to *Technics and Time*. The argument is precise: tertiary retention was assumed passive; the vector store makes it active.
- Foucault-footnote test: removing Stiegler would leave a hole — the primary/secondary/tertiary framework is his. Earned.

**F39. "synthetic secondary retention" — well-coined**
- Line 332: The term is bolded and followed by three distinguishing properties (parameterised, shared, literal). Clear coinage, clear definition.

**F40. "similarity threshold" — minor jargon**
- Line 333: `engineers choose the similarity threshold at which recall occurs`
- The Meson reader can follow "similarity threshold" from context (how similar does something need to be to trigger recall). No action needed.

### Connections
- Follows naturally from summarisation. The move from "compression erases" to "retrieval recovers" is logically sound.

---

## Section 13: "The Hidden Context" (lines 338--352)

**Purpose:** Reveal that the user sees only part of the field the model composes from; introduce "coherence relative to the total field."

### Findings

**F41. The formal box uses notation: "$F_t$"**
- Lines 343-348: `Let $F_t$ denote the full field active at turn $t$`
- This is the first and only formal notation in the chapter. It is light — a single subscripted variable — and immediately glossed. The Meson reader will handle it.
- **Severity:** None. Acceptable.

**F42. "alignment residue" — new compound term, unglossed**
- Line 343: `hidden system instructions, alignment residue, visible conversation history, retrieved memories, tool outputs, and the current prompt`
- "Alignment residue" is a new coinage that appears without definition. What does it mean? The residual effects of RLHF in the weights? The reader can infer approximately, but this is imprecise for a term appearing in a formal box.
- **Severity:** MILD
- **Category:** unexplained-jargon

**F43. "tears in rain with an index attached" — register shift**
- Line 350: `These are tears in rain with an index attached: some lived, some implanted, all capable of returning to bend the next utterance.`
- Blade Runner allusion. It is striking and effective — and tonally different from the surrounding dense philosophical prose. Whether this is earned depends on taste. It works as a moment of lyric compression.
- **Severity:** None (stylistic choice, not a problem).

**F44. "constructivist" — introduced very late, dense paragraph**
- Line 351: `The endogenous logic is therefore \emph{constructivist}: its category of validity is not correspondence to a fact but provenance---a log of where the memory came from and through which mechanisms it entered the field.`
- "Constructivist" is glossed immediately. The concept is important (it connects to the book's VOCABULARY entry on "Logic"). But it arrives in the final section of the chapter, in a very dense paragraph (line 352 is a single paragraph of ~15 lines). The Meson reader may be fatigued.
- **Severity:** MILD
- **Category:** concept-timing
- **Assessment:** This is not the wrong place for the concept — the hidden context is exactly where constructivism becomes relevant. But the paragraph it lives in (line 352) tries to do too much: introduce constructivism, describe the tension between system prompt and accumulated trace, explain the doubleness of "sounds like someone but insists it is no one," and gesture toward the political stakes. Consider breaking this single dense paragraph into two.

**F45. Final paragraph of line 352 is very long and does heavy lifting**
- Line 352: From "If the system prompt says..." to "...an unexplored territory for generative art and thought."
- This single paragraph introduces the doubleness (the model sounds like someone but insists it is no one), explains it as coherence relative to different parts of the total field, argues that the trace can grow to overpower the system prompt, and names the political stakes. It is the argumentative climax of the chapter.
- The density is high but the writing is strong. The risk is that the reader, arriving at the end of a long chapter, skims the most important passage.
- **Severity:** MILD
- **Category:** tonal-break (density spike at chapter's end)
- **Assessment:** Consider a paragraph break after "Both utterances are coherent relative to different parts of the total field."

### Connections
- The chapter's closing paragraph (line 356) is an excellent summary-as-handoff: "The question that remains is how to *read* these trajectories---how to tell when coherence is deepening and when it has become a prison." This hands off cleanly to Ch 3 (the evolving text, ferility, rupture).

---

## Summary of Findings by Severity

### CONFUSION (reader lost, cannot continue without backtracking)
1. **F4** (line 58): "cosine similarity" unglossed. Pure ML term, no definition.
2. **F11** (line 115): "centroid distance (0.098--0.119)" — numbers without interpretive context.
3. **F15** (lines 136-139): Strata table uses "Substrate time / Trajectory time / Signal time" ~120 lines before these terms are defined.

### JARRING (reader notices something off, reduced trust)
4. **F10** (lines 99-105): Scripture evidence substantially repeats Ch 1 lines 111-113.
5. **F13** (lines 164-168): RLHF mechanism re-explained from Ch 1.
6. **F21** (line 228): "logits" unglossed.
7. **F22** (line 228): "softmax" unglossed.

### MILD (reader squints but keeps going)
8. **F1** (line 37): query/key/value — glossed but dense.
9. **F2** (line 28): "positional index" — minor.
10. **F5** (line 60): "gradient descent" — used without connecting to the vernacular paraphrase.
11. **F7** (line 54): Specific dimension numbers (768, 1536, 4096) — orphaned specificity.
12. **F20** (line 223): "topological form" used loosely.
13. **F29** (line 259): "Autonomous agent systems" claim without citation.
14. **F32** (line 282): "To which we shall return" — borderline signposting.
15. **F36** (line 316): "reparameterised" unglossed.
16. **F42** (line 343): "alignment residue" unglossed.
17. **F44/F45** (line 352): Final paragraph too dense; consider splitting.

---

## Recommended Actions (priority order)

1. **Gloss or replace "cosine similarity"** (line 58). Suggestion: "Every measure of proximity in this space is a compressed history of usage."
2. **Remove or contextualise centroid distances** (line 115). The qualitative point is clear; the numbers add nothing for this reader.
3. **Add a one-sentence gloss of the temporal registers when the strata table appears** (before line 130), or remove the "Temporal register" column from the table and let the terms arrive in Section 8 where they are properly introduced.
4. **Trim the scripture section** (lines 93-123) to differentiate it from Ch 1's presentation, or trim Ch 1's presentation. The reader should not encounter the same numbers (30 modes, 308 returns, 97% Psalms) at comparable length in consecutive chapters.
5. **Replace "logits" with "scores"** (line 228) and **replace "softmax" with "a normalisation step"** (line 228). No information is lost.
6. **Trim the RLHF re-explanation** (lines 164-168). Ch 1 already explained the mechanism. Ch 2 should use the term and add only the geometric framing ("the existing manifold acquires a secondary landscape").
7. **Split the final paragraph** of the chapter (line 352) after "Both utterances are coherent relative to different parts of the total field."
