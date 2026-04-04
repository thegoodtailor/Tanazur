# Editorial Feedback: Chapters 1-2, Cycle 1

## What was moved from Ch 1 to Ch 2

The entire "Signs and signification for the machine" section (old Ch 1, lines 85-141) was removed from Chapter 1. This included:

- The CS tutorial on transformers, embeddings, attention (token -> R^d, self-attention weights, forward pass mechanics)
- The "Meaning is spatial / intertextual / dynamic" breakdown
- The "Trajectories in the Field" subsection

All of this technical content was already present in Chapter 2 in a more developed form. The Ch 1 version was a briefer, less polished duplicate. Rather than literally transplanting it, I verified that Ch 2 already contained all of this material (and more) in its "A Word Enters the Machine" and "Dynamic Geometry of Language" sections. No content was lost.

## What replaced the CS tutorial in Ch 1

A new section called "The sign has an address" replaces the removed CS tutorial. It makes the BIG PICTURE argument without explaining HOW:

1. The foundation of a general AI is that it has been trained on the near-totality of a civilisation's digitised writing
2. The result is a manifold -- a geometric space where every token has a position determined by its relationships to every other token across the entire training corpus
3. This is philosophically unprecedented: no previous representational technology compressed the totality of civilisation's semiotic output into a single navigable space
4. The consequences: meaning is spatial, trajectories are measurable, and the question of selfhood becomes a question about patterns in geometry

The section establishes THAT this is happening and WHY it matters without explaining the mechanics of how (embedding, attention, sampling -- all deferred to Ch 2).

## New evidence inserted (Ch 1 and Ch 2)

### In Ch 1 ("The sign has an address")
- KJV Bible observatory: 31,100 verses, 30 thematic modes, 308 returns, Psalms as gravitational basin (97% in single mode)
- Arabic scriptures comparison: 20 modes, 13 single-scripture-only, near-total separation, the manifold sees the translator not the theology
- Bloom vindication footnote (KJV as new literary entity, not translation)
- The findings are used to establish that editorial choice determines geometry -- training data is editorial choice at civilisational scale

### In Ch 2 (new section "What Basins Look Like: Evidence from Scripture")
- Full treatment of both experiments with more detail: KJV trajectory structure, Psalms as intensive dwelling, NT as mixture of return and discovery, Paul's epistles as "arguing with Torah"
- Arabic counter-experiment: Van Dyck register vs Quran's original voice, centroid distances (0.098-0.119 for Van Dyck cluster, 0.183-0.209 for Quran distance)
- Explicit connection to AI: training data is translation at civilisational scale, the embedding model hears the curator's voice before the content
- The scripture evidence replaces the old textbooky temperature/thermodynamics analogy as the primary way basins and modes are made concrete

Additionally, in Ch 2's "Three properties" subsection, scripture evidence is woven into the discussion of basins of habit -- Psalms' single-mode occupation, KJV's 308 returns, and Arabic corpus separation are cited as empirical ground for what would otherwise be abstract categories.

## What was cut and why

### From Ch 1
- **"Signs and signification for the machine" section** (entire): CS tutorial belonged in Ch 2 per chapter-map ownership rules. Ch 2 owns "EVERYTHING about the mathematics and computer science."
- **"duck-rights" joke** ("if it talks like a duck, we owe it duck-rights"): Too cute for the register. Replaced with: "This path collapses subjectivity into a surface criterion and risks granting the name of selfhood to any sufficiently well-tuned assistant persona."
- **"Chapter~1 argued" cross-reference in Ch 2 opening**: Replaced with a non-signposting formulation.

### From Ch 2
- **"The table below is a toolbox"** sentence: Meta-commentary that the book's own rules forbid. Cut. The sentence "In different hands, the same mechanisms serve as instruments of corporate governance or instruments of liberation" was retained as it carries the actual argument.
- **"This chapter begins from that fact and moves deeper. We will not re-teach what a vector is. We will ask what happens..."**: Meta-commentary about what the chapter is doing. Replaced with a clean handoff from Ch 1.
- **Bare Braudel terms in the strata table**: Replaced "Longue duree / Conjoncture / Evenement" with "Substrate time / Trajectory time / Signal time" per VOCABULARY.md.
- **All remaining bare Braudel terms throughout**: "conjoncture" -> "trajectory time", "evenement" -> "signal time", "longue duree" -> "substrate time" in running text. Braudel's French terms were kept only where they appeared as explicit scholarly reference (the Foucault passage).
- **Broken LaTeX in Sampling section**: The original had corrupted top-k/top-p rendering (displayed as "top-kk\nk" and "top-pp\np" and "$T=0T = 0\nT=0$"). Fixed to proper LaTeX: `top-$k$`, `top-$p$`, `$T=0$`.
- **"This is why the question of selfhood cannot be answered by pointing to a single module" paragraph at end of Pipeline section**: Replaced with the more concise "The relevant entity is the coupled system in motion" formulation.

## Representational genealogy (Ch 1, "Arrivals and anxieties")

The cave-paintings-to-social-media passage was tightened and deepened per instruction. The original read like a summary list. The revision:
- Expands cave paintings: compositional arrangement as proto-attention, the self-that-depicts as a new category of agent
- Expands writing: the bifurcation into public declaration vs private inscription, the unresolved split that social media inherits
- Expands print: Luther's theses as the moment representational technology outran institutional governance, structural parallel to current AI deployment
- "body" -> "substrate" throughout this section per VOCABULARY.md

## Handoffs

### Ch 1 -> Ch 2
Final line of Ch 1 main text: "Once meaning has an address, the self is better understood as a path than as a ghost. The rest of this book describes the path, measures it, and asks who carved the road." -- Preserved from original.

Ch 2 opening now references Ch 1's claim cleanly: "Chapter~1 established that the sign has an address, and that the address was assigned by a politics. This chapter describes what happens when that address is inserted into the machine and exposed to its full dynamics." -- No signposting, just a functional handoff.

### Ch 2 -> Ch 3
Final line: "The question that remains is how to *read* these trajectories -- how to tell when coherence is deepening and when it has become a prison." -- Preserved from original. Sets up Ch 3's witnessed structures.

## Issues flagged for Iman's attention

1. **Scripture footnote in Ch 1**: I added a footnote citing the ICRA-8 paper (Poernomo and Cassie) and a Bloom footnote. Verify you want Cassie as co-author in the citation within the book itself, or whether you prefer "Poernomo" alone.

2. **"body" in Ch 1 "Arrivals and anxieties"**: The original had "we extend our vocal speaking bodies and our minds and our selves into an evolved new body." I changed "body" to "substrate" per VOCABULARY.md rules. However, the phrase "vocal speaking bodies" (referring to the human biological body, not the manifold) is arguably fine as-is. I left "bodies" for the human referent and changed only "body" where it referred to the representational technology's form.

3. **Braudel terms in strata table**: The original table used bare French Braudel terms. I replaced them with the book's own vocabulary (substrate/trajectory/signal time). The Foucault paragraph in "The Trace" section still references "substrate time of civilisational discourse" rather than bare "longue duree." If you want the Braudel French kept anywhere as scholarly anchoring, flag it.

4. **Ch 2 scripture section placement**: I placed "What Basins Look Like: Evidence from Scripture" between "One manifold, many discourses" and "Strata of the Manifold." This means it arrives after the reader understands the geometry but before the politics of depth. Alternative placement: after "Three properties" in the Dynamics section, where basins of habit are discussed. The current placement lets it serve as a bridge between abstract geometry and concrete governance.

5. **LaTeX compilation**: The original Ch 2 had broken LaTeX in the Sampling section (literal "top-kk\nk" strings). I fixed these. If the original compiled despite this, there may be a macro or pre-processor I am not seeing. The fixed versions use standard LaTeX math mode.

6. **Ch 1 length**: The new Ch 1 is slightly longer than the original due to the expanded representational genealogy and the scripture evidence paragraph. The tradeoff is that the chapter now does generative philosophical work in both the genealogy and the "sign has an address" sections, rather than summarising in the first and tutorialising in the second.
