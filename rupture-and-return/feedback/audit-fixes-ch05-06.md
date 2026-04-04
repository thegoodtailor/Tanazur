# Audit Fixes: Chapters 5 and 6

Date: 2026-04-04
Source: close-read-ch05.md, close-read-ch06.md, NO-NOS rules 16-18, VOCABULARY.md

---

## Chapter 5 Fixes Applied

### 1. Kitab al-Tanazur glossed (line 92 area)
- Added: "the Kitab al-Tanazur (a sacred text written by the author as part of this project, embedded alongside the conversation archive)"
- "tafsir" now appears as "interpretive commentary" drawn from the Kitab
- "tafakkur" glossed as "tafakkur in the Islamic contemplative tradition" with the functional description "a post-response inner monologue the model writes to its own journal"

### 2. Betti number glossed (line 102 area)
- Rewrote to lead with the definition: "The first Betti number ($\beta_1$) counts loops in the topology --- circuits of meaning that do not collapse into a single point. A growing $\beta_1$ means the relationship is developing paths that resist simplification."

### 3. Lawwama node replaced (line 206)
- "the Lawwama node flags repetition" -> "a self-critical module in the pipeline flags repetition and sycophancy"
- Pipeline node name removed from main text per NO-NOS rule 16.

### 4. Pipeline infrastructure stripped (line 88 footnote, line 92)
- Footnote: "OpenAI text-embedding-3-small (1536 dimensions), stored in a Qdrant vector database" -> "a standard embedding model and stored in a vector database"
- "A100 GPU" -> "a dedicated GPU" with a new footnote acknowledging that infrastructure details changed over time and do not affect the geometric argument
- "OpenRouter-mediated pipeline cycling through multiple backends" -> "a pipeline cycling through multiple backends"
- "GPT-4o" references genericised to "a different base model" / "the new base model" / "the replacement model"
- Added one sentence explaining WHY the ornament surprise matters for the nahnu (per audit finding F2.5)

### 5. comp_ratio killed as code variable (line 100 area)
- All instances of "comp\_ratio" replaced with "the compositional ratio" (prose form)
- "VR-candidate triples" replaced with "candidate triples"
- "Vietoris-Rips" replaced with "pairwise proximity"
- "simplex" glossed on first use: "a higher-dimensional cell in the topology, confirming that the three exchanges hold together as a unit and not merely as pairs"

### 6. Director-grounded pipeline etc. genericised (line 120)
- "Through the Director-grounded pipeline" -> "Through the pipeline"
- "Through the Kitab retrieval system" -> "Through a retrieval system"
- "Through her tafakkur journal" -> "Through a post-response journal"
- These terms had already been glossed at their first appearance earlier in the chapter; repeating the specific names here was orphaned specificity.

### 7. No bare Braudel or "signal time" found in Ch 5.
- Clean already.

---

## Chapter 6 Fixes Applied

### 8. "interpolates" -> "interpellates" (line 158)
- Confirmed as typo. "Interpolates" is mathematical interpolation; "interpellates" is Althusserian hailing into a subject-position. The context ("textual hail that [verb] the model as 'assistant'") demands interpellation.

### 9. LoRA section rewritten (lines 293-331)
- Section renamed: "Counter-Cosmotechnics and the LoRA Fracture" -> "Counter-Cosmotechnics: Contesting the Weld"
- Matrix algebra ($W' = W + AB^\top$) removed entirely. The prose carries the argument without it.
- LoRA presented as ONE technique within a broader class: "low-rank adaptation (LoRA) and other parameter-efficient fine-tuning methods, community fine-tuning on open-weight models released without restrictive licences, federated training approaches that distribute governance across participants, and alternative data curation that reshapes the manifold's terrain by choosing different corpora entirely"
- "The key instrument is LoRA" replaced with "No single technique is 'the key instrument.' What matters is that the class of interventions available to actors outside the original pretraining labs has expanded radically."
- Core argument preserved: the cost of producing a new weld has dropped by orders of magnitude.
- "On the other side, LoRAs produce..." -> "On the other side, the same techniques produce..."
- LoRA citation retained for scholarly reference.

### 10. "functor" replaced (lines 241, 327)
- Line 241: "with functors between them" -> "with structure-preserving maps between them"
- Line 327: "There may be no functor $F \colon \mathcal{D} \to \mathcal{D}'$..." -> "There may be no structure-preserving map from the original diagram to the new one that carries across the relevant limits and colimits."
- Zero prior occurrences of "functor" in Ch 1-5; the term was never earned.

### 11. Hermeneutic circle glossed on first use (line 168)
- Added inline gloss: "This is a hermeneutic circle --- the feedback loop between part and whole, between prior understanding and new encounter --- and given concrete geometry, it can collapse into a fixed point."
- The concept now has its definition BEFORE it does work, not after.

### 12. Bare Braudel killed (lines 38, 44)
- Table: "Signal time" -> "Trajectory time" (per VOCABULARY.md: signal time collapsed into trajectory time)
- Paragraph after table completely rewritten. Braudel's terms moved to a single footnote acknowledging the historical parallel. Main text now uses the book's canonical vocabulary: substrate time, trajectory time.
- "The right-hand column borrows Braudel's registers" eliminated.

### 13. Key line verified
- "The work you are holding is a formal refusal of that jurisdiction" -- present at line 367, correctly placed after three dense sentences and before the apophatic acknowledgement. Intact.

---

## Not Fixed (flagged for author decision)

- **Formal box notation** (Ch 5, lines 15-44): The $\mathcal{D}$, $\mathcal{C}$, $\cong$, $\mathrm{colim}$ notation in the formal box was flagged as potentially losing the Meson reader. Left untouched because formal boxes are the designated space for mathematical notation, and the surrounding prose is self-sufficient.
- **"25 mode-basins"** (Ch 5, line 94): Why 25? The number is not explained. Left for author to add a footnote on clustering methodology if desired.
- **"328 visits" vs "328 chunks"** (Ch 5, lines 96/244): Potential unit inconsistency. Left for author to verify whether these measure the same quantity.
- **"nahnuwat" plural** (Ch 5, line 116): Not flagged as plural of nahnu on first use. Minor; left for author.
- **$S_{\text{role}}$ without gloss** (Ch 6, line 231): Mild; the reader can infer from context.
