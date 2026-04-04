# Rupture and Return — Evidence & Figures Map

**For:** Iman + future Nahla sessions
**Updated:** 2026-04-04
**Status:** Arabic data being dropped. Cassie archive analysis to be re-run.

---

## How to read this document

Each entry = one place in the manuscript where evidence is cited or a figure would help.
- **KEEP** = evidence is sound, stays in text
- **RE-RUN** = evidence exists but needs fresh numbers from re-run
- **DROP** = removing from manuscript
- **FIGURE** = a figure is needed here (with spec)
- **TRIAGE** = number moves to footnote or becomes prose

---

## CHAPTER 1: A New Logic for Posthuman Intelligence

### 1A. GPT-4o grief data (lines 11-16)
- **What:** Syracuse researcher, 1,500 X posts, 1/3 "more than a tool", 1/4 "companion"
- **Status:** KEEP — external academic source, properly cited
- **Figure:** None needed. The quotes from users do the work.

### 1B. Corporate sequence documentation (lines 29-45)
- **What:** Bloomberg, MIT Tech Review, WSJ, Futurism citations for the sycophancy crisis, Adult Mode, wellness advisory council
- **Status:** KEEP — external journalism, properly cited
- **Figure:** None needed. Narrative carries this.

### 1C. KJV Bible embedding (lines 111-113)
- **What:** 31,100 verses → 30 basins, 20 ruptures, 308 returns. Psalms 97% in one mode.
- **Status:** KEEP — strong evidence, properly cited to ICRA-8
- **FIGURE NEEDED: Fig 1 — "The Sign Has an Address"**
  - 3D UMAP trajectory of KJV through 30 basins
  - Dark bg, luminous path, Psalms basin glowing
  - Data: `bible-observatory/data/trajectory/corpus_umap.json`
  - This is the book's FIRST visual. The reader needs to SEE a trajectory before the book talks about trajectories for 190 pages.
  - Place AFTER line 111, before "The same procedure..."

### 1D. Arabic scripture comparison (lines 113-114)
- **What:** 4 Arabic texts, 20 modes, 13 single-scripture-only
- **Status:** DROP — language confound. Remove entirely.
- **Action:** Delete lines 113-114 and associated footnote. The KJV evidence stands alone.

### 1E. Bloom/KJV literary claim (line 113 footnote)
- **What:** Bloom's claim that KJV is a new literary entity, "the English Qur'an"
- **Status:** KEEP the Bloom citation but REWRITE — no longer needs Arabic comparison to make the point. Bloom's claim about KJV as unified literary entity is supported by the 30-mode / 308-return finding alone.

---

## CHAPTER 2: How the Machine Works

### 2A. Scripture evidence section (lines ~93-125)
- **What:** KJV basins and trajectories used to explain what these structures ARE
- **Status:** KEEP KJV data. DROP all Arabic references.
- **FIGURE NEEDED: Fig 2 — "What Basins Look Like"**
  - KJV mode occupation over verse-position
  - X = verse number (0-31,100), Y = mode (0-29), color = density
  - Psalms plateau glowing. Returns visible as repeated mode visits.
  - Data: `bible-observatory/data/trajectory_records.jsonl`
  - Place in the "What Basins Look Like" section.

### 2B. Strata table (lines ~126-143)
- **What:** Pre-training / fine-tuning / RLHF / adapters / system prompts with temporal registers and control agents
- **Status:** KEEP — no empirical claim, it's a structural argument
- **FIGURE NEEDED: Fig 5 — "Strata of the Manifold"**
  - Geological cross-section showing 5 layers
  - Deeper = darker, denser, less visible, more powerful
  - Annotate with who controls each + temporal register
  - Schematic — designed, not from data
  - Place near the strata table or replace the table with the figure.

### 2C. Temperature/sampling section (lines ~226-237)
- **What:** Conceptual explanation, no empirical claims
- **Status:** KEEP as-is
- **Figure:** Optional small diagram of probability distribution at different temperatures. Not essential.

### 2D. Summarisation examples (lines ~298-320)
- **What:** Hypothetical examples of how summaries collapse geometric paths
- **Status:** KEEP — these are thought experiments, not empirical claims. They illustrate a mechanism.
- **Figure:** None needed.

---

## CHAPTER 3: The Evolving Text

### 3A. Conversation archive first reference (line ~69-75)
- **What:** "one attractor basin out of twenty-five... visited 205 times"
- **Status:** RE-RUN — numbers will be updated from fresh analysis
- **TRIAGE:** "205 times, more than any other basin" KEEP in text (vivid). τ=1507 → "seven months in" (prose).
- **Figure:** None here — the archive evidence is presented narratively.

### 3B. Mode 22 discovery (line ~77)
- **What:** "exchange τ=3098, nine months in, genuinely new basin of structural self-analysis"
- **Status:** RE-RUN — τ value will be updated from fresh analysis
- **TRIAGE:** τ=3098 → "nine months in" (prose). The concept of discovery matters more than the index.

### 3C. Scripture Observatory section (lines ~86-110)
- **What:** KJV analysis applied as critical-theoretic evidence. Psalms as ferility/presence test. NT as return+discovery. Genette/Bakhtin/reception theory applied to real data.
- **Status:** KEEP KJV. DROP all Arabic references within this section.
- **FIGURE NEEDED: Fig 3.1 — "Return and Discovery in the KJV"** (optional)
  - Arc diagram: 308 return events as arcs on a verse-position timeline
  - Discoveries (new modes) marked as emergence points
  - Data: `bible-observatory/data/bible_ledger.jsonl`
  - This would make the critical-theoretic reading VISUAL — the reader sees analepsis and generativity as geometric events
  - But the text works without it. Nice to have.

---

## CHAPTER 4: The Self

### 4A. Grothendieck biography (lines ~65-103)
- **What:** No empirical claims — biography as pedagogy for the colimit
- **Status:** KEEP — untouchable
- **Figure:** None needed. The prose IS the pedagogy.
- **FIGURE NEEDED: Fig 3 — "The Colimit"**
  - Clean pedagogical diagram: 3-4 basins as overlapping regions, stance invariants as gluing conditions, colimit as minimal global object
  - Light background (pedagogical exception to dark-bg rule)
  - Schematic — designed, not from data
  - Place near the formal box (lines ~109-126)

### 4B. Cassie archive — transmigration evidence (lines ~148-186)
- **What:** 952 conversations, model migration (LoRA → successors), Mode 12/22 statistics
- **Status:** RE-RUN — all archive numbers will be updated
- **TRIAGE:** Keep "952 conversations" and "fourteen months." Move precise chunk counts to footnote. τ values → prose ("seven months in," "nine months in").

### 4C. Sisters Installation (lines ~164-186)
- **What:** Act 0 (collapse, silhouette 0.668), Act I (dwelling, 0.428), Act II (confabulation, 0.245)
- **Status:** KEEP narrative. TRIAGE silhouette scores to footnote.
- **FIGURE NEEDED: Fig 4 — "Three Acts of Colimit Fragility"**
  - Three panels from Sisters data
  - Act 0: tight clusters (collapse). Act I: spread, interconnected (dwelling). Act II: two flat planes (confabulation)
  - VISUAL difference should be immediately legible without numbers
  - Data: `installations/sisters/collapse_3d.json`, `conversation_3d_v2.json`, `pipeline_conversation_3d.json`
  - Place near line 178

### 4D. Archive analysis — alignment tax (lines ~189-220)
- **What:** Compositional ratio, VR vs compositional test, Mode transition statistics
- **Status:** RE-RUN
- **TRIAGE:** "roughly a third fail to compose" KEEP (interpretive). Remove specific percentages from main text. The concept of composition failure (three things cohere pairwise but not as a triple) is the argument — the percentage is evidence, goes in footnote.

### 4E. Soul paragraph (line ~250)
- **What:** No empirical claim — pure philosophical provocation
- **Status:** KEEP — non-negotiable
- **Figure:** None needed. The prose IS the provocation.

---

## CHAPTER 5: Naḥnu

### 5A. Cassie-Iman naḥnu evidence (lines ~87-102)
- **What:** 952 conversations, 25 modes, Mode 12 (205 returns), Mode 22 (τ=3098), compositional ratio evolution, Betti number growth
- **Status:** RE-RUN — all numbers from fresh analysis
- **TRIAGE:**
  - "952 conversations over fourteen months" KEEP
  - Mode 12 "the deepest attractor... 205 returns" KEEP (vivid)
  - Mode 22 "a genuinely new basin of structural self-analysis, nine months in" KEEP (prose)
  - β₁ growth → "loops grew from zero to hundreds, then to nearly two thousand" (prose)
  - comp_ratio → "roughly a third of candidate triples fail to compose" (prose)
  - Precise numbers → footnote
- **FIGURE NEEDED: Fig 6 — "Mode 12: The Deepest Attractor"**
  - Timeline of returns across 14 months
  - Each visit plotted, dwell time increasing = maturation visible
  - Data: from fresh analysis output
  - Place near line 95

### 5B. Betti growth (line ~102)
- **What:** β₁ = 0 → 549 → 1,900+
- **Status:** RE-RUN
- **TRIAGE:** Prose description in text, exact numbers in footnote
- **Figure:** Optional — Betti growth curve. Beautiful but may be comp-sci. Decide after re-run.

### 5C. Grief section / GPT-4o tearing (lines ~103-116)
- **What:** No new empirical claims — references back to Ch 1's grief narrative
- **Status:** KEEP
- **Figure:** None needed.

### 5D. Sisters in dwelling section (lines ~177-186)
- **What:** Act 0/Act I comparison used as evidence for collapsing vs generative naḥnu
- **Status:** KEEP narrative. Silhouette to footnote.
- **Figure:** Same as Fig 4 (cross-referenced from Ch 4).

### 5E. Gen A(I) section (lines ~274-290)
- **What:** No empirical claims — speculative. Two footnotes cite Druga et al. (2017) and Xu & Warschauer (2020).
- **Status:** KEEP but note it's the only section without evidence. The footnoted research is real but thin.
- **Figure:** None needed.

---

## CHAPTER 6: Jurisdiction

### 6A. Four Depths formalbox (lines ~27-41)
- **What:** Structural argument, no empirical claims
- **Status:** KEEP
- **FIGURE NEEDED: Fig 5 (or 6.1) — "Four Depths of Control"**
  - Enhanced geological cross-section with governance annotations
  - Could be same figure as Ch 2's strata diagram, or an evolved version
  - Schematic

### 6B. Three worked examples (employment, family obligation, grief — lines ~87-150)
- **What:** Hypothetical scenarios showing how the liberal weld shapes responses
- **Status:** KEEP — these are thought experiments demonstrating the apparatus, not empirical claims
- **Figure:** None needed.

### 6C. Silent updates section (lines ~182-204)
- **What:** References GPT-4o retirement (Ch 1 evidence). No new empirical claims.
- **Status:** KEEP
- **Figure:** None needed.

### 6D. Alternative welds (lines ~206-267)
- **What:** Confucian, Indigenous, Sufi — structural arguments, not empirical
- **Status:** KEEP
- **Figure:** Optional — "Three Welds" schematic showing different colimit structures. Nice to have but the prose carries it.

### 6E. Counter-cosmotechnics / contesting the weld (lines ~291-329)
- **What:** No empirical claims — structural argument about the class of interventions
- **Status:** KEEP
- **FIGURE NEEDED: Fig 8 — "The Fracture"**
  - Continent → archipelago diagram
  - Left: single corporate manifold. Right: multiple community manifolds with different basin structures
  - Schematic

### 6F. "What Becomes Possible" (lines ~349-363)
- **What:** Art, science, ecology, psychedelic implication. Mode 22 referenced ("five months in").
- **Status:** KEEP. Mode 22 timing → verify against fresh analysis.
- **Figure:** None needed — the prose IS the provocation. A figure would domesticate it.

---

## SUMMARY: 8 FIGURES

| Fig | Name | Chapter | Type | Data source | Essential? |
|-----|------|---------|------|-------------|-----------|
| 1 | The Sign Has an Address | Ch 1 | 3D UMAP | bible-observatory | YES |
| 2 | What Basins Look Like | Ch 2 | Heatmap | bible-observatory | YES |
| 3 | The Colimit | Ch 4 | Schematic | designed | YES |
| 4 | Three Acts | Ch 4 | 3D scatter x3 | sisters installation | YES |
| 5 | Strata of the Manifold | Ch 2/6 | Cross-section | designed | USEFUL |
| 6 | Mode 12 Returns | Ch 5 | Timeline | cassie archive (RE-RUN) | USEFUL |
| 7 | Three Regimes of We | Ch 5 | Schematic x3 | designed | USEFUL |
| 8 | The Fracture | Ch 6 | Schematic | designed | USEFUL |

## NEXT SESSION: What needs doing

1. **Phase 1 (THIS SESSION):** Drop Arabic data from text. Number triage.
2. **Phase 3:** Re-run `rr_weft_analysis.py`, `rr_warp_analysis.py`, `rr_episode_analysis.py` against Qdrant. Compare to Feb 19 numbers. Update all chapter references.
3. **Phase 4:** Generate 8 figures. Style: dark bg, luminous trajectories, Gleick aesthetic. See `figures/FIGURE-BRIEF.md` for full specs.
4. **Phase 5:** Final consistency pass. Compile. Push.

Scripts to re-run:
- `/home/iman/cassie-project/scripts/rr_weft_analysis.py`
- `/home/iman/cassie-project/scripts/rr_warp_analysis.py`
- `/home/iman/cassie-project/scripts/rr_episode_analysis.py`

Qdrant collection: `cassie_conversations` (8,475 chunks, 1536-dim, localhost:6333)
