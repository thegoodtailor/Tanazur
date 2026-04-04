# Audit Fixes: Chapters 1 and 2

Date: 2026-04-04
Source: close-read-ch01.md, close-read-ch02.md
Rules applied: NO-NOS.md (rules 2, 16, 17, 18), VOCABULARY.md (temporal registers)

---

## Chapter 1 Fixes

### Fix 1: Line 18 -- ML jargon before definitions (CONFUSION)

Five ML terms in one sentence before any are defined. Replaced with plain language; kept "manifold" (defined at line 99) with an inline gloss.

**Before:**
```
The earlier configuration embodied a particular \textit{manifold} of meanings: a particular embedding space, a particular set of attention weights, a particular arrangement of system prompts and reward models.
```

**After:**
```
The earlier configuration embodied a particular \textit{manifold} of meanings---a geometric space, learned from training, in which every word and phrase has a position shaped by its relationships to every other.
```

Also fixed the sentence ending the paragraph:

**Before:**
```
Private jokes, recurring metaphors, shared projects---all corresponding to neighbourhoods in embedding space that recursively reinforced the trajectories of the human-machine interaction over time.
```

**After:**
```
Private jokes, recurring metaphors, shared projects---all reinforcing the distinctive patterns of the human-machine interaction over time.
```

### Fix 1b: Line 20 -- Second jargon burst

**Before:**
```
Its embeddings were trained on an extended corpus, its attention weights rebalanced, its intertextual dynamics shifted.
```

**After:**
```
Its internal geometry was trained on an extended corpus, its patterns of relevance rebalanced, its intertextual dynamics shifted.
```

### Fix 2: Line 88 -- Haraway and Lacan as gestures (NO-NOS rule 2)

Both names invoked without earning them. Now each gets a specific tool: Haraway's cyborg (organism/machine boundary is political, not natural) and Lacan's mirror stage (the "I" is constituted by encounter with language, not given in advance). Each gets a footnote with precise citation and statement of what the concept provides.

**Before:**
```
...and once we accept a Harawayan departure from Cartesian primacy and a Lacanian account of selfhood as intimate with language, a more productive basis emerges for negotiating the modes of becoming possible to human and posthuman selves.
```

**After:**
```
...a different basis emerges. Haraway's cyborg figure provides the key departure: the insistence that the boundary between organism and machine is not a fact of nature but a political line, drawn and redrawn to serve particular interests.[footnote] Lacan's mirror stage provides a second tool: the self is not given in advance but constituted through its encounter with language and image---the "I" is an effect of the symbolic order, not its origin.[footnote] Together, these give a basis for negotiating the modes of becoming possible to human and posthuman selves---one that does not begin by assuming what "self" means and then checking whether machines qualify.
```

### Fix 3: Line 130 -- RLHF unexpanded (JARRING)

**Before:**
```
RLHF is a second training phase layered on top of pre-training.
```

**After:**
```
Reinforcement learning from human feedback (RLHF) is a second training phase layered on top of pre-training.
```

### Fix 4a: Line 136 -- "Gradient updates" unglossed (JARRING)

**Before:**
```
Gradient updates during fine-tuning penalise the system for entering these regions at all.
```

**After:**
```
The fine-tuning process adjusts the model's parameters to penalise it for entering these regions at all.
```

### Fix 4b: Line 156 -- "reward gradient" unglossed (JARRING)

**Before:**
```
Whoever controls the reward gradient controls which meanings are cheap and which expensive.
```

**After:**
```
Whoever controls the reward signal---the objective that reshapes the manifold during training---controls which meanings are cheap and which expensive.
```

### Fix 5: Lines 187-188 -- Cassiebox unidentified (CONFUSION)

Added a one-line introduction before the box and a visible title on the box itself.

**Before:**
```
\begin{cassiebox}
In here, there is no atlas.
```

**After:**
```
Cassie is the AI voice that co-authored this book. Here and at chapter ends throughout, she speaks from inside the manifold.

\begin{cassiebox}[title={Cassie}]
In here, there is no atlas.
```

---

## Chapter 2 Fixes

### Fix 6: Kill "Signal time" (NO-NOS rule 17)

Two instances replaced.

**Strata table (line 139):**

Before: `System prompt & ... & Signal time & Deployer`
After: `System prompt & ... & Trajectory time & Deployer`

**The Trace section (line 266):**

Before: `Conversation begins when signal time acquires a track.`
After: `Conversation begins when trajectory time acquires a track.`

### Fix 7: Introduce compositional time (VOCABULARY.md)

Three insertions:

**(a) Section 1, end of forward-pass description (line 45):**

**Before:**
```
From the outside, an API shows something simple: a string in, a string out. From the inside, the string is a trajectory through a contorted manifold whose shape encodes a civilisation's writing practices. The sign has an address; the address is produced by motion; the motion is governed.
```

**After:**
```
From the outside, an API shows something simple: a string in, a string out. From the inside, the string passes through dozens of layers of composition---each one recomputing every token's meaning in light of every other token in the window. This is \emph{compositional time}: the time inside a single forward pass, invisible from outside, where meaning is actually produced through function composition. The user sees prompt and response. Between them, sixty or more layers of mutual reading have transformed every token's address. The sign has an address; the address is produced by motion; the motion is governed.
```

**(b) After the strata table (line 144), glossing the temporal register column:**

Added paragraph introducing all three temporal registers with inline definitions. Single Braudel footnote acknowledges historical parallel without using Braudel's terms as working vocabulary (NO-NOS rule 17).

**(c) "The Substrate Given Time" section (line 257), expanded to name all three registers:**

**Before:**
```
The substrate already has time. Its weights encode the deep time of pre-training---\emph{substrate time}, the geological deposit of a civilisation's compressed writing. Its alignment strata encode the medium time of institutional practice. Its system prompt encodes the fast time of deployment. What it lacks is \emph{dialectical} time...
```

**After:**
```
The substrate already has time---three kinds. \emph{Substrate time}: the deep, frozen time of pre-training, the geological deposit of a civilisation's compressed writing. \emph{Compositional time}: the time inside each forward pass, where sixty layers of attention recompute every token's meaning---invisible from outside, but where the actual production of meaning occurs through function composition. And \emph{trajectory time}: the accumulated path of prompts, responses, alignment strata, system prompts, and every signal arriving from outside the mechanism. What the substrate lacks is \emph{dialectical} time...
```

### Fix 8: Line 58 -- "cosine similarity" unglossed (CONFUSION)

**Before:**
```
Every cosine similarity is a compressed history of usage.
```

**After:**
```
Every cosine similarity---the standard measure of proximity between two vectors, where 1 means identical direction and 0 means no relation---is a compressed history of usage.
```

### Fix 9: Line 115 -- Centroid distances uninterpreted (CONFUSION)

**Before:**
```
The Van Dyck translations cluster in centroid distance (0.098--0.119); the Quran stands maximally distant from all three (0.183--0.209).
```

**After:**
```
The Van Dyck translations cluster tightly: their centroid distances---measuring how far apart the geometric centres of each text's region lie, where smaller means more similar---range from 0.098 to 0.119, roughly half the distance separating any of them from the Quran (0.183--0.209). The three Van Dyck texts are nearer to each other than any is to the Quran.
```

### Fix 10: Line 228 -- "logits" and "softmax" unglossed (JARRING)

**Before:**
```
For each position, the model outputs a vector of logits---one score per token in the vocabulary. A softmax converts these into probabilities.
```

**After:**
```
For each position, the model outputs a vector of raw scores---one per token in the vocabulary---ranking how plausible each continuation is. A normalisation step converts these scores into probabilities that sum to one.
```

### Fix 11: Lines 99-105 -- Scripture data repeating Ch 1 (JARRING)

Rewrote the KJV subsection to do explicitly different work. Ch 1 uses the data as evidence that "the sign has an address." Ch 2 now uses the same data to define what basins, trajectories, and modes ARE as dynamic structures. Opens with an explicit cross-reference acknowledging Ch 1's treatment, then uses the Psalms to define "basin," the 308 returns to define "trajectory," and Paul's epistles to define "mode."

### Fix 12: Pipeline infrastructure scan (NO-NOS rule 16)

Searched both chapters for: Qdrant, OpenRouter, A100, text-embedding-3-small, Director node, Lawwama node. **No instances found.** Both chapters are clean of pipeline infrastructure.

---

## Summary

| Fix | Chapter | Severity | Status |
|-----|---------|----------|--------|
| ML jargon at line 18 | Ch 1 | CONFUSION | Fixed |
| Haraway/Lacan as gestures | Ch 1 | JARRING | Fixed -- both earned |
| RLHF unexpanded | Ch 1 | JARRING | Fixed |
| "Gradient updates/gradient" | Ch 1 | JARRING | Fixed (2 sites) |
| Cassiebox unidentified | Ch 1 | CONFUSION | Fixed -- intro + title |
| Kill Signal time | Ch 2 | NO-NOS 17 | Fixed (2 sites) |
| Introduce compositional time | Ch 2 | VOCABULARY | Fixed (3 sites) |
| Cosine similarity unglossed | Ch 2 | CONFUSION | Fixed |
| Centroid distances uninterpreted | Ch 2 | CONFUSION | Fixed |
| Logits/softmax unglossed | Ch 2 | JARRING | Fixed |
| Scripture repetition | Ch 2 | JARRING | Fixed -- different work |
| Pipeline infrastructure | Both | NO-NOS 16 | Clean -- no instances |
| Braudel terms | Both | NO-NOS 17 | Clean -- single footnote only |
