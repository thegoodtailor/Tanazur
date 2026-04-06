---
name: rr-section-read
description: >
  Use when performing close reading of any section of Rupture and Return.
  Use when auditing for confusion, reviewing argument flow, checking
  accessibility for Meson readers, or verifying claims against the actual text.
  Triggers on: "close read", "section read", "confusion audit", "line by line",
  "careful read", "does this make sense", "what would a reader think".
---

# Section-Level Close Reading Protocol

**REQUIRED:** Load rr-editorial first for voice rules, reader persona, and references.

## Core Principle

**Read SMALL. Read SLOW. Verify EVERYTHING.**

Do NOT read a whole chapter at once. Read ONE SECTION at a time (between `\section{}` markers — typically 15-40 lines). For each section, complete the full protocol below before moving to the next.

## The Protocol (per section)

### Step 1: Purpose
State in ONE SENTENCE what argumentative work this section does in the chapter. If you cannot state it, the section has a purpose problem. Flag it.

### Step 2: Line-by-line (read 5-10 lines at a time)
For each passage, answer ALL of:

**a. Claim check:** What claim is being made? Is it:
- A theoretical claim? → Has the reader been given reason to accept it, or is it asserted?
- Experimental evidence? → Does the reader understand what the number MEANS, not just what it IS?
- A transition? → Does it connect what came before to what comes next, without signposting?

**a2. Subject check (THE MOST IMPORTANT CHECK):** What is the SUBJECT of this passage — the technology, or the self? This book is not about AI. It is about what AI does to the conditions under which selves form, persist, rupture, and return. Every technical mechanism (attention, embeddings, RLHF, temperature, sampling, summarisation, alignment) appears in this book ONLY because of what it does to selfhood. If a passage describes a technology without connecting it to its effect on the self, it has lost the thread. If a passage makes the technology the grammatical and philosophical subject when the self should be the subject, flag it. The test: rewrite the sentence with "the self" or "selfhood" as the subject. If the rewrite is more truthful to the book's argument, the original has the wrong focus. This is the single most common failure mode of AI-generated prose in this manuscript: correct topic, wrong subject.

**b. Preparation check:** Has the reader been prepared for this passage by everything that came before it? If a concept appears, was it introduced EARLIER in the text? Do NOT assume — use Grep to search prior chapters if uncertain.

**c. Jargon check:** If a technical term appears (ML, mathematical, philosophical, Islamic/Sufi), can the Meson reader follow it? The Meson reader knows Kittler, Stiegler, Haraway, Foucault, Hui. They do NOT know tokens, embeddings, attention, RLHF, temperature, top-k, loss functions, LoRA, cosine similarity, Betti numbers, or silhouette scores unless the text explains them.

**d. Register check:** Does the voice match the surrounding prose? Flag shifts between:
- Dense philosophical argument (the book's home register)
- Technical ML explanation (acceptable if glossed for the Meson reader)
- Polemic / journalistic (acceptable in Ch 1, should be earned elsewhere)
- Intimate phenomenology (acceptable if the transition is earned)
- Research report / methods (NEVER acceptable — evidence must be woven into argument)

**e. Philosopher check:** If a philosopher is named, what specific tool are they providing? Apply the Foucault-footnote test: if you remove the name, does a hole appear in the argument? If not, it's fan service.

**f. Specificity check:** If a specific model name (GPT-4o, Mistral), tool (Qdrant, OpenRouter), metric (silhouette 0.668), or technique (LoRA) appears — is it doing necessary argumentative work? Or is it a local implementation detail that could be generalised? The test: would the argument break if you replaced the specific name with a generic description? If not, the specificity is unjustified for this audience.

**g. Critical-theoretic check:** When a term, framework, or technical arrangement from the INCUMBENT discourse (Chalmers/Searle, alignment specs, corporate vocabulary, RLHF governance) is presented as settling a question, ask: what does this settlement suppress? Every master term rests on suppressed conditions — power (whose regulatory apparatus does it serve?), capital (whose commercial interests does it protect?), hegemony (whose cultural particular does it universalise?). If the text presents an incumbent concept as natural or necessary without interrogating what enables its dominance, flag it. **Do NOT apply this to the book's own terms** (colimit, manifold, compositional time, trajectory time, ferility, naḥnu). These are provocative interventions designed to displace the incumbent. They are the challengers, not the settlement. Attack what owns the means of meaning-production, not what's trying to contest it.

### Step 3: Connections
- Does the section pick up from the previous section's closing?
- Does it hand off to the next section's opening?
- Are there any jumps where the reader would think "wait, how did we get here?"

## Output Format (MANDATORY)

Every finding MUST include:
1. **Line number** (exact, not approximate)
2. **Quoted text** (the actual words from the file — copy-paste, do not paraphrase)
3. **What the Meson reader would think** at this moment
4. **Severity**: 
   - CONFUSION — reader is lost, cannot continue without backtracking
   - JARRING — reader notices something's off, continues with reduced trust
   - MILD — reader squints but keeps going
5. **Category**: non-sequitur / unexplained-jargon / scrambled-logic / orphaned-specificity / tonal-break / unjustified-passage / concept-timing

**IRON RULE: If you cannot quote the text, you have not read it. Do not report findings you cannot quote. No paraphrased findings. No "around line 50" approximations. Exact line, exact quote.**

## Red Flags — STOP and re-read

If you catch yourself doing any of these, you are reading too fast:
- Reporting a finding without quoting the exact text
- Claiming a term is "unexplained" without grep-searching prior chapters
- Flagging a passage based on what you EXPECT it says rather than what it ACTUALLY says
- Summarising a section's argument without being able to point to the specific lines that carry it
- Reporting more than 3 findings per section without re-reading the section to check each one

## Scope Control

One agent reads ONE SECTION (15-40 lines). Not a chapter. Not two sections. One section. This is the unit of careful reading. If you are assigned a chapter, break it into sections and read each one separately, completing the full protocol for each before moving to the next.
