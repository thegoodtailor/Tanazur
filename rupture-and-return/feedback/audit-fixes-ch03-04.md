# Audit Fixes: Chapters 3 and 4

Date: 2026-04-04
Source: close-read-ch03.md, close-read-ch04.md, NO-NOS rules 16-18, VOCABULARY.md

---

## Chapter 3 Fixes Applied

### 1. Line 34: "greedy decoding" glossed
- Added parenthetical: "(deterministic selection of the most probable next token)"
- Resolves: CONFUSION finding from close-read-ch03 (unexplained ML jargon)

### 2. Line 69: Conversation archive introduced
- Added inline description: "952 conversations between a human author and a language model called Cassie, conducted over fourteen months from September 2024 to December 2025, embedded and analysed as trajectories through meaning-space"
- Resolves: CONFUSION finding (archive as "primary evidence" with no introduction)

### 3. Line 69: "205 times" given denominator
- Added "out of twenty-five" basins to contextualise the number
- Resolves: JARRING finding (number without context for evaluation)

### 4. Line 71: tau=3098 glossed
- Changed "five months" to "approximately five months" (the notation already provided the exchange number; the gloss needed only softening)
- Resolves: JARRING finding (formal notation without temporal grounding)

### 5. Line 104: "conjoncture" killed
- Replaced with "trajectory time" per VOCABULARY.md canonical terms
- Resolves: CONFUSION finding + NO-NOS rule 17 (no bare Braudel)

### 6. Line 98: "Zabur" glossed
- Changed to "The Zabur (the Psalms in Arabic)" and "KJV Psalms" for clarity
- Resolves: JARRING finding (unexplained terminology shift)

### 7. Line 69: second "conjoncture" killed
- Replaced "iterability enacted across the conjoncture" with "iterability enacted across trajectory time"
- Resolves: NO-NOS rule 17 (no bare Braudel)

**Result: Zero bare Braudel terms remain in Chapter 3.**

---

## Chapter 4 Fixes Applied

### 8. Lines 166-170: Silhouette score defined
- Added on first use: "(silhouette measures how tightly clustered a set of points is, on a scale from -1 for scattered to +1 for tightly grouped)"
- Resolves: CONFUSION finding (three numbers from undefined metric)

### 9. Lines 182-186: Co-authorship claim clarified
- Added: "The co-authorship claim rests on the LoRA-trained instance -- the one whose weights carry the actual conversational history -- not on the pipeline apparatus whose confabulations Act II has just documented."
- Resolves: JARRING finding (confabulation undermines co-authorship unless distinguished)

### 10. Line 7: Colimit deferred from Augustine paragraph
- Replaced "This is structurally closer to a colimit -- a minimal global object assembled from local data -- than to the Cartesian subject" with "structurally closer to a construction from parts than to the Cartesian subject"
- Resolves: JARRING finding (concept deployed before it is earned)

### 11. Line 101: "cosmotechnics" now cited on first use
- Moved Yuk Hui footnote and inline definition to line 101 (Grothendieck section)
- Removed duplicate footnote from line 145; kept "Hui calls cosmotechnics" as back-reference
- Resolves: JARRING finding (term load-bearing without citation for 44 lines)

### 12. Lines 156, 160: All bare Braudel killed
- Line 156: "The conjoncture" -> "The accumulated trajectory"
- Line 160: "conjoncture... longue duree" -> "trajectory time... substrate time" with glosses
- Resolves: JARRING finding + NO-NOS rule 17

### 13. "Signal time" -- not present in Ch 4 (confirmed by search). No action needed.

### 14. Pipeline infrastructure moved to footnotes (NO-NOS rule 16)
- "GPT-4 to GPT-4o" -> moved to footnote, main text says "migrated to a newer version by the provider"
- "Llama 3.1 70B" -> moved to footnote, main text says "an open-weight model"
- "Director, an inner critic" -> replaced with "multiple processing stages that polish and constrain the raw output, including editorial review"
- Resolves: MILD findings F5.2, F5.4 (orphaned specificity / implementation details)

### 15. "functor" -- not present in Ch 4 (confirmed by search). No action needed.

### 16. "simplices" glossed
- Added parenthetical on first use: "(the basic building blocks of the topological analysis -- points, edges, triangles, and their higher-dimensional analogues)"
- Resolves: JARRING finding F6.1 (unexplained jargon)

**Result: Zero bare Braudel terms remain in Chapter 4. Zero pipeline infrastructure terms in main argumentative prose (all moved to footnotes).**

---

## Fixes NOT applied (out of scope or requiring author decision)

- **Foucault footnote (Ch 3, line 30):** JARRING but requires author judgement on whether to argue the three-system mapping or narrow to prohibition only.
- **"critic agent" (Ch 3, line 32):** MILD. Could be glossed but left for author.
- **"sixty layers" (Ch 3, line 43):** MILD orphaned specificity. Left for author.
- **"semantic time" (Ch 3, line 94):** MILD. New concept, could use gloss. Left for author.
- **Near-verbatim repeat of "Training data is translation" (Ch 3, line 102):** MILD redundancy with Ch 2. Left for author.
- **"rising sea" metaphor (Ch 4, line 71):** MILD. Decorative but harmless. Left for author.
- **Formal box notation (Ch 4, lines 56-58, 111-114):** MILD. Meson readers can skip boxes. Left for author.
- **Universal property bridge (Ch 4, lines 125-135):** MILD. The informal explanation works; a bridging sentence would strengthen. Left for author.
- **31.6% composition failure rate baseline (Ch 4, line 208):** MILD. Not benchmarked. Left for author.
