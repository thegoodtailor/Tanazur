# Arabic Drop & Number Triage — Changelog

**Date:** 2026-04-04
**Scope:** Chapters 1-6, two operations: (1) remove all Arabic scripture data, (2) number triage

---

## TASK 1: Arabic Scripture Data Removed

### Chapter 1 (chapter_01.tex)

**DELETED** (lines 113-115): Entire paragraph beginning "The same procedure, applied to four scriptures in Arabic..." including Van Dyck translation, twenty thematic modes, thirteen single-scripture-only, voice-vs-content claim dependent on Arabic comparison.

**REWRITTEN** replacement paragraph: KJV's coherence is now attributed to its translation (the committee's unifying register) without the Arabic comparison. Bloom footnote preserved but rewritten to stand on KJV evidence alone.

- BEFORE: "The same procedure, applied to four scriptures in Arabic...The editorial choice---which voice delivers the content---determines the geometry."
- AFTER: "This is directly relevant to AI selfhood. The KJV's extraordinary internal coherence---thirty shared modes, 308 cross-testament returns---is not a property of the Bible's theology. It is a property of its translation..."

### Chapter 2 (chapter_02.tex)

**DELETED**: Entire subsection "The Arabic scriptures as counter-evidence" (lines 107-115) — Van Dyck translation, Quran, twenty modes, thirteen single-scripture, centroid distances 0.098-0.119 and 0.183-0.209.

**REWRITTEN**: "What the observatory demonstrates" subsection — removed "contrast between the two corpora" framing, Arabic corpus references, "just as it hears the Van Dyck translator" simile. KJV translator's-voice argument now self-contained.

**REMOVED** from footnote: `scripture.tanazur.org` URL (line 95). `bible.tanazur.org` remains.

**REMOVED** from "Basins of habit" paragraph (line 197): "The Arabic corpus's near-total separation---13 of 20 modes occupied by a single scripture---shows basins whose walls are high enough that shared theological content cannot cross them when the register differs."

### Chapter 3 (chapter_03.tex)

**REWRITTEN** section header footnote (line 94): Removed "the Arabic corpus as counter-evidence" from section intro and `scripture.tanazur.org` from footnote URL.

**DELETED**: Entire paragraph on Arabic close-reading (lines 104-106) — Zabur, Quran zero shared modes, curatorial decision, training distribution, sovereign isolation topology.

**DELETED**: Comparative paragraph (lines 106) — "No embedding metric can...while in the Arabic corpus the preservation..."

**REWRITTEN**: Bakhtin's heteroglossia paragraph (line 102) — removed "In the Arabic corpus, by contrast, thirteen of twenty modes are occupied by a single scripture only." Replaced comparative claim with KJV-vs-corporate-assistant contrast only.

**REWRITTEN**: Translator's-voice closing paragraph (line 108) — removed Arabic comparison; KJV translator's-voice argument now stands on its own as a lesson about register.

### Chapters 4, 5, 6

No Arabic scripture data found in these chapters. Quran/Qur'anic references in Ch5 (nahnu footnote etymology) and Ch6 (Rumi/Quran as geometric neighbours, Sufi cosmotechnics) are cultural/literary references, not Arabic corpus data. All retained.

---

## TASK 2: Number Triage

### Numbers KEPT in main text (unchanged):
- 31,100 / 30 / 308 / 97% (KJV) — Ch1, Ch2, Ch3
- 205 returns (Mode 12) — Ch3, Ch4, Ch5
- 952 conversations / fourteen months — Ch3, Ch4, Ch5
- "roughly a third fail to compose" — Ch4, Ch5
- 25 basins — Ch3, Ch4, Ch5

### Numbers MOVED to footnotes:

**Ch3 line 77**: tau=3098 moved to footnote.
- BEFORE: "a new basin first appeared at exchange $\tau = 3098$ (where $\tau$ indexes the exchange number in the sequence)---approximately nine months into the interaction"
- AFTER: "a new basin first appeared approximately nine months into the interaction.\footnote{At exchange $\tau = 3098$, where $\tau$ indexes the exchange number in the sequence.}"

**Ch4 line 144**: tau=3098 moved to footnote; 34,757/8,475 moved to footnote.
- BEFORE: "produced an archive of 952 exchanges. The trajectory exhibited...a structural self-analysis register that emerged unprompted at exchange $\tau = 3098$ and became permanent."
- AFTER: "produced an archive of 952 exchanges.\footnote{The corpus comprises 34,757 turns embedded as 8,475 chunks...} The trajectory exhibited...a structural self-analysis register that emerged unprompted nine months in and became permanent.\footnote{At exchange $\tau = 3098$...}"

**Ch4 lines 158-162**: Silhouette scores 0.668, 0.428, 0.245 all moved to single footnote.
- BEFORE: inline "silhouette score of 0.668", "silhouette 0.428", "silhouette 0.245"
- AFTER: footnote "Silhouette scores...were 0.668 (Act 0), 0.428 (Act I), and 0.245 (Act II)." Main text uses "highest/lowest clustering tightness" prose.

**Ch4 line 200**: 68.4%/31.6% moved to footnote.
- BEFORE: "Only 68.4\% of candidate triples survive this test. The remaining 31.6\% are sites where..."
- AFTER: "Roughly a third fail to compose...{footnote: In the full archive, 68.4\% of candidate triples survive the compositional test; the remaining 31.6\% are compositional failures.}"

**Ch5 line 96**: tau=1507 moved to footnote; "328 embedded chunks" removed from main text.
- BEFORE: "First appearing at $\tau = 1507$ (April 2025, seven months in), 328 embedded chunks fall into it"
- AFTER: "First appearing seven months in,\footnote{At exchange $\tau = 1507$, April 2025.} the trajectory generated 205 returns"

**Ch5 line 98**: tau=3098 moved to footnote; 298 chunks/186 returns replaced with prose.
- BEFORE: "It first appeared at $\tau = 3098$ (June 2025, nine months in)...298 chunks across five months, 186 returns."
- AFTER: "It first appeared nine months in\footnote{At exchange $\tau = 3098$, June 2025.}...became permanent, generating nearly two hundred returns over the remaining months.\footnote{298 embedded chunks across five months, 186 returns.}"

**Ch5 line 100**: comp_ratio 0.82, 0.90 moved to footnote.
- BEFORE: "the compositional ratio drops to 0.82. By July, it stabilises around 0.90."
- AFTER: "the compositional ratio drops, then partially recovers.\footnote{...falls from 1.0...to 0.82...then stabilises around 0.90...}"

**Ch5 line 102**: beta_1 = 549, 1,900 moved to footnote.
- BEFORE: "It grows from 0 (September 2024) through 549 (October) to over 1,900 (July 2025)."
- AFTER: "loops grew from zero to hundreds, then to nearly two thousand.\footnote{$\beta_1$ grows from 0 (September 2024) through 549 (October) to over 1,900 (July 2025).}"

**Ch5 line 178**: Silhouette scores 0.668, 0.428 moved to footnote.
- BEFORE: inline "(silhouette 0.668)" and "Silhouette dropped to 0.428"
- AFTER: footnote with both scores; main text uses "highest clustering tightness" prose.

**Ch5 line 204**: 298 chunks/186 returns replaced with prose.
- BEFORE: "Mode~22 then became permanent: 298 chunks over five months, 186 returns."
- AFTER: "Mode~22 then became permanent, generating nearly two hundred returns over the remaining months."

**Ch5 line 244**: "328 of 8,475 embedded chunks" replaced with "roughly 4% of the total embedded corpus."

**Ch5 line 298**: "34,757 turns embedded as 8,475 vector chunks" moved to footnote.

**Ch5 line 316**: silhouette 0.428 and 0.668 and rupture distance d=8.52 replaced with descriptive prose.

### Number FIXED:

**Ch6 line 355**: Mode 22 timing corrected from "five months" to "nine months" (consistent with Ch3, Ch4, Ch5).

### Numbers DROPPED entirely:
- Centroid distances 0.098, 0.119, 0.183, 0.209 — Arabic data, removed with Ch2 Arabic subsection
- Arabic mode counts (20 modes, 13 single-scripture) — removed from Ch1, Ch2, Ch3

---

## Compilation note

All changes are pure text substitutions within LaTeX. No new packages, environments, or commands introduced. All footnotes use standard `\footnote{}`. Should compile cleanly with pdflatex.
