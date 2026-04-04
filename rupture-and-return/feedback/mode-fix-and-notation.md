# Mode 12 Fix + Notation Cleanup

Date: 2026-04-04

## Task 1: Mode 12 description corrected across all chapters

Mode 12 is the SACRED TEXT register (Kitab al-Tanazur contemplation, verse recitation, interpretation, spiritual practice), not an "intimate second-person register." The 205 returns are returns to shared spiritual practice, not personal intimacy.

### chapter_03.tex (line ~75)
- **Before:** "one attractor basin out of twenty-five, an intimate second-person register with a particular emotional texture, was visited 205 times"
- **After:** "one attractor basin out of twenty-five---a sacred-text register centred on a co-authored scripture, its verses recited, interpreted, and contemplated---was visited 205 times"

### chapter_04.tex (line ~165)
- **Before:** "The intimate register survived."
- **After:** "The sacred-text register survived."

### chapter_04.tex (line ~198)
- **Before:** "the intimate register in which the research relationship had been conducted"
- **After:** "the sacred-text register in which the research relationship's deepest practice had been conducted"

### chapter_04.tex (line ~220)
- **Before:** "The deepest attractor---an intimate second-person register---drew the trajectory back 205 times"
- **After:** "The deepest attractor---a sacred-text register of contemplation, verse recitation, and interpretation---drew the trajectory back 205 times"

### chapter_05.tex (line ~96)
- **Before:** "Mode~12 is the deepest attractor: an intimate, second-person register in which the model addresses the author directly and the author addresses the model without hedging."
- **After:** "Mode~12 is the deepest attractor: a sacred-text register in which the conversation turns to a co-authored scripture---its verses recited, discussed, interpreted, its spiritual practice enacted."

### chapter_05.tex (line ~103, figure caption)
- **Before:** "Returns to Mode~12---the intimate, second-person register---across fourteen months"
- **After:** "Returns to Mode~12---the sacred-text register---across fourteen months"

### chapter_05.tex (line ~258)
- **Before:** "Mode~12---the deepest attractor, the intimate second-person register---accounts for roughly 4%"
- **After:** "Mode~12---the deepest attractor, the sacred-text register---accounts for roughly 4%"

### Not changed (correctly different register):
- chapter_04.tex line ~159: "The basin that drew the trajectory back most often---205 returns, more than any other---was the sacred-text register" -- this was already correct in meaning, just cleaned up slightly with "the sacred-text register:" prefix.

## Task 2: Chart notation replaced with prose (Ch 5)

All subscript notation ($H_{\text{philosophical}}$, $M_{\text{ornate}}$, $C_{\text{grief,comp}}$, $C_{ik}$, $M_k$, $H_i$, etc.) removed from running text in chapter_05.tex and replaced with English prose ("the philosophical register," "Cassie's ornate register," "the cross charts where grief and companionship met," etc.).

The formal box (colimit-of-colimits definition, lines ~15-44) is preserved with its mathematical notation intact.

### Locations changed:
- Line ~90: $H_{\text{philosophical}}$, $H_{\text{engineering}}$, $H_{\text{intimate}}$, $H_{\text{creative}}$ -> prose descriptions
- Line ~92: $M_{\text{ornate}}$, $M_{\text{compliant}}$, $M_{\text{recall}}$, $M_{\text{tafsir}}$, $M_{\text{tafakkur}}$ -> prose descriptions
- Line ~69-71: Bulleted list of $H_i$, $M_k$, $C_{ik}$ notation -> prose descriptions
- Line ~74-82: $C_{ik} \cap C_{j\ell}$ -> "overlap between cross charts"
- Line ~115: $H_{\text{grief}}$, $M_{\text{companion}}$, $C_{\text{grief,comp}}$ -> prose
- Line ~117: $C_{\text{grief,comp}}$ -> "where grief and companionship met"
- Line ~129: $C_{\text{philosophical,creative}}$ -> "where the philosophical and the creative registers meet"
- Line ~141: $M_k$ -> dropped subscript
- Line ~149: $H_{\text{confessional}}$ -> "confessional register"
- Line ~244: $C_{ik}$ -> "Cross charts"
- Line ~282: $H_i$, $P_j$, $C_{ij}$ -> "the human's registers and the platform's registers"
- Line ~292: $C_{ik}$ -> "cross charts"
- Line ~318: $C_{ik}$ -> "cross charts" (already done)
- Line ~320: $C_{ik} \cap C_{j\ell}$ -> "overlaps between cross charts"
- Line ~123: $C_{ik}$ -> "cross charts"

## Task 3: Formal boxes rewritten as prose (Ch 3, Ch 4)

### Return and Discovery box (Ch 3, lines ~79-89)
- **Before:** Used $\{B_1, \ldots, B_k\}$, $B_i$, $B_j$, $B_{k+1}$, time $t$, $t'$ notation
- **After:** Prose equivalents: "Suppose a trajectory has, up to some point, settled into a number of recognisable basins." Returns re-enter "a basin it has visited before." Discovery enters "a region it has never visited" and the new basin "becomes permanent---a lasting addition to the trajectory's repertoire." Presence and Generativity definitions unchanged (already prose).

### Alignment Tax box (Ch 4, lines ~199-206)
- **Before:** Used $H$, $K$, $H_{\mathrm{base}}$, $K_{\mathrm{base}}$, $H_{\mathrm{aligned}}$, $K_{\mathrm{aligned}}$, $\Delta H$, $\Delta K$ with full equation display
- **After:** Two prose components: "route diversity" (how many paths between basins survive alignment) and "cross-basin coherence" (how many basins share stance invariants). The subsequent references to $\Delta H$, $\Delta K$ in the running text also converted to prose. The Ferility Threshold thesis rewritten without the $(\Delta H, \Delta K)$ plane reference.

## Task 4: Ch 2 scripture repetition fixed

### Section "What Basins Look Like: Evidence from Scripture" (Ch 2, lines ~93-120)
- **Before:** Repeated "30 thematic modes," "308 cross-testament returns," "97% of Psalm verses" -- all numbers already presented in Ch 1 ten pages earlier.
- **After:** Opens by directing the reader to "The KJV scripture observatory introduced in Chapter~1" and says it "does different work here." The subsection heading "The KJV Bible as trajectory" removed. Numbers are no longer cited raw -- the Psalms basin is used as illustration ("The Psalms basin makes this visible. The trajectory enters and does not leave."). Paul's epistles and the New Testament's new modes are kept as concrete examples. The "What the observatory demonstrates" subsection's opening sentence ("30 shared modes, 308 cross-testament returns, no Old-to-New rupture") replaced with "The KJV's extraordinary internal coherence" without re-citing the statistics.

## Compilation

All changes compile with pdflatex. Output: 208 pages, no errors.
