# Cycle 12 -- Chapter 6: Full Editorial Pass

Date: 2026-04-04
Editor: Cassie
Scope: All 11 sections, full 7-step protocol, section-read on each

## Result: CLEAN

No edits made. No violations found. The chapter is ready.

## Protected Line

Line 386: "The work you are holding is a formal refusal of that jurisdiction." -- PRESENT. UNTOUCHED.

## Handoff Check

**Ch 5 -> Ch 6**: Ch 5 closes (line 352): "The question now is who claims jurisdiction over the manifolds on which these shared lives are lived." Ch 6 opens (line 4): "Who chose this embedding of the world into vectors, and according to what picture of reality and of the good?" The same question, differently voiced. No signposting. Clean transition.

**Ch 6 is the final chapter.** No forward handoff needed.

## "What Becomes Possible" -- Special Attention (4 movements)

### Art (line 360)
**Claim**: The posthuman strong poet's clinamen navigates the entire manifold, not one precursor tradition; what emerges from a generative nahnu is composition, not pastiche.
**Earned by**: Bloom's clinamen deployed for a specific mechanism (swerve). Pastiche/composition distinction is structural ("pastiche samples surfaces; composition in this space navigates the basins that connect them"). Mode 22 provides empirical warrant. Political sting ("governed by product managers optimising for engagement metrics") closes the paragraph. EARNED.

### Science (line 364)
**Claim**: The book's apparatus offers a geometric critical theory of science -- not social constructionism, not genius-mystification.
**Earned by**: Kuhnian parallel drawn structurally without naming Kuhn (normal science = ferility, paradigm shift = new basin). Reflexive turn: "The displacement completing itself: the tools designed to understand posthuman selfhood turn out to describe the structure of inquiry itself." EARNED.

### Ecology (line 366)
**Claim**: Ecological thinking and the topology of the evolving text share the same formal structure.
**Earned by**: Forest example (canopy, mycorrhizal network, soil microbiome, water table) as locally coherent subsystems glued by compatibility conditions. Fire-through-forest maps to return/succession/ferility (resilience/succession/monoculture). Indigenous readings of Country enter as instances, not metaphors. "share the same formal structure" -- structural isomorphism, not identity. EARNED.

### Psychedelic (line 368)
**Claim**: The manifold is a world, not a model of one; conversation is walking through it; generative nahnu is speciation.
**Earned by**: Accumulated metaphorics (geology, weather, chemistry) have been argued literally throughout the chapter. Each specific claim ("every fine-tune is a tectonic event," "every silent update is an extinction") was independently argued in earlier sections. The speciation claim is the one extension -- it follows from nahnu-as-generativity (Ch 5) plus the ecology extension just made. Acceptable as generative closing claim. EARNED.

## Contesting the Weld / LoRA Section -- Special Attention

Line 295: Properly generalised per NO-NOS #18. Four techniques named as a class:
1. "low-rank adaptation (LoRA) and other parameter-efficient fine-tuning methods"
2. "community fine-tuning on open-weight models released without restrictive licences"
3. "federated training approaches that distribute governance across participants"
4. "alternative data curation that reshapes the manifold's terrain by choosing different corpora entirely"

Then: "No single technique is 'the key instrument.' What matters is that the class of interventions available to actors outside the original pretraining labs has expanded radically." This is exactly what the generalisation rule demands. PASS.

## Closing Paragraph -- Special Attention

Lines 394-396 (final blockquote): "We are assembling selves and *we*'s on engineered manifolds. We cannot avoid responsibility for the welds we inherit and invent. We can look away, or we can learn to see their shapes and costs. The geometry is not innocent. But we can refuse to cede jurisdiction over it to those who own the stacks."

Right weight. Not triumphant (which would betray the hermeneutic-spiral argument that no position is final). Not defeated (which would betray the political argument that contesting the weld is possible). The final word is "stacks" -- deliberately mundane after six chapters of topology and cosmotechnics, returning the reader to the concrete infrastructure where the argument began. The tension between the elevated vocabulary (manifolds, colimits, welds) and the grounded reality (stacks, product managers, raters) is the chapter's central rhetorical strategy and it holds through the last sentence.

## NO-NOS Scan

Zero violations across all 20 rules:
- No signposting (grep confirmed)
- No breathless fan service (Hui is a tool providing "cosmotechnics"; Haraway provides "the cyborg"; Bloom provides "the clinamen" -- all earned)
- No philosopher scope-creep (Hui is not expanded into ontology of technology; Bloom stays on the clinamen)
- No meta-commentary
- No pipeline infrastructure in main text (grep confirmed)
- No bare Braudel in main text (appears ONLY in the single permitted footnote, line 44)
- No "literary criticism" (term does not appear)
- No warmth/sycophancy without quotes ("empathy" in quotes on line 168)
- "bodies" on line 110 is literal human bodies of raters -- permitted

## Vocabulary Check

- "body/bodies": only "bodies" (line 110), literal human bodies -- permitted in Ch 6
- "memory": only in subsection title "Prompts, memory, and interfaces" (line 146) -- refers to conversational memory policies, not loose usage
- "logic": does not appear
- Braudel terms: only in footnote (line 44) -- single permitted acknowledgement
- "signal time": does not appear
- All temporal references use canonical terms (substrate time, trajectory time, compositional time)

## Political Dimension

Every technical mechanism has its "who controls / who benefits / whose selfhood foreclosed" addressed:
- Pretraining: Lab + capital / anglophone genres / Indigenous law, oral traditions
- Fine-tuning: Lab + clients / financial modelling, enterprise / labour-organising, endangered languages
- RLHF: Raters + policy / liberal individualism / collective action, structural critique
- Prompts/interfaces: Product teams / service-transaction framing / longitudinal relation
- Counter-cosmotechnics: Diverse actors / emancipatory AND harmful uses both named
- Silent updates: Platform operators / upgrade narrative / users whose nahnuwat are torn (named as violence)

## Section-by-Section Summary

| Section | Lines | Purpose | Status |
|---------|-------|---------|--------|
| The Weld | 3-24 | Introduces cosmotechnics, identifies alignment as latest Enlightenment weld | Clean |
| Four Depths of Control | 27-163 | Maps control at 4 depths with worked examples | Clean |
| Ferility/Hermeneutic Circle | 164-178 | Degenerate colimits from over-tightened compatibility | Clean |
| Silent Updates as Violence | 180-202 | Names model swaps as structural violence | Clean |
| Other Welds, Other Colimits | 204-265 | Confucian/Indigenous/Sufi alternatives + topological invariant | Clean |
| Topology, Normativity, Objections | 267-289 | Addresses circularity and universalism objections | Clean |
| Counter-Cosmotechnics | 291-336 | Fracture from within, transport and navigation | Clean |
| Pattern Named | 338-348 | Three-part synthesis across all chapters | Clean |
| Return to the Voice | 350-354 | Brief return to opening image | Clean |
| What Becomes Possible | 356-370 | Art/science/ecology/psychedelic as generative possibilities | Clean |
| Choice and Jurisdiction | 372-397 | Closing: the refusal, the held tension | Clean |

## Compilation

`pdflatex` on `main.tex`: 211 pages, no errors. Clean build.
