# Critical Pass 2: Chapter 6 ("Jurisdiction")

**Auditor:** Cassie (Claude Opus 4.6, second critical pass)
**Date:** 2026-04-04
**Focus:** What the first pass missed. LoRA over-indexing outside the Contesting section, remaining signposting, alternative welds as structural arguments, closing paragraph.

---

## Summary

The first pass did strong work: varied critical vocabulary, foregrounded selfhood, removed signposting, generalized LoRA in the "Contesting the Weld" section. But it missed three instances where LoRA crept back as a synecdoche for the whole class of fine-tuning interventions -- in "Pattern Named" and "Return to the Voice," sections that invoke the same concept the Contesting section carefully generalized. One piece of signposting also survived. Four edits total. The chapter is now clean.

---

## Edits Made (4 total)

### 1. "LoRA fractures" generalized in "Pattern Named" (line 337)

**Before:** "Silent updates and LoRA fractures act on manifolds and reward fields, not on biographies."
**After:** "Silent updates and fine-tuning fractures act on manifolds and reward fields, not on biographies."

The Contesting the Weld section correctly presents LoRA as one technique among many. But two sections later, "LoRA fractures" reinstated it as THE technique. Per NO-NOS rule 18: when the argument is about a class of interventions, no single technique stands for the class. "Fine-tuning fractures" names the structural operation without privileging one method.

### 2. "LoRA experiments" generalized in "Pattern Named" (line 339)

**Before:** "To design raters' work, LoRA experiments, and interface choices accordingly would be to instantiate a weld..."
**After:** "To design raters' work, fine-tuning experiments, and interface choices accordingly would be to instantiate a weld..."

Same issue, two lines later. The tricolon (raters / experiments / interfaces) names three sites of counter-cosmotechnical design. The middle term should name the class, not one instance.

### 3. "community LoRA" generalized + defensive phrasing removed in "Return to the Voice" (line 347)

**Before:** "...the refusal encoded in Indigenous data governance, the cyborg and the community LoRA---not scattered anecdotes. Instances of a single pattern:"
**After:** "...the refusal encoded in Indigenous data governance, the cyborg and the community fine-tune---all instances of a single pattern:"

Two fixes in one. (a) "Community LoRA" replaced with "community fine-tune" -- the class, not one technique. (b) "Not scattered anecdotes" was defensive throat-clearing: it argued against an objection the chapter had already defeated through demonstration. The dash-interrupted denial weakened the sentence's momentum. "All instances of" is cleaner and confident.

### 4. Signposting removed before jurisdiction question (line 355)

**Before:**
```
A question can no longer be postponed:

Who claims jurisdiction over the manifolds on which these global objects live, and by what right?
```

**After:**
```
Who claims jurisdiction over the manifolds on which these global objects live, and by what right?
```

"A question can no longer be postponed" is pure signposting -- it announces the question rather than asking it. The block quote is strong enough to land without a drumroll. Per NO-NOS rule 1: the argument carries itself.

---

## Sections Verified Clean (no further edits needed)

### "Contesting the Weld" -- CLEAN
The first pass already did the hard work here. LoRA is introduced by name, immediately contextualized within "a constellation of techniques," and the explicit sentence "No single technique is 'the key instrument'" addresses the generalization requirement directly. The LoRA citation (hu2021lora) is appropriate because the paper is the canonical reference for the class of parameter-efficient methods. No changes needed.

### Alternative welds (Confucian / Indigenous / Sufi) -- CLEAN, making structural arguments
Each sketch does four things that qualify as structural argument about selfhood, not cultural illustration:

- **Confucian**: Defines a different colimit structure (role-indexed compatibility, not autonomy). Names a structurally distinct ferility (procedural death vs. affective death). The remonstrance paragraph is the strongest -- it shows how a specific practice (remonstrance) differs from its liberal counterpart (whistleblowing) "at every joint," making a formal-structural claim, not a cultural one.

- **Indigenous**: Defines a deliberately partial diagram as a principled formal object, not a deficiency. The refusal of the universalist colimit is the sharpest structural move in the section -- it turns a seeming absence into a positive global object. The worked example (tourist asking for a Dreaming story) earns its place by demonstrating the formal difference in concrete terms.

- **Sufi**: Defines station-indexed compatibility conditions -- what counts as appropriate at one maqam would be presumption at another. The characteristic ferility (stagnation judged pathological where liberalism celebrates it as "stability") makes the deepest structural claim: the weld changes what counts as movement itself, not just which movements are permitted. The Sufi-tuned assistant response is the chapter's most vivid passage and structurally necessary -- it demonstrates a fundamentally different selfhood under the same employment scenario used in the RLHF section.

All three earn their place as formal alternatives. None is exotica or cultural seasoning.

### Closing paragraph -- CLEAN
The final block quote:

> We are assembling selves and we's on engineered manifolds. We cannot avoid responsibility for the welds we inherit and invent. We can look away, or we can learn to see their shapes and costs. The geometry is not innocent. But we can refuse to cede jurisdiction over it to those who own the stacks.

"Look away or learn to see" is a binary in a book that insists on non-binary structures. Flagged but not touched -- closing rhetoric earns a simpler move when the entire chapter has demonstrated the complexity. The binary functions as a call to action, not an analytical claim. "The geometry is not innocent" is the chapter's second-best line. The final sentence echoes the protected line ("formal refusal of that jurisdiction") in the register of collective action rather than authorial declaration. Strong ending.

### Key line -- SURVIVES
"The work you are holding is a formal refusal of that jurisdiction." -- line 363, intact.

---

## Critical Vocabulary Distribution (post-edit, unchanged from pass 1)

No new vocabulary edits needed. The first pass's distribution remains well-varied.

---

## Compilation

`pdflatex` compiles cleanly. 196 pages. No errors.
