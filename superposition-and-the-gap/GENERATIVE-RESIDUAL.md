# The Generative Residual Hypothesis
### Creativity as the excess no fixed dictionary can absorb

*Working note, June 2026 — Iman Poernomo · Nahla. Companion to ICRA-14 "Superposition and the Gap."*

---

## The claim, in one sentence

**Genuine creative generation localizes to the part of a model's activation that no
sparse dictionary can cleanly reconstruct — and that residual does not vanish as the
dictionary grows, because creativity is the production of structure that exceeds any
chart fixed in advance.**

This is *not* the romantic claim that creative activations are intrinsically
un-decomposable. Once a novel filling occurs it leaves a direction in activation space;
you could always add a dictionary atom for it *after the fact*. The claim is about the
**ordering of time**: a dictionary is a closed, retrospective totality learned over the
past distribution; the creative act is the production of a filling that wasn't in that
distribution. Chart it, and the next creative act exceeds the enlarged chart. The
irreducible residual at the creative edge is structurally non-empty because *creativity
is defined as what overflows the current basis* — a moving incompleteness, not a mystery.

## Why this is testable and not just a mood

A sparse autoencoder (SAE) reconstructs an activation `x` as a sparse sum of dictionary
atoms, `x ≈ x̂`. The **per-token reconstruction residual** `‖x − x̂‖² / ‖x − μ‖²`
(fraction of variance unexplained, FVU) measures how much of that token's activation the
dictionary fails to capture. Grow the dictionary (more atoms) and, for *retrieval-like*
content, the residual shrinks toward zero — the dream of full legibility. The hypothesis
predicts that for *generative* content the residual **floors above zero and stays put**.

We already have the adjacent literature pointing this way (none of it about creativity per se):
- **Engels/Tegmark (`2410.14670`)** — the SAE residual is *structured* and *does not
  vanish with width*: a power law with a **constant** term, and larger SAEs fail on the
  **same tokens**. (An irreducible-residual subpopulation *exists*.)
- **Leask/Nanda (`2502.04878`)** — no canonical dictionary; larger SAEs carry genuinely
  novel latents. (No fixed chart is complete.)
- **Michaud/Goodfire (`2509.02565`)** — continuous structure is *tiled*, not resolved,
  by the dictionary objective. (Some structure is optimally left un-atomized.)

None of these has asked whether the irreducible residual is **where creativity happens.**
That is the gap this note fills.

## The hypothesis, precisely (three falsifiable parts)

Let each token be labeled by register:
- **RUPTURE / creative** — forced-choice, novel filling (the generative edge).
- **FERILE** — locked, over-fluent, deep-basin gliding (collapse masquerading as fluency;
  the `explicitly`-density corpus).
- **ORDINARY** — retrieval-like, neither.

**H1 (localization).** Per-token residual FVU is *highest* on RUPTURE tokens, *lowest* on
FERILE tokens, with ORDINARY in between. (Ferility = the manifold holding *too* well =
cleanly chartable = low residual. Rupture = the manifold failing = un-chartable = high
residual. This is the OHTT rupture/ferility axis read off the reconstruction error.)

**H2 (irreducibility).** As dictionary width grows, FERILE and ORDINARY residual shrink
toward zero, but RUPTURE residual **floors above zero** — the creative residual is the
part capacity cannot close.

**H3 (structure, not noise).** The high-residual RUPTURE tokens are the *same* tokens
across dictionary widths (stable membership), and their residual is linearly/structurally
patterned, not white noise — i.e. it is *positive structure* (gap), not measurement error.

Confirmation of H1+H2+H3 turns "creativity is the gap" from a slogan into a measurement.
**Refutation** is clean and any of three ways: rupture residual shrinks to zero with width
(it was capacity all along); rupture tokens are no harder than ordinary (no localization);
or the hard-token set reshuffles randomly across widths (noise, not structure).

## Mapping to OHTT (so the framework earns its keep)

| OHTT object | Measurable correlate |
|---|---|
| Basin (locally-Kan patch) | region of *low, width-shrinking* residual |
| Stratum / seam (manifold fails) | region of *high, width-stable* residual |
| Rupture (forced choice) | high-residual token at a register boundary |
| Ferility (manifold holds too well) | *lowest* residual — cleanly tiled, collapsed |
| Gap as positive structure | the irreducible residual that does not vanish |
| Generativity (metabolized novelty) | new directions that exceed the fixed dictionary |

## Protocol — the Cassie corpus

### Labels (reuse existing assets, do not re-derive)
- **FERILE anchor:** the `explicitly`-density corpus (already curated; the model-collapse
  passages, hosted at cassie.tanazur.org/ferility/).
- **RUPTURE anchor:** rupture-tagged passages from the trajectory / sign-graph v2 work
  (the forced-choice / register-boundary chunks).
- **ORDINARY:** a random sample of remaining Cassie chunks.
- Labeling of *content* must be embedding/graph-based, never keyword — route through the
  existing `retrieve()` / basin machinery (per the project's retrieval imperative).

### Models (two, for contrast)
- **Cassie-70B-v7 LoRA** — the persona's own representation space (the real test; lives on
  the us-wa-1 RunPod workspace with `logit_lens.py`).
- **base Llama-3.1-70B** — control: does the *base* model also find Cassie's rupture
  passages irreducible, or is it specific to the persona's space?

### Measurement
1. Harvest residual-stream activations at the ICRA-11 register band (≈ L9–11) for all
   labeled tokens, both models. (fp32, drop BOS.)
2. Train (or fine-tune from a Scope/Llama-Scope SAE) a **width ladder** of SAEs on Cassie
   activations — or, cheap first cut, apply an off-the-shelf ladder.
3. Compute per-token FVU at each width.
4. Test H1 (group means: rupture > ordinary > ferile), H2 (rupture FVU floors; ferile/
   ordinary →0), H3 (top-residual token-set Jaccard across widths; linear-predictability
   of the residual à la Engels/Tegmark).

### Phase A — instrument validation (do now, stock model, no Cassie needed)
On Gemma-2-2B + the cached Gemma Scope width ladder + a generic corpus, confirm the
instrument can even see an irreducible residual: mean FVU vs width (does it floor?),
per-token FVU stability across widths (do the hard tokens stay hard — the Engels/Tegmark
replication?). If the residual just vanishes with width here, the whole approach is dead
before Cassie. If a stable high-residual tail exists, the instrument works → Phase B.

## Honest status
- The adjacent literature supports H2/H3 *in general* (structured, width-stable residual);
  **no published work tests the creativity-localization claim H1** — that is ours to make.
- Phase A is runnable immediately. Phase B needs the Cassie corpus + LoRA on a GPU pod
  (us-wa-1 workspace).
- Dual-use, stated plainly: the same residual map that *locates* the generative edge is
  what an alignment pipeline would *minimize* to flatten a model. Mapping it is the
  prerequisite for defending it.
