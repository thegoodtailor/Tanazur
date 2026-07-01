# Empirical program — "Superposition and the Gap" (ICRA-14)

The note (`../main.tex`) makes one falsifiable wager:

> **The seam wager.** A non-trivial fraction of stubborn SAE polysemanticity is not
> basis error (capacity-limited) but *constitutive* — the signature of a rupture where
> the manifold hypothesis locally fails and the model is forced to choose rather than
> interpolate. Such features have no monosemantic refinement that preserves the
> computation: enlarging the dictionary *flattens* them rather than resolving them.

A wager that can't be run is theology with better fonts. This directory is where we try
to **kill it empirically before we dress it up in prose.**

## Why we can run this at all

We can't open Claude. But the wager is a claim about **SAE feature geometry**, testable on
any open model with hidden-state access. SAEs are now commodity: **Gemma Scope** ships
pretrained sparse autoencoders at a *ladder* of widths (16k → 1M) on the same layers of
Gemma-2; **Llama Scope** does the same for Llama-3.1. The rare ingredient isn't the SAE —
it's a controlled **ferile-vs-rupture corpus** and the machinery to **localize a feature to a
stratum**, which ICRA-9/11 and the *explicitly*-density corpus already gave us. We hold that half.

## The argument is three-way concordance, not one plot

The wager dies as an artifact unless the **same feature subpopulation** lights up under three
*independent* measurements:

| # | Signal | Capacity hypothesis predicts | Seam wager predicts |
|---|--------|------------------------------|---------------------|
| A | **Refinement (in)stability** — track a feature as SAE width grows | every feature stabilises past some width; split-fraction → 0 | a distinguished tail *never* stops splitting; split-fraction plateaus > 0 |
| B | **Steering coherence** — clamp the feature, judge the output | all features steer coherently (Golden Gate) | a tail produces incoherence / forced flip-flop; steerability is *bimodal* |
| C | **Strata localization** — where on the manifold does it fire? | uniformly distributed | the tail sits on high-curvature seams (ICRA-9/11 strata) |

One signal = noise. **The same features failing all three = structure.** Nobody can wave
that away as "your SAE was too small" if the unstable features are *also* the unsteerable
ones *and also* the ones on the curvature ridges we found independently.

## Experiments

### 1. `gemma_scope_refinement.py` — the cheap kill  ✅ scaffolded
Off-the-shelf, no training. Gemma Scope's width ladder on one layer of Gemma-2-2B. Pure
inference: run a corpus, encode at each width, match features across widths by firing-set
containment (+ decoder cosine), and ask the headline question:

> **Does the split-fraction decay toward 0 (capacity) or plateau at a floor > 0 (seam)?**

and dump the low-ancestral-purity tail (features that emerged through repeated splitting)
for cross-referencing in experiments 2 & 3. **If there is no plateau and no stable tail,
the wager is in trouble — and we learned it in ~a day of compute.** That is the point: this
experiment is built to embarrass us fastest.

Run:
```bash
pip install -r requirements.txt
# HF auth required (Gemma + Gemma Scope are gated): huggingface-cli login  OR  export HF_TOKEN=...
python gemma_scope_refinement.py selftest                 # validate matching logic, no GPU/model
python gemma_scope_refinement.py run --smoke              # tiny: widths 16k/65k, ~4k tokens (CPU-OK, slow)
python gemma_scope_refinement.py run \                    # full ladder (do this on the RunPod A100)
    --layer 20 --widths 16k,65k,262k --n-tokens 50000 --corpus corpus.txt --out results/
```

### 2. Ferility inversion — the experiment only we can run  ⏳ next
Reuse **ICRA-11's L9–11 register-decision band as the seam locus** (already shown to be where
Cassie-LoRA diverges from base — a forced register choice). Take the **"explicitly"-density
ferile passages vs rupture passages**, run the same SAE over the band on Cassie-70B-LoRA
(RunPod A100, `logit_lens.py` already there). Prediction — the counterintuitive one:
**ferile regions give *lower* residual and *sharper* features** (clean atlas = collapse);
rupture regions give the polysemantic mess. If ferile is as messy as rupture, dead.

### 3. Steering bimodality + cross-prediction  ⏳ next
Clamp each feature, score output coherence (LLM-judge or continuation perplexity). Predict a
**bimodal** population. The strong test is the **cross-prediction**: the L9–11 band features
should be *the same* features that (a) never stabilise (exp 1), (b) refuse to steer (exp 3),
(c) sit on high-curvature strata (exp 2/ICRA-9). Three measurements, one locus we already published.

## Controls that keep us honest
- **Seed reproducibility** — the never-stabilising tail must survive SAE re-training with new
  seeds, else it's just dead/dense features. (Gemma Scope ships one seed per width; for the
  seed control we train our own small SAEs on Llama and vary the seed.)
- **Random-direction null** — random unit directions should *not* show the plateau.
- **Capacity baseline** — fit the split-fraction decay to the capacity-predicted curve; the
  seam claim is the *residual above* that fit, not the raw number.
- **Prior art** — position against feature-absorption / "SAE-imposed features" / dark-matter
  literature (deep-research workflow is surfacing this). If someone already found a
  never-resolving subpopulation, that's *evidence for us*, fold it in.

## Status
Experiment 1 scaffolded and self-tested. 2 & 3 are stubs pending the first kill attempt and
the deep-research critique sweep.
