#!/usr/bin/env python3
"""
Experiment 1 — the cheap kill for the *seam wager* (ICRA-14, "Superposition and the Gap").

Question
--------
As we walk an SAE up a ladder of dictionary widths on a fixed layer, does the
per-feature *split-fraction* decay toward 0 (capacity hypothesis: every feature
eventually finds its final monosemantic form) or plateau at a floor > 0 (seam
wager: a distinguished subpopulation never stops splitting)?

Method
------
1. Capture the residual stream at one layer of an open model (default Gemma-2-2B)
   over a text corpus.
2. Encode those activations with a ladder of pretrained Gemma Scope JumpReLU SAEs
   of increasing width on the same layer (16k -> 65k -> 262k -> ...).
3. For each consecutive width pair (w, w2), match each *parent* feature at width w
   to its *children* at width w2 by firing-set containment (fraction of the parent's
   firing tokens captured by each child), sanity-checked with decoder-direction cosine.
   - purity(parent) = max child containment.  ~1.0 => preserved intact ("stable").
                                                 low  => fragmented ("splitting").
4. Headline metric per transition: split_fraction = fraction of (sufficiently active)
   parents with purity < tau.  Capacity predicts split_fraction -> 0 along the ladder;
   the seam wager predicts a plateau.
5. Trace each finest-width feature back along its best lineage to the smallest width;
   ancestral_purity = product of containments along that lineage.  The low-ancestral-
   purity tail = features that emerged through repeated splitting = seam candidates,
   dumped to JSON for cross-referencing with steering (exp 3) and strata (exp 2).

This file is deliberately self-contained: a tiny JumpReLU encoder + direct .npz
loading, no sae_lens dependency.  `selftest` validates the matching maths on
synthetic data with no model or GPU.

Auth: Gemma and Gemma Scope are gated on HuggingFace.  `huggingface-cli login`
or `export HF_TOKEN=...` first.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Heavy / optional imports are done lazily inside functions so `selftest` runs
# with only numpy + scipy present.

SCOPE_REPO = "google/gemma-scope-2b-pt-res"     # ungated; SAEs by layer/width/average_l0
DEFAULT_MODEL = "unsloth/gemma-2-2b"            # ungated, weight-identical to google/gemma-2-2b
DEFAULT_TARGET_L0 = 70                          # pick the SAE whose avg L0 is nearest this

# ----------------------------------------------------------------------------
# JumpReLU SAE (Gemma Scope) — matches the DeepMind reference encoder.
# ----------------------------------------------------------------------------
class JumpReLUSAE:
    """Minimal Gemma Scope JumpReLU SAE.  params.npz keys:
    W_enc (d_model, d_sae), W_dec (d_sae, d_model), b_enc (d_sae,),
    b_dec (d_model,), threshold (d_sae,)."""

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        self.W_enc = W_enc
        self.W_dec = W_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.threshold = threshold
        self.d_model, self.d_sae = W_enc.shape

    @classmethod
    def from_npz(cls, path):
        p = np.load(path)
        return cls(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], p["threshold"])

    def encode_torch(self, x):
        """x: (..., d_model) torch tensor on same device/dtype as the SAE buffers.
        Returns JumpReLU activations (..., d_sae)."""
        import torch
        W_enc = torch.as_tensor(self.W_enc, dtype=x.dtype, device=x.device)
        b_enc = torch.as_tensor(self.b_enc, dtype=x.dtype, device=x.device)
        thr = torch.as_tensor(self.threshold, dtype=x.dtype, device=x.device)
        pre = x @ W_enc + b_enc
        mask = pre > thr
        return mask * torch.nn.functional.relu(pre)

    def decoder_directions_unit(self):
        """Unit-normalised decoder rows (d_sae, d_model) for cosine matching."""
        W = self.W_dec.astype(np.float32)
        n = np.linalg.norm(W, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return W / n


# ----------------------------------------------------------------------------
# Feature matching across two widths  (pure numpy/scipy — the testable core)
# ----------------------------------------------------------------------------
def _to_csr_binary(fire_lists, n_tokens, d_sae):
    """fire_lists: list over tokens of arrays of active feature indices.
    Returns scipy.sparse CSR (n_tokens x d_sae) binary firing matrix."""
    from scipy import sparse
    rows, cols = [], []
    for t, feats in enumerate(fire_lists):
        if len(feats):
            rows.extend([t] * len(feats))
            cols.extend(feats)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_tokens, d_sae))


def match_widths(A_small, A_large, min_parent_fires=5):
    """A_small (N x d1), A_large (N x d2) binary CSR firing matrices over the SAME
    N tokens.  For each parent feature p (column of A_small) compute, over its
    firing tokens, the containment fraction in each child c (column of A_large):
        containment(p, c) = |fire(p) & fire(c)| / |fire(p)|.
    Returns dict with, per parent: n_fires, purity(=max containment), best_child,
    and effective number of children (exp of entropy of the containment dist).
    Parents with < min_parent_fires firing tokens are marked inactive (purity=nan)."""
    from scipy import sparse
    A_small = A_small.tocsc()
    A_large = A_large.tocsc()
    # co-firing counts: (d1 x d2) sparse = A_small^T @ A_large
    co = (A_small.T @ A_large).tocsr()          # counts of tokens where both fire
    parent_fires = np.asarray(A_small.sum(axis=0)).ravel()  # |fire(p)|

    d1 = A_small.shape[1]
    purity = np.full(d1, np.nan, dtype=np.float64)
    best_child = np.full(d1, -1, dtype=np.int64)
    n_eff = np.full(d1, np.nan, dtype=np.float64)

    for p in range(d1):
        npf = parent_fires[p]
        if npf < min_parent_fires:
            continue
        start, end = co.indptr[p], co.indptr[p + 1]
        if start == end:
            purity[p] = 0.0
            n_eff[p] = 0.0
            continue
        children = co.indices[start:end]
        counts = co.data[start:end].astype(np.float64)
        contain = counts / npf                  # containment fraction per child
        j = int(np.argmax(contain))
        purity[p] = float(contain[j])
        best_child[p] = int(children[j])
        # effective #children from the containment distribution (normalised)
        q = contain / contain.sum()
        ent = -np.sum(q * np.log(np.clip(q, 1e-12, None)))
        n_eff[p] = float(np.exp(ent))
    return {
        "parent_fires": parent_fires,
        "purity": purity,
        "best_child": best_child,
        "n_eff_children": n_eff,
    }


def split_fraction(match, tau=0.8):
    """Fraction of *active* parents with purity < tau (i.e. that split)."""
    pur = match["purity"]
    active = ~np.isnan(pur)
    if active.sum() == 0:
        return float("nan"), 0
    return float(np.mean(pur[active] < tau)), int(active.sum())


# ----------------------------------------------------------------------------
# Activation capture
# ----------------------------------------------------------------------------
def load_corpus(args):
    if args.corpus and Path(args.corpus).exists():
        text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if chunks:
            return chunks
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train", streaming=True)
        out, key = [], None
        for ex in ds:
            if key is None:
                key = "text" if "text" in ex else list(ex.keys())[0]
            if ex[key].strip():
                out.append(ex[key].strip())
            if len(out) >= 2000:
                break
        if out:
            return out
    # Built-in fallback: enough diverse English to smoke-test the pipeline.
    return _BUILTIN_CORPUS


def collect_activations(model_name, layer, texts, n_tokens, device, hidden_offset=1,
                        max_seq=256, dtype="float32"):
    """Run the model over `texts`, capture residual stream at `layer`
    (= hidden_states[layer + hidden_offset]), return (acts float32 ndarray (T x d_model), T).

    Gemma-2 + Gemma Scope fidelity: the SAEs were trained on activations from the
    model run with its native *softcapped* attention.  HF's default `sdpa` backend
    silently alters that softcapping, which inflates observed L0 far above the SAE's
    nominal sparsity.  We therefore force `attn_implementation="eager"` and fp32 so
    that observed L0 lands near the SAE's average_l0."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, output_hidden_states=True,
        attn_implementation="eager",
    ).to(device).eval()

    collected, total = [], 0
    with torch.no_grad():
        for txt in texts:
            if total >= n_tokens:
                break
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_seq)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hs = out.hidden_states[layer + hidden_offset][0]   # (seq, d_model)
            # Drop position 0 (BOS): it is a massive-activation / attention-sink
            # outlier that fires thousands of SAE features and inflates observed L0
            # far above the SAE's nominal average_l0 (which excludes it).
            if hs.shape[0] > 1:
                hs = hs[1:]
            collected.append(hs.float().cpu().numpy())
            total += hs.shape[0]
    acts = np.concatenate(collected, axis=0)[:n_tokens]
    return acts, acts.shape[0]


def resolve_sae_path(repo, layer, width, target_l0):
    """Gemma Scope organises residual SAEs as
    `layer_{L}/width_{W}/average_l0_{N}/params.npz` (multiple sparsities per width).
    Pick the available average-L0 nearest to `target_l0`.  Returns (subpath, l0).
    NB: regex here parses an opaque *filename*, not content semantics."""
    import re
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo)
    prefix = f"layer_{layer}/width_{width}/average_l0_"
    cands = []
    for f in files:
        if f.startswith(prefix) and f.endswith("/params.npz"):
            m = re.search(r"average_l0_(\d+)/params\.npz$", f)
            if m:
                cands.append((int(m.group(1)), f))
    if not cands:
        raise FileNotFoundError(f"no SAE for layer {layer} width {width} in {repo}")
    cands.sort(key=lambda t: abs(t[0] - target_l0))
    return cands[0][1], cands[0][0]


def encode_all_widths(acts, layer, widths, device, batch=4096, min_act=0.0,
                      scope_repo=SCOPE_REPO, target_l0=DEFAULT_TARGET_L0):
    """For each width: download the nearest-L0 Gemma Scope SAE, encode acts, return a
    binary CSR firing matrix and the SAE (for decoder cosine).  Activations above
    JumpReLU threshold are already > 0; `min_act` is an extra floor if desired."""
    import torch
    from huggingface_hub import hf_hub_download
    results = {}
    acts_t = torch.as_tensor(acts)
    if device == "cuda":
        torch.cuda.empty_cache()            # release the model's cached GPU memory first
    # Cap on the (batch x d_sae) pre-activation intermediate, in elements.
    # At d_sae=1e6 a batch of 4096 would allocate 16GB *per copy* and OOM the GPU;
    # bound it so the intermediate stays ~<=1GB regardless of width.
    PREACT_ELEM_CAP = 256_000_000
    for w in widths:
        sub, l0 = resolve_sae_path(scope_repo, layer, w, target_l0)
        path = hf_hub_download(scope_repo, sub)
        sae = JumpReLUSAE.from_npz(path)
        dev = device
        W_enc = torch.as_tensor(sae.W_enc, dtype=torch.float32, device=dev)
        b_enc = torch.as_tensor(sae.b_enc, dtype=torch.float32, device=dev)
        thr = torch.as_tensor(sae.threshold, dtype=torch.float32, device=dev)
        eff_batch = max(128, min(batch, PREACT_ELEM_CAP // max(1, sae.d_sae)))
        fire_lists = []
        with torch.no_grad():
            for i in range(0, acts_t.shape[0], eff_batch):
                x = acts_t[i:i + eff_batch].to(dev).float()
                pre = x @ W_enc + b_enc
                a = (pre > thr) * torch.nn.functional.relu(pre)
                a = (a > min_act)
                idx = a.nonzero(as_tuple=False).cpu().numpy()  # (k,2): row,col
                # bucket by row
                byrow = {}
                for r, c in idx:
                    byrow.setdefault(int(r), []).append(int(c))
                for r in range(x.shape[0]):
                    fire_lists.append(np.asarray(byrow.get(r, []), dtype=np.int64))
                del x, pre, a, idx
        A = _to_csr_binary(fire_lists, len(fire_lists), sae.d_sae)
        results[w] = {"firing": A, "sae": sae, "d_sae": sae.d_sae, "l0": l0}
        del W_enc, b_enc, thr                # free this width's SAE weights before the next
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  width {w}: SAE avg_l0={l0} d_sae={sae.d_sae}, "
              f"observed mean L0={A.sum()/A.shape[0]:.1f}", flush=True)
    return results


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
@dataclass
class TransitionResult:
    w_small: str
    w_large: str
    split_fraction: float
    n_active_parents: int
    median_purity: float
    median_n_eff_children: float


def run(args):
    widths = [w.strip() for w in args.widths.split(",") if w.strip()]
    if args.smoke:
        widths = widths[:2] if len(widths) >= 2 else ["16k", "65k"]
        args.n_tokens = min(args.n_tokens, 4000)
        print(f"[smoke] widths={widths} n_tokens={args.n_tokens}")
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"device={device} model={args.model} layer={args.layer} widths={widths}")

    texts = load_corpus(args)
    print(f"corpus: {len(texts)} chunks")
    acts, T = collect_activations(args.model, args.layer, texts, args.n_tokens,
                                  device, hidden_offset=args.hidden_offset)
    print(f"captured {T} token activations, d_model={acts.shape[1]}")

    enc = encode_all_widths(acts, args.layer, widths, device, min_act=args.min_act,
                            scope_repo=args.scope_repo, target_l0=args.target_l0)

    transitions = []
    for a, b in zip(widths[:-1], widths[1:]):
        m = match_widths(enc[a]["firing"], enc[b]["firing"],
                         min_parent_fires=args.min_parent_fires)
        sf, n_active = split_fraction(m, tau=args.tau)
        pur = m["purity"][~np.isnan(m["purity"])]
        neff = m["n_eff_children"][~np.isnan(m["n_eff_children"])]
        tr = TransitionResult(
            w_small=a, w_large=b, split_fraction=sf, n_active_parents=n_active,
            median_purity=float(np.median(pur)) if pur.size else float("nan"),
            median_n_eff_children=float(np.median(neff)) if neff.size else float("nan"),
        )
        transitions.append(tr)
        print(f"  {a}->{b}: split_fraction(purity<{args.tau})={sf:.3f}  "
              f"median_purity={tr.median_purity:.3f}  "
              f"median_n_eff_children={tr.median_n_eff_children:.2f}  "
              f"(n_active={n_active})", flush=True)

    # Headline read
    sfs = [t.split_fraction for t in transitions]
    verdict = _verdict(sfs)
    print(f"\nSPLIT-FRACTION TRAJECTORY: {[round(s,3) for s in sfs]}")
    print(f"VERDICT: {verdict}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model, "layer": args.layer, "widths": widths,
        "n_tokens": T, "tau": args.tau, "device": device,
        "transitions": [asdict(t) for t in transitions],
        "split_fraction_trajectory": sfs,
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out/'summary.json'}")
    try:
        _plot(sfs, widths, out / "split_fraction.png")
        print(f"wrote {out/'split_fraction.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print("\nNOTE: split-fraction is the headline. A monotone decay toward ~0 supports "
          "the capacity hypothesis; a plateau at a floor > 0 supports the seam wager. "
          "Either way, the low-purity tail feeds experiments 2 (strata) and 3 (steering).")


def _verdict(sfs):
    if len(sfs) < 2 or any(np.isnan(sfs)):
        return "inconclusive (need >=3 widths for a trajectory)"
    decay = sfs[0] - sfs[-1]
    floor = sfs[-1]
    if floor < 0.05:
        return "leans CAPACITY (split-fraction collapsed toward 0)"
    if decay < 0.05 and floor > 0.1:
        return "leans SEAM (split-fraction plateaus at a floor > 0)"
    return (f"mixed (decay={decay:.3f}, floor={floor:.3f}) — needs more widths / "
            "seed control / strata cross-check before either side can claim it")


def _plot(sfs, widths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{a}->{b}" for a, b in zip(widths[:-1], widths[1:])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(sfs)), sfs, "o-", color="#D4A84B", lw=2)
    ax.set_xticks(range(len(sfs)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("split-fraction (purity < tau)")
    ax.set_title("Refinement instability along the width ladder")
    ax.axhline(0.0, color="#0C1020", lw=0.6, ls=":")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# ----------------------------------------------------------------------------
# Self-test — validates the matching maths with NO model / GPU / network.
# ----------------------------------------------------------------------------
def selftest(args=None):
    from scipy import sparse

    def csr(fire_lists, d):
        return _to_csr_binary([np.asarray(f, dtype=np.int64) for f in fire_lists],
                              len(fire_lists), d)

    N = 20
    # width-small: feature 0 fires on tokens 0..9; feature 1 fires on 10..14 (stable).
    small = [[] for _ in range(N)]
    for t in range(0, 10):
        small[t].append(0)
    for t in range(10, 15):
        small[t].append(1)
    A_small = csr(small, 2)

    # width-large: feature 0 SPLITS into child 0 (tokens 0..4) and child 1 (tokens 5..9);
    # parent feature 1 is PRESERVED intact as child 2 (tokens 10..14).
    large = [[] for _ in range(N)]
    for t in range(0, 5):
        large[t].append(0)
    for t in range(5, 10):
        large[t].append(1)
    for t in range(10, 15):
        large[t].append(2)
    A_large = csr(large, 3)

    m = match_widths(A_small, A_large, min_parent_fires=1)
    p0 = m["purity"][0]
    p1 = m["purity"][1]
    neff0 = m["n_eff_children"][0]
    neff1 = m["n_eff_children"][1]

    ok = True
    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {name}: got {got:.4f}, want {want:.4f}")

    print("selftest: split parent (0) vs stable parent (1)")
    check("purity(split parent 0) == 0.5", p0, 0.5)       # max child captures 5/10
    check("purity(stable parent 1) == 1.0", p1, 1.0)      # one child captures 5/5
    check("n_eff_children(split parent 0) == 2.0", neff0, 2.0)
    check("n_eff_children(stable parent 1) == 1.0", neff1, 1.0)

    sf, n_active = split_fraction(m, tau=0.8)
    check("split_fraction(tau=0.8) == 0.5", sf, 0.5)      # 1 of 2 parents splits
    check("n_active_parents == 2", float(n_active), 2.0)

    # inactivity guard
    m2 = match_widths(A_small, A_large, min_parent_fires=100)
    sf2, n2 = split_fraction(m2, tau=0.8)
    check("all-inactive -> n_active 0", float(n2), 0.0)

    print("VERDICT helper:", _verdict([0.6, 0.3, 0.02]))
    print("VERDICT helper:", _verdict([0.42, 0.40, 0.39]))
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
_BUILTIN_CORPUS = [
    "The lighthouse keeper recorded the tides each morning, noting how the grey water "
    "climbed the rocks and fell away again, indifferent to the ledger he kept.",
    "In category theory a colimit glues a diagram of objects into a single universal "
    "object; the gluing is determined up to unique isomorphism by a universal property.",
    "She tuned the old radio past the static until a voice resolved, reading shipping "
    "forecasts in a flat cadence: Dogger, Fisher, German Bight, rising slowly.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using "
    "light energy captured by chlorophyll within the thylakoid membranes of chloroplasts.",
    "The defendant maintained his account under cross-examination, though the timeline "
    "he offered did not reconcile with the bank records entered into evidence.",
    "A sparse autoencoder is trained to reconstruct activations while keeping most of "
    "its hidden units inactive, so that each active unit aligns with a single feature.",
    "The bazaar smelled of saffron and diesel; vendors called prices over one another "
    "while a boy threaded a tray of glasses through the crowd without spilling a drop.",
    "Glaciers store roughly two thirds of the world's fresh water, and their retreat "
    "alters the timing of meltwater that downstream cities depend upon each summer.",
    "He revised the sonnet a final time, choosing a closing rhyme he had in fact fixed "
    "before the third line was written, then arranging the lines to arrive there.",
    "Quantum error correction encodes a logical qubit across many physical qubits so "
    "that local noise can be detected and corrected without measuring the logical state.",
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run the refinement-instability experiment")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--layer", type=int, default=20)
    r.add_argument("--widths", default="16k,65k,262k",
                   help="comma list of Gemma Scope widths on this layer")
    r.add_argument("--scope-repo", default=SCOPE_REPO,
                   help="ungated Gemma Scope residual-SAE repo")
    r.add_argument("--target-l0", type=int, default=DEFAULT_TARGET_L0,
                   help="pick the SAE whose average L0 is nearest this value")
    r.add_argument("--n-tokens", type=int, default=50000)
    r.add_argument("--corpus", default=None, help="path to a UTF-8 text file (blank-line separated)")
    r.add_argument("--hf-dataset", default=None, help="optional HF dataset id, e.g. wikitext")
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--hidden-offset", type=int, default=1,
                   help="hidden_states index offset: resid_post(layer)=hidden_states[layer+offset]")
    r.add_argument("--tau", type=float, default=0.8, help="purity threshold for 'split'")
    r.add_argument("--min-parent-fires", type=int, default=5)
    r.add_argument("--min-act", type=float, default=0.0)
    r.add_argument("--smoke", action="store_true", help="tiny run for a pipeline check")
    r.add_argument("--out", default="results")

    sub.add_parser("selftest", help="validate matching maths (no model/GPU/network)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        return selftest(args)
    if args.cmd == "run":
        run(args)
        return 0
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
