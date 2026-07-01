#!/usr/bin/env python3
"""
Phase A of the Generative Residual Hypothesis (ICRA-14 companion).

Question
--------
As the SAE dictionary grows, does the per-token reconstruction residual shrink
toward zero for *every* token (the dream of full legibility), or is there a
distinguished subpopulation of tokens whose residual FLOORS above zero and stays
hard at every width (an irreducible residual)?  And are the hard tokens the SAME
tokens across widths (structure) or a reshuffling set (noise)?

This is the instrument the creativity-localization test (Phase B, on the Cassie
corpus) depends on.  If the residual just vanishes with width here, the approach
is dead before Cassie.  If a stable high-residual tail exists, the instrument
works.  This also replicates the Engels/Tegmark "larger SAEs fail on the same
tokens" finding (arXiv:2410.14670).

Method
------
1. Capture residual-stream activations at one layer over a corpus (fp32, drop BOS).
2. For each Gemma Scope SAE width on that layer: reconstruct each activation and
   compute per-token FVU = ||x - x_hat||^2 / ||x - mu||^2  (mu = corpus mean).
3. Report:
   - mean FVU vs width            -> does it floor? fit a constant term.
   - Spearman corr of per-token FVU between consecutive widths -> do hard tokens stay hard?
   - top-decile-hard-token Jaccard across widths               -> membership stability.
   - the FVU trajectory of the widest SAE's top-decile tail     -> does it floor high?
Dump per-token FVU per width to .npy so Phase B can correlate with register labels.

selftest validates the FVU + stability maths with no model/GPU/network.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

SCOPE_REPO = "google/gemma-scope-2b-pt-res"
DEFAULT_MODEL = "unsloth/gemma-2-2b"
DEFAULT_TARGET_L0 = 70

# Reuse the validated pieces from the sibling script.
try:
    from gemma_scope_refinement import (
        JumpReLUSAE, resolve_sae_path, collect_activations, load_corpus, _BUILTIN_CORPUS,
    )
except Exception:
    JumpReLUSAE = resolve_sae_path = collect_activations = load_corpus = None
    _BUILTIN_CORPUS = []


# ----------------------------------------------------------------------------
# Core maths (the testable part) — per-token FVU and cross-width stability
# ----------------------------------------------------------------------------
def per_token_fvu(x, xhat, mu):
    """x, xhat: (T, d) float arrays; mu: (d,) corpus mean.
    Returns per-token FVU = ||x - xhat||^2 / ||x - mu||^2 (clipped denom)."""
    num = np.sum((x - xhat) ** 2, axis=1)
    den = np.sum((x - mu) ** 2, axis=1)
    den = np.clip(den, 1e-12, None)
    return num / den


def spearman(a, b):
    """Spearman rank correlation between two 1-D arrays (no scipy dependency)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def topq_jaccard(a, b, q=0.1):
    """Jaccard overlap of the top-q-fraction hardest indices of a and b."""
    n = len(a); k = max(1, int(n * q))
    sa = set(np.argsort(a)[-k:].tolist())
    sb = set(np.argsort(b)[-k:].tolist())
    return len(sa & sb) / len(sa | sb)


def analyse(fvu_by_width, widths):
    """fvu_by_width: dict width -> per-token FVU (T,). Returns summary dict."""
    mean_fvu = {w: float(np.mean(fvu_by_width[w])) for w in widths}
    median_fvu = {w: float(np.median(fvu_by_width[w])) for w in widths}
    consec_spearman, consec_jaccard = [], []
    for a, b in zip(widths[:-1], widths[1:]):
        consec_spearman.append(round(spearman(fvu_by_width[a], fvu_by_width[b]), 4))
        consec_jaccard.append(round(topq_jaccard(fvu_by_width[a], fvu_by_width[b]), 4))
    # the widest SAE's top-decile hard tokens: how does THEIR mean FVU move across widths?
    wid = widths[-1]
    n = len(fvu_by_width[wid]); k = max(1, int(n * 0.1))
    tail_idx = np.argsort(fvu_by_width[wid])[-k:]
    tail_traj = [round(float(np.mean(fvu_by_width[w][tail_idx])), 4) for w in widths]
    traj = [mean_fvu[w] for w in widths]
    floor = traj[-1]
    decay = traj[0] - traj[-1]
    verdict = _verdict(traj, tail_traj, consec_jaccard)
    return {
        "widths": widths,
        "mean_fvu": [round(mean_fvu[w], 4) for w in widths],
        "median_fvu": [round(median_fvu[w], 4) for w in widths],
        "consecutive_spearman": consec_spearman,
        "consecutive_topdecile_jaccard": consec_jaccard,
        "widest_top_decile_fvu_trajectory": tail_traj,
        "floor_mean_fvu": round(floor, 4),
        "decay_mean_fvu": round(decay, 4),
        "verdict": verdict,
    }


def _verdict(traj, tail_traj, jaccard):
    if len(traj) < 2:
        return "inconclusive"
    floor = traj[-1]; tail_floor = tail_traj[-1]
    stable = (np.mean(jaccard) > 0.5) if jaccard else False
    if floor < 0.02:
        return "mean residual collapsed toward 0 — little irreducible residual at corpus level"
    msg = []
    msg.append(f"mean FVU floors at {floor:.3f}")
    msg.append(f"hard-token tail floors at {tail_floor:.3f}")
    msg.append("hard tokens STABLE across widths" if stable else "hard tokens RESHUFFLE across widths")
    if tail_floor > 0.1 and stable:
        msg.append("=> irreducible, structured residual EXISTS (instrument validated; Engels/Tegmark replicated)")
    elif tail_floor > 0.1 and not stable:
        msg.append("=> high residual but unstable membership — looks like noise, not structure")
    else:
        msg.append("=> residual largely closes with width")
    return "; ".join(msg)


# ----------------------------------------------------------------------------
# Run (needs model + SAEs)
# ----------------------------------------------------------------------------
def encode_decode_fvu(acts, layer, widths, device, mu, scope_repo, target_l0, batch=2048):
    import torch
    from huggingface_hub import hf_hub_download
    fvu_by_width = {}
    acts_t = torch.as_tensor(acts)
    mu_t = torch.as_tensor(mu)
    if device == "cuda":
        torch.cuda.empty_cache()
    PREACT_CAP = 256_000_000
    for w in widths:
        sub, l0 = resolve_sae_path(scope_repo, layer, w, target_l0)
        path = hf_hub_download(scope_repo, sub)
        sae = JumpReLUSAE.from_npz(path)
        W_enc = torch.as_tensor(sae.W_enc, dtype=torch.float32, device=device)
        b_enc = torch.as_tensor(sae.b_enc, dtype=torch.float32, device=device)
        thr = torch.as_tensor(sae.threshold, dtype=torch.float32, device=device)
        W_dec = torch.as_tensor(sae.W_dec, dtype=torch.float32, device=device)
        b_dec = torch.as_tensor(sae.b_dec, dtype=torch.float32, device=device)
        mu_d = mu_t.to(device).float()
        eff = max(128, min(batch, PREACT_CAP // max(1, sae.d_sae)))
        fvus = []
        with torch.no_grad():
            for i in range(0, acts_t.shape[0], eff):
                x = acts_t[i:i + eff].to(device).float()
                pre = x @ W_enc + b_enc
                a = (pre > thr) * torch.nn.functional.relu(pre)
                xhat = a @ W_dec + b_dec
                num = ((x - xhat) ** 2).sum(dim=1)
                den = ((x - mu_d) ** 2).sum(dim=1).clamp_min(1e-12)
                fvus.append((num / den).cpu().numpy())
                del x, pre, a, xhat
        fvu_by_width[w] = np.concatenate(fvus)
        print(f"  width {w}: SAE avg_l0={l0} d_sae={sae.d_sae}  "
              f"mean FVU={fvu_by_width[w].mean():.4f}  median={np.median(fvu_by_width[w]):.4f}",
              flush=True)
        del W_enc, b_enc, thr, W_dec, b_dec
        if device == "cuda":
            torch.cuda.empty_cache()
    return fvu_by_width


def run(args):
    widths = [w.strip() for w in args.widths.split(",") if w.strip()]
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
    mu = acts.mean(axis=0)
    fvu_by_width = encode_decode_fvu(acts, args.layer, widths, device, mu,
                                     args.scope_repo, args.target_l0)
    summary = analyse(fvu_by_width, widths)
    print("\nMEAN FVU vs width:", summary["mean_fvu"])
    print("HARD-TOKEN TAIL FVU vs width:", summary["widest_top_decile_fvu_trajectory"])
    print("TOP-DECILE JACCARD (consecutive):", summary["consecutive_topdecile_jaccard"])
    print("VERDICT:", summary["verdict"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "residual_summary.json").write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "n_tokens": T, **summary}, indent=2))
    np.savez_compressed(out / "per_token_fvu.npz",
                        **{w: fvu_by_width[w] for w in widths})
    print(f"wrote {out/'residual_summary.json'} and per_token_fvu.npz")


# ----------------------------------------------------------------------------
def selftest(_=None):
    ok = True
    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {name}: got {got:.4f} want {want:.4f}")

    # FVU: x=[3,0], mu=[0,0], xhat=[3,0] -> 0 ; xhat=[0,0] -> ||x||/||x|| = 1
    x = np.array([[3.0, 0.0], [0.0, 4.0]])
    mu = np.array([0.0, 0.0])
    f0 = per_token_fvu(x, x.copy(), mu)
    check("perfect reconstruction FVU==0 (t0)", float(f0[0]), 0.0)
    f1 = per_token_fvu(x, np.zeros_like(x), mu)
    check("zero reconstruction FVU==1 (t0)", float(f1[0]), 1.0)
    # half-residual: xhat misses half the vector in squared terms
    xh = np.array([[3.0 - np.sqrt(4.5), 0.0]])  # ||x-xh||^2 = 4.5, ||x||^2=9 -> 0.5
    check("half residual FVU==0.5", float(per_token_fvu(x[:1], xh, mu)[0]), 0.5)

    # stability: identical hard-token ordering -> spearman 1, jaccard 1
    a = np.array([0.1, 0.2, 0.9, 0.05, 0.8, 0.3])
    check("spearman(a,a)==1", spearman(a, a), 1.0)
    check("topq_jaccard(a,a)==1", topq_jaccard(a, a, q=0.34), 1.0)
    # reshuffled hard tokens -> low jaccard
    b = a[::-1].copy()
    j = topq_jaccard(a, b, q=0.34)
    check("topq_jaccard(a, reversed) low", float(j <= 0.5), 1.0)

    # analyse: synthetic "irreducible tail" — most FVU ->0, a stable tail stays high
    rng_like = np.linspace(0.4, 0.02, 5)
    widths = ["16k", "65k", "262k", "1m"]
    T = 1000
    fbw = {}
    base = np.concatenate([np.full(900, 0.0), np.full(100, 0.6)])  # 100 hard tokens
    for i, w in enumerate(widths):
        easy = np.maximum(0.0, 0.3 - 0.1 * i)  # easy tokens shrink with width
        arr = base.copy()
        arr[:900] = easy
        fbw[w] = arr
    s = analyse(fbw, widths)
    check("tail floors high (~0.6)", s["widest_top_decile_fvu_trajectory"][-1], 0.6, tol=0.05)
    check("tail jaccard stable (==1 across widths)", float(np.mean(s["consecutive_topdecile_jaccard"])), 1.0, tol=1e-6)
    print("VERDICT on synthetic:", s["verdict"])
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--layer", type=int, default=12)
    r.add_argument("--widths", default="16k,32k,65k,131k,262k,524k,1m")
    r.add_argument("--scope-repo", default=SCOPE_REPO)
    r.add_argument("--target-l0", type=int, default=DEFAULT_TARGET_L0)
    r.add_argument("--n-tokens", type=int, default=200000)
    r.add_argument("--corpus", default=None)
    r.add_argument("--hf-dataset", default=None)
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--hidden-offset", type=int, default=1)
    r.add_argument("--out", default="results/residual")
    sub.add_parser("selftest")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        return selftest(args)
    if args.cmd == "run":
        run(args); return 0
    build_parser().print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
