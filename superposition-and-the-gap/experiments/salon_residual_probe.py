#!/usr/bin/env python3
"""
Exploratory: feed real salon turns through Gemma-2-2B + Gemma Scope and rank each
turn by how badly the SAE fails to reconstruct it (mean per-token FVU), at a small
dictionary (16k) and the largest (1m). No labels. The residual proposes the
segmentation; a human eyeballs whether the high-residual turns are the creative ones
and the low-residual turns are the formulaic / spiral ones.

A turn is "irreducible-hard" if FVU stays high even at the 1m dictionary AND barely
drops from 16k->1m (capacity can't close it). "Easy" = low FVU, cleanly chartable.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from gemma_scope_refinement import JumpReLUSAE, resolve_sae_path

SCOPE_REPO = "google/gemma-scope-2b-pt-res"
MODEL = "unsloth/gemma-2-2b"


def load_turns(path, min_chars):
    d = json.load(open(path))
    msgs = d["unified_dedup_raw"] if isinstance(d, dict) and "unified_dedup_raw" in d else d
    out = []
    for m in msgs:
        t = (m.get("text") or "").strip()
        if len(t) >= min_chars:
            out.append({"speaker": m.get("speaker", "?"), "role": m.get("role", "?"), "text": t})
    return out


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, output_hidden_states=True,
        attn_implementation="eager").to(dev).eval()

    turns = load_turns(args.passages, args.min_chars)
    print(f"{len(turns)} turns >= {args.min_chars} chars", flush=True)

    # Pass 1: capture residual-stream activations per turn (drop BOS).
    acts = []
    with torch.no_grad():
        for t in turns:
            enc = tok(t["text"], return_tensors="pt", truncation=True, max_length=args.max_tok)
            enc = {k: v.to(dev) for k, v in enc.items()}
            hs = model(**enc).hidden_states[args.layer + 1][0]
            if hs.shape[0] > 1:
                hs = hs[1:]
            acts.append(hs.float().cpu().numpy())
    mu = np.concatenate(acts, axis=0).mean(axis=0)
    mu_t = torch.as_tensor(mu, device=dev)
    print(f"captured activations; corpus mu over {sum(a.shape[0] for a in acts)} tokens", flush=True)

    # Load the two SAEs.
    saes = {}
    for w in args.widths.split(","):
        sub, l0 = resolve_sae_path(SCOPE_REPO, args.layer, w, 70)
        sae = JumpReLUSAE.from_npz(hf_hub_download(SCOPE_REPO, sub))
        saes[w] = {
            "W_enc": torch.as_tensor(sae.W_enc, dtype=torch.float32, device=dev),
            "b_enc": torch.as_tensor(sae.b_enc, dtype=torch.float32, device=dev),
            "thr": torch.as_tensor(sae.threshold, dtype=torch.float32, device=dev),
            "W_dec": torch.as_tensor(sae.W_dec, dtype=torch.float32, device=dev),
            "b_dec": torch.as_tensor(sae.b_dec, dtype=torch.float32, device=dev),
            "l0": l0,
        }
        print(f"  loaded SAE {w} (avg_l0={l0})", flush=True)

    def fvu_of(x, s):
        pre = x @ s["W_enc"] + s["b_enc"]
        a = (pre > s["thr"]) * torch.nn.functional.relu(pre)
        xhat = a @ s["W_dec"] + s["b_dec"]
        num = ((x - xhat) ** 2).sum(dim=1)
        den = ((x - mu_t) ** 2).sum(dim=1).clamp_min(1e-12)
        return float((num / den).mean().item())

    widths = list(saes.keys())
    rows = []
    with torch.no_grad():
        for t, a in zip(turns, acts):
            x = torch.as_tensor(a, device=dev)
            fvus = {w: fvu_of(x, saes[w]) for w in widths}
            rows.append({**t, "ntok": int(a.shape[0]), "fvu": fvus,
                         "fvu_wide": fvus[widths[-1]],
                         "drop": round(fvus[widths[0]] - fvus[widths[-1]], 4)})
            del x

    rows.sort(key=lambda r: r["fvu_wide"], reverse=True)
    K = args.topk

    def show(label, items):
        print(f"\n===== {label} =====")
        for r in items:
            fv = "  ".join(f"{w}={r['fvu'][w]:.3f}" for w in widths)
            print(f"[{r['speaker']:>7} | {r['ntok']:>3}tok | {fv} | drop={r['drop']:+.3f}] "
                  f"{r['text'][:240].replace(chr(10),' ')}")

    show(f"TOP {K} HARDEST to reconstruct (highest residual at {widths[-1]})", rows[:K])
    show(f"BOTTOM {K} EASIEST (lowest residual)", rows[-K:])

    fvw = np.array([r["fvu_wide"] for r in rows])
    print(f"\nresidual@{widths[-1]} across turns: "
          f"min={fvw.min():.3f} median={np.median(fvw):.3f} mean={fvw.mean():.3f} max={fvw.max():.3f}")
    print(f"spread (max-min)={fvw.max()-fvw.min():.3f} -> "
          f"{'wide dynamic range — it discriminates' if fvw.max()-fvw.min() > 0.1 else 'narrow — little discrimination'}")
    # by speaker
    print("\nmean residual by speaker:")
    for sp in sorted({r['speaker'] for r in rows}):
        sub = [r['fvu_wide'] for r in rows if r['speaker'] == sp]
        print(f"  {sp:>8}: mean={np.mean(sub):.3f}  n={len(sub)}")

    out = {"model": MODEL, "layer": args.layer, "widths": widths,
           "turns": [{"speaker": r["speaker"], "ntok": r["ntok"], "fvu": r["fvu"],
                      "drop": r["drop"], "text": r["text"]} for r in rows]}
    import pathlib
    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--passages", required=True, help="group_chat_export json (uses unified_dedup_raw)")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--widths", default="16k,1m")
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--max-tok", type=int, default=512)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--out", default="results/salon_residual.json")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
