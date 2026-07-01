#!/usr/bin/env python3
"""
Whole-book persistence loops, with the ACTUAL passages forming them.

Method (lean, sequential, CPU-friendly — no parallelism):
  1. Corpus = all 10 chapters in order. Strip LaTeX to readable prose; keep
     per-chapter token offsets so any global token index maps back to a chapter.
  2. tiktoken cl100k_base. Bounded rolling-context masked-prefix trajectory:
     point i = embedding of decode(tokens[max(0, i-WIN) : i]); stride chosen so
     total points <= 900.
  3. Embed with text-embedding-3-small, batched 100, SEQUENTIAL.
  4. gudhi RipsComplex over cosine distance, max_dimension=2,
     max_edge_length = 90th percentile of pairwise distances. Extract H1 bars,
     rank by lifespan/birth ratio.
  5. For the top 3 H1 loops: take birth+death simplex vertices (APPROXIMATE cycle
     support — gudhi does not return the minimal generating cycle), map each
     vertex's token offset -> chapter, decode the ~WIN-token passage at that vertex.

Output: audiobook/book_loops.json — per loop: ratio, birth, persistence, and the
list of {chapter, passage_text} for its support vertices.

These are GEOMETRIC returns, not certified themes.

    python scripts/book_loops.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import tiktoken
import gudhi
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = sorted(glob.glob(str(ROOT / "chapters" / "ch-*.tex")))
OUT = ROOT / "audiobook" / "book_loops.json"
WIN = 256                # rolling-context window (tokens) for the masked prefix
MAX_POINTS = 900
EMBED_MODEL = "text-embedding-3-small"
BATCH = 100


def load_openai_key() -> str:
    env = Path("/home/iman/cassie-project/.env").read_text(encoding="utf-8")
    for line in env.splitlines():
        if line.startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found")


def strip_latex(src: str) -> str:
    """Drop comments / cite / footnote / commands / math markers; keep readable prose."""
    s = src
    # line comments (LaTeX % not escaped)
    s = re.sub(r"(?<!\\)%.*", "", s)
    # remove \footnote{...} and \cite{...} (and starred / optional-arg variants) — balanced-ish
    # iterate a few times to catch simple nesting
    for _ in range(3):
        s = re.sub(r"\\footnote\{[^{}]*\}", "", s)
        s = re.sub(r"\\(?:cite|citep|citet|ref|label|eqref|autoref)\*?(?:\[[^\]]*\])?\{[^{}]*\}", "", s)
    # \chapter{X}, \section{X}, \subsection{X}, \textit{X} etc -> keep the inner text
    for cmd in ("chapter", "section", "subsection", "subsubsection", "paragraph"):
        s = re.sub(rf"\\{cmd}\*?\{{([^{{}}]*)\}}", r" \1. ", s)
    # keep argument of inline emphasis/formatting commands
    s = re.sub(r"\\(?:textit|textbf|emph|texttt|textsc|underline)\{([^{}]*)\}", r"\1", s)
    # math: drop $...$ and \(...\) and \[...\] markers (keep nothing — they're symbolic)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", " ", s, flags=re.S)
    # environments: drop the \begin{..}/\end{..} wrappers, keep inner text
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", s)
    # remaining commands with a braced arg -> keep the arg
    for _ in range(2):
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
    # bare commands (\noindent, \\, \item, \theendnotes, etc.) -> space
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("\\\\", " ").replace("~", " ")
    # latex quotes & leftover braces
    s = s.replace("``", '"').replace("''", '"').replace("`", "'")
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"---", "—", s)
    s = re.sub(r"--", "–", s)
    # collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()


def main() -> int:
    enc = tiktoken.get_encoding("cl100k_base")

    # 1. build corpus with per-chapter token offsets
    all_tokens: list[int] = []
    offsets = []  # (chapter_label, start_tok, end_tok)
    for f in CHAPTERS:
        label = Path(f).stem  # ch-05-no-beneath
        prose = strip_latex(Path(f).read_text(encoding="utf-8"))
        toks = enc.encode(prose)
        start = len(all_tokens)
        all_tokens.extend(toks)
        offsets.append((label, start, len(all_tokens)))
        print(f"  {label}: {len(toks)} prose tokens (offset {start})")
    total = len(all_tokens)
    print(f"  corpus total prose tokens: {total}")

    def tok_to_chapter(i: int) -> str:
        for label, s, e in offsets:
            if s <= i < e:
                return label
        return offsets[-1][0]

    # 2. trajectory points (masked-prefix, bounded rolling context)
    stride = max(8, total // MAX_POINTS)
    idxs = list(range(1, total, stride))   # i>=1 so the prefix is non-empty
    if len(idxs) > MAX_POINTS:
        idxs = idxs[:MAX_POINTS]
    print(f"  stride={stride}, points={len(idxs)}")
    texts = [enc.decode(all_tokens[max(0, i - WIN):i]) for i in idxs]

    # 3. embed sequentially, batched — cache to .npy so a rerun NEVER re-spends credits
    cache = ROOT / "audiobook" / f"_book_loops_emb_s{stride}.npy"
    if cache.exists():
        X = np.load(cache)
        if X.shape[0] == len(idxs):
            print(f"  loaded cached embeddings {X.shape} from {cache.name} (0 API calls)")
        else:
            X = None
    else:
        X = None
    if X is None:
        client = OpenAI(api_key=load_openai_key())
        vecs = []
        for b in range(0, len(texts), BATCH):
            chunk = texts[b:b + BATCH]
            resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            vecs.extend([d.embedding for d in resp.data])
            print(f"  embedded {min(b + BATCH, len(texts))}/{len(texts)}", flush=True)
        X = np.asarray(vecs, dtype=np.float32)
        np.save(cache, X)
        print(f"  cached embeddings -> {cache.name}")
    X = X.astype(np.float64)
    # cosine distance matrix
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.clip(sim, -1.0, 1.0, out=sim)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices(len(D), k=1)
    max_edge = float(np.percentile(D[iu], 90))
    print(f"  max_edge_length (90th pctile cosine dist) = {max_edge:.4f}")

    # 4. Rips persistence.  A dim-2 Rips on ~900 points at the 90th-pctile edge length
    # enumerates billions of triangles -> OOM.  Standard lean fix: build the 1-skeleton,
    # COLLAPSE edges (preserves H1 persistence exactly), THEN expand to dim 2.
    rc = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=1)   # 1-skeleton only (bounded memory)
    st.collapse_edges()                            # critical: keeps H1, slashes simplices
    st.expansion(2)                                # now safe to fill triangles
    st.compute_persistence()
    h1 = []  # (birth, death, ratio)
    for dim, (birth, death) in st.persistence():
        if dim == 1 and death != float("inf"):
            ratio = (death - birth) / birth if birth > 1e-9 else float("inf")
            h1.append((birth, death, ratio))
    h1.sort(key=lambda t: t[2], reverse=True)
    print(f"  H1 bars (finite): {len(h1)}; top ratios: {[round(r,3) for *_ ,r in h1[:5]]}")

    # 5. for top-3 loops, recover approximate cycle support
    # gudhi doesn't return generating cycles; we approximate support by collecting the
    # vertices of the simplices whose filtration value == the loop's birth (the edges that
    # close the cycle) plus the death simplex's vertices.
    skel = list(st.get_filtration())  # (simplex_vertices, filt_value)

    def support_vertices(birth, death):
        verts = set()
        for simplex, fv in skel:
            if abs(fv - birth) < 1e-9 or abs(fv - death) < 1e-9:
                if len(simplex) >= 2:  # edges / triangles only
                    verts.update(simplex)
        # cap to a handful, ordered by trajectory position
        return sorted(verts)

    loops_out = []
    for rank, (birth, death, ratio) in enumerate(h1[:3], 1):
        verts = support_vertices(birth, death)
        passages = []
        seen_pos = set()
        for v in verts:
            gi = idxs[v]                       # global token index of this point
            if gi in seen_pos:
                continue
            seen_pos.add(gi)
            passage = enc.decode(all_tokens[max(0, gi - WIN):gi]).strip()
            passages.append({"chapter": tok_to_chapter(gi), "passage_text": passage})
        loops_out.append({
            "loop": rank,
            "ratio": round(ratio, 4),
            "birth": round(birth, 4),
            "persistence": round(death - birth, 4),
            "n_support_vertices": len(passages),
            "passages": passages,
        })
        chs = sorted({p["chapter"] for p in passages})
        print(f"  loop {rank}: ratio={ratio:.3f} birth={birth:.3f} "
              f"persist={death-birth:.3f} vertices={len(passages)} chapters={chs}")

    OUT.write_text(json.dumps({
        "method": "rolling-context masked-prefix trajectory; Rips H1 over cosine dist; "
                  "cycle support APPROXIMATE (gudhi gives no minimal generator)",
        "window_tokens": WIN, "stride": stride, "n_points": len(idxs),
        "max_edge_length": round(max_edge, 4),
        "loops": loops_out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
