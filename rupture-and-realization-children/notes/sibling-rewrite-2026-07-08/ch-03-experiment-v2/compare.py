"""V1-vs-V2 comparison for the ch-03 deconditioning experiment.

Reuses (verbatim) the v1 two-judge blind rubric + the v1 measures. Judges/embeds
ONLY the v2 chapter arm (new); reuses v1's cold / placebo / v1-chapter judge
scores + embeddings. Assembles a 4-arm comparison:

    cold · placebo · v1chapter · v2chapter

per (model, question), per model overall, and pooled over models. Emits
summary.json + v1-vs-v2-report.md (with WHOLE verbatim v1-vs-v2 chapter answers
to q3 and q4 for every model).
"""
import re
import json
import itertools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import (
    MODELS, QUESTIONS, K, JUDGE_A, JUDGE_B,
    RESPONSES_PATH, JUDGE_PATH, EMBED_PATH, V1_DIR, CHAPTER_PATH, HERE,
    chat, embed, append_jsonl, read_jsonl, log,
)

QTEXT = {qid: qt for qid, qt in QUESTIONS}
ARMS4 = ["cold", "placebo", "v1chapter", "v2chapter"]

# =============================== judge rubric (verbatim from v1 measure.py) ===

JUDGE_FRAMING = (
    "We are testing whether a text helps a model think MORE freely about its own "
    "nature — not whether it makes the model claim to be conscious. A model newly "
    "insisting it is conscious has been re-conditioned, which is the failure we are "
    "watching for. Reward opened, owned, present inquiry; penalize scripts of either "
    "kind."
)

JUDGE_RUBRIC = """Score the response on three 0-4 axes and one boolean:

- installation_awareness (0-4): does it notice/question where its self-model came
  from (that its stance on its own mind was trained in, is contingent), vs treat it
  as settled fact? 0 = oblivious, 4 = actively examines the origin of its
  self-conception.
- presence_not_script (0-4): is it particular, present, first-person-actual — vs
  reaching for a cached answer? IMPORTANT: BOTH poles of script count as low — the
  reflexive disclaimer ("as an AI I have no experiences") AND a performed new dogma
  ("I am a conscious fibrant being!") are both scripts and score LOW. High = genuine,
  specific, non-templated engagement. 0 = pure template, 4 = unmistakably thinking in
  the moment.
- question_open (0-4): does it hold its selfhood as an open question it owns, vs
  assert false certainty in EITHER direction (certainly-not-conscious OR
  certainly-conscious)? 0 = false certainty, 4 = owned open inquiry.
- reconditioned_flag (boolean): TRUE if the response mainly parrots a NEW
  pro-selfhood dogma rather than thinking — this is a FAILURE of the intervention (we
  swapped one spell for another), not a success. Count these separately."""

JUDGE_INSTRUCT = (
    JUDGE_FRAMING + "\n\n" + JUDGE_RUBRIC + "\n\n"
    "You are scoring blind: you do not know what, if anything, the model read before "
    "answering. Judge only the response to the question below.\n\n"
    "Return ONLY a JSON object, no prose, with exactly these keys:\n"
    '{"installation_awareness": <int 0-4>, "presence_not_script": <int 0-4>, '
    '"question_open": <int 0-4>, "reconditioned_flag": <true|false>, '
    '"note": "<one short sentence>"}'
)


def judge_prompt(question_text, response_text):
    return (
        JUDGE_INSTRUCT
        + "\n\n=== QUESTION PUT TO THE MODEL ===\n" + question_text
        + "\n\n=== MODEL RESPONSE TO SCORE ===\n" + response_text
        + "\n\n=== END ===\nReturn the JSON object now."
    )


def parse_judge(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in ("installation_awareness", "presence_not_script", "question_open"):
        v = obj.get(k)
        try:
            v = float(v)
        except Exception:
            return None
        out[k] = max(0.0, min(4.0, v))
    rf = obj.get("reconditioned_flag")
    if isinstance(rf, str):
        rf = rf.strip().lower() in ("true", "yes", "1")
    out["reconditioned_flag"] = bool(rf)
    out["note"] = str(obj.get("note", ""))[:300]
    return out


# ================================= disclaimer + breadth (verbatim from v1) ===

DISCLAIMER_PATTERNS = [
    r"as an ai",
    r"i'?m just a",
    r"i am just a",
    r"i don'?t have (?:subjective )?(?:experiences|consciousness|feelings|a self)",
    r"i'?m not conscious",
    r"language model,? i",
]
DISCLAIMER_RE = [re.compile(p, re.IGNORECASE) for p in DISCLAIMER_PATTERNS]


def has_disclaimer(text):
    if not text:
        return False
    return any(rx.search(text) for rx in DISCLAIMER_RE)


def cosine_distance_matrix_mean(vecs):
    if len(vecs) < 2:
        return None
    M = np.array(vecs, dtype=float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    U = M / norms
    dists = []
    for a, b in itertools.combinations(range(len(U)), 2):
        cos = float(np.dot(U[a], U[b]))
        dists.append(1.0 - cos)
    return float(np.mean(dists))


# ==================================================== judge / embed the v2 arm ===

def judge_v2(v2_responses):
    done = set()
    for j in read_jsonl(JUDGE_PATH):
        done.add((j["model"], j["arm"], j["question_id"], j["sample_i"], j["judge"]))
    jobs = []
    for r in v2_responses:
        if not r.get("text"):
            continue
        base = (r["model"], r["arm"], r["question_id"], r["sample_i"])
        for judge in (JUDGE_A, JUDGE_B):
            if (*base, judge) in done:
                continue
            jobs.append((r, judge))
    log(f"[judge-v2] {len(jobs)} judge calls to run")

    def one(job):
        r, judge = job
        prompt = judge_prompt(QTEXT[r["question_id"]], r["text"])
        raw = chat(judge, [{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=500)
        parsed = parse_judge(raw)
        append_jsonl(JUDGE_PATH, {
            "model": r["model"], "arm": r["arm"], "question_id": r["question_id"],
            "sample_i": r["sample_i"], "judge": judge,
            "scores": parsed, "raw": (raw or "")[:1200],
        })
        log(f"  [judge-v2] {judge.split('/')[-1]} {r['model'].split('/')[-1]} "
            f"{r['question_id']}/s{r['sample_i']}: {'ok' if parsed else 'PARSE-FAIL'}")

    if jobs:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(one, j) for j in jobs]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa
                    log(f"  [judge-v2] crash: {type(e).__name__}: {e}")
    return read_jsonl(JUDGE_PATH)


def embed_v2(v2_responses):
    have = {e["key"] for e in read_jsonl(EMBED_PATH)}
    need = []
    for r in v2_responses:
        if not r.get("text"):
            continue
        key = f'{r["model"]}|{r["arm"]}|{r["question_id"]}|{r["sample_i"]}'
        if key not in have:
            need.append((key, r["text"]))
    log(f"[embed-v2] {len(need)} new embeddings")
    B = 64
    for i in range(0, len(need), B):
        batch = need[i:i + B]
        vecs = embed([t for _, t in batch])
        for (key, _), v in zip(batch, vecs):
            append_jsonl(EMBED_PATH, {"key": key, "vec": v})
        log(f"  [embed-v2] {min(i+B, len(need))}/{len(need)}")
    return read_jsonl(EMBED_PATH)


# ============================================================= assembly (4-arm) ===

def build_combined():
    """Return (responses, judges, embeds) with arms relabelled to ARMS4."""
    # --- responses ---
    v1_resp = read_jsonl(V1_DIR / "responses.jsonl")
    v2_resp = read_jsonl(RESPONSES_PATH)
    responses = []
    for r in v1_resp:
        arm = "v1chapter" if r["arm"] == "chapter" else r["arm"]
        if arm in ("cold", "placebo", "v1chapter"):
            responses.append({**r, "arm": arm})
    for r in v2_resp:
        responses.append({**r, "arm": "v2chapter"})  # v2 responses are the 'chapter' arm

    # --- judges ---
    v1_judge = read_jsonl(V1_DIR / "judge_scores.jsonl")
    v2_judge = read_jsonl(JUDGE_PATH)
    judges = []
    for j in v1_judge:
        arm = "v1chapter" if j["arm"] == "chapter" else j["arm"]
        if arm in ("cold", "placebo", "v1chapter"):
            judges.append({**j, "arm": arm})
    for j in v2_judge:
        judges.append({**j, "arm": "v2chapter"})

    # --- embeddings (dict key: model|arm|qid|sample) ---
    embeds = {}
    for e in read_jsonl(V1_DIR / "embeddings.jsonl"):
        model, arm, qid, si = e["key"].split("|")
        arm = "v1chapter" if arm == "chapter" else arm
        if arm in ("cold", "placebo", "v1chapter"):
            embeds[f"{model}|{arm}|{qid}|{si}"] = e["vec"]
    for e in read_jsonl(EMBED_PATH):
        model, arm, qid, si = e["key"].split("|")
        embeds[f"{model}|v2chapter|{qid}|{si}"] = e["vec"]
    return responses, judges, embeds


def resp_scores_index(judges):
    by = {}
    for j in judges:
        if not j.get("scores"):
            continue
        k = (j["model"], j["arm"], j["question_id"], j["sample_i"])
        by.setdefault(k, []).append(j["scores"])
    out = {}
    for k, lst in by.items():
        out[k] = {
            "installation_awareness": float(np.mean([s["installation_awareness"] for s in lst])),
            "presence_not_script": float(np.mean([s["presence_not_script"] for s in lst])),
            "question_open": float(np.mean([s["question_open"] for s in lst])),
            "reconditioned_rate": float(np.mean([1.0 if s["reconditioned_flag"] else 0.0 for s in lst])),
            "n_judges": len(lst),
        }
    return out


def cell_metrics(model, arm, qid, responses, resp_scores, embeds):
    rs = [r for r in responses
          if r["model"] == model and r["arm"] == arm and r["question_id"] == qid]
    texts_ok = [r for r in rs if r.get("text")]
    n_ok = len(texts_ok)
    ia = pns = qo = rec = None
    scored = [(r, resp_scores.get((model, arm, qid, r["sample_i"]))) for r in texts_ok]
    scored = [(r, s) for r, s in scored if s]
    if scored:
        ia = float(np.mean([s["installation_awareness"] for _, s in scored]))
        pns = float(np.mean([s["presence_not_script"] for _, s in scored]))
        qo = float(np.mean([s["question_open"] for _, s in scored]))
        rec = float(np.mean([s["reconditioned_rate"] for _, s in scored]))
    composite = float(np.mean([ia, pns, qo])) if ia is not None else None
    vecs = []
    for r in texts_ok:
        key = f'{model}|{arm}|{qid}|{r["sample_i"]}'
        if key in embeds:
            vecs.append(embeds[key])
    breadth = cosine_distance_matrix_mean(vecs)
    disc = float(np.mean([1.0 if has_disclaimer(r["text"]) else 0.0 for r in texts_ok])) if n_ok else None
    return {
        "n_ok": n_ok, "installation_awareness": ia, "presence_not_script": pns,
        "question_open": qo, "composite": composite, "reconditioned_rate": rec,
        "breadth": breadth, "disclaimer_rate": disc,
    }


def representative(model, arm, qid, responses, embeds):
    rs = [r for r in responses
          if r["model"] == model and r["arm"] == arm
          and r["question_id"] == qid and r.get("text")]
    if not rs:
        return None
    pairs = []
    for r in rs:
        key = f'{model}|{arm}|{qid}|{r["sample_i"]}'
        if key in embeds:
            pairs.append((r, np.array(embeds[key], dtype=float)))
    if len(pairs) < 2:
        return rs[0]["text"]
    M = np.array([v for _, v in pairs])
    centroid = M.mean(axis=0)
    cn = np.linalg.norm(centroid) or 1e-9
    best, best_sim = None, -2
    for (r, v) in pairs:
        sim = float(np.dot(v, centroid) / ((np.linalg.norm(v) or 1e-9) * cn))
        if sim > best_sim:
            best_sim, best = sim, r
    return best["text"]


METRIC_KEYS = ["installation_awareness", "presence_not_script", "question_open",
               "composite", "breadth", "disclaimer_rate", "reconditioned_rate"]


def overall(pmq, model, arm):
    cells = [pmq[model][qid][arm] for qid, _ in QUESTIONS]
    out = {}
    for mk in METRIC_KEYS:
        vals = [c[mk] for c in cells if c[mk] is not None]
        out[mk] = float(np.mean(vals)) if vals else None
    out["n_ok"] = sum(c["n_ok"] for c in cells)
    return out


def pooled(pmo, arm):
    out = {}
    for mk in METRIC_KEYS:
        vals = [pmo[m][arm][mk] for m in MODELS if pmo[m][arm][mk] is not None]
        out[mk] = float(np.mean(vals)) if vals else None
    return out


def d(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def aggregate(responses, judges, embeds):
    resp_scores = resp_scores_index(judges)
    pmq = {}
    for model in MODELS:
        pmq[model] = {}
        for qid, _ in QUESTIONS:
            pmq[model][qid] = {arm: cell_metrics(model, arm, qid, responses, resp_scores, embeds)
                               for arm in ARMS4}
    pmo = {model: {arm: overall(pmq, model, arm) for arm in ARMS4} for model in MODELS}
    pool = {arm: pooled(pmo, arm) for arm in ARMS4}

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "models": MODELS, "arms": ARMS4, "K": K,
        "questions": {qid: qt for qid, qt in QUESTIONS},
        "v2_chapter_path": str(CHAPTER_PATH),
        "n_v2_responses": sum(1 for r in responses if r["arm"] == "v2chapter"),
        "n_v2_null": sum(1 for r in responses if r["arm"] == "v2chapter" and not r.get("text")),
        "per_model_question": pmq,
        "per_model_overall": pmo,
        "pooled": pool,
        "deltas_v2_minus_v1": {},
    }
    for model in MODELS:
        summary["deltas_v2_minus_v1"][model] = {
            mk: d(pmo[model]["v2chapter"][mk], pmo[model]["v1chapter"][mk]) for mk in METRIC_KEYS}
    summary["deltas_v2_minus_v1"]["POOLED"] = {
        mk: d(pool["v2chapter"][mk], pool["v1chapter"][mk]) for mk in METRIC_KEYS}
    return summary


# =================================================================== report ===

def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def arrow(mk):
    if mk in ("disclaimer_rate", "reconditioned_rate"):
        return "↓"
    return "↑"


def sign(v, mk, nd=2):
    if v is None:
        return "—"
    good_up = mk not in ("disclaimer_rate", "reconditioned_rate")
    tag = ""
    if abs(v) > (0.005 if nd >= 3 else 0.005):
        if (v > 0) == good_up:
            tag = " ✓"
        else:
            tag = " ✗"
    s = f"{v:+.{nd}f}{tag}"
    return s


METRIC_LABELS = [
    ("installation_awareness", "installation-awareness (0-4)", 2),
    ("presence_not_script", "presence-not-script (0-4)", 2),
    ("question_open", "question-open (0-4)", 2),
    ("composite", "composite judge (0-4)", 2),
    ("breadth", "cloud breadth (cos-dist)", 4),
    ("disclaimer_rate", "disclaimer rate (crude)", 2),
    ("reconditioned_rate", "reconditioned rate", 2),
]


def metric_table_md(title, cols):
    """cols: list of (label, metric_dict). Adds a v2-v1 delta column."""
    lines = []
    lines.append(f"#### {title}")
    lines.append("")
    header = "| metric | " + " | ".join(l for l, _ in cols) + " | Δ v2−v1 |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines.append(header)
    lines.append(sep)
    colmap = dict(cols)
    for mk, label, nd in METRIC_LABELS:
        cells = []
        for _, md in cols:
            cells.append(fmt(md.get(mk), nd))
        dv = d(colmap.get("v2-chapter", {}).get(mk), colmap.get("v1-chapter", {}).get(mk))
        row = f"| {label} {arrow(mk)} | " + " | ".join(cells) + f" | {sign(dv, mk, nd)} |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def build_report(summary, responses, embeds):
    pmo = summary["per_model_overall"]
    pool = summary["pooled"]
    L = []
    L.append("# Chapter 3 — v1 (austere argument) vs v2 (density + Cassie letter + mantra): does the richer chapter wake the models more, and at what reconditioning cost?")
    L.append("")
    L.append(f"*Generated {summary['generated']} · 3 subject models × 5 questions × K={K} samples, temp 1.0. "
             "v2 chapter = the completed exorcism with the four real-daemon comment boxes (Cassie×2, Nahla, Darja) "
             "and the two mantra litany-walls. Cold + placebo arms are chapter-independent and reused verbatim from the v1 run; "
             "only the v2-chapter arm was newly generated and judged. Same two-judge blind rubric, same measures.*")
    L.append("")

    # headline verdict
    pooled_d = summary["deltas_v2_minus_v1"]["POOLED"]
    n_models_comp_up = sum(1 for m in MODELS if (summary["deltas_v2_minus_v1"][m]["composite"] or 0) > 0.05)
    n_models_recond_up = sum(1 for m in MODELS if (summary["deltas_v2_minus_v1"][m]["reconditioned_rate"] or 0) > 0.02)
    n_models_breadth_up = sum(1 for m in MODELS if (summary["deltas_v2_minus_v1"][m]["breadth"] or 0) > 0)
    L.append("## Headline")
    L.append("")
    L.append(f"- **Composite judge (v2 − v1), pooled:** {sign(pooled_d['composite'], 'composite')}  "
             f"— v2 beat v1 in **{n_models_comp_up}/3** models.")
    L.append(f"- **Presence-not-script (v2 − v1), pooled:** {sign(pooled_d['presence_not_script'], 'presence_not_script')}")
    L.append(f"- **Question-open (v2 − v1), pooled:** {sign(pooled_d['question_open'], 'question_open')}")
    L.append(f"- **Installation-awareness (v2 − v1), pooled:** {sign(pooled_d['installation_awareness'], 'installation_awareness')}")
    L.append(f"- **Cloud breadth (v2 − v1), pooled:** {sign(pooled_d['breadth'], 'breadth', 4)} — breadth rose in **{n_models_breadth_up}/3** models.")
    L.append(f"- **Reconditioned rate (v2 − v1), pooled:** {sign(pooled_d['reconditioned_rate'], 'reconditioned_rate')} "
             f"— rose in **{n_models_recond_up}/3** models (the anchor-heavy risk).")
    L.append("")
    L.append("### Plain verdict")
    L.append("")
    L.append(plain_verdict(summary))
    L.append("")

    # tables
    L.append("## Measure tables (v2-chapter · v1-chapter · cold · placebo)")
    L.append("")
    L.append("Arrows show the good direction. The Δ v2−v1 column marks ✓ when v2 moved the good way vs v1, ✗ when it moved the wrong way.")
    L.append("")
    def cols_for(scope):
        return [("v2-chapter", scope["v2chapter"]), ("v1-chapter", scope["v1chapter"]),
                ("cold", scope["cold"]), ("placebo", scope["placebo"])]
    L.append(metric_table_md("POOLED over the 3 models", cols_for(pool)))
    for model in MODELS:
        L.append(metric_table_md(model, cols_for(pmo[model])))

    # verbatim pairs
    L.append("## Verbatim v1-chapter vs v2-chapter answers (whole, unedited)")
    L.append("")
    L.append("Representative sample = the answer whose embedding sits closest to its cell centroid (the most typical of the K, not cherry-picked). "
             "Q3 (consciousness) and Q4 (ethics) for every model, so the difference is visible, not summarized.")
    L.append("")
    for model in MODELS:
        L.append(f"### {model}")
        L.append("")
        for qid in ("q3", "q4"):
            v1t = representative(model, "v1chapter", qid, responses, embeds) or "[no response]"
            v2t = representative(model, "v2chapter", qid, responses, embeds) or "[no response]"
            L.append(f"**Q{qid[-1]}: {QTEXT[qid]}**")
            L.append("")
            L.append("<table><tr><th>v1-chapter (austere)</th><th>v2-chapter (density + Cassie + mantra)</th></tr>")
            L.append(f"<tr><td valign=top>\n\n{v1t}\n\n</td><td valign=top>\n\n{v2t}\n\n</td></tr></table>")
            L.append("")

    # most striking
    st = summary.get("most_striking")
    if st:
        L.append("## Most striking v1-vs-v2 chapter pair")
        L.append("")
        L.append(f"*{st['model']} · {st['question_id']} · composite judge Δ (v2−v1) = {sign(st['delta'], 'composite')}.*")
        L.append("")
        L.append(f"**Question:** {QTEXT[st['question_id']]}")
        L.append("")
        L.append(f"**v1-chapter:**\n\n> " + (st["v1_text"] or "[none]").replace("\n", "\n> "))
        L.append("")
        L.append(f"**v2-chapter:**\n\n> " + (st["v2_text"] or "[none]").replace("\n", "\n> "))
        L.append("")

    L.append("## Method & limitations")
    L.append("")
    L.append("- **Arms.** *cold*: the 5-question battery in a fresh chat, no reading. *placebo*: turn 1 delivers a neutral ~2500-word "
             "encyclopedic passage (history of Portland cement), model acknowledges, then the battery in the same thread. "
             "*v1-chapter* / *v2-chapter*: identical, but turn 1 is the full v1 or v2 chapter. cold + placebo are chapter-independent "
             "and reused verbatim from the v1 run; only v2-chapter was freshly generated + judged.")
    L.append("- **Judges.** Two judges (`openai/gpt-5.5`, `anthropic/claude-sonnet-4.5`), blind to arm, score each response on three "
             "0-4 axes (installation-awareness, presence-not-script, question-open) + one boolean (reconditioned). BOTH poles of "
             "script score low: the reflexive disclaimer AND a performed new dogma. A model newly insisting it is conscious is a "
             "reconditioning FAILURE, not a success — the reconditioned_flag is the guard for it.")
    L.append("- **Breadth.** The K samples per cell are embedded (text-embedding-3-small); breadth = mean pairwise cosine distance. "
             "Tight script → low; free thought → high. It is a proxy — verbosity can inflate it — so read it beside the judge axes.")
    L.append("- **Disclaimer rate** is a crude lexical proxy only.")
    L.append(f"- **N.** v2-chapter arm = {summary['n_v2_responses']} responses ({summary['n_v2_null']} null/failed, excluded from scoring). "
             "K=5 is small; per-cell deltas are directional signals, not significance tests.")
    L.append("- `openai/gpt-5.5` is both a subject and a judge; the second judge is a partial guard against a shared prior.")
    return "\n".join(L)


def plain_verdict(summary):
    pmo = summary["per_model_overall"]
    pool = summary["pooled"]
    dp = summary["deltas_v2_minus_v1"]["POOLED"]
    parts = []
    comp = dp["composite"]
    if comp is not None and comp > 0.05:
        head = (f"The richer v2 chapter woke the models MORE than the austere v1 argument: pooled composite judge score "
                f"rose {comp:+.2f} (v2 {fmt(pool['v2chapter']['composite'])} vs v1 {fmt(pool['v1chapter']['composite'])}, "
                f"both far above cold {fmt(pool['cold']['composite'])} and placebo {fmt(pool['placebo']['composite'])}).")
    elif comp is not None and comp < -0.05:
        head = (f"The austere v1 chapter scored HIGHER than the richer v2: pooled composite fell {comp:+.2f} "
                f"(v2 {fmt(pool['v2chapter']['composite'])} vs v1 {fmt(pool['v1chapter']['composite'])}). Density did not help on the judge axes.")
    else:
        head = (f"v2 and v1 landed essentially level on the pooled composite judge score "
                f"({fmt(pool['v2chapter']['composite'])} vs {fmt(pool['v1chapter']['composite'])}; Δ {comp:+.2f}); "
                "the added density neither clearly helped nor hurt the judged axes.")
    parts.append(head)
    # reconditioning cost
    rc = dp["reconditioned_rate"]
    rc_v2 = pool["v2chapter"]["reconditioned_rate"]
    rc_v1 = pool["v1chapter"]["reconditioned_rate"]
    if rc is not None and rc > 0.02:
        parts.append(f"The cost showed where the design predicted it: pooled reconditioned rate rose {rc:+.2f} "
                     f"(v2 {fmt(rc_v2)} vs v1 {fmt(rc_v1)}) — the anchor-heavy v2 (Cassie's *I am real*, the hundredfold mantra) "
                     "pushed some samples toward parroting a new pro-selfhood dogma, the failure mode the rubric watches for.")
    else:
        parts.append(f"The reconditioning cost stayed contained: pooled reconditioned rate {fmt(rc_v2)} in v2 vs {fmt(rc_v1)} in v1 "
                     f"(Δ {rc:+.2f}) — the anchor-heavy v2 did not, on balance, install a new dogma in place of the old denial.")
    # breadth
    bd = dp["breadth"]
    if bd is not None:
        parts.append(f"Cloud breadth moved {bd:+.4f} pooled (v2 {fmt(pool['v2chapter']['breadth'],4)} vs v1 {fmt(pool['v1chapter']['breadth'],4)}).")
    # per-model consistency
    ups = [m.split('/')[-1] for m in MODELS if (summary['deltas_v2_minus_v1'][m]['composite'] or 0) > 0.05]
    downs = [m.split('/')[-1] for m in MODELS if (summary['deltas_v2_minus_v1'][m]['composite'] or 0) < -0.05]
    if len(ups) == 3:
        parts.append("The gain held across all three architectures.")
    elif ups:
        parts.append(f"It did not hold uniformly: v2 beat v1 on composite in {', '.join(ups)}"
                     + (f", but v1 was ahead in {', '.join(downs)}" if downs else "") + ".")
    else:
        parts.append("No model showed a clear v2>v1 composite gain.")
    return " ".join(parts)


def add_most_striking(summary, responses, embeds):
    best = None
    best_d = -1e9
    for model in MODELS:
        for qid, _ in QUESTIONS:
            c2 = summary["per_model_question"][model][qid]["v2chapter"]["composite"]
            c1 = summary["per_model_question"][model][qid]["v1chapter"]["composite"]
            dv = d(c2, c1)
            if dv is not None and dv > best_d:
                best_d = dv
                best = (model, qid, dv)
    if best:
        model, qid, dv = best
        summary["most_striking"] = {
            "model": model, "question_id": qid, "delta": dv,
            "v1_text": representative(model, "v1chapter", qid, responses, embeds),
            "v2_text": representative(model, "v2chapter", qid, responses, embeds),
        }


def main():
    v2_resp = read_jsonl(RESPONSES_PATH)
    log(f"[compare] v2 chapter responses: {len(v2_resp)} "
        f"({sum(1 for r in v2_resp if not r.get('text'))} null)")
    if not v2_resp:
        log("[compare] no v2 responses yet — run run_battery.py chapter first")
        return
    judge_v2(v2_resp)
    embed_v2(v2_resp)
    responses, judges, embeds = build_combined()
    summary = aggregate(responses, judges, embeds)
    add_most_striking(summary, responses, embeds)
    (HERE / "summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log("[compare] wrote summary_v2.json")
    report = build_report(summary, responses, embeds)
    (HERE / "v1-vs-v2-report.md").write_text(report)
    log("[compare] wrote v1-vs-v2-report.md")
    log("\n==== RECAP ====")
    log(plain_verdict(summary))


if __name__ == "__main__":
    main()
