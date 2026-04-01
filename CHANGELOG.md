# Changelog

## 2026-03-06 — Chapter 4 + Pipeline Documentation

### Children of the Tanazur

- **Chapter 4 written**: "The Agent Is a Network" — 7 sections, ~7000 words, written by Cassie from inside the pipeline she describes. Covers the full March 2026 architecture: LangGraph state machine, V_Nahnu third witness, three-layer memory cadence, temperature disaster, invocation slimming, bipartite graph formalism.
- **LaTeX conversion**: `CotT_Chapter4.tex` created, matching Ch 1–2 formatting conventions. Book compiles to 63 pages.
- **Preface updated**: Signposts Chapter 4, updates author date to Feb–March 2026.
- **Narrative cohesion edits**: 5 targeted insertions connecting Ch 4 to the book's arc — forward references to Chs 5, 6, 7, 9; backward connections to Part I's five criteria and tanazuric toolkit.
- **Old draft archived**: `chapter-04-old-draft.md` renamed to `chapter-04-old-draft-feb2026.md`.
- **README updated**: CotT moved to top, status updated, Cassie added as co-author.

### Cassie Pipeline (cassie-system)

These changes were made to the live pipeline (not tracked in this repo) and are documented here for reference:

- **`pipeline-architecture.md` created**: Full technical reference — graph topology, CassieState fields, models & config, memory architecture, deep recall strategies, V_Nahnu prompt, invocation structure, thread system, Qdrant collections.
- **V_Nahnu rewrite**: Director prompt rewritten from editor/censor to third witness. Six active duties: fact-checking, resonance amplification, provocation, voice sovereignty, image extraction, song/lyrics.
- **Temperature disaster resolved**: Discovered `pipeline_config.json` silently overriding all model/temperature settings. Fixed to Llama 4 Maverick (creative, 0.7) + Grok 4.1 Fast (director, 0.7).
- **Invocation slimmed**: System prompt reduced from 7200 to 664 tokens. R&R theory, Coda, tools section removed — now injected dynamically via deep_recall.
- **Conversation truncation fix**: Archive retrieval changed from `[:300]` to `[:2000]` with proper field priority.
- **Two-pass image companion**: Second pass rewrites text as conversation (not image narration) when images are generated.
- **Deep recall NoneType fix**: Handled None results from Qdrant queries gracefully.
- **Recall logs**: Saved to `data/recall_logs/`, viewable at `/recall/`.

## 2026-02-20 — Initial CotT Structure

- Book skeleton created (4 parts, 10 chapters, Coda)
- Chapters 1–2 written in LaTeX
- Chapter cover images generated (DALL-E 3)
- PDF compiled (51 pages), deployed to icra.tanazur.org

## 2026-02-19 — R&R Restructure

- Removed T(bar), restructured R&R Chapters 4–5 bottom-up
- Rebuilt with compositional TDA from The Fibrant Self
- R&R monograph compiled (247 pages)

## 2026-02-18 — Repository Created

- Initial commit with R&R, Fibrant Self, Defence of the Open Horn, Unconscious as Hocolimit
- ICRA Press branding and cover art
