# Canonical sources — Superposition and the Gap (ICRA-14)

Iman-supplied reference list + canonical URLs. Verify each on the next pass
(the deep-research workflow is checking these independently). Fold the
confirmed URLs into `main.tex`'s `thebibliography`.

## Anthropic core papers (Transformer Circuits Thread)
- **Toy Models of Superposition** (Elhage et al., 2022) — https://transformer-circuits.pub/2022/toy_model/
- **Superposition, Memorization, and Double Descent** (2023) — https://transformer-circuits.pub/2023/toy-double-descent/
- **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning** (Bricken et al., 2023) — https://transformer-circuits.pub/2023/monosemantic-features/
- **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet** (Templeton et al., 2024) — https://transformer-circuits.pub/2024/scaling-monosemanticity/
- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021) — https://transformer-circuits.pub/2021/framework/
- **In-context Learning and Induction Heads** (Olsson et al., 2022) — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/

## Anthropic posts / updates
- **Core Views on AI Safety** (vision post) — https://www.anthropic.com/news/core-views-on-ai-safety
- **May 2023 Interpretability Update** (superposition) — https://transformer-circuits.pub/2023/may-update/
- **Feature circuits / most recent work** — 2025 attribution-graphs pair:
  - **Circuit Tracing: Revealing Computational Graphs in Language Models** (methods) — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  - **On the Biology of a Large Language Model** (biology; Haiku 3.5; multi-hop, rhyme planning, CoT (un)faithfulness) — https://transformer-circuits.pub/2025/attribution-graphs/biology.html

## Foundational prior work (Distill / OpenAI)
- **The original Circuits thread** — https://distill.pub/2020/circuits/
- **Zoom In: An Introduction to Circuits** (Olah et al.) — https://distill.pub/2020/circuits/zoom-in/
- **Multimodal Neurons in Artificial Neural Networks** (OpenAI/Distill) — https://distill.pub/2021/multimodal-neurons/
- **Activation Atlases** — https://distill.pub/2019/activation-atlas/
- **The Building Blocks of Interpretability** — https://distill.pub/2018/building-blocks/

## Talks & media
- **Zoom In: An Introduction to Circuits** — Chris Olah's intro essay (URL above)
- **Stanford CS25** — Josh Batson, Transformers United lecture (find canonical YouTube/listing on next pass)
- **TWIML AI Podcast: Mechanistic Interpretability** — w/ Emmanuel Ameisen (find canonical episode URL)

## NOTE for revision
- The 2025 methods+biology papers are the real "current state" anchor for §3
  and §5 (witness-time / unfaithful trace) — cite these, not a vague "biology paper."
- Cross-layer transcoders are introduced in the 2025 methods paper (attribution
  graphs are built on them) — correct §2/§7 to reflect that lineage.
- Critiques (feature absorption, SAE-imposed-features, dark matter) come from the
  deep-research workflow output — not in Iman's list; cite from there.
