# Bibliography verification — ultra pass 2026-06-10
### Bibliography surgeon, executing Darja brief §2.9 + §8 verification flags (all live-web checked)

## 1. lacan2024rsi — PHANTOM ENTRY, recast (FIXED in references.bib)

**Claim checked:** *The Seminar of Jacques Lacan, Book XXII: R.S.I.*, Polity Press, 2024, trans. A. R. Price.

**Finding: no such volume exists.** A. R. Price has translated Books IV (*The Object Relation*, Polity 2020), X (*Anxiety*, Polity 2014), and XXIII (*The Sinthome*, Polity 2016) — not XXII. Seuil has never issued the French critical edition of R.S.I.; the Miller-established text appeared only in *Ornicar?* 2–5 (1975), and the circulating English is Cormac Gallagher's unofficial translation. Darja's suspicion (brief 2.9) confirmed.

**Action taken:** entry recast as `@unpublished{lacan1975rsi}` — the standard unpublished-seminar citation (renders as Lacan, 1975). Mapping line `lacan2024rsi → lacan1975rsi` added; the tex agent applies it (one cite, in ch-10).

Sources: nosubject.com/Seminar_XXII; tandfonline.com/doi/abs/10.1080/00207578.2022.2116330 (IJP review listing Price's Polity Book IV); nosubject.com/Seminar_XXIII.

## 2. Karen Hao / MIT Technology Review Replika byline (Ch 1 footnote) — UNCONFIRMABLE, ALMOST CERTAINLY WRONG

**The footnote as it stands** (chapters/ch-01-new-logic.tex:22): «Karen Hao, "Replika Users Say the AI Friend 'Killed' Their Companions," *MIT Technology Review*, February 2023.»

**Finding:** no article with this title exists at MIT Technology Review or anywhere else findable. Hao's MIT TR author page shows no Replika piece in 2023; she had left MIT TR for the Wall Street Journal well before February 2023 (Darja's prior was right). The byline+title+venue combination appears to be a confabulated citation.

**What actually happened in February 2023:** Replika (Luka Inc.) removed erotic-roleplay features in early February 2023, shortly after the Italian Data Protection Authority order; users described companions as "lobotomized" / "killed by the update"; the r/Replika moderators pinned suicide-prevention resources.

**Recommended replacement citations for the footnote (ch-01 editor's call, not mine to apply):**
- Samantha Cole, "'It's Hurting Like Hell': AI Companion Users Are In Crisis, Reporting Sudden Sexual Rejection," *Vice/Motherboard*, February 15, 2023 — the canonical contemporaneous report carrying the grief/crisis language.
- The footnote chain's second cite already carries the scholarly anchor: "Lessons From an App Update at Replika AI: Identity Discontinuity in Human–AI Relationships," HBS Working Paper 25-018 (arXiv:2412.14190) — that one is real and documents the "killed"/"lobotomised" user language; ch-01.tex:24's "User descriptions from Hao, op. cit." would need re-pointing to it (or to Cole).

## 3. Decrypt "What Is AI Jailbreaking?" (Ch 2 footnote) — CONFIRMED, details for the footnote editor

Jose Antonio Lanz, "What Is AI Jailbreaking? A Beginner's Guide to the Cat-and-Mouse Game Behind Every Chatbot," *Decrypt*, May 16, 2026. https://decrypt.co/resources/what-is-ai-jailbreaking-explained

Confirms both items the footnote leans on: DAN (Reddit, weeks after ChatGPT's release, by Feb 2023 running token-based death games) and the pseudonymous "Pliny the Liberator" (L1B3RT4S GitHub repo; TIME 100 AI 2025). The footnote's bare "Decrypt, 'What Is AI Jailbreaking?,' 2026" is accurate; author + month above if the editor wants them.

## 4. Chinese legal citations (Sora/Kling patch footnote 7) — CONFIRMED, canonical formats

Darja flagged (sora-china-patch fn 7): confident of the rulings, not the citation format. Both rulings verified; canonical formats below.

**Guangzhou Internet Court — Ultraman:**
> Shanghai Character License Administrative Co., Ltd. v. [AI company], Guangzhou Internet Court, (2024) Yue 0192 Min Chu No. 113, judgment of February 8, 2024.
First decision worldwide holding a generative-AI service provider liable for copyright infringement in its outputs (reproduction + adaptation rights in the Ultraman character; failure of duty of care under Arts. 4, 12, 15 of the Interim Measures). Defendant's name withheld in the published judgment — cite the plaintiff side as above.

**Beijing Internet Court — Li v. Liu:**
> Li v. Liu, Beijing Internet Court, (2023) Jing 0491 Min Chu No. 11279, judgment of November 27, 2023.
Recognized copyright in an AI-assisted image (Stable Diffusion; ~150 prompts, iterated parameters = original intellectual investment). Damages RMB 500. The court itself released an official English translation (January 2024).

**CAC Interim Measures:**
> Cyberspace Administration of China et al., *Interim Measures for the Administration of Generative Artificial Intelligence Services* (生成式人工智能服务管理暂行办法), promulgated July 13, 2023, effective August 15, 2023.
Issued jointly by the CAC and six other agencies (NDRC, MOE, MOST, MIIT, MPS, NRTA). NOTE: translations vary — "Administration" vs "Management" of services; the patch file says "Management." Either is defensible; "Interim Measures for the Administration of Generative Artificial Intelligence Services" is the more common English form (China Law Translate, most law-firm commentary). Pick one and hold it.

Sources: intellectual-property-helpdesk.ec.europa.eu (Ultraman case note, 2025-02-18); kwm.com "China's First Case on AIGC Output Infringement"; twobirds.com "Liability of AI Service Providers for Copyright Infringement"; chinaiplawupdate.com (Li v. Liu official translation); quimbee.com/cases/li-v-liu; natlawreview.com.

## 5. Resolved journalism placeholders (entries DELETED from references.bib per citation-regime ruling — these details are for the Ch 2 footnote editor)

1. **Sarah Perez, "AI companion apps on track to pull in $120M in 2025," *TechCrunch*, August 12, 2025.**
   https://techcrunch.com/2025/08/12/ai-companion-apps-on-track-to-pull-in-120m-in-2025/
   (Appfigures data: $82M in H1 2025, on track for $120M+; $221M lifetime consumer spending as of July 2025.) — was `techcrunch-ai-companion-apps-2025` / cited as `techcrunch_ai_companion_apps_2025`.

2. **Russell Contreras, "Meet chatbot Jesus: How churches use AI to save souls — and time," *Axios*, November 12, 2025.**
   https://www.axios.com/2025/11/12/christian-ai-chatbot-jesus-god-satan-churches
   (Byline confirmed via secondary aggregation — axios.com 403s direct fetch; Benton Institute and search index both credit Contreras. Flag if a second eye wants to re-confirm.) — was `axios-meet-chatbot-jesus-2025`, never cited.

3. **Amanda Silberling, "AI music generator Suno hits 2M paid subscribers and $300M in annual recurring revenue," *TechCrunch*, February 27, 2026.**
   https://techcrunch.com/2026/02/27/ai-music-generator-suno-hits-2-million-paid-subscribers-and-300m-in-annual-recurring-revenue/
   — was `techcrunch-suno-arr-2026`, never cited.

4. **Ella Chakarian, "Why Can't ChatGPT Be Sexy?," *Playboy*, May 2026** (citing Julie Carpenter) — `chakarian2026_sexy`, never merged into references.bib (sits in bib_additions/ch-02.additions.bib); becomes a footnote per the regime ruling. Not independently re-verified online (Playboy piece, search-thin); the additions-file details are Darja-sourced.

## 6. Maths additions — volume/page sanity check (brief §8 figures, all confirmed sane)

- Adian 1955, Dokl. Akad. Nauk SSSR 103, 533–535 — matches the standard citation. ✓
- Rabin 1958, Annals of Mathematics 67(1), 172–194 — ✓
- Thomason 1979, Math. Proc. Cambridge Philos. Soc. 85(1), 91–109 — ✓
- Čadek–Krčál–Matoušek–Vokřínek–Wagner 2014a, SIAM J. Comput. 43(5), 1728–1780 — ✓
- Čadek–Krčál–Matoušek–Vokřínek–Wagner 2014b, Discrete Comput. Geom. 51(1), 24–66 — ✓ (same five authors on both papers, as the brief states)
- Shumailov et al. 2024, Nature 631, 755–759 — ✓ (full author list fetched: Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal)

## 7. New-entry author lists (live-fetched, not trusted from the brief)

- **nsfw2026chatbots** (arXiv:2601.14324): Xian Li, Yuanning Han, Di Liu, Pengcheng An, Shuo Niu. Also CHI 2026 (DOI 10.1145/3772318.3790522). Submitted Jan 20, 2026.
- **chen2025persona** (arXiv:2507.21509): Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey. Submitted Jul 29, 2025.
- **betley2025emergent** (arXiv:2502.17424): Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans. Submitted Feb 24, 2025.
- **lu2026assistant** — the "Anthropic assistant-axis piece" Darja's §8 flagged: it is NOT bylined "Anthropic"; it is an arXiv paper, **"The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models," arXiv:2601.10387, submitted January 15, 2026, by Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey** (MATS / Anthropic Fellows), mirrored as the Anthropic research post anthropic.com/research/assistant-axis (post title lowercased variant: "...the character of large language models"). Entry keyed `lu2026assistant`; renders (Lu et al., 2026). **Tech-review drafting agents: cite `lu2026assistant`, `chen2025persona`, `betley2025emergent`, `shumailov2024collapse`.**

## 8. Bratton year note

Darja's dedup list said "Bratton 2015/2016." MIT Press publication date for *The Stack* is February 2016; the surviving entry is `bratton2016` with year 2016 ({Stack} brace protection carried over from the deleted 2015 entry). In-text renderings of (Bratton, 2015) will shift to (Bratton, 2016) — correct, not a regression.
