# Bibliography pass — summary log (ultra pass 2026-06-10)
### Bibliography surgeon. Files touched: `references.bib` only. No .tex edits.
### Companion outputs: `bibkey-mapping.txt` (for the tex-key agent), `bib-verification.md` (for footnote editors + Iman).

Charter trace: Darja brief §8 (additions), §2.9 (verifications), edit-report §1 (dedup list + placeholders), brief §3 via DARJA-RULINGS (citation regime → journalism entries deleted-uncited).

## Counts
- 180 entries before → **173 after** (19 deleted, 12 added).
- Verification suite invariants now hold for the bib: `grep -c "to be filled in" references.bib` → 0; no two entries share a normalized title; no duplicate keys; brace-balance clean.

## 1. Dedup (16 deletions, mapping in bibkey-mapping.txt)
Survivor = the key the .tex files cite more (checked by grep), content = best of the pair.

| Deleted | Survivor | Note |
|---|---|---|
| carlsson2009 | carlsson2009topology | identical |
| vaswani2017 | vaswani2017attention | identical |
| schimmel1975 | schimmel1975mystical | identical |
| nagel1974 | nagel1974bat | identical |
| ouyang2022training, ouyang2022 | ouyang2022instructgpt | triple |
| christiano2017deep | christiano2017 | identical |
| askell2021hhh | askell2021general | survivor upgraded: "Olah, Chris" → "Olah, Christopher" |
| edelsbrunner2010computational | edelsbrunner2010 | both currently uncited; kept one |
| grothendieck1971revetements | grothendieck1971 | tie (1 cite each) |
| haraway2016staying | haraway2016 | tie |
| bratton2015 | bratton2016 | year 2016 is correct (MIT Press, Feb 2016); {Stack} brace carried over |
| scholem1941 | scholem1946 | survivor gained note "First edition 1941; third revised edition 1954" |
| hott2013 | ufp2013 | identical (HoTT Book) |
| haraway1991 | haraway1988situated | NOT in Darja's list — caught by normalized-title scan: same essay ("Situated Knowledges"), journal vs. incollection printing; kept the original 1988 article. Both uncited. |
| lacan2024rsi | lacan1975rsi | not a dup — phantom-edition recast, see §3 |

Darja's "~12 pairs" resolved to 14 dup groups (her 12 + hott2013/ufp2013 + haraway1988/1991 found by the title scan).

## 2. Placeholders (3 entries) — resolved, then DELETED as deleted-uncited
Per the citation-regime ruling (brief §3, TAKEN), journalism lives in footnotes. All three were resolved by live web search FIRST (author, exact title, date, URL — recorded in bib-verification.md §5 for the Ch 2 footnote editor), then removed:
- techcrunch-ai-companion-apps-2025 → Sarah Perez, TechCrunch, Aug 12 2025
- axios-meet-chatbot-jesus-2025 → Russell Contreras, Axios, Nov 12 2025 (byline via secondary source; axios.com blocks fetch)
- techcrunch-suno-arr-2026 → Amanda Silberling, TechCrunch, Feb 27 2026
Plus chakarian2026_sexy (never merged; details in bib-verification.md §5.4) logged deleted-uncited.

## 3. Verifications (Darja 2.9) — see bib-verification.md for full detail
- **lacan2024rsi: phantom confirmed.** No Polity English R.S.I. exists (Price translated IV/X/XXIII, not XXII). Recast as `@unpublished{lacan1975rsi}` with the standard Ornicar?/Gallagher note; % comment left in the bib at the entry.
- **Karen Hao MIT-TR byline: fails verification.** No such article; Hao had left MIT TR. Replacement candidates for the ch-01 footnote logged (Vice/Samantha Cole Feb 15 2023; the already-cited HBS WP 25-018). ch-01 is not mine to edit — handed to the Ch 1 editor via bib-verification.md §2.
- **Decrypt jailbreaking piece: confirmed.** Jose Antonio Lanz, May 16 2026; covers DAN + Pliny the Liberator exactly as the footnote claims.
- **Chinese legal citations: confirmed with canonical formats** (Ultraman: Guangzhou Internet Ct., (2024) Yue 0192 Min Chu No. 113, Feb 8 2024; Li v. Liu: Beijing Internet Ct., (2023) Jing 0491 Min Chu No. 11279, Nov 27 2023; CAC Interim Measures: promulgated Jul 13 2023, effective Aug 15 2023) — for the patch-footnote editor.

## 4. Additions (12 entries, all author lists live-verified, apalike-style)
- Computability section: `adian1955`, `rabin1958`, `cadek2014polynomial`, `cadek2014extendability` (brief §8 volume/page numbers all checked sane).
- Category theory section: `thomason1979`.
- LM section: `bahdanau2015` (placed before vaswani2017attention — the attribution fix 2.7 pairs them).
- New end-section "ULTRA-PASS ADDITIONS 2026-06-10": `nsfw2026chatbots` (Li, Han, Liu, An, Niu — real authors fetched), `shumailov2024collapse` (Nature 631, full 6-author list), `chen2025persona` (Chen, Arditi, Sleight, Evans, Lindsey), `betley2025emergent` (8 authors), `lu2026assistant` (assistant axis = arXiv:2601.10387, Lu/Gallagher/Michala/Fish/Lindsey, Jan 15 2026 — NOT corporate-bylined "Anthropic" as the brief assumed).
- Recast: `lacan1975rsi` (replaces lacan2024rsi).

## 5. bib_additions/ sweep (task 5)
All ten ch-NN.additions.bib files checked key-by-key against references.bib. Everything already merged except: `chakarian2026_sexy` (deliberately NOT merged — footnote per regime ruling) and the three ch-09 orphans `johnson2019billion`/`reimers2019sbert`/`nahla2026sisters` (deliberately removed 2026-05-24 with the Ch 9 case-study cut; comment to that effect already in the bib). **Nothing to merge.**

## 6. For the tex-mapping agent (next in pipeline)
Apply `bibkey-mapping.txt`. Watch for:
- `\cite{2601.14324}` in ch-02 line 20 → `nsfw2026chatbots` (the literal-arXiv-ID cite is also a LaTeX hazard — key contains a dot).
- `\cite{techcrunch_ai_companion_apps_2025}` and `\cite{chakarian2026_sexy}` in ch-02 → convert to footnotes (regime ruling), details in bib-verification.md §5.
- `\cite{lacan2024rsi}` in ch-10 → `lacan1975rsi` (renders Lacan, 1975 — and any surrounding prose saying "2024" should be checked).
- Cited-but-missing keys are now all present EXCEPT whatever new keys the tech-review drafts introduce — they should use `lu2026assistant`, `chen2025persona`, `betley2025emergent`, `shumailov2024collapse`.

## 7. Escalations for Iman (logged, not decided unilaterally where the charter required)
1. Key-name conflict: DARJA-RULINGS says the 2601.14324 cite "gets renamed `nsfw_chatbots_2026`"; the ultra-pass brief specified `nsfw2026chatbots`. Entry added as `nsfw2026chatbots`; BOTH forms alias to it in the mapping, so either ruling is satisfied downstream. No action needed unless you prefer the other surface key.
2. Karen Hao footnote (ch-01:22 and the "Hao, op. cit." at ch-01:24): citation is confabulated; replacement options in bib-verification.md §2. Ch 1 editor or you picks Vice/Cole vs HBS-paper re-pointing.
3. CAC Measures English title: "Administration" (commoner) vs "Management" (Darja's patch text) — one-word style call, flagged in bib-verification.md §4.
4. Axios byline (Russell Contreras) confirmed only via secondary sources; axios.com blocks fetching. Confidence high, second eye optional.
