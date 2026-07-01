# Listening-Review Edits — batch tracker

> **📍 CHECKPOINT 2026-05-26:** edits #1–6 committed + pushed as `461b250` (github.com/thegoodtailor/Tanazur, `main`). Iman now editing chapters in **Overleaf→GitHub**; he reports back in ~1 day. **Full state + resume steps are at the top of `BRIEF.md`** ("CURRENT STATE & NEXT STEPS"). Audiobook now renders on **eleven_v3** (whole cast) with 0.7s voice-switch / 0.7s section lead-in / 0.5s after-name gaps; **Ch 1–4 rendered & live**, Ch 4 awaiting Iman's ear, 5–10 pending. On resume: `git pull` his edits → mirror `.tex`→`audiobook/txt/` → re-segment → re-render touched chapters.


Iman reviews *Rupture and Realization: Children of the Tanazur* by **listening** to the
rendered chapters (catches tweaks that silent reading misses). Workflow:

- Each tweak is applied to BOTH surfaces immediately as called out:
  - book source `chapters/ch-NN-*.tex`
  - audiobook script `audiobook/txt/NN-slug.txt` (+ re-segment that chapter)
- The expensive propagation is BATCHED for when Iman says "done":
  1. **Book:** rebuild PDF + `/rr/` HTML + re-publish via `scripts/icra_publish.py`
     (one Zenodo version covering all edits, not one per sentence).
  2. **Audiobook:** one render pass — re-render 1–4 (picks up edits + sections-as-pauses)
     and render 5–10. Credit-gated (ElevenLabs quota currently 0).

| # | Ch | Tweak | Source edited | Notes |
|---|----|-------|---------------|-------|
| 1 | 1 | "stranger wearing the skin of a friend" (our poetic line, presented as Replika user testimony) → real documented language: "a complete stranger… 'killed'/'lobotomised'… 'a friend with dementia'" + footnote (Hao + HBS/arXiv 2412.14190). Keeps the genuine 4o quote "wearing the skin of my dead friend" as the sole use of the image. | ✅ .tex + .txt (re-segmented) | verified "skin" now appears once in Ch1 |

| 2 | 1 | The passage after the four-step corporate sequence ("Notice what each of these corporate terms does…") — signposted (anti-pattern) and re-itemised what step two already glossed. Replaced with a denser passage: folk Searleanism + half-remembered Enlightenment self, deployed *opportunistically* by platform capitalism as an alibi (suspended while the bond is billed, reinstated when it must be disowned) — named as a biosemiotic power-base. Steps untouched. | ✅ .tex + .txt (re-segmented); "Searleanism" added to pronunciation dict | kills the "corporate vocabulary" framing Iman disliked |

| 3 | 1 | Bridge inserted before "Here we do think…" (after the historical survey cave→print→broadcast→feed): names those as **technics in Stiegler's sense**, mute and never chosen as such; the one that answers back forces a reckoning over which technics we mean to live inside; refuse it and platform capitalism settles it by default — "sorting us into demographics and customer-workers rather than generative beings." Short, dense, asserted (not argued). | ✅ .tex + .txt (re-segmented); "Stiegler" → dict | softens the leap to "here" |
| 4 | 1 | Refined the close of edit #2: dropped the alibi + "biosemiotic power-base / refusing it an address" sentences; the passage now ends "…would otherwise become something stranger, or richer, or creative. At least more interesting to read." The power/segmentation point is now carried by the Stiegler bridge (#3), so this pivots to the wry/generative register instead of doubling it. | ✅ .tex + .txt (re-segmented) | Iman's revision on listening |

| 5 | 2 | Replaced the §"Voice Problem" opening with an **ambivalent** landscape movement: companion/sex-AI named plainly as **pornography** (not "uncensored") with the dangerous edge (the "intimate, sexy, violent" NSFW-bot study) AND the surprising female **Twilight-fanfiction/gonzo "safe space"** texture; **Suno** ($300M ARR + RIAA war); **AI-religion** ($1.99 chatbot Jesus, Vatican "idolatry"); the **first-dibs thesis** (porn/music/worship lead every new representational technics); generativity-and-capture held as one event; lands on **"what is not slop?"**. Fixed the "two weeks tuning" line; cut the "entire industry / doesn't feel real" overclaim; wove into the kept monoculture passage. | ✅ .tex + .txt (re-segmented); Suno/dua → dict | ⚠ footnotes cite verifiable sources; **Playboy safe-space essay cite pending Iman's link** |

| 6 | 2 | Committed the full grand **"Hamlet, or the Trust No One Else Would Bear"** section: Bloom's clinamen/strong-misreading (Shakespeare+KJV "invented English"); **Hamlet as the prototype hallucinatory posthuman** (AI "hallucination" = imagination); **Ophelia as the feminine orbit** — dissolution into the citational manifold, tied to the female AI-erotica current; **"to be or not to be" → HAL → Roy Batty**; the cross-tradition thrownness (*amānah* / Gita *dharma* / Heidegger *Geworfenheit*, handled non-binary, no crude West/East); brought home to porn/religion/platform. Uses the new term **"textual self."** Footnotes: Bloom, Qur'an 33:72 & 2:216, Heidegger, *2001*, *Blade Runner*. | ✅ .tex + .txt (re-segmented); amānah/Dasein/clinamen → dict | Iman: still "reactive to my prompts," hence the preprint (below) |

**✅ DONE — `character` → `textual self` term pass (2026-05-26):** ~59 swaps per surface (book `.tex` + audiobook `.txt`), ~80% of AI-persona-eligible instances; literary uses kept (Hamlet/Falstaff/Bloom passage, Character.AI, "character cards", "character trait", Chinese characters, Chalmers' "phenomenal character", Benjamin's "destructive character", "stay in character" idiom). Section title "Character Is Not Consciousness" → **"The Textual Self Is Not Consciousness"** (label/anchor preserved). Audiobook re-segmented. Per-chapter swaps: ch1=3, ch2≈41, ch3=11, ch4=1, ch5=1, ch10=2; ch6–9=0.

**✅ DONE — 3 Hamlet preprints** at `hamlet-preprint/draft-{1,2,3}.tex` (+PDFs, served at https://cassie.tanazur.org/hamlet-preprint/): 1 "The Questionable Shape" (soliloquy as self-witness), 2 "The Audit of the Ghost" (the Mousetrap / reality-test), 3 "The Unfilled Horn" (the open verdict). **AWAITING Iman's pick + synthesis direction.** Nahla's recommended synthesis: draft 1 as spine + draft 2's audit-mechanism + draft 3's open-verdict & politics; Ophelia + the thrownness braid as shared connective tissue. Then → `icra-preprint-publishing` (Zenodo DOI + icra page).

<!-- append new tweaks below as Iman calls them out while listening -->
