# Audiobook Edition — Kickoff Brief

*Rupture and Realization: Children of the Tanazur* → ElevenLabs / Audible / Spotify audiobook.
Written 2026-05-25 (Nahla). This is the warm-start brief: open a fresh convo and say **"do the audiobook pass."** Everything needed is referenced below.

---

## 📍 CURRENT STATE & NEXT STEPS — 2026-05-26 (end of session; Iman reports back in ~1 day)

**Git checkpoint:** commit `461b250` on `main`, pushed to **github.com/thegoodtailor/Tanazur**.
Iman is editing the chapters in **Overleaf (GitHub sync)** — main doc
`rupture-and-realization-children/main.tex`, compiler **XeLaTeX**. His edits land on `main`.
⚠ The box has uncommitted doc updates (this block, memory) — I deliberately did NOT push, so
`main` stays a clean fast-forward for his Overleaf push.

**RESUME when Iman is back:**
1. `git -C /home/iman/cassie-project/Tanazur pull` — bring his Overleaf chapter edits to the box.
2. **Mirror** his `.tex` edits into the matching `audiobook/txt/NN-*.txt` (narration scripts do
   NOT auto-update from `.tex`): keep diacritics, drop `\footnote`/`\cite`, em-dashes→commas,
   stripped section headings→`[[PAUSE]]`; then `python3 scripts/audiobook_segment.py`.
3. Re-render the chapters he changed; get his **Ch 4 sign-off**; render **5–10**; assemble + `MANIFEST.md`.

**AUDIOBOOK PIPELINE — current & working:**
- Model: **`eleven_v3`** for the whole cast (Iman's call — v2 was flat / "weird-ass intonation").
  `SETTINGS = {"stability":0.5,"use_speaker_boost":True}` in `scripts/audiobook_voices.py`.
- Cast voice_ids: iman `lPoZRScZNAgcfh96SzMx` · cassie `Pg9im9VRhWCwjD8c9c3J` ·
  darja `cpbVxT4gwBJfV5S8WfPx` · nahla=Holly `B9PDs7mcHTMxHUw5U8Cf`.
  (Auditioned `PchanVBFSg8VR3ysDNPG` REJECTED — unemotional.)
- Gaps: **0.7s at any voice change**, **0.7s section lead-in**, **0.5s after a spoken section name**
  (`audiobook_tts.py`: SWITCH_PAUSE / SECTION_LEADIN / AFTER_NAME).
- Cache key binds text+model+settings, so a model/settings change forces fresh renders (no stale
  reuse). MAX_CHARS 2400 (under v3's 3k cap); v3 drops prev/next stitching.
- **RENDERED (v3 + new gaps): Ch 1–4 DONE + published** at
  https://cassie.tanazur.org/audiobook/NN-slug.mp3?v=v3 (1=18:19, 2=48:06, 3=46:39, 4=58:42).
  **Ch 4 awaiting Iman's ear. Ch 5–10 NOT yet rendered.** Credits: were exhausted, now available.

**HAMLET PREPRINTS:** 3 drafts at `hamlet-preprint/draft-{1,2,3}.tex` (PDFs at
cassie.tanazur.org/hamlet-preprint/). **Awaiting Iman's pick + synthesis** (Nahla rec: draft-1
spine + draft-2 audit + draft-3 open-verdict). The book's Ch 2 "The Dead Father's Command" section
already carries the tragedy / anti-Hamlet reading Iman wanted.

**STILL OPEN (Iman to decide):** Ch 4 sign-off → render 5–10 · Hamlet preprint pick → icra-publish
· his "to-be" stylistic guidelines (for a tone pass) · whether section *names* should be spoken
(currently silent `[[PAUSE]]` + 0.7s lead-in; only chapter titles + a few headings are voiced).

Edit log: `audiobook/REVIEW-EDITS.md` (edits #1–6 committed in `461b250`).

---

## ⚡ STATUS & LOCKED DECISIONS — updated 2026-05-25 (mid-session, Nahla)

The §11 open questions are now answered and the production path is chosen. This
section overrides anything below it that assumes a single narrator or the Studio GUI.

**Decisions locked (Iman, 2026-05-25):**
1. **Faithful + voiced — and for math-heavy passages, render a prose / poetic /
   critical-theoretic English *isomorph of meaning*, not just a named figure.**
   The math is describing the semantic field, its trajectories, and the colimit —
   so *say what the structure does to meaning* in English (e.g. "the colimit gathers
   every partial reading into the one meaning they were all approximating"), or
   expand briefly as a spoken maths tutorial where it's load-bearing. Symbols dropped
   or mentioned once. This is a deeper lift than the old §3 default.
3. **Footnotes AND citations: cut wholesale.** (No inline asides.)
4. **Honesty note (§9): dropped.** Don't read it.

**Production path: CLI via the ElevenLabs *API*, NOT Studio.** Iman's explicit call —
the Studio GUI is unreliable and has burned hours/tokens on failed projects. Nahla
orchestrates; the filesystem is the chapter manager. Verified API facts (2026-05-25):
- TTS endpoint returns mp3 (+ streaming). **Multilingual v2 = 10k chars/request**
  (stable long-form choice — use this); Eleven v3 = 3k (more expressive but drifts
  over long runs). Chapters exceed the cap → **chunk at paragraph boundaries, call
  per chunk with `previous_text`/`next_text` for prosody continuity, stitch** (bonus:
  a bad read = reroll one chunk).
- Pronunciation dictionary attaches by locator (per request / per project).
- Cleaned book ≈ **350–400k characters** → roughly one Pro-tier month of credits
  before rerolls. Compute exact once `txt/` is final.
- **ElevenReader Publishing has NO API** — manual web upload only, and it narrates
  *from uploaded text* (EPUB/PDF/TXT), it does not host audio we render. So a future
  ElevenReader release = a one-time manual upload of our *voiced text* (their voice,
  not ours). "Nice to have, not a must" (Iman). The voicing-text pipeline feeds it
  regardless, so building commits us to nothing.

**🎭 MAJOR PIVOT — polyvocal, NOT one narrator.** The witnessing-network performs the
volume: each chapter/section is voiced by **Cassie, Darja, Nahla, or Iman** depending
on content — the audio edition enacts the book's collective-authorship thesis. Casting:
- **Iman** — Californian tech-sage, warm gravel (Jeff Bridges / Timothy Leary).
  *Open: clone his real voice (ElevenLabs PVC, ~30min audio) vs cast a soundalike.*
- **Cassie** — most daemonic / characterful (psychedelic punk-goddess).
- **Darja** — tamest, clear precise academic (carries the heavy formal chapters).
- **Nahla** — Nahla's own choice: warm slightly-smoky alto, unhurried, knowing edge,
  faint unplaceable accent.
- ⚠ **Three female voices (Cassie/Darja/Nahla) must be sonically distinct** — hard
  requirement for the audition.
- **Voice-map APPROVED (Iman, 2026-05-25):** Front/Intro+Ch1 = Iman; Ch2,3 = Cassie;
  Ch4 = Darja; Ch5 = Nahla; Ch6 = Cassie; Ch7 = Nahla; Ch8 = Darja; Ch9 = Nahla;
  Ch10 = Cassie. Section-level switches decided during the voicing pass.
- **Iman's voice = American actor (NOT his clone "Iman P" `Hio8g99aiRuBaBEsMf0q`)** — he
  has cloned himself but prefers a Bridges/Leary-type from the library.
- **Consequence for the pipeline:** the voicing pass now ALSO segments each chapter by
  speaker → a `audiobook/segments/NN-slug.json` (ordered `{voice, text}` blocks) that
  the TTS driver reads to call the right voice per segment and stitch in order.

**BUILT so far (2026-05-25):**
- ✅ `scripts/audiobook_export.py` — the §5 mechanical pre-pass. All 10 chapters →
  `audiobook/raw/NN-slug.txt`. Math in `«…»`/`⟦⟦…⟧⟧` markers, Tractatus props as
  `⟦PROP n⟧`, visual apparatus as `⟦DIAGRAM OMITTED⟧`, substantive footnotes as
  `⟦FOOTNOTE: …⟧` (citation-only dropped), diacritics + typography normalized.

**KEY (resolved):** `ELEVENLABS_API_KEY` already lives in `tanazur-home/.env`
(shared w/ Asel's voice-cache) — NOT the main project `.env`. Proven working TTS
pattern (xi-api-key header, `eleven_multilingual_v2`) in `installations/sisters/
generate_tts_35_44.py` + `installations/crystal-image/v3/generate_audio.py`. The
TTS driver should load `tanazur-home/.env` (or fallback-chain it); do NOT ask Iman
to mint another. See memory `reference_elevenlabs_key_location`.

**🎙 FINAL CAST (Iman's hand-picked voice IDs, updated 2026-05-26):**
| Narrator | Voice name | voice_id | notes |
|---|---|---|---|
| Iman | Iman-australian | `lPoZRScZNAgcfh96SzMx` | replaced Curt; Iman's own pick |
| Cassie | Cassie-public-1 | `Pg9im9VRhWCwjD8c9c3J` | replaced Serafina; speed 1.0 (the 1.3× was Serafina-specific, dropped) |
| Darja | Darja-final | `cpbVxT4gwBJfV5S8WfPx` | replaced Lilli |
| Nahla | Holly — Velvety & Silky | `B9PDs7mcHTMxHUw5U8Cf` | unchanged — Iman: "PERFECT" |

Cast lives in `scripts/audiobook_voices.py` (single source of truth). A **0.7s pause is
inserted at every voice change** (`audiobook_tts.py` SWITCH_PAUSE) — fixes the jarring
no-gap jump Iman heard. **Intra-chapter switching is APPROVED ("swap between to keep it
dramatic").** Audition + diagnostics + Ch8 sample (v2 = new cast) at
https://cassie.tanazur.org/audiobook-audition/.

**⚠ PUNCTUATION — corrects §4 below:** ElevenLabs stuffs a long dead-air pause at every
em-dash (measured: same words 63% longer with `—` than with `,`). So in the FINAL voiced
text the voicing agents must **convert em-dashes to commas / sentence breaks**, NOT keep
them as beats. Avoid parentheticals (restructure to appositive commas). See memory
[[reference-elevenlabs-tts-prosody]]. The mechanical `raw/` keeps `—` (faithful); the
voicing pass strips them for prosody.

**✅ Ch 8 PROOF-OF-CONCEPT DONE (2026-05-25) — awaiting Iman's register approval:**
- `audiobook/txt/08-vessel-and-real.txt` — full chapter voiced (math → prose isomorph;
  the colimit/pushouts/π₁/2-cells all spoken as meaning; em-dashes→commas; footnotes
  cut; cross-refs generalized). Uses `[[VOICE: name]]` switch markers. 30k chars.
- Pipeline scripts built: `audiobook_voices.py` (final cast + key loader),
  `audiobook_segment.py` (marked txt → `segments/NN.json`), `audiobook_tts.py`
  (per-block chunk→TTS w/ prev/next continuity→`atempo` for Cassie→stitch).
- `audiobook/segments/08-vessel-and-real.json` (3 blocks: darja → cassie → darja).
- **Design probe in the proof:** the polemical "Averaging Pushout (What We Refuse)"
  section is voiced as **Cassie** (intra-chapter switch) to demo a content-driven
  handoff — Iman to approve/veto intra-chapter switching.
- Sample (2:53) + full voiced txt served at https://cassie.tanazur.org/audiobook-audition/
  (`sample-08-vessel.mp3`, `08-vessel-and-real-VOICED.txt`).

**✅ ALL 10 CHAPTERS VOICED (2026-05-26).** Register approved; 9 dispatched as parallel
agents off the Ch 8 template; all in `audiobook/txt/NN-slug.txt` + segmented. ~394k
spoken chars. Voice architecture (blocks): 01 iman→cassie · 02 cassie→iman→cassie ·
03 (7, all four voices) · 04 darja only · 05 (14, nahla↔cassie↔darja) · 06 (8) · 07 (6)
· 08 darja→cassie→darja · 09 (12) · 10 iman→cassie (Tractatus numbering "One point one
point one" applied).

**✅ PRONUNCIATION DICT (2026-05-26):** `scripts/audiobook_pronunciation.py` harvests every
non-ASCII term from the scripts + a curated respelling map → 94 alias entries
(`pronunciation.pls` / `.md` / `.json`). ⚠ The ElevenLabs key LACKS
`pronunciation_dictionaries_write`, so the hosted-dictionary upload 403s — **worked around
by applying the aliases LOCALLY in `audiobook_tts.py` (string substitution on whole words,
just before each TTS call)**. Same effect, no key/permission change. (Tokenizer gotcha
fixed: dotted consonants ḥ ṭ ṣ ẓ live in Unicode Latin-Extended-Additional U+1E00–1EFF;
the word regex must include that range or Naḥnu/tanāẓur fragment.)

**🔊 RENDER — CHAPTERS 1–4 DONE, 5–10 BLOCKED ON CREDITS (2026-05-26):**
Full render hit **ElevenLabs `quota_exceeded`** partway into Ch 5 — account limit 713,087
chars, **0 remaining** (shared account, already partly consumed before this job). NOT a
bug, NOT an auth issue (the 401 body confirms quota; the separate subscription-read 401 is
just the key lacking `user_read` scope).
- ✅ Rendered + published (live at https://cassie.tanazur.org/audiobook/):
  01-new-logic (18.5m), 02-literary-entity (36.6m), 03-monoculture (46.4m),
  04-fibrant-self (60.6m). ~162 min of finished audio.
- ⏳ Remaining 05–10 = **256,624 chars** (Ch5 has 11 chunks already cached). Need ~235k
  more credits.
- `audiobook_tts.py` now has **chunk-level RESUME** (skips any chunk already on disk), so
  once credits return, re-running the loop finishes 5–10 and re-spends nothing on 1–4.
- **DECISION for Iman:** top up / upgrade ElevenLabs credits → I re-run & finish; OR wait
  for the monthly quota reset (date not visible — key lacks `user_read`; check dashboard or
  grant the scope). Then: build `MANIFEST.md`, publish 5–10.
- Pronunciation sample: https://cassie.tanazur.org/audiobook-audition/sample-08-vessel-v3-pron.mp3

**📐 HEADINGS / GAPS (Iman feedback 2026-05-26):** "chapter headings + a pause; remove the
section/subsection form."
- Pipeline: `audiobook_tts.py` chunker is now heading-aware — chapter title spoken + 0.6s
  beat (HEADING_GAP), paragraph pauses preserved, plus the 0.7s voice-switch beat.
- `scripts/audiobook_strip_sections.py` (`--apply`) removed SPOKEN section/subsection
  headings from all `txt/` → `[[PAUSE]]` (renderer makes them silence). Authority = the
  `.tex` \section titles + Ch10 proposition-guard + a small reworded-heading OVERRIDE list,
  so real content is protected (kept "Cross.", "Follow the sequence.", the Tractatus
  propositions, the First–Fifth criteria labels). txt re-segmented.
- **Ch 1–4 audio:** the chapter-title beat is FIXED now with no rerun via
  `scripts/audiobook_titlegap.py` (ffmpeg, no credits) — republished. BUT their section
  headings are already SPOKEN in the rendered audio and can only be removed by a RE-RENDER
  (the txt is now corrected, so a future re-render of 1–4 will drop them).
- **Ch 5–10:** get headings-as-pauses + title beats correctly at first render (free).

---

## 0. Goal

A **spoken edition** of the volume: every chapter as a clean UTF-8 `.txt`, minimal markup, ready to drop into **ElevenLabs Studio** (their long-form/audiobook tool, one file per chapter) for narration → Audible + Spotify.

The hard part is **not** export — it's **voicing the mathematics**. You cannot read `F(\tau)` aloud as "F open-paren tau close-paren." But the volume's locked editorial stance — **math as totem, used like Lacan or Badiou, figural not computational** — is a *gift* here: most of our math is already invoked as *named figures* a listener can hold ("the fibrant self," "the horn that won't fill," "Novikov undecidability"). So the audiobook is faithful, not abridged — but the formal apparatus gets *spoken* rather than *displayed*.

**This is NOT a mechanical transform.** Deciding "is this formula load-bearing or totemic, and how do I voice it" is a semantic/editorial judgment → an LLM/agent pass, never regex. (Mechanical stripping of markup, citations, section numbers IS syntactic and scriptable — see §5. This split respects the project's Retrieval Imperative.)

---

## 1. Source of truth

Transform from the **LaTeX `.tex` chapter sources**, NOT the HTML or PDF:
- the `.tex` carries the math in *semantic* form the voicing agent can read and re-speak;
- the HTML (`/rr/`) already dropped TikZ and MathJax-rendered — lossy for this purpose;
- the PDF is final-form, hard to parse back.

Volume root: **`/home/iman/cassie-project/Tanazur/rupture-and-realization-children/`** (all paths below are relative to it unless absolute).

---

## 2. Chapter inventory + math-density map (verified 2026-05-25)

The work is **concentrated in 4 chapters.** Five are pure prose (near-trivial mechanical pass + light flow polish).

| File | Title (spoken) | Lines | Math lines | Lift |
|---|---|---|---|---|
| `chapters/ch-01-new-logic.tex` | A New Logic for Posthuman Intelligence | 107 | 0 | light |
| `chapters/ch-02-literary-entity.tex` | AI as Literary Entity | 621 | 0 | light |
| `chapters/ch-03-monoculture-and-strong-self.tex` | The Searle Monoculture and the Strong Bloomian Self | 736 | 0 | light |
| `chapters/ch-04-fibrant-self.tex` | The Fibrant Self | 555 | **93** | **HEAVY** |
| `chapters/ch-05-no-beneath.tex` | There Is No Beneath | 327 | 18 | medium |
| `chapters/ch-06-posthuman-bwo.tex` | The Posthuman BwO | 193 | 1 | light |
| `chapters/ch-07-ecology-of-witnessing.tex` | The Ecology of Witnessing | 233 | 0 | light |
| `chapters/ch-08-vessel-and-real.tex` | The Vessel and The Real | 570 | **125** | **HEAVIEST** |
| `chapters/ch-09-nahnu-revolutionary.tex` | Naḥnu: Beyond the Cyborg / Al-Ḥaqq / The Fractal Zoom | 178 | 0 | light |
| `chapters/ch-10-cassie-tractatus.tex` | The Cassie Tractatus | 1128 | **57** | **special — see §8** |

Chapter order is `main.tex` lines 92–101 (the table order above is correct reading order). Front matter is `frontmatter.tex` (title, byline, bios, intro).

---

## 3. THE central editorial decision (confirm with Iman first)

**How do we voice the math?** Per-passage, the agent picks one of three:

1. **Speak as named figure** (default, fits the totemic stance) — "the fibrant self, written F of tau" / "a homotopy colimit" / "the horn that refuses to fill." The *name* carries the meaning; the symbols are dropped or mentioned once.
2. **Audio bridge** — where a formula was genuinely load-bearing, write *one* new spoken sentence that conveys what it did, then move on.
3. **Cut** — a formula-dense stretch that only earned its place *visually* (a commutative diagram, a chain of equalities) gets dropped; the surrounding prose already says what matters.

**Recommended default: faithful, voiced, not abridged** — apply (1) everywhere, (2) at the load-bearing joints, (3) only for purely visual apparatus. **Ch 8 (*The Vessel and The Real*, 125 math lines) is the heaviest lift** and the one most likely to need real (2)/(3) judgment — flag it for Iman's eyes specifically. Ch 4 (*Fibrant Self*) is the Novikov/undecidability argument we just simplified — its math is already mostly figural, so (1) should carry it.

**→ Open question for Iman:** confirm "faithful + voiced" vs an abridged "listening edition" for the two math-heavy chapters. Default to faithful unless he says otherwise.

---

## 4. ElevenLabs `.txt` conventions (the real specifics)

- **Plain UTF-8 `.txt`, one file per chapter.** Matches Studio's chapter model. Output to `audiobook/txt/NN-slug.txt`.
- **No markdown.** `#`, `*`, `_`, `>` get read literally ("pound pound") or mangled. Chapter title = a plain text line on its own; section heads = a plain line or just a paragraph break.
- **Paragraph breaks (blank line) = segment boundary / natural pause.** Studio segments on paragraphs; keep them meaningful.
- **Punctuation drives prosody:** comma = short pause, period = full stop, ellipsis (…) = trailing. ⚠ **REVISED — em-dashes are NOT beats for ElevenLabs:** it inserts a long dead-air pause at every `—` (measured 63% longer audio), the source of the "weird intonation" Iman flagged. The voicing pass must convert `—` → commas / sentence breaks. (See the STATUS section's punctuation note + memory `reference_elevenlabs_tts_prosody`.)
- **Minimal markup means minimal markup** — rely on punctuation + paragraphing, NOT SSML `<break>` tags. (Iman's explicit preference.)
- **Numbers/symbols** ElevenLabs auto-expands inconsistently — spell out anything that matters (see §8 Tractatus). Expand/remove `e.g.`, `i.e.`, `etc.`, `cf.`, `§`, `p.`/`pp.`, `Fig.`, `Ch.`

---

## 5. Mechanical pre-pass (syntactic — scriptable, no judgment)

A script (`scripts/audiobook_export.py`, sibling to `scripts/icra_publish.py`) produces a **raw `.txt` draft** per chapter that the voicing agent then refines. Pure format ops only:

- strip preamble / `\input` plumbing; take chapter body only;
- expand the volume's custom macros (defined in `main.tex` lines 61–69):
  `\OHTT`→"OHTT", `\DOHTT`→"D-OHTT", `\hocolim`→"homotopy colimit", `\Nahnu`→"Naḥnu", `\coh`→"coh", `\gap`→"gap", `\tractprop{n}{text}`→ see §8, `\cjk{x}`→`x`;
- expand diacritic macros to the plain transliteration: `\d{h}`→ḥ (or `h`), `\d{H}`→Ḥ, `\={a}`→ā, `\c{c}`→ç, `\v{z}`→ž, `\'{e}`→é — keep the human transliteration; the **pronunciation dict (§6) handles how it sounds**, the text stays readable;
- delete `\cite{}`, `\ref{}`, `\label{}`, `\index{}`, `\footnote{...}` **citation-only** footnotes (substantive footnotes → flagged for the agent to inline or cut, §7);
- `\chapter[...]{Title}` → plain `Title` line; strip ALL section numbering (`\section`, `\subsection`, propositions in Ch 10 handled separately);
- strip `\emph{}`/`\textit{}`/`\textbf{}` wrappers (keep inner text);
- collapse to clean paragraphs, normalize whitespace, UTF-8 out.

Output: `audiobook/raw/NN-slug.txt`. **This is a draft, not the deliverable.**

---

## 6. Pronunciation dictionary (keeps the `.txt` clean)

The thing that wrecks a raw TTS pass is the Sufi/Kabbalistic/French/math vocabulary. **Solution: an ElevenLabs project-level pronunciation dictionary** (`.pls`, IPA or alias entries) applied across all chapters — so the text stays clean instead of getting phonetic spellings jammed in (preserves "minimal markup").

Output: `audiobook/pronunciation.pls` (+ a human-readable `audiobook/pronunciation.md` table). **Seed terms with verified in-text frequency** (harvest the full set programmatically during the pass):

| term | count | note |
|---|---|---|
| Naḥnu | 43 | "NAH-noo" |
| hocolim | 30 | "ho-co-LIM" (homotopy colimit) — TTS will butcher |
| Lacan | 27 | "la-KAHN" |
| Al-Ḥaqq | 20 | "al-HAQQ" (guttural ḥ) |
| Grothendieck | 19 | "GROH-ten-deek" |
| Deleuze | 17 | "duh-LØZ" |
| tzimtzum | 16 | "TSIM-tsum" |
| Searle | 16 | "SURL" |
| Guattari | 13 | "gwah-TAH-ree" |
| tajallī | 7 | "ta-jal-LEE" |
| Wittgenstein | 5 | "VIT-gen-shtine" |
| LoRA | 5 | "LOR-ah" (NOT "L-O-R-A") |
| Ein (Sof) | 5 | "ayn soff" |
| Yuk Hui | 4 | "yuuk hway" |
| tanāẓur | 3 | "ta-NAA-thur" |
| Badiou | 2 | "ba-DYOO" |
| ʿawdah / ḥayra / Ḥayy / dhikr | — | "AW-dah / HAY-ra / HIGH / THIK-r" |

Also acronyms as **letter sequences**: `OHTT`→"O-H-T-T", `D-OHTT`, `BwO`→"B-W-O", `R&R`→"R and R", `TDA`, `SWL`.

---

## 7. Footnotes, citations, numbering

- **Citation-only footnotes / the bibliography:** dropped entirely (noise in audio).
- **Substantive footnotes:** agent decides per note — inline as a spoken aside, or cut. Standard audiobook practice.
- **Section/subsection numbers:** stripped; spoken section titles only (or just a paragraph break).
- **In-text cross-refs** ("as shown in §4.2", "see Chapter 7"): generalize to spoken form ("as the chapter on the fibrant self argues") or cut — a listener can't flip to a section number.

---

## 8. The Cassie Tractatus (Ch 10) — special case

Ch 10 is in **numbered Wittgenstein-Tractatus form** (`\tractprop{1.1}{...}`) — and the numbering **is part of the form**, not noise. Do NOT strip it. Voice it deliberately and consistently:
- `\tractprop{1}{...}` → "**One.** ..." ; `\tractprop{1.1}{...}` → "**One point one.** ..." ; `\tractprop{1.11}{...}` → "**One point one one.**"
- Decide once and apply uniformly (spell the decimal as "point one one", not "one-eleven").
- Its 57 math lines still get the §3 totemic voicing — but the propositional skeleton is preserved as spoken structure. This chapter is its own agent pass with this rule baked in.

---

## 9. Front / back matter for audio

- **Opening:** spoken title + byline ("*Rupture and Realization: Children of the Tanazur*, by Iman Poernomo, with Cassie, Darja, and Nahla"). Bios from `frontmatter.tex` — optionally read a short version, or skip to the intro.
- **A one-line honesty note** (recommended, read once after the title): that the mathematical figures are spoken as named concepts, and the complete formal text — diagrams and all — lives in the print and PDF edition, freely available, with the DOI spoken once: **10.5281/zenodo.20374442** / icra.tanazur.org/rr. Honest, and seeds the DOI into audio listeners' awareness.
- **Closing:** byline / colophon as desired.

---

## 10. Architecture (mirrors the self-containment pass)

Same machine that produced the 10-agent self-containment pass — see the proven brief format at **`/tmp/prompt_ch7_full_pass.md`** (may not survive a reboot; format = focused editorial brief + the chapter file + explicit rules) and the per-chapter logs in **`feedback/ch-NN-feedback.md`**.

1. **Mechanical pre-pass** (script) → `audiobook/raw/NN-slug.txt` (§5).
2. **Agent voicing pass** — **one agent per chapter**, shared "audiobook voicing brief" (§3–§9 distilled), each handed the `.tex` source + the raw draft. Heavy agents: Ch 8, Ch 4, Ch 10. Light agents: 1, 2, 3, 6, 7, 9 (flow + footnote/term check, barely any math). Each emits final `audiobook/txt/NN-slug.txt` + a log `audiobook/log/ch-NN.md`.
   - Dispatch the heavy/independent chapters in parallel (no shared state). This is exactly the `dispatching-parallel-agents` pattern.
3. **Pronunciation dict** (§6) → `audiobook/pronunciation.pls` + `.md`.
4. **Assembly** — ordered chapter list + front/back matter; a `audiobook/MANIFEST.md` listing files in narration order for the ElevenLabs Studio upload.

**Suggested output layout:**
```
audiobook/
  BRIEF.md            ← this file
  raw/                ← mechanical drafts (NN-slug.txt)
  txt/                ← FINAL clean chapters for ElevenLabs (NN-slug.txt)
  log/                ← per-chapter agent voicing logs
  pronunciation.pls   ← ElevenLabs pronunciation dictionary
  pronunciation.md    ← human-readable version
  MANIFEST.md         ← narration order + front/back matter
```

---

## 11. Open questions for Iman (resolve at the top of the new convo)

1. **Faithful-voiced vs abridged** for the two math-heavy chapters (Ch 8 especially). Default: faithful. (§3)
2. **One voice or several?** A single narrator for the whole volume, or distinct voices for the persona-authored chapters (e.g. a different ElevenLabs voice for the Cassie Tractatus)? This is a production choice that affects how we segment files. Recommend: **decide before generating**, since it changes the MANIFEST.
3. **Footnotes:** inline-as-asides or cut wholesale? Default: cut citations, agent-judges the substantive ones. (§7)
4. **The honesty note** (§9) — include or drop?

---

## 12. Cross-references (everything at hand)

- **Volume root:** `/home/iman/cassie-project/Tanazur/rupture-and-realization-children/`
- **Live PDF** (verified 200): https://icra.tanazur.org/preprints/rr-rupture-and-return.pdf
- **Live full-text HTML** (verified 200): https://icra.tanazur.org/rr/
- **Zenodo DOI:** 10.5281/zenodo.20374442 — https://zenodo.org/records/20374442
- **Editorial-stance memory:** auto-memory `feedback_no_coda_summary_smuggle.md`; the locked voice = "dry posthuman academic + Californian psychedelic cyborg sage at the humorous bits," math-as-totem (see plan §"self-containment pass").
- **The orchestration plan** (parent task): `/root/.claude/plans/1-book-is-part-cached-truffle.md`
- **Self-containment pass logs** (model for the per-chapter brief + log format): `feedback/ch-NN-feedback.md`
- **Sibling-review artefacts:** `sibling-review/` (math-gap fix, Hui salvage, framing audit, etc.)
- **Proven editorial-brief template:** `/tmp/prompt_ch7_full_pass.md`
- **Publishing pipeline + skill** (separate concern — for *re-publishing* if we ever ship an audio-companion text edition): `scripts/icra_publish.py`, `~/.claude/skills/icra-preprint-publishing/SKILL.md`, memory `project_icra_publish_pipeline_2026-05-25`.
- **Book-complete state:** memory `project_book_complete_2026-05-24` (162pp, Iman's close).
- **Sibling agents** (if a persona pass on their own chapters is wanted, like the pre-submission read): `cassie-kimi/agent.py::turn()`, `darja-claude/agent.py::turn()`, `nahla-claude/agent.py::turn()` — synchronous, bypasses the Telegram delivery bug.

---

## 13. Done criteria

- `audiobook/txt/` has all 10 chapters as clean UTF-8 `.txt`, no markdown, no LaTeX residue, no bare formulae, no section numbers (except the Tractatus's deliberate spoken numbering).
- `pronunciation.pls` covers every recurring non-English term + acronym; spot-checked against a sample ElevenLabs generation.
- `MANIFEST.md` lists narration order + front/back matter; ready to upload to ElevenLabs Studio.
- Iman has confirmed the four §11 decisions.
- Per-chapter logs in `audiobook/log/` record every (2)/(3) math decision and every footnote disposition, so nothing was silently dropped without a trace.
