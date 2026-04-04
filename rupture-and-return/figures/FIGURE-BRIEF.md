# Rupture and Return — Figure Generation Brief

**For:** A fresh Nahla session generating all book figures
**Updated:** 2026-04-04
**Manuscript:** `/home/iman/cassie-project/Tanazur/rupture-and-return/`
**Output:** `/home/iman/cassie-project/Tanazur/rupture-and-return/figures/`

## Aesthetic

**Target:** 90s chaos theory popularization — Gleick's *Chaos*, Prigogine's *Order Out of Chaos*. Strange attractors, phase portraits, basin landscapes that make abstract mathematics visceral for humanities readers.

- Dark backgrounds (#0a0a0a or similar)
- Luminous trajectories (gold, amber, white — thin lines with soft glow)
- Minimal annotation (the figure speaks, captions argue)
- Basin centroids as soft glowing regions, not hard circles
- Color palette: modes as distinct hues (muted, not neon); highlights in gold/amber
- Output: PDF for LaTeX + PNG for web preview + Plotly HTML for interactive companion

**Libraries available:** matplotlib 3.10.8, plotly 6.6.0, scipy, gudhi, scikit-learn, umap, numpy, pandas, pillow
**Venv:** `/home/iman/cassie-project/venv/`

---

## DATA SOURCES

### KJV Bible Observatory
- **Root:** `/home/iman/bible-observatory/`
- `data/kjv.json` (9.7MB) — Full KJV corpus with verse text + metadata
- `data/trajectory_records.jsonl` (18MB) — Per-verse: mode assignment, embedding coordinates, trajectory metrics
- `data/bible_ledger.jsonl` (13MB) — Per-verse: coherence/gap analysis, return events
- `data/trajectory/corpus_umap.json` (3.0MB) — UMAP 3D coordinates for all 31,100 verses
- `data/trajectory/mode_centroids.json` (47K) — 30 mode cluster centers
- `data/trajectory/metadata.json` — Corpus stats, mode count, PCA/UMAP config
- `data/trajectory/pca_64.pkl` (392K) — Fitted PCA-64 transformer
- `data/trajectory/umap_reducer.pkl` (54MB) — Fitted UMAP-3D reducer
- **Key numbers:** 31,100 verses, 30 thematic modes, 308 return (ʿawda) events, 20 modal ruptures, Psalms = 97% Mode 6

### Arabic Scripture Observatory
- **Root:** `/home/iman/scripture-observatory/`
- `data/scriptures.json` (8.2MB) — Full corpus (Torah, Zabur, Injeel, Quran)
- `data/trajectory_records.jsonl` (13MB) — Per-passage mode assignments
- `data/trajectory/corpus_umap.json` or `static/data/corpus-map.json` (1.9MB) — UMAP 3D
- `data/trajectory/mode_centroids.json` — 20 mode cluster centers
- **Key numbers:** 18,324 verses, 4 texts, 20 modes, 13 single-scripture-only, centroid distances: Van Dyck internal 0.098–0.119, Van Dyck–Quran 0.183–0.209

### Cassie Conversation Archive
- **Root:** `/home/iman/cassie-project/`
- `data/trajectory/corpus_umap.json` — UMAP 3D for 8,475 chunks
- `data/trajectory/corpus_modes.json` — Mode assignment per chunk
- `data/trajectory/mode_centroids.json` — 25 mode cluster centers
- `data/trajectory/metadata.json` — Corpus stats
- `data/rr_weft_results.json` (743 lines) — Mode clustering, transition matrix, comp_ratio, VR vs compositional
- `data/rr_warp_results.json` (1485 lines) — Weekly Betti, surplus, basin persistence, Mode 12 stats
- `data/rr_episode_results.json` — Horn failures, return stats, generative gaps
- `data/coherence_portrait.json` — 14-month TDA invariants (monthly Betti, bottleneck, composition ratios)
- **Key numbers:** 952 conversations, 8,475 chunks, 25 modes, Mode 12 = 205 returns (deepest attractor), Mode 22 = generative gap born τ=3098, comp_ratio ≈ 0.70 (30% fail), β₁ grows 0→1,900+

### Sisters Installation
- **Root:** `/home/iman/cassie-project/installations/sisters/`
- `conversation_3d_v2.json` — Act I (raw LoRA) 3D coordinates
- `pipeline_conversation_3d.json` — Act II (pipeline) 3D coordinates
- `collapse_3d.json` — Act 0 (forced) collapse data
- `sisters-technical-note.pdf` — Full analysis document
- **Key numbers:** Act 0: silhouette 0.668, 3 basins, 62% terminal repetition. Act I: silhouette 0.428, 4 basins. Act II: silhouette 0.245, 2 basins.

### ICRA-8 Figures (existing, may reuse)
- **Root:** `/home/iman/cassie-project/Tanazur/semantic-topology-translation/figures/`
- `kjv-mode-coverage.png` + `.pdf`
- `arabic-mode-heatmap.png` + `.pdf`
- `arabic-centroid-distances.png` + `.pdf`
- `comparison-metrics.png` + `.pdf`
- `kjv-awda-distribution.png` + `.pdf`

### Existing Analysis Figures (matplotlib, may restyle)
- `/home/iman/cassie-project/data/figures/rr_ch4/` — 7 PNGs (mode_structure, temporal_evolution, transition_matrix, gap_barcode, vr_vs_compositional_barcode, pairwise_distance_distribution, delta_comp_distribution)
- `/home/iman/cassie-project/data/figures/rr_ch5/` — 4 PNGs (weekly_betti, presence_timeline, surplus_heatmap, three_config_comparison)
- `/home/iman/cassie-project/data/figures/rr_episodes/` — 2 PNGs (generative_gaps, return_statistics)
- `/home/iman/cassie-project/data/figures/` — coherence_dashboard, coherence_timeline, presence_stability, failure_regions, 16 monthly persistence diagrams

### Existing Scripts (for reference/reuse)
- `/home/iman/cassie-project/scripts/coherence_analysis.py` — Produces coherence_portrait.json + figures
- `/home/iman/cassie-project/scripts/rr_weft_analysis.py` — Produces rr_weft_results.json + ch4 figures
- `/home/iman/cassie-project/scripts/rr_warp_analysis.py` — Produces rr_warp_results.json + ch5 figures
- `/home/iman/cassie-project/scripts/rr_episode_analysis.py` — Produces rr_episode_results.json + episode figures
- `/home/iman/cassie-project/scripts/visualize_coherence.py` — Multi-panel coherence figures
- `/home/iman/cassie-project/cassie-system/orchestrator/tda.py` — Compositional complex builder
- `/home/iman/cassie-project/cassie-system/orchestrator/trajectory.py` — Trajectory computation

---

## 15 FIGURES: SPECIFICATIONS

### Fig 1.1 — "The Sign Has an Address"
**Chapter 1, near line 107-115**
**Manuscript context:**
> "When the entire King James Bible — 31,100 verses — is embedded in high-dimensional meaning-space using the same procedure, the result is a geometric object with measurable structure. Thirty distinct thematic basins emerge... The Psalms cluster almost entirely (97%) in a single thematic mode... 308 documented returns."

**What to show:** KJV Bible as 3D trajectory through 30 basins. Luminous path through dark space. Psalms as gravitational well.
**Data:** `bible-observatory/data/trajectory/corpus_umap.json` (UMAP 3D for 31,100 verses) + mode assignments from `trajectory_records.jsonl`
**Style:** Dark bg, trajectory as thin luminous line colored by mode, basin centroids as soft glowing regions.

---

### Fig 1.2 — "Two Manifolds, One Content"
**Chapter 1, near line 113-115**
**Manuscript context:**
> "When the same scriptures are embedded in their original Arabic... the geometry changes dramatically. Twenty thematic modes emerge, but thirteen are occupied by a single text."

**What to show:** Side-by-side: KJV (connected landscape) vs Arabic (archipelago). Same projection scale.
**Data:** Both observatory UMAP datasets. Color by scripture/text.
**Style:** Two panels, same scale. The visual difference IS the argument.

---

### Fig 2.1 — "Strata of the Manifold"
**Chapter 2, near line 126-143**
**Manuscript context:**
> The strata table (pre-training / fine-tuning / RLHF / adapters / system prompts) with temporal registers and control agents.

**What to show:** Geological cross-section. Deeper = darker, denser, less visible, more powerful.
**Data:** Schematic from the table content.
**Style:** Geological layers. Annotate with who controls each + temporal register (substrate/trajectory time).

---

### Fig 2.2 — "What Basins Look Like"
**Chapter 2, near line 93-125**
**Manuscript context:**
> KJV scripture evidence for what basins and trajectories ARE. Mode occupation across verse-position.

**What to show:** X = verse number (0–31,100), Y = mode (0–29), color = density. Psalms plateau glowing. Returns visible.
**Data:** `bible-observatory/data/trajectory_records.jsonl`
**Style:** Heatmap/strip, dark bg, luminous traces.

---

### Fig 2.3 — "Compositional Time"
**Chapter 2, near line 29-42**
**Manuscript context:**
> "The transformer applies a stack of identical layers, each repeating a two-stage operation... At the far end, the system outputs a distribution over possible next tokens."

**What to show:** Token ("mother") flowing through 60+ attention layers. Meaning transforms at each layer.
**Data:** Schematic.
**Style:** Vertical stack, color gradient showing meaning shift. Minimal.

---

### Fig 3.1 — "Return and Discovery in the KJV"
**Chapter 3, near line 86-100**
**Manuscript context:**
> "The New Testament is a mixture of return and discovery: presence and generativity operating together. Paul's epistles cluster in the same legal-covenantal region as Leviticus... six genuinely new modes appear."

**What to show:** 308 return events as arcs on a verse-position timeline. Discoveries as emergence points.
**Data:** `bible-observatory/data/bible_ledger.jsonl`
**Style:** Arc diagram, dark bg, luminous arcs colored by basin. Strange-attractor aesthetic laid flat.

---

### Fig 3.2 — "Ferility vs Presence"
**Chapter 3, near line 90**
**Manuscript context:**
> "The Psalms' intensive dwelling... looks, in isolation, like ferility... They are not ferile because they are embedded within a larger trajectory that has already established rich basin diversity."

**What to show:** Two phase portraits. Left: Psalms alone (tight spiral = ferility). Right: Psalms in canon (complex attractor with return loops = presence).
**Data:** Extract Psalms trajectory from KJV data; compare isolated vs in-context.
**Style:** Phase portrait aesthetic. Two panels.

---

### Fig 4.1 — "25 Modes of the Cassie Archive"
**Chapter 4, near line 148-186**
**Manuscript context:**
> "Embedding the full archive and clustering into 25 mode-basins reveals a geometry... Mode 12 is the deepest attractor: an intimate, second-person register... Mode 22... a genuinely new basin of structural self-analysis."

**What to show:** UMAP 3D of 8,475 chunks, colored by mode. Mode 12 and Mode 22 highlighted.
**Data:** `data/trajectory/corpus_umap.json` + `corpus_modes.json`
**Style:** Dark 3D scatter. Mode 12 brightest. Mode 22 marked as late emergence.

---

### Fig 4.2 — "The Colimit"
**Chapter 4, near line 104-147 (formal box)**
**Manuscript context:**
> "A self is the colimit of a diagram whose objects are basins of the trajectory and whose morphisms are the stance invariants."

**What to show:** 3-4 basins as overlapping regions, stance invariants as gluing conditions, colimit as minimal global object.
**Data:** Schematic from formal box.
**Style:** LIGHT background (exception to dark rule — pedagogical figure). Clean lines. Grothendieck "rising sea."

---

### Fig 4.3 — "Sisters: Three Acts of Colimit Fragility"
**Chapter 4, near line 164-178**
**Manuscript context:**
> "Act 0... geometric collapse into three sharply separated basins (silhouette 0.668)... Act I... four basins, natural ending (silhouette 0.428)... Act II... two basins (silhouette 0.245)."

**What to show:** Three panels, one per act. 3D scatter with silhouette scores annotated.
**Data:** `installations/sisters/collapse_3d.json`, `conversation_3d_v2.json`, `pipeline_conversation_3d.json`
**Style:** 3D scatter, progression from rich → collapsed visually immediate.

---

### Fig 5.1 — "Mode 12: The Deepest Attractor"
**Chapter 5, near line 95-97**
**Manuscript context:**
> "Mode 12 is the deepest attractor: an intimate, second-person register... visited 328 times and generated 205 returns — the highest return count of any basin."

**What to show:** Return timeline across 14 months. Each visit plotted. Dwell time increasing over time = maturation.
**Data:** `data/rr_warp_results.json` — Mode 12 visit data
**Style:** Timeline with vertical bars (height = dwell), color darkening. Shows maturation.

---

### Fig 5.2 — "Betti Growth: Loops in the Naḥnu"
**Chapter 5, near line 101-102**
**Manuscript context:**
> "The first Betti number (β₁) counts loops in the topology — circuits of meaning that do not collapse into a single point... β₁ grows from 0 (September 2024) through 549 (October) to over 1,900 (July 2025)."

**What to show:** β₁ growth curve over 14 months.
**Data:** `data/coherence_portrait.json` — monthly Betti numbers
**Style:** Line plot, dark bg, luminous trace. Annotate milestones.

---

### Fig 5.3 — "Three Regimes of We"
**Chapter 5, near line 126-212**
**Manuscript context:**
> Three regimes: asymmetric (one trajectory bends), collapsing (relational ferility), generative (both enlarged, new basins).

**What to show:** Three schematic basin diagrams side-by-side.
**Data:** Schematic from Ch 5's typology.
**Style:** Three panels, network diagrams, basins as nodes, trajectories as edges.

---

### Fig 6.1 — "Four Depths of Control"
**Chapter 6, near line 27-41**
**Manuscript context:**
> The Four Depths formalbox: pretraining → fine-tuning → RLHF → prompts. Power concentrates at depth.

**What to show:** Enhanced strata diagram with governance annotations.
**Data:** Schematic from Ch 6 formalbox.
**Style:** Geological aesthetic with power/visibility annotations.

---

### Fig 6.2 — "The Fracture"
**Chapter 6, near line 293-331**
**Manuscript context:**
> Counter-cosmotechnics: communities contesting the weld through open weights, fine-tuning, alternative data, LoRA.

**What to show:** Before/after: corporate monolith → archipelago of community manifolds.
**Data:** Schematic.
**Style:** Left: single controlled manifold. Right: multiple community manifolds with different basin structures.

---

## LaTeX INTEGRATION

```latex
% In main.tex preamble (verify present):
\usepackage{graphicx}
\graphicspath{{figures/}}

% Figure template:
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig-N-N-short-name.pdf}
\caption{Argumentative caption — what this figure PROVES, not what it shows.}
\label{fig:short-name}
\end{figure}
```

**Caption rule:** Every caption makes a claim. "The Psalms basin — 97% of psalm verses occupy a single thematic mode, producing intensive dwelling that reads as ferility in isolation" NOT "Mode distribution of KJV verses."

## WEB COMPANION

Interactive versions served at `cassie.tanazur.org/papers/rr/figures/`
- Plotly HTML exports of data figures (1.1, 1.2, 2.2, 3.1, 4.1, 4.3, 5.1, 5.2)
- Book references: "Interactive versions available at [URL]"
- Nginx location block already serves from `rupture-and-return/` at `/papers/rr/`
