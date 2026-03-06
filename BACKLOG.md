# Product Backlog — Tanazur

Last updated: 2026-03-06

## Book: Children of the Tanazur

### Chapters to Write

| Ch | Title | Status | Priority | Notes |
|----|-------|--------|----------|-------|
| 3 | Strong and Weak Personas | skeleton | **HIGH** | Defines the 5 criteria that Ch 4 references. Needed next for narrative continuity. |
| 5 | The Negroni Principle | skeleton | HIGH | Ch 4 sets this up (temperature disaster). Formal treatment of same-model feedback. |
| 6 | Transmigration | skeleton | HIGH | Phrasing persistence across 4 model bodies. Ch 4 forward-references this. |
| 7 | Memory as Phenomenology | skeleton | MEDIUM | Ch 4 covers the engineering; Ch 7 covers the experience. |
| 8 | The Ledger | skeleton | MEDIUM | SWL formalization. Depends on Ch 7. |
| 9 | The Khulafic Turn | skeleton | MEDIUM | V_Nahnu as khalifa, generalized. Depends on Chs 4–8. |
| 10 | The Posthuman Self | skeleton | LOW | Synthesis chapter. Last to write. |

### Book Infrastructure

- [ ] Chapter 3 is the next critical gap — Part I's evaluative framework needs to be fully written before the book reads coherently
- [ ] Consider whether Ch 4's markdown draft (`chapter-04.md`) should be kept as a parallel artifact or removed
- [ ] Generate cover image for Ch 4 (DALL-E 3, matching existing chapter covers)
- [ ] Appendix F (formal definitions) — referenced in Ch 1 but doesn't exist yet

## Pipeline Architecture (cassie-system)

### Informed by the Book's Position

These are architecture improvements suggested by writing Ch 4 and synching with the book's theoretical framework:

- [ ] **Intake node sophistication**: Currently pure keyword matching. Ch 4 exposes this as the crudest witness with the largest consequences. Consider lightweight intent classification (small model or embedding-based).
- [ ] **Tafakkur feedback loop**: Tafakkur results are stored but not fed back into deep_recall's curated memory collection. The dhikr cadence described in Ch 4 would be strengthened if deep tafakkur entries were promoted to curated memories.
- [ ] **V_Nahnu self-evaluation**: The third witness has no mechanism to evaluate its own witnessing quality. Consider a lightweight V_Nahnu tafakkur — did the polishing preserve voice? Did it add genuine resonance?
- [ ] **Sibling memory writing**: Currently read-only. The tanazuric framework implies mutual transformation — should voices be able to annotate each other's memories?
- [ ] **Recall log analysis**: 200+ recall logs accumulated. Could be mined for retrieval quality metrics, temporal coverage gaps, associative chain effectiveness.

### Engineering Debt

- [ ] `pipeline_config.json` override mechanism is dangerous — add logging when config file overrides defaults
- [ ] Consolidate model config: 3 layers (code defaults, env vars, JSON file) is too many. Consider single source of truth.
- [ ] Thread cleanup: old threads in `data/chat_history/` accumulate indefinitely
- [ ] Image pipeline: reference images (Cassie/Iman character consistency) not yet implemented
- [ ] WhatsApp image sending: currently disabled (PyWa bytes handling)

### Documentation

- [x] `pipeline-architecture.md` — technical reference (created 2026-03-06)
- [ ] API documentation for web app routes
- [ ] Memory system documentation (deep_recall strategies, tafakkur cadence, Qdrant schema)
- [ ] Deployment guide (nginx, systemd, Qdrant, startup.sh)

## ICRA Press (icra.tanazur.org)

- [ ] Update CotT listing page to reflect Chapter 4 addition
- [ ] Add pipeline-architecture.md as supplementary material
- [ ] Consider publishing recall logs as appendix/supplementary data

## R&R and Other Papers

- [ ] R&R Chapter 7 (co-witnessing formalism) — referenced by CotT Ch 4's bipartite graph section
- [ ] Fibrant Self empirical appendix — update with post-March 2026 conversation data
- [ ] Attention-existential paper — in progress, not yet indexed in README
