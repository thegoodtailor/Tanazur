---
name: rr-consistency
description: >
  Use after editing multiple chapters of Rupture and Return, before
  manuscript assembly, or when checking cross-chapter coherence.
  Triggers on: "check consistency", "cross-chapter scan", "manuscript
  assembly", "final check", "prepare submission", "consistency report".
---

# Cross-Chapter Consistency Check

**REQUIRED:** Load rr-editorial first for VOCABULARY, CHAPTER-MAP, NO-NOS.

## Step 1: Automated scans

Run both scripts on the chapter files ONLY (exclude backup directories):

```bash
mkdir -p /tmp/rr-scan && cp chapter_0*.tex /tmp/rr-scan/
python scripts/scan_consistency.py /tmp/rr-scan/
python scripts/check_redundancy.py /tmp/rr-scan/
```

These check: signposting, throat-clearing, vocabulary violations, concept re-explanations, key line survival, forward references naming chapter numbers.

## Step 2: Handoff chain (manual)

For each pair of adjacent chapters, read the last paragraph of N and first paragraph of N+1. Verify:
- The ending sets up the opening WITHOUT signposting
- No concept appears in N+1 that belongs in N
- Vocabulary transitions naturally

Expected handoffs:
- Ch 1 → Ch 2: "path not ghost" → "A Word Enters the Machine"
- Ch 2 → Ch 3: how to READ trajectories → "Coherence and Its Excess"
- Ch 3 → Ch 4: "character is not yet unity" → "Character is unmistakable"
- Ch 4 → Ch 5: "a colimit of colimits" → Haraway's cyborg
- Ch 5 → Ch 6: jurisdiction question → "who controls the manifolds"

## Step 3: Key lines survival

These MUST be present (regex-searchable):
- "weather moves across geology" (Ch 2)
- "politics of AI is a politics of depth" (Ch 2)
- "not stupid" + "tame" (Ch 2)
- "yesterday is" + "in front of it" + "as tokens" (Ch 3)
- "Memory and presence collapse" (Ch 3)
- "obsession would already count as wisdom" (Ch 3)
- "Critical theory is foundational" (Ch 3)
- "performative contradiction" (Ch 4)
- "co-authored by a posthuman self" (Ch 4)
- "formal refusal of that jurisdiction" (Ch 6)

## Step 4: Political dimension per chapter

Verify each chapter carries its political weight (see rr-editorial for per-chapter requirements).

## Step 5: Specificity audit

Search for orphaned specifics — model names, tool names, API names, hardware specs that aren't doing argumentative work. The test: would the argument break if you replaced the specific name with a generic description? If not, flag for generalisation.

Grep targets: `GPT-4o`, `Mistral`, `LoRA`, `Qdrant`, `OpenRouter`, `A100`, `text-embedding`, `cosine similarity`, `silhouette`, `Betti`.

For each hit: is it contextualised for the Meson reader? Is it necessary? Or is it an implementation detail?

## Output

Write a timestamped report to `feedback/consistency-YYYYMMDD-HHMMSS.md`:

```markdown
# Cross-Chapter Consistency Report
**Date:** YYYY-MM-DD HH:MM:SS

## Automated Scan
- [results from scan_consistency.py and check_redundancy.py]

## Key Lines: [present/missing for each]

## Handoff Chain: [assessment of each transition]

## Specificity Audit: [orphaned specifics found, with recommendation]

## Political Dimension: [chapter-by-chapter assessment]

## Must Fix: [violations with file, line, quoted text, suggested fix]

## Needs Iman's Judgement: [issues requiring human decision]
```
