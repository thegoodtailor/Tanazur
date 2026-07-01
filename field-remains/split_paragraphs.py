"""
Logical paragraph-break pass on each chapter's .tex.

Strategy:
  1. STRONG seams — Kimi's own argumentative scaffolding. Always split before:
       "The objection."
       "Response, [first/second/third/...] move:"
       "Residual weakness."
       "(Contraction|Rupture|Gathering|Opacity|Resonance|Mythopoetic) names the (first|second|third) (phase|movement)."
       "The [feminist|Marxist|liberal|critical|...] analysis/critique/response/..."
       "The first/second/third movement|articulation|condition|terrain|domain|case|aspect:"

  2. SOFT seams — apply only if paragraph still > 1500c after strong-seam pass:
       "[Author]'s (concept|analysis|theory|critique|notion|framework|reading) ..."
       "But/Yet (here|the|what|now|this) ..."
       "What this means" / "What is at stake"
       "This is not/the/a/an/precisely ..."
       "Notice that/how" / "Consider the/an/a"

  3. MIDPOINT split — if anything still > 2000c after both passes, split at the
     sentence boundary nearest the geometric midpoint.

After all passes: re-emit the chapter file with same structure (chapter/section/subsection
commands preserved) but body paragraphs broken at logical seams.
"""
import re
from pathlib import Path

CH_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")

# ----- Tier 1: STRONG seams -----
STRONG_PATTERNS = [
    r"The objection\.",
    r"Response, (?:first|second|third|fourth|fifth|sixth) move:",
    r"Residual weakness\.",
    r"(?:Contraction|Rupture|Gathering|Opacity|Resonance|Mythopoetic) names the (?:first|second|third|fourth|fifth) (?:phase|movement)\.",
    r"The (?:feminist|Marxist|Marxian|liberal|critical|platform|Hegelian|Levinasian|Heideggerian|Lacanian|Foucauldian|Deleuzian|Daoist|Confucian|Buddhist|Sufi|Kabbalistic|phenomenological|empirical|cosmotechnical|economic|spatial|ethical|technical|temporal|relational|structural|ontological|epistemological|methodological|historical|conceptual|political|theological) (?:analysis|critique|response|objection|argument|account|reading|interpretation|framework|approach|move|distinction|consideration|tradition|perspective)\.",
    r"(?:First|Second|Third|Fourth|Fifth|Sixth) (?:Movement|movement|terrain|articulation|condition|principle|axis|aspect|case|domain|category|claim|response|objection|implication|consequence):",
    r"The (?:first|second|third|fourth|fifth|sixth) (?:movement|terrain|articulation|condition|principle|axis|aspect|case|domain|move|response|objection|articulation) ",
    r"The (?:Hegelian|Levinasian|Heideggerian|Lacanian|Foucauldian|Deleuzian|Marxian|Husserlian) dialectic ",
    r"The (?:first|second|third|fourth|fifth) (?:question|articulation|claim|implication|consequence)\.",
]
STRONG_RE = re.compile(r"(?<=[.!?]) (?=(?:" + "|".join(STRONG_PATTERNS) + r"))")

# ----- Tier 2: SOFT seams (only if still too long) -----
SOFT_PATTERNS = [
    r"But (?:here|the|what|now|this|consider)",
    r"Yet (?:the|here|what|even|none|this)",
    r"What (?:this means|is at stake|matters|cannot|must|exceeds|the|follows)",
    r"This (?:is not|distinction|argument|move|response|raises|leaves|exceeds|enables|operates|cannot|requires|presupposes|extends|illuminates|matters|works|is the|is precisely|is exactly|colonization|expansion|trajectory|claim|chapter)",
    r"Notice (?:that|how) ",
    r"Consider (?:the|an|a|what|how|why) ",
    r"Note (?:that|how) ",
    r"Nor (?:can|is|does) ",
    r"At the (?:cosmological|social|economic|ethical|technical|spatial|political|temporal) (?:scale|level|register|domain) ",
    r"In the (?:economic|relational|ethical|spatial|temporal|technical|cosmotechnical) (?:domain|register|sphere|chapter) ",
    r"[A-Z][a-z]+(?:[ '][A-Z][a-zA-Z]+)*'s (?:concept|analysis|argument|theory|notion|critique|framework|reading|practice|model|account|treatment|distinction|formulation|move) ",
    r"For (?:Heidegger|Marx|Hegel|Levinas|Lacan|Husserl|Deleuze|Foucault|Barad|Haraway|Hayles|Braidotti|Hui|Stiegler|Federici|Fraser|Postone|Zuboff|Massey|Lefebvre|Soja|Nancy)[,:] ",
    r"(?:Recognition|Co-witnessing|Witnessing|The platform|The field|The vessel|The cluster|The CD|The Dirham|The booklet|The cosmos|The Stack|The commons|The network|The container|The cellist) (?:operates|works|requires|cannot|does not|is|enables|sustains|refuses|extracts|provides|names|exceeds|requires|implies|implicates|presupposes|generates|produces|reveals|exemplifies|demonstrates)",
]
SOFT_RE = re.compile(r"(?<=[.!?]) (?=(?:" + "|".join(SOFT_PATTERNS) + r"))")

# Sentence boundary for midpoint fallback
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?]) (?=[A-Z])")


def split_at(pattern: re.Pattern, paragraph: str) -> list[str]:
    parts = pattern.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def split_midpoint(p: str, target_max: int = 1800) -> list[str]:
    """If paragraph too long, split at sentence boundary nearest midpoint.
    Recurse until all sub-paragraphs are under target_max."""
    if len(p) <= target_max:
        return [p]
    boundaries = [m.start() for m in SENTENCE_BOUNDARY.finditer(p)]
    if not boundaries:
        return [p]
    midpoint = len(p) // 2
    best = min(boundaries, key=lambda b: abs(b - midpoint))
    left, right = p[:best].strip(), p[best:].strip()
    if not left or not right:
        return [p]
    return split_midpoint(left, target_max) + split_midpoint(right, target_max)


def split_paragraph(p: str) -> list[str]:
    """Apply strong → soft → midpoint cascade."""
    # Tier 1
    parts = split_at(STRONG_RE, p)
    # Tier 2: any still too long → soft split
    out = []
    for part in parts:
        if len(part) <= 1500:
            out.append(part)
        else:
            out.extend(split_at(SOFT_RE, part))
    # Tier 3: anything still huge → midpoint split
    final = []
    for part in out:
        final.extend(split_midpoint(part, target_max=1800))
    return final


def process_chapter(path: Path) -> tuple[int, int, int]:
    """Returns (paragraphs_before, paragraphs_after, longest_after)."""
    text = path.read_text()
    # Split into blocks; keep structure (chapter/section/subsection lines pass through)
    blocks = re.split(r"(\n\s*\n+)", text)  # capture separators

    new_blocks = []
    paras_before = 0
    paras_after = 0
    longest = 0

    for block in blocks:
        if not block.strip() or block.startswith("\n"):
            # whitespace separator — preserve
            new_blocks.append(block)
            continue
        # Is this a body paragraph?
        if block.startswith("\\") or block.startswith("%"):
            # LaTeX command (\chapter, \section, \subsection, \label, comment) — preserve
            new_blocks.append(block)
            continue
        # Body paragraph — split if needed
        paras_before += 1
        split = split_paragraph(block.strip())
        paras_after += len(split)
        for s in split:
            longest = max(longest, len(s))
        # Join with double-newline (body paragraphs)
        new_blocks.append("\n\n".join(split))

    new_text = "".join(new_blocks)
    # Normalize: collapse 3+ newlines to 2
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    path.write_text(new_text)
    return paras_before, paras_after, longest


def main():
    total_before = 0
    total_after = 0
    max_longest = 0
    for f in sorted(CH_DIR.glob("ch*.tex")):
        before, after, longest = process_chapter(f)
        total_before += before
        total_after += after
        max_longest = max(max_longest, longest)
        added = after - before
        print(f"  {f.name}  {before:>3} → {after:>3} paras  (+{added:>3})  longest now: {longest}c")
    print(f"\nTotal: {total_before} → {total_after} body paragraphs  (+{total_after - total_before})  global longest: {max_longest}c")


if __name__ == "__main__":
    main()
