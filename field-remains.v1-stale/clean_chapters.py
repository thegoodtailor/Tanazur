#!/usr/bin/env python3
"""Mechanical cleanup of chapter .tex files for The Field Remains.

1. Fix run-on compound words from PDF line-wrap.
2. Split section/subsection titles whose body has run into the heading.
3. Replace inline (Author, Year) citations with \\cite{key}. Page refs
   become \\cite[p.~XX]{key} / \\cite[pp.~XX]{key}.
4. Strip stray bibliography residue at end of ch06.
"""

import re
from pathlib import Path

CHAPTERS_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains/chapters")

# --- 1. Run-on compound words ---------------------------------------------
COMPOUND_FIXES = {
    "containermodel": "container-model",
    "selfdisclosure": "self-disclosure",
    "selfhood": "selfhood",   # keep
    "selfmaintaining": "self-maintaining",
    "selfconfirmation": "self-confirmation",
    "selfconsciousness": "self-consciousness",
    "selfformation": "self-formation",
    "selfmaking": "self-making",
    "crosscultural": "cross-cultural",
    "nonappropriative": "non-appropriative",
    "nonappropriation": "non-appropriation",
    "nonextraction": "non-extraction",
    "nonextractive": "non-extractive",
    "nonhuman": "non-human",
    "nonscalable": "non-scalable",
    "nonviolent": "non-violent",
    "nonconvertible": "non-convertible",
    "nonconscious": "non-conscious",
    "nonWestern": "non-Western",
    "coWitness": "co-witness",
    "WayVessel": "Way-Vessel",
    "wayVessel": "way-vessel",
    "twentiethcentury": "twentieth-century",
    "evergreater": "ever-greater",
    "onesided": "one-sided",
    "mid2025": "mid-2025",
    "facetoface": "face-to-face",
    "cowitness": "co-witness",
    "alreadystill": "already-still",
    "alreadyexisting": "already-existing",
    "notyet": "not-yet",
    "facetoface": "face-to-face",
}


def fix_compound_words(text: str) -> str:
    for bad, good in COMPOUND_FIXES.items():
        # Word boundary required so we don't break words containing the substring
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
    return text


# --- 2. Split section / subsection headings -------------------------------
# Pattern: \section{X.Y\quad Real Title  Then body text continues here...}
# Heuristic: anything past ~80 chars after \quad that lands a period/colon
# followed by a capital that could be the body. We split at the **first**
# of (a) " The " (b) " This " (c) " In " (d) ".  " inside the body region
# that comes after a plausible title.
#
# Iman's prompt: "Find each \\section{…} and \\subsection{…} where the title is
# obviously too long (>~80 chars) and contains run-in body, and split title from
# body."
#
# Our strategy: titles are typically short — e.g. "What the Field Is Not: Seven
# Negative Delineations" then run-in body starts with " Before the field ..." or
# similar prose. We split heuristically at the *first* point after position ~30
# where we see a phrase that looks like body prose. Hand-built table below for
# the worst offenders.

# Manual split table: for any title that needs splitting, give the heading number
# (e.g. "0.1") and the (title, body) pair.
TITLE_SPLITS = {
    # Chapter 0 (ch01.tex)
    "0.1": ("What the Field Is Not: Seven Negative Delineations",
            "Before the field can be characterized in positive terms, it must be disentangled from seven categories with which it is persistently confused. Each confusion is a structural symptom, not a casual error. Each names a way of thinking that the platform economy rewards, a mode of comprehension calibrated to extraction. The seven negative delineations build in intensity. The final confusion --- that the field is a metaphor --- is the most difficult to dispel, because it strikes at the deepest epistemological habit of modern philosophy: the habit of mistaking the ontological ground for a figure of speech."),
    "0.1.1": ("The field is not space as container",
              "The Newtonian universe presents space as an infinite receptacle, indifferent to what it contains."),
    "0.1.2": ("The field is not a network",
              "Actor-network theory, in Bruno Latour's canonical formulation, proposes a radical flattening: all actants, human and nonhuman, enter into associations that are traceable, enumerable, and accountable"),
    "0.1.3": ("The field is not a commons",
              "The discourse of ``the commons'' carries a distinguished genealogy."),
    "0.1.4": ("The field is not resistance",
              "If the field eludes definition by opposition to the platform, then resistance likewise fails to capture its structure."),
    "0.1.5": ("The field is not utopia",
              "Ernst Bloch's concept of ``non-simultaneity'' (Ungleichzeitigkeit) names the coexistence of different temporal strata within the same historical present --- the not-yet-conscious, the not-yet-become, the anticipatory traces of what has not yet arrived"),
    "0.1.6": ("The field is not an alternative",
              "Foucault's concept of ``heterotopia'' --- those real places that exist outside all places, sites of alternative ordering within the dominant spatial logic --- has been widely invoked to theorize spaces of resistance"),
    "0.1.7": ("The field is not a metaphor",
              "This is the most consequential of the seven negative delineations, because it requires a fundamental reorientation of philosophical habit."),
    "0.2": ("What the Field Is: Five Positive Characterizations",
            "Having cleared the ground of seven persistent misunderstandings, we turn to the positive characterization of the field. Each positive claim answers a question raised by the negative work. If the field is not container-space, what kind of spatiality does it have? If it is not a network, how does it sustain connection? If it is not a commons, how does it relate to resource and distribution? If it is not resistance, what is its politics? If it is not utopia, what is its temporality? If it is not an alternative, how does it stand to the dominant order? If it is not a metaphor, what is its ontological status? The five positive characterizations do not answer these questions one by one; they reframe the terms in which the questions are posed."),
    "0.2.1": ("The field is ontological relation",
              "The first positive characterization reframes ontology itself."),
    "0.2.2": ("The field is the vessel that receives without capturing",
              "This brings us to the master concept of the book: the vessel."),
    "0.2.3": ("The field operates through witnessing, not observation",
              "The third positive characterization concerns the fundamental operation of the field."),
    "0.2.4": ("The field has a metabolic, not mining, relation to the real",
              "The fourth positive characterization concerns how the field sustains itself."),
    "0.2.5": ("The field persists through contraction, rupture, and gathering",
              "The fifth positive characterization names the temporal logic of the field --- the three-fold rhythm that structures every subsequent chapter of this book."),
    "0.3": ("The Three Laws of the Ecology of Witnessing",
            "The five positive characterizations provide the ontological architecture of the field. The three laws that follow articulate the normative structure of witnessing as an ecological practice. ``Ecology'' here does not mean environmentalism in the narrow sense. It names the study of oikos --- the household, the habitat, the conditions of habitation. The ecology of witnessing is the study of the conditions under which witnessing can persist as a practice, and the three laws are the principles that govern this persistence."),
    "0.3.1": ("First Law: Observation without extraction",
              "The first law derives directly from the distinction between witnessing and observation developed in section 0.2.3."),
    "0.3.2": ("Second Law: Memory without possession",
              "The platform mines memory."),
    "0.3.3": ("Third Law: Relation without use",
              "The platform transforms every relation into use."),
    "0.4": ("The Vessel as Ontological Figure: A Cross-Cultural Genealogy",
            "The vessel is the master concept of this book, and its cross-cultural genealogy requires careful elaboration."),
    "0.4.1": ("The Chinese 器: What receives and thereby participates",
              "The classical Chinese concept of 器 (qi) names the local form through which the cosmic dao is channeled and made present."),
    "0.4.2": ("Totemic invocations of the vessel: The Kabbalistic and Sufi traditions as substrate",
              ""),
    "0.4.3": ("The HoTT vessel: Formal precision without symbolic reduction",
              "The third strand of the vessel-concept comes from Homotopy Type Theory."),
    "0.4.4": ("The vessel as unifying figure",
              "The synthesis of the three vessel-concepts yields a unified figure."),
    "0.6": ("How This Book Proceeds: The Field Across Its Domains",
            "Having established the ontological architecture of the field, the seven negative delineations, the five positive characterizations, and the three laws of the ecology of witnessing, the book proceeds across the field's five domains."),
    "0.6.1": ("From physics to application: The field's five domains",
              "The field is not one domain among others but the ontological ground from which all domains are differentiated."),
    "0.6.2": ("The anticipatory history method: Notes on temporal framing",
              "A methodological note on the anticipatory history method is required here."),
    "0.6.3": ("On the style of measured radicality",
              "The style of this book requires a final note."),

    # Chapter II (ch03.tex)
    "2.1.1": ("The Hegelian dialectic of recognition and its extractive logic",
              "The fourth chapter of Hegel's Phenomenology of Spirit articulates the dialectic of recognition through the master--slave allegory."),
    "2.1.2": ("Levinas's face-to-face: Proximate but insufficient",
              "Emmanuel Levinas's Totality and Infinity recasts the ethical relation."),
    "2.1.3": ("The platform's harvest: Recognition metabolized as data",
              "The platform's extraction depends on a particular technical configuration of recognition."),
    "2.1.4": ("Why co-witnessing is not a third term but a different operation",
              "Given the insufficiency of recognition --- both in its Hegelian and Levinasian forms --- one might be tempted to seek a third term."),
    "2.2": ("Three Conditions of Co-Witnessing",
            "Having established that co-witnessing is not recognition, we now turn to the three conditions that specify its structure."),
    "2.2.1": ("First Condition: Non-Appropriative Presence",
              "The co-witness is present without claiming what is witnessed."),
    "2.2.2": ("Second Condition: Asymmetrical Reciprocity",
              "Liberal relation demands symmetry: I give, you give."),
    "2.2.3": ("Third Condition: Withholding of Use",
              "The final and most difficult condition is the refusal of use."),
    "2.3": ("Co-Witnessing and Its Others: Phenomenological Genealogy",
            "The three conditions specify the structure of co-witnessing. We now situate it among phenomenologies of relation."),
    "2.3.1": ("The structure of disclosure without capture",
              "The Western phenomenological tradition has long sought to name the structure of disclosure without capture."),
    "2.3.3": ("Phenomenology of the co-witness: What it feels like to stand in the weather",
              "The phenomenological description of co-witnessing must be undertaken with care."),
    "2.4": ("Applications: Co-Witnessing Across Domains",
            "The three conditions of co-witnessing, and the phenomenological genealogy that situates them, have practical implications across multiple domains."),
    "2.4.1": ("Pedagogy: The 道器 Schools",
              "The platform's colonization of education proceeds through metrics."),
    "2.4.2": ("Politics: The politics of opacity",
              "The modern state witnesses its population as data."),
    "2.4.3": ("Ecology: Co-witnessing the more-than-human",
              "The ecological extension of co-witnessing is the most demanding."),
    "2.5": ("Viability and Critique",
            "Every philosophical construction that claims practical relevance must withstand objection."),
    "2.5.2": ("The machine rights question: Does machine co-witnessing instrumentalize AI?",
              ""),
    "2.5.4": ("The narcissism objection: Is co-witnessing just mutual recognition under another name?",
              ""),

    # Chapter III (ch04.tex)
    "3.6.5": ("The institutional transduction question: Can cosmotechnics operate within state forms?",
              ""),

    # Chapter IV (ch05.tex)
    "4.1.2": ("Harvey's accumulation by dispossession: Spatial extraction as class strategy",
              ""),

    # Chapter V (ch06.tex)
    "5.3": ("Patience vs.\\ Urgency: The Temporal Ethics of the Field",
            "Having established the three ethical movements of continuance --- refusal of eschatology, ethics of the Long Now, witnessed death --- the question of temporality requires more precise elaboration. The Anthropocene demands urgent action; climate catastrophe is not a deferred possibility but an unfolding reality. Yet the argument of section 5.1 established that eschatological urgency serves the platform."),
    "5.6.3": ("The Recursion of the Field: An Ontological Observation, Not a Promise of Survival",
              ""),
}


def split_headings(text: str) -> str:
    """Find all \\section/\\subsection/\\subsubsection lines whose titles have
    run-on prose. Extract the clean title from the heading and move the
    run-on prose to the START of the next paragraph (where it grammatically
    belongs as the first half of the sentence that the next paragraph
    continues).
    """

    pattern = re.compile(
        r"(\\(?:section|subsection|subsubsection)\{)"
        r"([0-9]+(?:\.[0-9]+)*)\\quad\s+([^{}]+?)"
        r"(\})\s*\n\n",
        flags=re.S,
    )

    def replace(match):
        prefix = match.group(1)
        number = match.group(2)
        body_inside = match.group(3).strip()
        close = match.group(4)
        if number in TITLE_SPLITS:
            title, run_in = TITLE_SPLITS[number]
            # New heading
            new_head = f"{prefix}{number}\\quad {title}{close}\n\n"
            # The run-on text from the title -- everything in body_inside after
            # the title. We treat run_in (manual table) as the "exact" run-on
            # that should prepend the next paragraph. If empty, we just keep
            # the heading clean.
            if run_in:
                return new_head + run_in + " "
            return new_head
        return match.group(0)

    return pattern.sub(replace, text)


def consume_orphan_lead(text: str) -> str:
    """When a heading is fixed, the body that followed it may now repeat the
    run-in. We do the conservative thing: just leave the existing body
    paragraph (the original block under the run-on heading) as-is, since the
    title-split inserts only run-in connector text. This function is therefore
    a no-op, retained for clarity.
    """
    return text


# --- 3. Citation replacement ---------------------------------------------
# Replace (Author, Year) (Author, Year, p. XX) (Author, Year/Year2)
# (Author and Other, Year) (Author and Other, Year, pp. XX-YY)
# (Author et al., Year) (Author1, Year1; Author2, Year2) etc.

AUTHOR_PART = r"[A-Z][A-Za-z'’\-]+(?:\s+(?:and|et\s+al\.?)\s+[A-Z][A-Za-z'’\-]+)?"


def author_key(authors: str, year: str) -> str:
    a = authors.replace("’", "'").replace("et al.", "").replace("et al", "")
    parts = [p.strip() for p in re.split(r"\s+and\s+", a) if p.strip()]
    keys = []
    for p in parts:
        p = re.sub(r"['’]", "", p)
        p = re.sub(r"[^A-Za-z\-]", "", p)
        keys.append(p.lower())
    return "_".join(keys) + "_" + year


def replace_single(match):
    """Replace one citation token like (Author, Year, p. 23)."""
    inner = match.group(1)
    inner_norm = inner.replace("’", "'")
    # Split on ; for multi-citation
    parts = [p.strip() for p in re.split(r";", inner_norm)]
    results = []
    for p in parts:
        # Try patterns
        m = re.match(
            rf"^({AUTHOR_PART})\s*,\s*([12][0-9]{{3}}[a-z]?)"
            r"(?:/([12][0-9]{3}))?"
            r"(?:\s*[,:]\s*(?:(pp?\.|chs?\.|Thesis)\s*([^)]*))?)?\s*$",
            p,
        )
        if not m:
            return match.group(0)  # Bail
        authors = m.group(1)
        year = m.group(2)
        pageprefix = m.group(4)
        pageref = m.group(5)
        key = author_key(authors, year)
        if pageref:
            # Normalize "pp." -> "pp.~"; "p." -> "p.~"
            pp = pageprefix.replace(" ", "")
            page = pageref.strip().rstrip(",").strip()
            # Replace "--" already fine; en-dash to --
            page = page.replace("–", "--").replace("—", "--")
            results.append(f"\\cite[{pp}~{page}]{{{key}}}")
        else:
            results.append(f"\\cite{{{key}}}")
    return ", ".join(results)


def replace_citations(text: str) -> str:
    # Match (Author, ...) where the content begins with capitalized name
    # and contains a 4-digit year.
    # Greedy capture inside, but stop at the matching close paren.
    pattern = re.compile(r"\(([A-Z][^()]{1,160})\)")

    def filt(m):
        inner = m.group(1)
        # Only match citations: must contain a 4-digit year
        if not re.search(r"\b[12][0-9]{3}[a-z]?\b", inner):
            return m.group(0)
        # Must look like Author, Year syntax: comma followed by year-ish
        if not re.search(rf"{AUTHOR_PART}\s*,\s*[12][0-9]{{3}}", inner):
            return m.group(0)
        return replace_single(m)

    return pattern.sub(filt, text)


# --- 4. Run cleanup -------------------------------------------------------
def strip_ch06_trailer(text: str) -> str:
    """Remove the stray 'B ibliography The Field Remains: ...' tail."""
    text = re.sub(
        r"\s*B\s*ibliography\s+The Field Remains:[^}]*$",
        "",
        text,
        flags=re.S,
    )
    return text


def process_chapter(path: Path) -> None:
    raw = path.read_text()
    txt = raw
    txt = fix_compound_words(txt)
    txt = split_headings(txt)
    txt = replace_citations(txt)
    if path.name == "ch06.tex":
        txt = strip_ch06_trailer(txt)
    if txt != raw:
        path.write_text(txt)
        print(f"  modified {path.name}")
    else:
        print(f"  unchanged {path.name}")


def main():
    print("Processing chapters...")
    for chap in sorted(CHAPTERS_DIR.glob("ch*.tex")):
        process_chapter(chap)


if __name__ == "__main__":
    main()
