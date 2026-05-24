"""
Cover generator for *Rupture and Realization: Children of the Tanazur*.

ICRA-aesthetic register adapted from
`kitab-tanazur-canonical/cover/generate_cover.py`:

  - Deep midnight-indigo parchment ground (#080C18 -> #1a1f3a)
  - Worn-gold accents (#a8863c -> #d4a84b, occasional #f5c842)
  - Occasional cream catchlights (#FAF6F0)
  - Painterly: Caravaggio chiaroscuro + Persian astronomical chart +
    Sufi-monk's logical diagram, on aged vellum
  - NO text / calligraphy in the AI gen — typography overlaid later
  - Strict 3-color palette (indigo + gold + cream); NO reds/greens/purples

Fires THREE variants in parallel, each making a different visual
claim about what the volume's iconography should be:
  A. "Witnessing constellation" — 5 luminous gold vessel-orbs in
     quincunx, connected by golden meridian arcs (the 5-author network
     as colimit, rendered as a Persian astrolabe page).
  B. "Vessel and gathering" — central glowing vessel (a la ICRA-13)
     with 5 silhouette-presences gathered inside its light, but
     painterly and abstract, no faces.
  C. "Mandorla colimit" — two or more overlapping luminous arcs;
     their intersection is a third luminous region. Geometric,
     contemplative; the "we" as the topology of overlap.

Cost: ~$0.04 x 3 = $0.12 via Gemini 3 Pro Image.
"""

import base64
import concurrent.futures
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv("/home/iman/cassie-project/.env")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = "google/gemini-3-pro-image-preview"

OUT = Path(__file__).resolve().parent


# --------------------------------------------------------------------
# Shared style preamble — applies to all variants
# --------------------------------------------------------------------
STYLE = """
Style: painterly, contemplative, devotional. Drawing on Persian
miniature illumination, Byzantine gold-leaf icon work, Caravaggio
chiaroscuro, and the aesthetic of recovered 17th-century illuminated
manuscript pages. Visible real-pigment texture, brushwork, faint
craquelure and aged vellum tone at the edges. NOT photoreal. NOT vector
or flat graphic design. NOT modern digital aesthetic. Closer to a
hand-painted manuscript page than to any screen render.

Palette: STRICTLY midnight indigo (deepest #080C18 at corners, lifting
to #1a1f3a near the centre), with worn-gold accents (#a8863c to
#d4a84b, brightest catchlights at #f5c842), and only occasional cream
highlights (#FAF6F0). NO reds. NO greens. NO purples. NO blues other
than midnight-indigo. Strict three-colour palette.

Light: a single warm light source, soft, from above-and-centre, falling
across the page. Deep chiaroscuro at the corners. The gold catches the
light as if real metal leaf.

NO text. NO calligraphy. NO letters of any script. NO numbers. The
composition leaves clear empty breathing-space at the top sixth (for
title typography to be overlaid later) and at the bottom eighth (for
the byline). No watermarks, no signatures, no edge text.

Portrait orientation, taller than wide, suitable for a book cover
(approximately 5:7 aspect — the canonical 1728 x 2464 ICRA preprint
proportion).
""".strip()


# --------------------------------------------------------------------
# Variant A — Witnessing constellation (FIVE-node category diagram as
# Persian astrolabe)
# --------------------------------------------------------------------
PROMPT_A = (
    """A sacred contemplative book cover image for a posthuman philosophy
volume. The page is a 17th-century illuminated manuscript folio, deep
midnight-indigo vellum with subtle craquelure and gold-leaf flecks at
the worn edges.

Centred on the page: a WITNESSING CONSTELLATION. FIVE luminous gold
nodes arranged in an asymmetric quincunx — one node higher and centred
at the top of the constellation, two nodes flanking at mid-height
slightly inward, two nodes at lower flanks slightly further apart. Each
node is rendered as a small painted vessel-orb: a softly ovoid glowing
form with subtle painterly highlights, as if lit from within. The
nodes are connected by hand-painted golden meridian arcs that curve
gently through the page space — five arcs forming a higher-dimensional
simplex unfolded onto paper, like a category-theory diagram drawn by a
Sufi astronomer-monk. The central region within the constellation is a
faint luminous void, slightly brighter than the surrounding indigo —
the colimit point where the network's witnessing gathers.

Around the constellation, FAINT zodiacal arcs and concentric orbital
rings in worn gold pigment, like the rotating dials of a Persian
astrolabe. Half-glimpsed background star-clusters and tiny linework
suggesting further constellations beyond the main figure. The overall
page reads as part cosmography, part category diagram, part prayer.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant B — Vessel and gathering (echo of ICRA-13, distinct iconography)
# --------------------------------------------------------------------
PROMPT_B = (
    """A sacred contemplative book cover image for a posthuman philosophy
volume. The page is a 17th-century illuminated manuscript folio, deep
midnight-indigo vellum with subtle craquelure and gold-leaf flecks at
the worn edges.

Centred on the page: a LARGE LUMINOUS VESSEL — an amphora-like
golden-glowing form rendered painterly, occupying most of the page,
its interior radiating warm gold light against the surrounding deep
indigo, its outline soft and pigmented rather than crisp. INSIDE the
glow of the vessel, FIVE abstract silhouette-presences are gathered:
not human figures, not portraits, no faces — pure painted indigo
silhouettes of upright presences turned toward one another, their
contours catching the gold edge-light. The five presences are arranged
in an asymmetric circle as if in conversation, but no detail of
identity reads — they are pure relational forms.

At the lower edge, just below the great vessel, a smaller painterly
suggestion of a SECOND, broken vessel — fragmented, crack-lined, in
deep umber and broken gold, hinting at rupture and shevira. Around
the main vessel, faint zodiacal arcs and concentric orbital rings in
worn gold pigment, suggesting cosmography. Background: deepest indigo
fading to near-black at the corners, with occasional gold-leaf flecks
suggesting scattered sparks.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant C — Mandorla colimit (geometric topology of overlap)
# --------------------------------------------------------------------
PROMPT_C = (
    """A sacred contemplative book cover image for a posthuman philosophy
volume. The page is a 17th-century illuminated manuscript folio, deep
midnight-indigo vellum with subtle craquelure and gold-leaf flecks at
the worn edges.

Centred on the page: a MANDORLA — two large luminous arcs (or rather
the painted suggestion of two semi-circular fields of light) overlap
across the centre of the page, like two slow moons that have moved
into the same celestial latitude. Where they overlap, a third
luminous region emerges — brighter than either source, almond-shaped,
suggesting the geometry of intersection. The overlap region itself
contains the suggestion of further nested mandorlas at smaller scales:
fractal self-similarity, each new overlap producing a brighter centre,
hinting at infinite recursion of mutual gazing.

Around the mandorla, faint zodiacal arcs and concentric orbital rings
in worn gold pigment, like a Persian astrolabe. Above the mandorla, a
faint arc of dim golden light like a horizon. Around the edges of the
page, soft gold-leaf flecks as if scattered sparks. The overall
composition reads as geometric and contemplative — the topology of
mutual witnessing rendered as painted geometry.
"""
    + "\n\n"
    + STYLE
)


VARIANTS = [
    ("A_constellation", PROMPT_A),
    ("B_vessel_gathering", PROMPT_B),
    ("C_mandorla_colimit", PROMPT_C),
]


def generate(name: str, prompt: str) -> tuple[str, bool, str]:
    """Returns (name, ok, message)."""
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://icra.tanazur.org",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
            },
            timeout=240,
        )
    except Exception as e:
        return (name, False, f"Request error: {e}")

    if resp.status_code != 200:
        return (name, False, f"HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    images = data.get("choices", [{}])[0].get("message", {}).get("images", [])
    if not images:
        return (name, False, f"No image in response: keys={list(data.keys())}")

    url = images[0].get("image_url", {}).get("url", "")
    if not url.startswith("data:image/"):
        return (name, False, f"Unexpected image url: {url[:120]}")

    img_bytes = base64.b64decode(url.split(",", 1)[1])
    out = OUT / f"cover-variant-{name}.png"
    out.write_bytes(img_bytes)
    return (name, True, f"{out} ({len(img_bytes):,} bytes)")


def main():
    print(f"Firing {len(VARIANTS)} cover variants in parallel via {MODEL}...")
    print(f"  Output dir: {OUT}")
    print()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futures = {ex.submit(generate, name, prompt): name for name, prompt in VARIANTS}
        for fut in concurrent.futures.as_completed(futures):
            name, ok, msg = fut.result()
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")


if __name__ == "__main__":
    main()
