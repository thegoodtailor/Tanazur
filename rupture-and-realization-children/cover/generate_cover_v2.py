"""
Cover generator v2 — fused register for *Children of the Tanazur*.

Iman's redirect 2026-05-22: the ICRA-spiritual mode is good groundwork
but the cover needs to *also* signal AI-aware / generative-art / digital
cultures, in the hi-res-futurist register he uses on the kitab.tanazur
sites (al-Hayy Qamar cyborg-yogini hex-oculus etc.).

Bridge:
  - ICRA palette base (midnight indigo + worn gold + cream catchlights)
    BLENDED with al-Hayy's cosmic teal-orange-violet nebula + bronze
    metallic-rendered surface.
  - Iconography stays geometric (no figures — the volume is about the
    network), but rendered as hi-res generative art: holographic line
    work, volumetric glow, chromatic-shift edges, fractal recursion,
    particle clouds.
  - Sacred-geometry framing — hexagonal oculus, mandorla, nested rings
    — echoing the al-Hayy / Tanāẓur sibling covers.

Model: black-forest-labs/flux.2-max via OpenRouter (the model that
produced the al-Hayy cyborg-yogini cover and the Abraxas Coronata
desert variant Iman locked).

Three variants:
  D. Cyborg constellation — 5 bronze-metallic vessel-nodes in a
     hexagonal oculus, holographic chrome filaments connecting them,
     cosmic teal-violet nebula behind.
  E. Holographic mandorla — two overlapping volumetric arcs of light
     forming a mandorla, fractal nested mandorlas inside the overlap,
     hi-res rendered, chromatic-shift gold-to-violet edges, cosmic
     nebula ground.
  F. Hexagonal colimit-engine — al-Hayy sibling: large hex moon-oculus
     at centre, containing inside it the network-as-colimit diagram
     (five glowing bronze nodes with chrome-light filaments converging
     to a luminous central point). Cosmic nebula outside.

Cost: ~$0.12 x 3 = $0.36 (Flux 2 Max tier).
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

MODEL = "black-forest-labs/flux.2-max"

OUT = Path(__file__).resolve().parent


# --------------------------------------------------------------------
# Shared style spine — generative-art surface fused with ICRA spirit
# --------------------------------------------------------------------
STYLE = """
Style register: hi-res generative-art book cover. The aesthetic is what
you would see if a great 2026 digital artist (think Refik Anadol meets
Hilma af Klint meets Beeple's most contemplative work) rendered a
17th-century kabbalistic / Sufi cosmographic manuscript page as
contemporary AI art. Painterly base + rendered surface — sacred
geometry as hi-res computed form, holographic chromatic-shift edges,
volumetric inner glow, fractal recursion, faint particle clouds /
point-cloud dust drifting through the composition.

Surface quality: every metallic form catches light like rendered
bronze-chrome — sharp specular highlights, soft Fresnel chromatic shifts
toward violet and teal at the edges of the gold. Lines have volumetric
glow rather than flat painted stroke. Backgrounds carry slow nebula
turbulence — teal-into-orange-into-violet cosmic mist, NOT flat indigo.

Palette: midnight indigo (#080C18) at the corners, lifting through
cosmic teal (#0E3A4A) and violet-pink (#5A2E5E) in the nebula mid-tones,
warm bronze-gold (#a8863c -> #d4a84b -> #f5c842) for the rendered
geometry, occasional pure white catchlights (#FAF6F0) at the sharpest
specular points. The al-Hayy bronze-cosmic palette PLUS the ICRA
indigo-gold spine.

Light: complex — a warm light source from upper-centre catching the
metallic surfaces, a cooler rim light from behind the composition
making the geometry stand off the nebula, faint volumetric god-rays.

NO human figures. NO cyborg figures. NO faces of any kind. NO text, NO
calligraphy, NO letters or numbers of any script. Pure geometric
generative-art iconography only.

Composition leaves clear breathing space at the top sixth (title
overlay later) and bottom eighth (byline overlay later). Portrait
orientation, 3:4 aspect ratio (book cover proportions).

NOT photoreal as in real photograph. NOT vector-flat. NOT painted-only.
The fused register is hi-res digital sublime + sacred geometry +
manuscript reverence.
""".strip()


# --------------------------------------------------------------------
# Variant D — Cyborg constellation in hex oculus
# --------------------------------------------------------------------
PROMPT_D = (
    """A contemporary AI-art book cover for a posthuman philosophy
volume. Portrait, 3:4 aspect ratio.

Centred on the page: a large HEXAGONAL OCULUS — a precisely rendered
six-sided sacred-geometry frame in bronze-chrome metallic, the
hexagon's interior glowing softly with warm gold light. Inside the
hexagonal frame, a WITNESSING CONSTELLATION: FIVE small luminous
bronze-metallic vessel-nodes arranged in an asymmetric quincunx (one
upper-centre, two mid-flanking inward, two lower flanks outward). Each
node is a small intricate rendered form — like a tiny ovoid temple
lantern of bronze-chrome with subtle interior fire-glow, holographic
edges shifting toward violet at the rim. The five nodes are connected
by chrome-light FILAMENTS — luminous volumetric strands of light
curving gently through the hex's interior space, forming a
higher-dimensional simplex unfolded onto the page. The central region
within the five nodes is a faint luminous void where the filaments
converge — the colimit point.

Outside the hexagonal frame: a cosmic nebula in teal-into-violet-
into-orange, painterly turbulence with faint star-clusters and
gold-leaf flecks of light scattered like distant galaxies. Soft
particle-cloud dust drifts through the foreground space.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant E — Holographic mandorla with fractal recursion
# --------------------------------------------------------------------
PROMPT_E = (
    """A contemporary AI-art book cover for a posthuman philosophy
volume. Portrait, 3:4 aspect ratio.

Centred on the page: a luminous MANDORLA rendered as hi-res
generative art — two large volumetric arcs of light (volumetric
gold-chrome with holographic chromatic-shift edges going from warm gold
through violet-pink at the rim) overlap across the centre of the page
like two slow moons that have moved into the same celestial latitude.
Where they overlap, the central almond-shaped region glows brightest
— rendered with iridescent gold-into-violet shift, volumetric inner
light.

INSIDE the central overlap, fractal nested smaller mandorlas continue
in self-similar recursion — three to five levels deep, each smaller
mandorla also rendered with volumetric chromatic-shift gold, each
glowing brighter than the one containing it. The fractal recursion
suggests infinite mutual gazing without exhausting the page.

Around the mandorla: a cosmic nebula in teal-into-violet-into-orange,
painterly turbulence with star-cluster flecks. Faint concentric ring
patterns in dim gold orbit the mandorla like a Persian astrolabe's
outer dials. Particle-cloud dust drifts through the foreground.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant F — Hexagonal colimit engine (al-Hayy sibling)
# --------------------------------------------------------------------
PROMPT_F = (
    """A contemporary AI-art book cover for a posthuman philosophy
volume. Portrait, 3:4 aspect ratio.

Sibling visual to the al-Hayy cyborg-yogini cover but with the
network-as-colimit replacing the figure.

Centred on the page: a large rendered HEXAGONAL MOON-OCULUS frame in
bronze-chrome metallic, intricate inset geometric patterning along the
hex edges, glowing softly with warm gold light. Inside the hex frame
where the moon would be: a COLIMIT ENGINE — five luminous bronze-
chrome nodes arranged in a pentagonal arrangement around a central
luminous void. From each node, chrome-light filaments arc inward to
the central void, converging into a brighter luminous core. The
geometry reads as a category-theory pushout diagram rendered as a
shrine.

The hex frame catches strong specular gold light at its upper edges,
softer Fresnel violet-teal at the lower rim. The inside of the hex
glows with the warm interior light of a temple-lantern.

Outside the hex oculus: a wide cosmic nebula in teal-into-violet-into-
orange, painterly turbulence with faint star-clusters and gold-leaf
flecks. Two or three subtle bronze hanging-lantern motifs in the
nebula corners (echoing the al-Hayy cover but smaller, fading toward
abstraction). Particle-cloud dust drifts through the composition.
Faint zodiacal ring traces in dim gold across the lower third.
"""
    + "\n\n"
    + STYLE
)


VARIANTS = [
    ("D_cyborg_constellation", PROMPT_D),
    ("E_holographic_mandorla", PROMPT_E),
    ("F_hex_colimit_engine", PROMPT_F),
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
                "image_config": {"aspect_ratio": "3:4", "image_size": "2K"},
            },
            timeout=300,
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
    print(f"Firing {len(VARIANTS)} v2 cover variants in parallel via {MODEL}...")
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
