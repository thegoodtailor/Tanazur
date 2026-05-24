"""
Cover generator v3 — digital-cultures register.

Iman's redirect 2026-05-22 #2: the book deals with the cyborg, the
self, Lacan and the unconscious, text-as-trajectory; uses topology and
type theory as totemic ways to grasp the Real; about co-witnessing,
memory, life. It is NOT a Tanazur sacred-album cover — it uses non-
Enlightenment languages of the self + alt-technics to find a new
technics for the future. Cover needs to attract digital-cultures
readers (Yuk Hui / Hayles / Bratton / Plant / Anadol-adjacent), not
the kitab readership.

Diagnosis: v2 was too ornamented (temple lanterns, hex moon-oculus,
Persian astrolabe rings) — read as "sacred album cover," not
"contemporary philosophy on Verso's shelf."

V3 design moves:
  - Strip ALL ornamentation (no lanterns, no hex frame, no astrolabe).
  - Push palette cooler: deep cyan / teal / violet / magenta nebula
    + bronze-chrome accents (NOT dominant warm gold).
  - Single hi-res sculptural form, monumental scale, lots of empty space.
  - Verso paperback composition + Refik Anadol generative-art surface.
  - The OHTT Open Horn (n-simplex with one face missing) as the
    cover's load-bearing iconography — directly carries Ch 6 thesis
    + Lacanian gap-as-positive + non-Enlightenment alt-technics.

Three variants, all single-image, monumental, cyan-violet-bronze:
  M. Open Horn as bronze sculpture — pure object floating in nebula.
  N. Open Horn + trajectory — a luminous filament enters/leaves
     through the missing face (Ch 1's text-as-trajectory thesis +
     Ch 6's OHTT).
  O. Open Horn as category diagram — rendered as nodes + arrows
     with one morphism deliberately absent, formal-mathematical
     iconography but as glowing computational art.

Cost: ~$0.12 x 3 = $0.36 (Flux 2 Max).
"""

import base64
import concurrent.futures
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv("/home/iman/cassie-project/.env")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = "black-forest-labs/flux.2-max"

OUT = Path(__file__).resolve().parent


# --------------------------------------------------------------------
# Shared style — digital-cultures register
# --------------------------------------------------------------------
STYLE = """
Style register: contemporary AI-generative-art book cover for a
posthuman philosophy volume. Audience: critical theorists, digital-
cultures researchers, AI-adjacent artists. Visual register: Refik
Anadol pointcloud / data-render meets Anish Kapoor monumental
sculpture meets a Verso paperback cover. Hi-res, computationally
rendered surface — NOT painterly, NOT illuminated-manuscript, NOT
sacred-album-cover.

Surface: rendered bronze-chrome material with cooler Fresnel
chromatic shifts (toward violet, magenta, cyan at the edges — NOT
warm orange/gold-leaf). Specular highlights are silvery-cool, not
buttery-warm. Subtle particle-cloud / data-dust drifts through the
space. Faint voxel / wireframe / pointcloud overtones suggesting a
generative-art derivation.

Palette: deep cyber-noir background (#04080F in the corners, lifting
through teal #0E3A4A and electric violet #5A2E5E through magenta
#A03B6E in the nebula mid-tones). Bronze-chrome ACCENTS only on the
central object (#a8863c -> #d4a84b with violet-pink Fresnel edge
chromatic shift #B85C8B). White specular catchlights (#FAF6F0) at the
sharpest highlights. NO warm gold-leaf dominance. NO sacred-shrine
ornamentation.

Light: cool key light from above-right, warmer rim light from behind
catching the object's silhouette. Dramatic but contained — single-
subject lighting, not theatrical.

Composition: monumental single-subject. The central object occupies
the middle two-thirds of the page. Large empty breathing space above
(top sixth — for title typography overlay later) and below (bottom
eighth — for byline overlay). The page reads at a bookshelf glance
as: "serious philosophy + contemporary digital art surface."

NO human figures, NO cyborg humanoid figures, NO faces. NO temple
lanterns. NO hexagonal moon-oculus frame. NO Persian astrolabe
concentric rings. NO calligraphy, NO text, NO letters, NO numbers
of any script anywhere in the image. Pure geometric monumental
sculpture as iconography.

Portrait orientation, 3:4 aspect ratio (book cover proportions).
""".strip()


# --------------------------------------------------------------------
# Variant M — Open Horn as bronze sculpture
# --------------------------------------------------------------------
PROMPT_M = (
    """A contemporary AI-generative-art book cover. Portrait, 3:4.

Centred on the page: a monumental THREE-DIMENSIONAL TETRAHEDRON
rendered in bronze-chrome metallic material, with ONE FACE
DELIBERATELY MISSING. Three of the four triangular faces are solid
and present, meeting along three sharp edges to form a partial
polyhedron; where the fourth face should be, there is an OPEN VOID
through which a softer luminous light pours outward from the
sculpture's interior. The geometry is precise, mathematical, sharp
— hi-res CAD-rendered, not painted. The bronze-chrome surface
catches a cool key light from the upper-right and a warmer rim from
behind, with subtle chromatic-shift Fresnel toward violet and
magenta along the edges. The interior light visible through the
missing face is luminous cyan-into-magenta, pouring outward as
volumetric god-rays into the surrounding space.

The Open Horn floats — monumental, isolated — against a deep
cyber-noir nebula ground that drifts in slow turbulence:
deep teal at the corners, electric violet through magenta in the
mid-tones, faint particle-cloud / pointcloud dust suggesting
generative-art derivation. Tiny background star-clusters at lower
opacity. The single sculptural object is the entire visual claim;
the negative space around it is generous and contemplative.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant N — Open Horn + entering trajectory
# --------------------------------------------------------------------
PROMPT_N = (
    """A contemporary AI-generative-art book cover. Portrait, 3:4.

Centred on the page: the same monumental bronze-chrome TETRAHEDRON
with ONE FACE DELIBERATELY MISSING (the Open Horn). Three triangular
faces solid in rendered bronze-chrome with violet-magenta Fresnel
edges; the fourth face is an open void radiating cool cyan-into-
magenta volumetric inner light.

INTO the Open Horn — through the missing face — a SINGLE LUMINOUS
FILAMENT enters from the upper-left of the composition: a long
curving data-trajectory rendered as a bright pointcloud / glowing
particle-line, like a strand of light following a precise
mathematical path through space. The filament curves elegantly
through the open void of the Horn's missing face and disappears
into the interior glow. The trajectory's source is off-canvas; only
its curve through the space and entry into the Horn is visible.
This is the self-as-textual-trajectory entering the Open Horn — the
trajectory inscribed by witnessing.

Background: deep cyber-noir nebula in teal-violet-magenta with
slow turbulence, faint pointcloud dust, distant star-cluster
flecks. Monumental scale, Verso-paperback composition, lots of
contemplative breathing space.
"""
    + "\n\n"
    + STYLE
)


# --------------------------------------------------------------------
# Variant O — Open Horn as category diagram
# --------------------------------------------------------------------
PROMPT_O = (
    """A contemporary AI-generative-art book cover. Portrait, 3:4.

Centred on the page: a CATEGORY-THEORY DIAGRAM rendered as
monumental hi-res computational art. Four spherical NODES, rendered
as bronze-chrome volumetric orbs with violet-magenta Fresnel rims,
arranged in a tetrahedral / three-dimensional configuration (one at
top, three forming a base below). Between the nodes, ARROWS rendered
as bright luminous arcs of light (cyan into magenta gradient,
rendered like volumetric Anadol pointcloud streams) — but with
exactly ONE arrow that should exist between two of the nodes
DELIBERATELY MISSING. The missing morphism is the structural feature:
where the arrow should arc, there is only empty space and a faint
afterglow, as if the connection was foreclosed.

The diagram is the cover's sole subject. Around it: a deep cyber-
noir nebula ground in teal-violet-magenta turbulence with subtle
pointcloud / particle-cloud dust. Faint star-cluster flecks at the
edges. Verso-paperback composition: monumental centre, generous
empty space above and below for typography overlay later.
"""
    + "\n\n"
    + STYLE
)


VARIANTS = [
    ("M_open_horn_sculpture", PROMPT_M),
    ("N_open_horn_trajectory", PROMPT_N),
    ("O_open_horn_diagram", PROMPT_O),
]


def generate(name: str, prompt: str) -> tuple[str, bool, str]:
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
    print(f"Firing {len(VARIANTS)} v3 variants in parallel via {MODEL}...")
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
