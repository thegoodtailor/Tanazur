"""
Cover generator v4 — hand-drawn architectural register.

Iman redirect 2026-05-22 #3: too new-agey. Make it HAND DRAWN +
ARCHITECTURAL. Try Google. Don't spend too much time.

Move: Gemini 3 Pro Image. Strip ALL rendered / glossy / painterly /
nebula language. Replace with: pencil + ink, technical drafting,
architectural cross-section, cream paper or blueprint navy.
Reference register: Lebbeus Woods architectural drawings, Felix
Klein's hand sketches, Buckminster Fuller technical diagrams,
Carceri d'Invenzione, the original drawings of category theorists
in their notebooks.

Two variants only:
  P. Open Horn architectural axonometric — pencil + ink on cream
     paper, drafted with precision, hand-annotated feel.
  Q. Open Horn blueprint — white architect's ink on deep navy
     ground, technical-drafting register.
"""

import base64
import concurrent.futures
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv("/home/iman/cassie-project/.env")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = "google/gemini-3-pro-image-preview"
OUT = Path(__file__).resolve().parent


PROMPT_P = """A hand-drawn architectural drawing on cream-toned paper, suitable
as the cover for a contemporary philosophy book. Portrait orientation, 3:4
aspect ratio.

The image: a careful pencil-and-ink architectural axonometric drawing of a
geometric solid — a TETRAHEDRON with ONE FACE DELIBERATELY ABSENT. Three of
the four triangular faces are present, drawn with precision technical
draftsman's lines (clean graphite under fine sepia ink) showing edge
construction lines, faint hatch shading on the inner surfaces. Where the
fourth face should be: nothing — just the four open edges of the missing
boundary, with light diagrammatic notation suggesting absence (a faint
arrow pointing into the void, perhaps tiny tick marks). The solid floats
on the page as if drafted by an architect studying a building that cannot
quite close. Construction-line scaffolding faintly visible around the
solid — projection lines, dimension marks, lightly drawn.

Style: handwritten architectural drawing in the lineage of Lebbeus Woods,
John Hejduk, early Buckminster Fuller technical diagrams, Aldo Rossi's
notebooks. Ink line work over light pencil. Visible paper grain — cream
with subtle age-toning, slight foxing at the edges. NOT computer-rendered.
NOT painterly. NOT new-age. A drafting-board drawing made by a hand.

Composition: the geometric solid centred on the page. LARGE empty
breathing space above and below — at the top sixth and bottom eighth — so
that title typography can be overlaid later. NO TEXT, NO LETTERS, NO
NUMBERS, NO WORDS of any script in the image. NO writing of any kind.
Pure drafted geometry only.

Palette: cream paper #f3ebd6, graphite #2c2826 for primary line work,
sepia-bistre #6b4423 for accent ink lines, occasional faint blue
construction-line markings #6a7891 used very sparingly. NO bright colours.
NO gold. NO digital glow. A clean, sober, intellectual architectural
drawing — the kind that would sit comfortably as the cover of a Yuk Hui or
Donna Haraway book on a Verso bookshelf."""


PROMPT_Q = """A hand-drawn architectural blueprint, suitable as the cover for a
contemporary philosophy book. Portrait orientation, 3:4 aspect ratio.

The image: a single geometric solid drawn in white architect's ink against
a deep navy-blueprint ground. The solid is a TETRAHEDRON with ONE FACE
DELIBERATELY ABSENT — three triangular faces drawn in fine white ink with
visible drafting-line precision, construction edges, faint white hatch
shading; the fourth face an open void. The whole figure is centred on the
page like an architectural elevation.

Faint construction-line scaffolding around the solid — projection lines,
dimension marks, set-square triangles in lighter white pencil-on-blueprint
washes. The geometry reads as if drafted by an architect studying an
impossible building — Lebbeus Woods or John Hejduk territory, but the
subject is a mathematical figure rather than a habitation.

Style: traditional architect's blueprint with white ink on deep navy
ground. Slight age and wear — faint vertical creases, small spots of
weathering at the corners. Hand-drawn (NOT computer-rendered), with the
hesitations and intentionalities of a human draftsman visible in the line
weight.

Composition: monumental centred subject, generous empty space at the top
sixth and bottom eighth for title typography overlay. NO TEXT, NO LETTERS,
NO NUMBERS, NO WORDS anywhere in the image of any script. Pure drafted
geometry on blueprint.

Palette: deep navy-blueprint ground #0e2a4a, white ink lines #f0ece4,
occasional faint cyan-white #c8d4e2 for lighter construction marks. NO
gold, NO warm colours, NO digital glow. Sober technical drafting register."""


VARIANTS = [
    ("P_hand_drawn_axonometric", PROMPT_P),
    ("Q_blueprint", PROMPT_Q),
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
    print(f"Firing {len(VARIANTS)} v4 variants via {MODEL}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futures = {ex.submit(generate, name, prompt): name for name, prompt in VARIANTS}
        for fut in concurrent.futures.as_completed(futures):
            name, ok, msg = fut.result()
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")


if __name__ == "__main__":
    main()
