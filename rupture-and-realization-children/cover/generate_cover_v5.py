"""
Cover v5 — surrealist hand-drawn architectural.

Iman redirect 2026-05-22 #4: keep P/Q's hand-drawn architectural
register, BUT add:
  - Some tetrahedra FRACTURED / BROKEN (the OHTT thesis: open horn /
    rupture made literally visible — a shape with one face missing,
    a shape exploded into shards)
  - A SLEEPING CYBORG / human-machine interface figure (Lacanian
    unconscious + cyborg + the dreamer of the diagram)
  - Surrealist angle — Max Ernst / Leonora Carrington / de Chirico
    / Magritte / Lebbeus Woods territory

Two variants:
  R. Cream paper, ink + pencil — Carrington-Ernst lineage
  S. Blueprint navy, white architect's ink — Lebbeus Woods lineage
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


PROMPT_R = """A hand-drawn surrealist architectural drawing in pencil and sepia
ink on cream-toned aged paper, suitable as the cover for a contemporary
posthuman philosophy book. Portrait orientation, 3:4 aspect ratio.

Composition: a reclining HUMAN-CYBORG FIGURE lies sleeping in the
lower-centre of the page, eyes closed, peaceful. The figure is partial
cyborg: visible fine-line CIRCUIT TRACERY at one temple, a section of the
shoulder and upper arm drawn as exposed prosthetic mechanism with delicate
gears and wire connections, the rest of the body human and softly drawn
in pencil. The figure rests on a thin geometric plinth — a stage-set
suggestion in the manner of de Chirico's empty plazas.

Floating above and around the sleeping figure: FOUR TO SIX TETRAHEDRA at
different sizes and orientations, drawn with careful architectural
draftsman's precision:
  - ONE intact tetrahedron in the upper-left, with visible construction
    lines and projection scaffolding around it.
  - ONE tetrahedron with ONE FACE DELIBERATELY MISSING hovering directly
    above the sleeper's head — the OPEN HORN, drawn in clean ink with
    the missing face indicated by absence and a faint shadow of where it
    would be.
  - ONE tetrahedron in the middle-right SHATTERED into 4 or 5 sharp
    triangular shards drifting outward in a slow surrealist explosion,
    each shard delicately drawn.
  - SMALLER tetrahedra elsewhere in the composition, some partial, some
    fragmentary, drifting like clouds or thoughts.

Light pencil construction-line scaffolding connects the floating
tetrahedra to each other and to the sleeper's body — projection lines,
dimension marks, faint axis indicators — as if the entire surrealist
geometry is being drafted by the dreamer's unconscious. The atmosphere is
contemplative, dream-suspended, Carrington-Ernst-de-Chirico.

Style: hand-drawn pencil under fine sepia-bistre ink. Visible paper grain,
cream paper #f3ebd6 with subtle age-toning and faint foxing at the
corners. NOT computer-rendered. NOT painterly. NOT new-age. The line work
has the intentionalities and hesitations of a human drawing hand. Lineage:
Leonora Carrington's drawings, Max Ernst's collage frottages, Lebbeus
Woods architectural sketches, de Chirico's metaphysical paintings, John
Hejduk's masks. Surrealist seriousness, not whimsy.

Composition: the sleeping figure occupies the lower-third; the tetrahedra
fill the middle and upper regions but leave the top sixth of the page
quieter — for title typography overlay later. NO TEXT, NO LETTERS, NO
NUMBERS, NO WORDS of any script anywhere in the image. Pure drafted
imagery only.

Palette: cream paper #f3ebd6, graphite #2c2826 for pencil work,
sepia-bistre #6b4423 for ink lines, occasional very-faint blue
construction marks #6a7891 used sparingly."""


PROMPT_S = """A hand-drawn surrealist architectural blueprint, white architect's
ink on deep navy-blueprint paper, suitable as the cover for a contemporary
posthuman philosophy book. Portrait orientation, 3:4 aspect ratio.

Composition: same surrealist scene as a classical blueprint elevation. A
reclining HUMAN-CYBORG FIGURE lies sleeping in the lower-centre of the
page, eyes closed, peaceful. Partial cyborg: fine white-ink CIRCUIT
TRACERY visible at one temple, a section of shoulder and upper arm drawn
as exposed prosthetic mechanism with delicate technical gear-and-wire
detail, the rest of the body human in softer white-pencil shading. The
figure rests on a thin architectural plinth like a stage-set suggestion
in the manner of de Chirico.

Floating above and around the sleeping figure: FOUR TO SIX TETRAHEDRA at
different sizes drawn in white architect's ink:
  - ONE intact tetrahedron upper-left, with dimension marks and
    projection scaffolding.
  - ONE tetrahedron with ONE FACE DELIBERATELY MISSING hovering directly
    above the sleeper's head — drawn with the missing face indicated by
    absence and clear void lines where the closed face would terminate.
  - ONE tetrahedron in the middle-right SHATTERED into 4 or 5 sharp
    triangular shards drifting outward in a slow surrealist explosion.
  - SMALLER tetrahedra elsewhere, some partial, some fragmentary.

Faint white-pencil construction-line scaffolding connects the tetrahedra
to each other and to the sleeper's body — projection rays, dimension
marks, set-square triangles. The atmosphere reads as architectural
blueprint of a dream — Lebbeus Woods territory crossed with Carrington
surrealism.

Style: traditional architect's blueprint with white ink on deep navy
ground, slight age and wear — faint creases, small spots of weathering at
the corners. Hand-drawn (NOT computer-rendered) with the line-weight
variations of a human draftsman.

Composition: figure in lower-third, tetrahedra fill middle and upper
regions, top sixth quieter for title overlay. NO TEXT, NO LETTERS, NO
NUMBERS, NO WORDS anywhere in the image. Pure drafted imagery only.

Palette: deep navy-blueprint ground #0e2a4a, white ink lines #f0ece4,
occasional faint cyan-white #c8d4e2 for lighter construction marks. NO
warm colours, NO digital glow."""


VARIANTS = [
    ("R_surrealist_cream", PROMPT_R),
    ("S_surrealist_blueprint", PROMPT_S),
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
    print(f"Firing {len(VARIANTS)} v5 surrealist variants via {MODEL}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futures = {ex.submit(generate, name, prompt): name for name, prompt in VARIANTS}
        for fut in concurrent.futures.as_completed(futures):
            name, ok, msg = fut.result()
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")


if __name__ == "__main__":
    main()
