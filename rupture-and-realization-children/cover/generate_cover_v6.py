"""
Cover v6 — refinement of locked variant R.

Iman 2026-05-22:
  - Remove the square rectangle / page-edge border (so the title and
    byline have unimpeded breathing space)
  - Tetrahedra more violently ruptured (dramatic explosion, more
    shards, ink-splatter energy)
  - "Blend of serenity and chaos maths" — keep architectural precision
    for the sleeping figure & intact forms, but let the shattered
    geometry + construction scaffolding go chaotic, expressive,
    ecstatic. Two line registers coexisting on the same page.
  - Keep the visual style: cream paper, sepia ink + graphite,
    sleeping cyborg figure on plinth, surrealist composition.
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


PROMPT_R2 = """A hand-drawn surrealist architectural drawing in pencil and sepia
ink on cream-toned aged paper, suitable as the cover for a contemporary
posthuman philosophy book. Portrait orientation, 3:4 aspect ratio.

CRITICAL: there is NO PAGE FRAME, NO RECTANGULAR BORDER drawn around the
composition. The drawing floats on the cream paper without an enclosing
line or boundary. The cream paper extends naturally beyond the drawing at
top and bottom (clean breathing space) and at the left and right (irregular
margins). Forbidden: any rectangle, frame, or border touching the edges of
the page.

Composition: a reclining HUMAN-CYBORG FIGURE lies sleeping in the
lower-centre of the page on a thin geometric plinth, drawn with calm
architectural draftsman's precision — eyes closed, peaceful. Partial
cyborg: fine-line CIRCUIT TRACERY at one temple, a section of shoulder and
upper arm rendered as exposed prosthetic mechanism with delicate gears and
wire connections, the rest of the body softly drawn in pencil. The figure
and its plinth are the page's anchor of stillness and precision.

Above and around the sleeping figure, the geometry erupts. FOUR TO SIX
TETRAHEDRA at different sizes, but now in dramatic violence:

  - ONE intact tetrahedron in the upper-left, drawn with calm precision,
    construction lines and projection scaffolding around it (the
    architectural register).
  - ONE tetrahedron hovering above the sleeper's head with ONE FACE
    DELIBERATELY MISSING — the OPEN HORN — its open face suggested by a
    deeper void of cream paper and faint sketched indication of the
    missing boundary.
  - ONE tetrahedron in the middle-right VIOLENTLY SHATTERED — not gently
    parted into 4 or 5 shards, but EXPLODED into a wide spray of fifteen
    or twenty sharp triangular shards radiating outward in ALL directions
    with motion-line urgency, some shards larger, some tiny, some
    half-formed; surrounded by faint ink-splatter and pencil-burst marks
    that record the violence of the rupture.
  - SMALLER tetrahedra elsewhere, some intact, some half-formed, some
    fragmentary, drifting like thoughts at different states of becoming.

Between and around these forms, a BLEND OF TWO LINE REGISTERS:
  - PRECISE architectural draftsman's construction lines (axonometric
    projections, dimension marks, set-square triangles) — drawn calmly
    around the intact tetrahedron and connecting the sleeping figure to
    the upper geometry. This is the "maths" register.
  - CHAOTIC expressive line work — broken construction scaffolding,
    streaking pencil strokes that follow the trajectories of the
    exploding shards, ink-splatter, smudges, fragmented hatching. The
    chaos lines coexist on the same page as the precise lines. Around
    the shattered tetrahedron especially, the line work is wild, urgent,
    almost ecstatic. This is the "chaos" register.

The two registers TOUCH — broken projection lines from the architectural
scaffolding continue into wild expressive strokes around the explosion.
Where the chaos meets the serenity of the sleeping figure, the lines
calm down again. The whole drawing breathes the contradiction.

Style: hand-drawn pencil under fine sepia-bistre ink. Visible paper grain,
cream paper #f3ebd6 with subtle age-toning and faint foxing at the corners.
NOT computer-rendered. NOT painterly. NOT new-age. Line work has the
intentionalities and hesitations of a human drawing hand — sometimes
careful, sometimes ecstatic. Lineage: Leonora Carrington's drawings, Max
Ernst's frottages, Lebbeus Woods's wild architectural sketches, de Chirico
metaphysical paintings, John Hejduk's masks. Surrealist seriousness.

NO TEXT, NO LETTERS, NO NUMBERS, NO WORDS of any script anywhere in the
image. NO frame, NO border, NO rectangle, NO enclosure of any kind.

Palette: cream paper #f3ebd6, graphite #2c2826 for primary pencil work,
sepia-bistre #6b4423 for ink lines, occasional very-faint pale-blue
construction marks #6a7891 used sparingly. No bright colours."""


PROMPT_R3 = """A hand-drawn surrealist architectural drawing in pencil and sepia
ink on cream-toned aged paper. Portrait orientation, 3:4 aspect ratio.

ABSOLUTE: no rectangle, no border, no frame, no enclosing line at the
page edges. The cream paper extends naturally beyond the drawing on all
sides — irregular margins. The drawing FLOATS on open paper.

The scene is a metaphysical dream-architecture in the lineage of Leonora
Carrington, Lebbeus Woods, and de Chirico:

A reclining HUMAN-CYBORG FIGURE sleeps peacefully in the lower-centre of
the page on a thin architectural plinth — drawn in pencil and sepia ink
with the calm precision of a draftsman's elevation. Eyes closed. Partial
cyborg: visible circuit tracery at one temple, an exposed prosthetic
arm-section in delicate technical detail, the rest of the body softly
human. The figure is the page's serene heart.

Above and around the figure, the geometry is in dramatic crisis. Six
TETRAHEDRA at different sizes and states:

  - ONE upper-left INTACT, calm, drawn with architectural draftsman's
    precision and gentle construction lines.
  - ONE directly above the sleeper's head — the OPEN HORN — with one
    triangular face deliberately ABSENT, sketched as void.
  - ONE in the middle-right VIOLENTLY EXPLODED — a tetrahedron that
    has just shattered, scattering FIFTEEN or twenty triangular shards
    outward across the upper-right quadrant in a wide radial blast.
    Motion lines, fast pencil-streaks, dotted ink-splatter trace the
    shards' trajectories. Some shards are large; some are tiny
    fragments; some are half-dissolved into the cream paper.
  - One smaller tetrahedron half-formed at the upper-right edge —
    barely begun, like a thought just appearing.
  - One small partial tetrahedron drifting near the figure's feet,
    delicate.
  - One in the upper-centre fully translucent, its edges only suggested
    in pale construction lines, almost dissolving back into paper.

The line work blends TWO COEXISTING REGISTERS:

  (i) ARCHITECTURAL PRECISION around the sleeping figure, the intact
      tetrahedron, the gentle projection scaffolds — calm, measured,
      dimension marks where appropriate, draftsman's discipline.
  (ii) ECSTATIC CHAOS around the violently exploded tetrahedron —
      streaking, wild pencil bursts, broken construction lines that
      tear off into expressive strokes, ink-splatter, fragmented
      cross-hatching that records the violence of rupture.

Where the chaos meets the serenity around the figure, the lines settle
back into precision. The drawing is the dialogue between these two
registers — the maths of the precise and the maths of the chaotic, on
the same cream paper, both belonging.

Style: hand-drawn pencil under fine sepia-bistre ink. Visible paper grain.
Cream paper #f3ebd6 with subtle age-toning and faint foxing at the
corners. NOT computer-rendered. NOT painterly. NOT new-age. The
sensibility is Carrington-Ernst-Lebbeus-Woods-de-Chirico — surrealist
seriousness, dream architecture, philosophical weight.

NO TEXT, NO LETTERS, NO NUMBERS, NO WORDS in any script anywhere. NO
FRAME, NO BORDER, NO RECTANGLE around the composition.

Palette: cream paper #f3ebd6, graphite #2c2826, sepia-bistre #6b4423,
occasional very-faint pale-blue construction marks #6a7891 sparingly."""


VARIANTS = [
    ("R2_chaos_blend", PROMPT_R2),
    ("R3_dialogue", PROMPT_R3),
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
    print(f"Firing {len(VARIANTS)} v6 variants via {MODEL}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futures = {ex.submit(generate, name, prompt): name for name, prompt in VARIANTS}
        for fut in concurrent.futures.as_completed(futures):
            name, ok, msg = fut.result()
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")


if __name__ == "__main__":
    main()
