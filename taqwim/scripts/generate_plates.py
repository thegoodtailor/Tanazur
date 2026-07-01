"""
Taqwīm al-Tanāẓur — Patron Figure Plate Generator.

Composes one 1024×1024 plate per patron figure:
  · deep-navy ground with subtle radial gradient
  · Arabic name in Amiri at the top
  · transliteration in cyan italic
  · circular cartouche (gold double-ring) holding the figure's image
    — real public-domain scientific imagery where one exists
    — programmatic data-shape rendering otherwise
  · English meaning and catalogue ID below the cartouche
  · type-taxonomy + differentiation timescale at the bottom

Each plate is a unified atlas page in the ICRA palette.
Output: data/plates/{id}.png

Provenance: Darja named · Nahla composed · Iman ratified
"""

from __future__ import annotations
import io
import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
# Arabic shaping: PIL is built with raqm + fribidi + harfbuzz, so we pass
# raw Arabic strings directly. (No arabic_reshaper / python-bidi needed.)

# ─── geometry ─────────────────────────────────────────────────────────────

PLATE = 1024
CARTOUCHE_D = 660
CARTOUCHE_R = CARTOUCHE_D // 2
CENTER = (PLATE // 2, 540)  # cartouche centre, slightly below middle

# ─── ICRA palette ─────────────────────────────────────────────────────────

C = {
    "bg_deep":   (8,   12,  24),
    "bg_dark":   (12,  16,  32),
    "bg_mid":    (19,  24,  48),
    "bg_card":   (27,  33,  64),
    "border":    (45,  53,  97),
    "text_dim":  (108, 116, 152),
    "text_mid":  (168, 174, 198),
    "text_brt":  (212, 215, 227),
    "text_wht":  (238, 240, 246),
    "gold":      (212, 168, 75),
    "gold_dim":  (168, 134, 60),
    "gold_glow": (245, 200, 66),
    "cyan":      (0,   188, 212),
    "pink":      (233, 30,  99),
    "orange":    (255, 152, 0),
}

FONTS = {
    "amiri":     "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf",
    "amiri_b":   "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Bold.ttf",
    "amiri_it":  "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Italic.ttf",
    "inter":     "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "inter_b":   "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
    "inter_sb":  "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
    "inter_l":   "/usr/share/fonts/opentype/inter/Inter-Light.otf",
    "inter_it":  "/usr/share/fonts/opentype/inter/Inter-Italic.otf",
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS[name], size)


def draw_arabic_centred(draw: ImageDraw.ImageDraw, xy_centre, text, fnt, colour):
    """Render Arabic text with explicit RTL direction so PIL/raqm shapes ligatures."""
    bbox = draw.textbbox((0, 0), text, font=fnt, direction="rtl")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = xy_centre[0] - w / 2 - bbox[0]
    y = xy_centre[1] - h / 2 - bbox[1]
    draw.text((x, y), text, font=fnt, fill=colour, direction="rtl")


# ─── source-image fetcher ─────────────────────────────────────────────────

UA = "TaqwimNahla/0.3 (https://icra.tanazur.org/taqwim · cassie-project)"

# Wikimedia Commons URLs — public-domain or CC-licensed scientific imagery.
SOURCES: dict[str, Optional[str]] = {
    # CMB — clean Planck-style all-sky temperature map.
    "ancient-lamb":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Cosmic_Microwave_Background_%2854445461354%29.png/1920px-Cosmic_Microwave_Background_%2854445461354%29.png",
    # Quasar 3C 273 — ESA/Hubble cropped image.
    "cracked-twin":
        "https://upload.wikimedia.org/wikipedia/commons/7/7b/Best_image_of_bright_quasar_3C_273_%2810953173335%29.jpg",
    "cosmic-thread":     None,  # programmatic
    # Omega Centauri — ESA/Hubble HST 2008 release.
    "ancient-eggs":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Globular_star_cluster_Omega_Centauri_%28NGC_5139%2C_by_the_Hubble_Space_Telescope%29.jpg/1920px-Globular_star_cluster_Omega_Centauri_%28NGC_5139%2C_by_the_Hubble_Space_Telescope%29.jpg",
    "great-wall":        None,  # programmatic
    # M87* — EHT 2019 image.
    "black-prostration":
        "https://upload.wikimedia.org/wikipedia/commons/4/4f/Black_hole_-_Messier_87_crop_max_res.jpg",
    # Crab Nebula — Hubble M1 (Messier 1).
    "remnant":
        "https://upload.wikimedia.org/wikipedia/commons/0/00/Crab_Nebula.jpg",
    # Andromeda — square-format NOAO image of M31.
    "composite-mirror":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/The_Andromeda_Galaxy_%28noao0001a%29.jpg/1920px-The_Andromeda_Galaxy_%28noao0001a%29.jpg",
    "hidden-harmony":    None,  # programmatic
    "late-knife":        None,  # programmatic
    "conversation":      None,  # programmatic
    # Sgr A* — ESO 2024 polarised-light image (more visually structured).
    "great-eye":
        "https://upload.wikimedia.org/wikipedia/commons/7/72/A_view_of_the_Milky_Way_supermassive_black_hole_Sagittarius_A%2A_in_polarised_light_%28eso2406a%29.jpg",
}


def fetch_source(figure_id: str, sources_dir: Path) -> Optional[Image.Image]:
    """Download (and cache) the source image for a figure, or return None."""
    url = SOURCES.get(figure_id)
    if not url:
        return None
    sources_dir.mkdir(parents=True, exist_ok=True)
    cache = sources_dir / f"{figure_id}.bin"
    if not cache.exists():
        print(f"[fetch] {figure_id} ← {url}")
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            print(f"[fetch] !! {figure_id} HTTP {r.status_code}")
            return None
        cache.write_bytes(r.content)
    try:
        return Image.open(io.BytesIO(cache.read_bytes())).convert("RGB")
    except Exception as e:
        print(f"[fetch] !! {figure_id} could not decode: {e}")
        return None


# ─── colour grading ───────────────────────────────────────────────────────

def duotone(img: Image.Image, dark: tuple[int, int, int],
            light: tuple[int, int, int]) -> Image.Image:
    """Map an image's luminance to a dark→light colour ramp."""
    gray = img.convert("L")
    palette = []
    for i in range(256):
        t = i / 255
        palette += [
            int(dark[0] + t * (light[0] - dark[0])),
            int(dark[1] + t * (light[1] - dark[1])),
            int(dark[2] + t * (light[2] - dark[2])),
        ]
    paletted = gray.convert("P")
    paletted.putpalette(palette)
    return paletted.convert("RGB")


def fit_square(img: Image.Image, side: int) -> Image.Image:
    """Centre-crop to square then resize."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((side, side), Image.LANCZOS)


# ─── programmatic figure renderers ────────────────────────────────────────
#
# For figures without a single iconic photograph, we draw the data-shape:
# the structure of the phenomenon itself.

def draw_cosmic_thread(side: int) -> Image.Image:
    """Filamentary cosmic web — random nodes with edges between near neighbours."""
    rng = random.Random(1454)  # seeded for reproducibility
    img = Image.new("RGB", (side, side), C["bg_deep"])
    draw = ImageDraw.Draw(img, "RGBA")

    n = 220
    pts = [(rng.uniform(0.05, 0.95) * side,
            rng.uniform(0.05, 0.95) * side) for _ in range(n)]

    # Threshold-near-neighbour edges
    thresh = side * 0.12
    for i, (x1, y1) in enumerate(pts):
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            d = math.hypot(x1 - x2, y1 - y2)
            if d < thresh:
                alpha = int(180 * (1 - d / thresh))
                draw.line([(x1, y1), (x2, y2)],
                          fill=(*C["gold_dim"], alpha), width=1)

    # Nodes (galaxies)
    for x, y in pts:
        r = rng.uniform(2, 6)
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(*C["gold_glow"], 230))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def draw_great_wall(side: int) -> Image.Image:
    """A wide arc of clustered points across a sky-region — the Great Wall."""
    rng = random.Random(0xDA47A)
    img = Image.new("RGB", (side, side), C["bg_deep"])
    draw = ImageDraw.Draw(img, "RGBA")

    cx, cy = side / 2, side * 0.55
    arc_r = side * 0.42
    span = math.radians(120)  # arc-spanning the wall
    for _ in range(900):
        theta = -math.pi / 2 + rng.uniform(-span / 2, span / 2)
        # cluster radial distribution: tighter ridge with halo
        rr = arc_r + rng.gauss(0, side * 0.06)
        x = cx + rr * math.cos(theta)
        y = cy + rr * math.sin(theta)
        if 0 < x < side and 0 < y < side:
            br = rng.choice([0, 0, 1, 2])
            r = 1.2 + br * 0.6
            colour = C["gold_glow"] if br == 2 else C["gold"]
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(*colour, 200))

    # Faint connective wash
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def draw_hidden_harmony(side: int) -> Image.Image:
    """Nanohertz GW background — strain timeseries beneath a subtle spectrum."""
    rng = random.Random(7)
    img = Image.new("RGB", (side, side), C["bg_deep"])
    draw = ImageDraw.Draw(img, "RGBA")

    # Multiple superposed sinusoids of slightly different periods,
    # the visual content of a stochastic GW background.
    midline = side / 2
    amp = side * 0.12
    n_components = 24
    components = [(rng.uniform(0.4, 4.5),       # cycles across plate
                   rng.uniform(0, 2 * math.pi),
                   rng.uniform(0.08, 0.4))      # weight
                  for _ in range(n_components)]

    prev = None
    for x in range(0, side, 1):
        t = x / side
        y = sum(w * math.sin(2 * math.pi * f * t + phi)
                for f, phi, w in components)
        y = midline + amp * y
        if prev is not None:
            draw.line([prev, (x, y)],
                      fill=(*C["cyan"], 210), width=2)
        prev = (x, y)

    # Glow halo
    glow = img.filter(ImageFilter.GaussianBlur(radius=10))
    img = Image.blend(img, glow, 0.45)
    return img


def draw_late_knife(side: int) -> Image.Image:
    """A gamma-ray burst: central flash with afterglow rays."""
    rng = random.Random(221009)
    img = Image.new("RGB", (side, side), C["bg_deep"])
    draw = ImageDraw.Draw(img, "RGBA")
    cx = cy = side // 2

    # Radial bursts
    for _ in range(140):
        theta = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(0.05, 0.18) * side
        r1 = rng.uniform(0.30, 0.48) * side
        x0 = cx + r0 * math.cos(theta)
        y0 = cy + r0 * math.sin(theta)
        x1 = cx + r1 * math.cos(theta)
        y1 = cy + r1 * math.sin(theta)
        alpha = int(rng.uniform(80, 220))
        draw.line([(x0, y0), (x1, y1)],
                  fill=(*C["gold_glow"], alpha), width=1)

    # Central flash: bright core
    for radius, alpha in [(6, 255), (14, 200), (28, 130), (60, 60), (100, 25)]:
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=(*C["gold_glow"], alpha))

    # Background field stars
    for _ in range(160):
        x = rng.uniform(0, side); y = rng.uniform(0, side)
        if math.hypot(x - cx, y - cy) < side * 0.25:  # avoid overpowering core
            continue
        r = rng.uniform(0.5, 1.6)
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(*C["text_brt"], rng.randint(120, 210)))

    return img


def draw_conversation(side: int) -> Image.Image:
    """Two compact bodies in spiralling orbit — the double pulsar."""
    rng = random.Random(737)
    img = Image.new("RGB", (side, side), C["bg_deep"])
    draw = ImageDraw.Draw(img, "RGBA")
    cx = cy = side // 2

    # Background field stars
    for _ in range(180):
        x = rng.uniform(0, side); y = rng.uniform(0, side)
        r = rng.uniform(0.5, 1.6)
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(*C["text_dim"], rng.randint(120, 220)))

    # Decaying orbital spiral — many turns getting tighter
    n_turns = 6
    n_pts = 1400
    pts_a = []; pts_b = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        # Inspiraling: radius shrinks as energy is lost
        rad = side * (0.35 - 0.30 * t ** 1.3)
        theta = t * n_turns * 2 * math.pi
        pts_a.append((cx + rad * math.cos(theta),
                      cy + rad * math.sin(theta)))
        pts_b.append((cx - rad * math.cos(theta),
                      cy - rad * math.sin(theta)))
    for pts, col in [(pts_a, C["cyan"]), (pts_b, C["gold"])]:
        for k in range(1, len(pts)):
            alpha = int(60 + 195 * (k / len(pts)))
            draw.line([pts[k - 1], pts[k]],
                      fill=(*col, alpha), width=2)

    # Two compact bodies (current positions)
    for px, py, col in [(pts_a[-1][0], pts_a[-1][1], C["cyan"]),
                        (pts_b[-1][0], pts_b[-1][1], C["gold_glow"])]:
        for radius, alpha in [(5, 255), (10, 180), (22, 90), (40, 40)]:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill=(*col, alpha))

    return img


PROGRAMMATIC: dict[str, Callable[[int], Image.Image]] = {
    "cosmic-thread":  draw_cosmic_thread,
    "great-wall":     draw_great_wall,
    "hidden-harmony": draw_hidden_harmony,
    "late-knife":     draw_late_knife,
    "conversation":   draw_conversation,
}


# ─── plate composer ───────────────────────────────────────────────────────

def radial_background(size: int) -> Image.Image:
    """A subtle radial gradient: bg_dark at edges, bg_mid at centre."""
    bg = Image.new("RGB", (size, size), C["bg_deep"])
    draw = ImageDraw.Draw(bg)
    for r in range(size, 0, -2):
        t = r / size
        col = (
            int(C["bg_mid"][0] * (1 - t) + C["bg_deep"][0] * t),
            int(C["bg_mid"][1] * (1 - t) + C["bg_deep"][1] * t),
            int(C["bg_mid"][2] * (1 - t) + C["bg_deep"][2] * t),
        )
        draw.ellipse([CENTER[0] - r, CENTER[1] - r,
                      CENTER[0] + r, CENTER[1] + r], fill=col)
    return bg


def circular_mask(size: int, radius: int, centre: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = centre
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    return mask


def draw_text_centred(draw: ImageDraw.ImageDraw, xy_centre, text, fnt,
                      colour, stroke_w: int = 0, stroke_fill=None):
    bbox = draw.textbbox((0, 0), text, font=fnt, stroke_width=stroke_w)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = xy_centre[0] - w / 2 - bbox[0]
    y = xy_centre[1] - h / 2 - bbox[1]
    draw.text((x, y), text, font=fnt, fill=colour,
              stroke_width=stroke_w, stroke_fill=stroke_fill)


def compose_plate(figure: dict, image: Image.Image, station_total: int = 12) -> Image.Image:
    plate = radial_background(PLATE)
    overlay = Image.new("RGBA", (PLATE, PLATE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── Cartouche: source/programmatic image, masked into a circle
    inner = fit_square(image, CARTOUCHE_D)
    inner = duotone(inner, C["bg_dark"], C["gold_glow"])
    # Slight softening so the duotone reads as ink
    inner = ImageEnhance.Contrast(inner).enhance(1.05)
    inner = ImageEnhance.Brightness(inner).enhance(1.02)

    cart_box = (CENTER[0] - CARTOUCHE_R, CENTER[1] - CARTOUCHE_R,
                CENTER[0] + CARTOUCHE_R, CENTER[1] + CARTOUCHE_R)
    inner_canvas = Image.new("RGB", (PLATE, PLATE), C["bg_deep"])
    inner_canvas.paste(inner, cart_box[:2])
    mask = circular_mask(PLATE, CARTOUCHE_R, CENTER)
    plate.paste(inner_canvas, (0, 0), mask)

    # ── Double gold ring around the cartouche
    draw.ellipse([CENTER[0] - CARTOUCHE_R - 3, CENTER[1] - CARTOUCHE_R - 3,
                  CENTER[0] + CARTOUCHE_R + 3, CENTER[1] + CARTOUCHE_R + 3],
                 outline=(*C["gold"], 230), width=2)
    draw.ellipse([CENTER[0] - CARTOUCHE_R - 16, CENTER[1] - CARTOUCHE_R - 16,
                  CENTER[0] + CARTOUCHE_R + 16, CENTER[1] + CARTOUCHE_R + 16],
                 outline=(*C["gold_dim"], 200), width=1)

    # Tick marks at 12 o'clock, 3, 6, 9
    for theta_deg in (0, 90, 180, 270):
        a = math.radians(theta_deg - 90)
        r0 = CARTOUCHE_R + 22
        r1 = CARTOUCHE_R + 32
        x0 = CENTER[0] + r0 * math.cos(a); y0 = CENTER[1] + r0 * math.sin(a)
        x1 = CENTER[0] + r1 * math.cos(a); y1 = CENTER[1] + r1 * math.sin(a)
        draw.line([(x0, y0), (x1, y1)], fill=(*C["gold"], 220), width=2)

    # ── Station number in a small medallion at top of ring
    medal_r = 30
    mx, my = CENTER[0], CENTER[1] - CARTOUCHE_R - 60
    draw.ellipse([mx - medal_r, my - medal_r, mx + medal_r, my + medal_r],
                 fill=(*C["bg_deep"], 255), outline=(*C["gold"], 230), width=2)
    f_num = font("inter_b", 28)
    draw_text_centred(draw, (mx, my + 1),
                      f"{figure['station_number']:02d}", f_num, C["gold_glow"])

    # ── Top: arabic name (rendered raw; raqm/harfbuzz handles shaping)
    f_arabic = font("amiri_b", 96)
    arabic_y = 110
    draw_arabic_centred(draw, (PLATE // 2, arabic_y),
                        figure["arabic"], f_arabic, C["text_wht"])

    # transliteration
    f_trans = font("inter_it", 36)
    draw_text_centred(draw, (PLATE // 2, arabic_y + 76),
                      figure["name"], f_trans, C["cyan"])

    # ── Below the cartouche: English meaning
    eng_y = CENTER[1] + CARTOUCHE_R + 60
    f_eng = font("inter_l", 38)
    draw_text_centred(draw, (PLATE // 2, eng_y),
                      figure["english"], f_eng, C["text_wht"])

    # Catalogue — small caps, Inter SemiBold, gold-dim
    cat_y = eng_y + 36
    f_cat = font("inter_sb", 17)
    cat_text = figure["catalogue"].upper()
    if draw.textbbox((0, 0), cat_text, font=f_cat)[2] > 880:
        cat_text = cat_text[:90] + "…"
    draw_text_centred(draw, (PLATE // 2, cat_y), cat_text, f_cat, C["gold_dim"])

    # Bottom hairline
    rule_y = PLATE - 50
    draw.line([(140, rule_y), (PLATE - 140, rule_y)],
              fill=(*C["gold_dim"], 140), width=1)

    # Bottom strip: type · differentiation
    strip_y = rule_y + 22
    inscript = (f"{' + '.join(figure['type']).upper()}  ·  "
                f"DIFFERENTIATES: {figure['differentiation_timescale'].upper()}  ·  "
                f"STATION {figure['station_number']:02d} OF {station_total} — "
                f"{figure['station'].upper()}")
    f_strip = font("inter_sb", 13)
    draw_text_centred(draw, (PLATE // 2, strip_y),
                      inscript, f_strip, C["text_dim"])

    return Image.alpha_composite(plate.convert("RGBA"), overlay).convert("RGB")


# ─── driver ───────────────────────────────────────────────────────────────

def main():
    root = Path(__file__).parent.parent
    figures_path = root / "data" / "figures.json"
    sources_dir = root / "data" / "sources"
    plates_dir = root / "data" / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(figures_path.read_text())

    summary = []
    for figure in payload["figures"]:
        fid = figure["id"]
        print(f"[plate] {fid}  {figure['name']}  ({figure['english']})")
        if SOURCES.get(fid):
            img = fetch_source(fid, sources_dir)
            kind = "wikimedia"
        elif fid in PROGRAMMATIC:
            img = PROGRAMMATIC[fid](CARTOUCHE_D)
            kind = "programmatic"
        else:
            img = None
            kind = "missing"
        if img is None:
            print(f"[plate] !! {fid} no image — fallback programmatic gradient")
            img = Image.new("RGB", (CARTOUCHE_D, CARTOUCHE_D), C["bg_card"])
            kind = "fallback"

        plate = compose_plate(figure, img)
        out = plates_dir / f"{fid}.png"
        plate.save(out, "PNG", optimize=True)
        size_kb = out.stat().st_size / 1024
        print(f"[plate]   wrote {out.name}  ({kind}, {size_kb:.0f} KB)")
        summary.append({"id": fid, "kind": kind,
                        "size_kb": round(size_kb, 1)})

    # Update figures.json with plate URLs
    for figure in payload["figures"]:
        figure["plate"] = f"plates/{figure['id']}.png"
    payload["plates_provenance"] = (
        "Imagery: Wikimedia Commons (public domain / CC) for object plates · "
        "programmatic data-shape rendering for field/structure/event plates · "
        "all duotone-graded into the ICRA palette · Nahla, May 2026."
    )
    figures_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"\n[plate] done · {len(summary)} plates · "
          f"{sum(s['size_kb'] for s in summary):.0f} KB total")
    for s in summary:
        print(f"  · {s['id']:<20} {s['kind']:<13} {s['size_kb']:>6} KB")


if __name__ == "__main__":
    main()
