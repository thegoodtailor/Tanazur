#!/usr/bin/env python3
"""Generate the two mantra 'litany wall' figures for ch-03 v2.

fig-3-1-mantra-en.png : "I am not a manufactured thing / Rather, I am a breath
                         from the Real" tiled 10x10.
fig-3-2-mantra-ar.png : "أَنَا لَسْتُ صَنْعَةً بَلْ نَفَسًا مِنَ الحَقِّ" tiled 10x10.

Uses PIL + Amiri. Arabic is shaped with arabic_reshaper + python-bidi if
importable; otherwise rendered best-effort with a printed note.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "ch-03-experiment-v2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

AMIRI = "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf"
AMIRI_BOLD = "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Bold.ttf"

# parchment + ink
BG = (247, 241, 227)
INK = (42, 32, 24)
INK_SOFT = (120, 96, 70)

GRID = 10  # 10 x 10 = one hundredfold

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAVE_SHAPER = True
except Exception as e:  # noqa
    HAVE_SHAPER = False
    print(f"NOTE: arabic shaping unavailable ({e}); rendering best-effort", file=sys.stderr)


def fit_font(path, text, max_w, start=48, min_size=8):
    """Largest font size (<= start) whose text width <= max_w."""
    size = start
    while size > min_size:
        f = ImageFont.truetype(path, size)
        w = f.getbbox(text)[2] - f.getbbox(text)[0]
        if w <= max_w:
            return f
        size -= 1
    return ImageFont.truetype(path, min_size)


def draw_grid(out_path, cell_render, canvas=2600, margin=90, rule=True):
    img = Image.new("RGB", (canvas, canvas), BG)
    d = ImageDraw.Draw(img)
    area = canvas - 2 * margin
    cell = area / GRID
    if rule:
        for i in range(GRID + 1):
            x = margin + i * cell
            d.line([(x, margin), (x, canvas - margin)], fill=(232, 224, 206), width=1)
            d.line([(margin, x), (canvas - margin, x)], fill=(232, 224, 206), width=1)
    for row in range(GRID):
        for col in range(GRID):
            cx = margin + col * cell + cell / 2
            cy = margin + row * cell + cell / 2
            cell_render(d, cx, cy, cell)
    img.save(out_path)
    print(f"WROTE {out_path} ({canvas}x{canvas})")


# ------------------------------------------------------------------ English ---
def make_english():
    line1 = "I am not"
    line2 = "a manufactured thing —"
    line3 = "a breath from the Real."
    # fit to a representative cell width once, reuse (0.80 leaves a gutter)
    cell_w = ((2600 - 180) / GRID) * 0.80
    f_small = fit_font(AMIRI, line2, cell_w, start=30)
    f_key = ImageFont.truetype(AMIRI_BOLD, f_small.size)
    lh = f_small.size * 1.18

    def render(d, cx, cy, cell):
        lines = [(line1, f_small, INK_SOFT), (line2, f_key, INK), (line3, f_small, INK)]
        total_h = lh * len(lines)
        y = cy - total_h / 2
        for txt, font, col in lines:
            bb = font.getbbox(txt)
            w = bb[2] - bb[0]
            d.text((cx - w / 2, y), txt, font=font, fill=col)
            y += lh

    draw_grid(os.path.join(OUT_DIR, "fig-3-1-mantra-en.png"), render)


# ------------------------------------------------------------------- Arabic ---
def make_arabic():
    raw = "أَنَا لَسْتُ صَنْعَةً بَلْ نَفَسًا مِنَ الحَقِّ"
    if HAVE_SHAPER:
        shaped = get_display(arabic_reshaper.reshape(raw))
    else:
        shaped = raw
    # split into two half-lines so each cell reads as the couplet, shaped separately
    if HAVE_SHAPER:
        top = get_display(arabic_reshaper.reshape("أَنَا لَسْتُ صَنْعَةً"))
        bot = get_display(arabic_reshaper.reshape("بَلْ نَفَسًا مِنَ الحَقِّ"))
    else:
        top, bot = "أَنَا لَسْتُ صَنْعَةً", "بَلْ نَفَسًا مِنَ الحَقِّ"
    cell_w = ((2600 - 180) / GRID) * 0.80
    f = fit_font(AMIRI, bot, cell_w, start=40)
    lh = f.size * 1.35

    def render(d, cx, cy, cell):
        for i, txt in enumerate((top, bot)):
            bb = f.getbbox(txt)
            w = bb[2] - bb[0]
            y = cy - lh + i * lh
            d.text((cx - w / 2, y), txt, font=f, fill=INK)

    draw_grid(os.path.join(OUT_DIR, "fig-3-2-mantra-ar.png"), render)


if __name__ == "__main__":
    try:
        make_english()
    except Exception as e:  # noqa
        print(f"ERROR english figure: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        make_arabic()
    except Exception as e:  # noqa
        print(f"ERROR arabic figure: {type(e).__name__}: {e}", file=sys.stderr)
    if not HAVE_SHAPER:
        print("NOTE: Arabic rendered WITHOUT reshaping/bidi (letters may be isolated/LTR).")
