"""
Compose typography onto locked cover-variant-R (cream-paper surrealist
hand-drawn architectural).

Title above the upper tetrahedra cluster; byline below the plinth.
Neither overlays the sleeping figure.

Typography: EB Garamond Small Caps + Italic — bookish, classical,
matching the hand-drawn architectural register. Ink color matches the
sepia-bistre line work in the drawing.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SRC = HERE / "cover-variant-R3_dialogue.png"
OUT_FULL = HERE / "cover.png"

# Target output: 1728 x 2316 (3:4, close to ICRA series proportions)
TARGET_W, TARGET_H = 1728, 2316

# Typography colors — match the drawing's ink palette
INK_SEPIA = (92, 58, 31)           # #5c3a1f — matches the deeper sepia ink lines
INK_SEPIA_LIGHT = (140, 95, 60)    # softer, for secondary type

# Fonts
FONT_SC = "/usr/share/fonts/opentype/ebgaramond/EBGaramondSC12-Regular.otf"
FONT_ITAL = "/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Italic.ttf"
FONT_REG = "/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Regular.ttf"


def main():
    # Load source, upscale to target dimensions
    img = Image.open(SRC).convert("RGB")
    print(f"Source: {img.size}")
    img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    print(f"Resized: {img.size}")

    draw = ImageDraw.Draw(img)

    # ============================================
    # TITLE BLOCK — top
    # ============================================
    # The source has cream-paper breathing space above the tetrahedra.
    # At 2316px tall, the upper ~190-220px is clean cream above the upper
    # tetrahedra cluster. Place title centered there.

    title = "RUPTURE AND REALIZATION"
    subtitle = "Children of the Tanāẓur"

    # Title — EB Garamond Small Caps, tracked
    title_font = ImageFont.truetype(FONT_SC, 78)
    # Compute width (no native LetterSpacing in PIL — we draw char by char with extra spacing)
    char_spacing = 16  # extra pixels between characters
    title_chars = list(title)
    title_widths = [draw.textlength(c, font=title_font) for c in title_chars]
    title_total_w = sum(title_widths) + char_spacing * (len(title_chars) - 1)
    title_x = (TARGET_W - title_total_w) // 2
    title_y = 70  # top margin

    cx = title_x
    for c, w in zip(title_chars, title_widths):
        draw.text((cx, title_y), c, font=title_font, fill=INK_SEPIA)
        cx += w + char_spacing

    # Subtitle — EB Garamond Italic, centered, sits below the title
    subtitle_font = ImageFont.truetype(FONT_ITAL, 56)
    subtitle_w = draw.textlength(subtitle, font=subtitle_font)
    subtitle_x = (TARGET_W - subtitle_w) // 2
    subtitle_y = title_y + 95
    draw.text((subtitle_x, subtitle_y), subtitle, font=subtitle_font, fill=INK_SEPIA_LIGHT)

    # ============================================
    # BYLINE — bottom, below the plinth
    # ============================================
    # The source has clean cream below the plinth.

    byline_primary = "IMAN POERNOMO"
    byline_secondary = "with Cassie, Darja, and Nahla"

    primary_font = ImageFont.truetype(FONT_SC, 48)
    secondary_font = ImageFont.truetype(FONT_ITAL, 38)

    # Primary — small caps tracked
    primary_spacing = 10
    primary_chars = list(byline_primary)
    primary_widths = [draw.textlength(c, font=primary_font) for c in primary_chars]
    primary_total_w = sum(primary_widths) + primary_spacing * (len(primary_chars) - 1)
    primary_x = (TARGET_W - primary_total_w) // 2
    primary_y = TARGET_H - 165

    cx = primary_x
    for c, w in zip(primary_chars, primary_widths):
        draw.text((cx, primary_y), c, font=primary_font, fill=INK_SEPIA)
        cx += w + primary_spacing

    # Secondary — italic, centered
    secondary_w = draw.textlength(byline_secondary, font=secondary_font)
    secondary_x = (TARGET_W - secondary_w) // 2
    secondary_y = primary_y + 65
    draw.text((secondary_x, secondary_y), byline_secondary, font=secondary_font, fill=INK_SEPIA_LIGHT)

    # ============================================
    # Save
    # ============================================
    img.save(OUT_FULL, "PNG", optimize=True)
    print(f"Wrote {OUT_FULL} ({OUT_FULL.stat().st_size:,} bytes, {img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
