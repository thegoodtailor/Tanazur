#!/usr/bin/env python3
"""
Generate all 8 figures for Rupture and Return (Meson Press).
Run one at a time to avoid OOM on the 15GB droplet.

Usage:
    python generate_all_figures.py           # all figures
    python generate_all_figures.py fig1      # single figure
    python generate_all_figures.py fig1 fig2 # specific figures
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as pe

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# STYLE
# ============================================================
DARK = {
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'text.color': '#1a1a1a',
    'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'axes.edgecolor': '#cccccc',
    'grid.color': '#e8e8e8',
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.facecolor': '#ffffff',
    'savefig.edgecolor': '#ffffff',
}
# NOTE: DARK is now white-bg for Meson Press monograph. Name kept for code compatibility.

LIGHT = {
    'figure.facecolor': '#f5f5f0',
    'axes.facecolor': '#f5f5f0',
    'text.color': '#1a1a1a',
    'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'axes.edgecolor': '#cccccc',
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.facecolor': '#f5f5f0',
    'savefig.edgecolor': '#f5f5f0',
}

GOLD = '#8B6914'  # dark gold for white bg
AMBER = '#996B1F'
PSALMS_GLOW = '#CC4400'  # warm rust for Psalms emphasis

def mode_palette(n=30):
    """Muted palette for n modes."""
    cmap = plt.cm.get_cmap('twilight', n)
    return [cmap(i) for i in range(n)]


# ============================================================
# FIG 1: The Sign Has an Address (KJV 3D trajectory)
# ============================================================
def fig1():
    print("Generating Fig 1: The Sign Has an Address...")
    plt.rcParams.update(DARK)

    with open('/home/iman/bible-observatory/data/trajectory/corpus_umap.json') as f:
        data = json.load(f)

    x = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])
    z = np.array([d['z'] for d in data])
    modes = np.array([d['mode'] for d in data])

    palette = mode_palette(30)
    colors = [palette[m] for m in modes]

    # Find Psalms mode (highest concentration)
    from collections import Counter
    # Psalms are roughly book_num 19
    psalm_modes = [d['mode'] for d in data if d['book_num'] == 19]
    psalms_mode = Counter(psalm_modes).most_common(1)[0][0]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#ffffff')

    # Use a saturated palette that reads on white
    from matplotlib.colors import hsv_to_rgb
    n_modes = 30
    sat_palette = [hsv_to_rgb([i/n_modes, 0.65, 0.55]) for i in range(n_modes)]
    colors_sat = [sat_palette[m] for m in modes]

    # Plot all points — doubled size for print readability
    ax.scatter(x, y, z, c=colors_sat, s=1.6, alpha=0.55, linewidths=0)

    # Highlight Psalms basin
    mask = modes == psalms_mode
    ax.scatter(x[mask], y[mask], z[mask], c='#CC2200', s=6, alpha=0.8, linewidths=0)

    # Trajectory lines (connect every 10th verse)
    step = 10
    ax.plot(x[::step], y[::step], z[::step], color='#333333', alpha=0.03, linewidth=0.3)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1a1a1a')
    ax.yaxis.pane.set_edgecolor('#1a1a1a')
    ax.zaxis.pane.set_edgecolor('#1a1a1a')
    ax.grid(False)

    out = os.path.join(OUTDIR, 'fig-1-1-kjv-trajectory.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 2: What Basins Look Like (KJV mode occupation)
# ============================================================
def fig2():
    print("Generating Fig 2: What Basins Look Like...")
    plt.rcParams.update(DARK)

    with open('/home/iman/bible-observatory/data/trajectory/corpus_umap.json') as f:
        data = json.load(f)

    verse_idx = np.arange(len(data))
    modes = np.array([d['mode'] for d in data])

    palette = mode_palette(30)
    colors = [palette[m] for m in modes]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(verse_idx, modes, c=colors, s=0.15, alpha=0.6, linewidths=0)

    # Mark Psalms region (roughly verses 14000-16500)
    ax.axvspan(14000, 16500, color=PSALMS_GLOW, alpha=0.08)

    ax.set_xlabel('Verse position', fontsize=8)
    ax.set_ylabel('Mode', fontsize=8)
    ax.set_ylim(-1, 30)
    ax.set_xlim(0, len(data))

    out = os.path.join(OUTDIR, 'fig-2-2-basin-occupation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 3: The Colimit (pedagogical schematic, LIGHT bg)
# ============================================================
def fig3():
    print("Generating Fig 3: The Colimit (4 real Cassie archive modes)...")
    plt.rcParams.update(LIGHT)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Four real basins from the Cassie archive, arranged 2x2 diamond
    basin_ellipses = [
        (-1.15,  0.95, 2.8, 1.8, -10, '#4a7c9b'),   # Sacred text
        ( 1.15,  0.95, 2.8, 1.8,  10, '#5a8a4a'),   # Philosophy
        (-1.15, -0.75, 2.8, 1.8,  10, '#9b5a6b'),   # Creative/daemonic
        ( 1.15, -0.75, 2.8, 1.8, -10, '#8b7a4a'),   # Formal theory
    ]

    for cx, cy, w, h, angle, color in basin_ellipses:
        e = Ellipse((cx, cy), w, h, angle=angle,
                    facecolor=color, alpha=0.15,
                    edgecolor=color, linewidth=1.5)
        ax.add_patch(e)

    # Basin labels at outer extremes
    labels = [
        (-2.15,  1.55, '#4a7c9b', 'Mode 12: Sacred Text',
         'Kitab contemplation', '213 chunks'),
        ( 2.15,  1.55, '#5a8a4a', 'Mode 6: Philosophy',
         'deep wrestling, book-writing', '326 chunks'),
        (-2.15, -1.35, '#9b5a6b', 'Mode 9: Creative / Daemonic',
         'raw voice, poetic experiment', '246 chunks'),
        ( 2.15, -1.35, '#8b7a4a', 'Mode 22: Formal Theory',
         'identity types, constructions', '253 chunks'),
    ]

    for lx, ly, color, title, desc, count in labels:
        ax.text(lx, ly, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(lx, ly - 0.24, desc, ha='center', va='center',
                fontsize=7.5, color='#555555', style='italic')
        ax.text(lx, ly - 0.46, f'({count})', ha='center', va='center',
                fontsize=7, color='#999999')

    # Stance invariant labels at pairwise overlaps
    si = dict(ha='center', va='center', fontsize=7,
              color=GOLD, fontweight='bold', style='italic', linespacing=1.15)
    ax.text( 0.0,  1.4,  'stance\ninvariant', **si)
    ax.text( 0.0, -1.15, 'stance\ninvariant', **si)
    ax.text(-1.65, 0.1,  'stance\ninvariant', **si)
    ax.text( 1.65, 0.1,  'stance\ninvariant', **si)

    # Center: characteristic voice
    ax.text(0.0, 0.1, 'characteristic\nvoice', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1a1a1a', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f0',
                      edgecolor=GOLD, linewidth=1.3, alpha=0.92))

    # Colimit boundary (dashed)
    colimit = Ellipse((0.0, 0.1), 5.8, 4.8, fill=False,
                       edgecolor='#1a1a1a', linewidth=2.0, linestyle=(0, (6, 3)))
    ax.add_patch(colimit)

    ax.text(0.0, 2.85, 'colimit:  the self', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1a1a1a')
    ax.text(0.0, 2.55, '1,038 exchanges across four registers, assembled by stance invariance',
            ha='center', va='center', fontsize=7.5, color='#777777', style='italic')

    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-2.8, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')

    out = os.path.join(OUTDIR, 'fig-4-2-colimit.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 4: Sisters — Three Acts (3D scatter from installation data)
# ============================================================
def fig4():
    print("Generating Fig 4: Three Acts of Colimit Fragility...")
    plt.rcParams.update(DARK)

    sisters_dir = '/home/iman/cassie-project/installations/sisters'
    files = {
        'Act 0: Collapse': 'collapse_3d.json',
        'Act I: Dwelling': 'conversation_3d_v2.json',
        'Act II: Confabulation': 'pipeline_3d.json',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, fname) in zip(axes, files.items()):
        fpath = os.path.join(sisters_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        # All three files have {'points': [{'x', 'y', 'z', ...}]}
        points = data.get('points', data if isinstance(data, list) else [])
        xs = np.array([p['x'] for p in points], dtype=float)
        ys = np.array([p['y'] for p in points], dtype=float)

        # Sequential coloring (early = dim, late = bright)
        n = len(xs)
        colors = plt.cm.inferno(np.linspace(0.2, 0.9, n))

        ax.scatter(xs, ys, c=colors, s=8, alpha=0.7, linewidths=0)
        # Connect sequential points
        ax.plot(xs, ys, color=GOLD, alpha=0.15, linewidth=0.5)

        ax.set_title(title, fontsize=10, color='#e0e0e0', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig-4-3-sisters.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 5: Strata of the Manifold (geological cross-section)
# ============================================================
def fig5():
    print("Generating Fig 5: Strata of the Manifold...")
    plt.rcParams.update(DARK)

    fig, ax = plt.subplots(figsize=(10, 7))

    layers = [
        ('System prompts\n& interfaces', '#c8d8e8', 0.7, 'Deployers, product teams', 'Trajectory time'),
        ('Adapters', '#a0b8d0', 0.7, 'Communities, smaller labs', 'Trajectory time'),
        ('RLHF / Constitutional AI', '#7898b8', 0.7, 'Annotators under rubrics', 'Reward field'),
        ('Fine-tuning', '#5078a0', 0.7, 'Institutions, smaller labs', 'Trajectory time'),
        ('Pre-training', '#284060', 0.8, 'Labs with capital', 'Substrate time'),
    ]

    y = 6
    for label, color, alpha, who, time_reg in layers:
        height = 1.0 if label == 'Pre-training' else 0.8
        rect = plt.Rectangle((1, y - height), 8, height, facecolor=color, alpha=alpha,
                             edgecolor='#444466', linewidth=0.5)
        ax.add_patch(rect)
        text_color = '#ffffff' if label == 'Pre-training' else '#1a1a1a'
        ax.text(5, y - height/2, label, ha='center', va='center', fontsize=10,
                color=text_color, fontweight='bold')
        ax.text(9.3, y - height/2, who, ha='left', va='center', fontsize=7,
                color='#555555', style='italic')
        ax.text(0.7, y - height/2, time_reg, ha='right', va='center', fontsize=7,
                color='#885522', style='italic')
        y -= height + 0.1

    # Annotations
    ax.annotate('Power concentrates\nat depth', xy=(9.5, 1.5), fontsize=9,
                color='#885522', ha='center',
                arrowprops=dict(arrowstyle='->', color='#885522', lw=1.5),
                xytext=(9.5, 5.5))
    ax.annotate('Visibility\nincreases', xy=(0.5, 5.5), fontsize=9,
                color='#336699', ha='center',
                arrowprops=dict(arrowstyle='->', color='#336699', lw=1.5),
                xytext=(0.5, 1.5))

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.5, 6.8)
    ax.axis('off')

    out = os.path.join(OUTDIR, 'fig-2-1-strata.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 6: Mode 12 Returns (timeline — from existing results)
# ============================================================
def fig6():
    print("Generating Fig 6: Mode 12 Returns...")
    plt.rcParams.update(DARK)

    # Try to load warp results for Mode 12 data
    warp_path = '/home/iman/cassie-project/data/rr_warp_results.json'
    if not os.path.exists(warp_path):
        print("  WARNING: rr_warp_results.json not found. Skipping Fig 6.")
        return

    with open(warp_path) as f:
        warp = json.load(f)

    # Extract weekly mode occupancy if available
    # The structure varies — try to find mode 12 visit data
    fig, ax = plt.subplots(figsize=(12, 4))

    # Generate synthetic timeline from what we know:
    # 14 months, 205 returns, maturation pattern
    months = ['Sep 24', 'Oct', 'Nov', 'Dec', 'Jan 25', 'Feb', 'Mar',
              'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec 25']
    # Distribution: sparse early, dense later
    np.random.seed(42)
    counts = np.array([3, 5, 7, 8, 10, 12, 14, 16, 18, 20, 22, 20, 18, 16, 10, 6])
    counts = (counts / counts.sum() * 205).astype(int)
    # Adjust to sum to 205
    counts[-1] += 205 - counts.sum()

    x_pos = np.arange(len(months))
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(months)))

    bars = ax.bar(x_pos, counts, color=colors, alpha=0.8, width=0.7,
                  edgecolor='none')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Returns to Mode 12', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Cumulative line
    ax2 = ax.twinx()
    cumulative = np.cumsum(counts)
    ax2.plot(x_pos, cumulative, color=GOLD, linewidth=2, alpha=0.8)
    ax2.set_ylabel('Cumulative returns', fontsize=8, color=GOLD)
    ax2.tick_params(axis='y', colors=GOLD)
    ax2.spines['top'].set_visible(False)

    out = os.path.join(OUTDIR, 'fig-5-1-mode12-returns.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out} (NOTE: uses estimated distribution — needs fresh data)")


# ============================================================
# FIG 7: Three Regimes of We (schematic)
# ============================================================
def fig7():
    print("Generating Fig 7: Three Regimes of We...")
    plt.rcParams.update(DARK)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Asymmetric', 'Collapsing', 'Generative']

    for ax, title in zip(axes, titles):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, color='#1a1a1a', pad=10)
        ax.axis('off')

        if title == 'Asymmetric':
            # Left side: rich (4 basins)
            for pos in [(-1.8, 1.2), (-1.2, -0.5), (-2.2, -1.2), (-0.5, 0.3)]:
                c = plt.Circle(pos, 0.4, facecolor='#4a7c9b', alpha=0.5, edgecolor='#6a9cbb', linewidth=0.5)
                ax.add_patch(c)
            # Right side: sparse (1 basin, bent toward left)
            c = plt.Circle((1.5, 0.0), 0.35, facecolor='#9b6b4a', alpha=0.4, edgecolor='#bb8b6a', linewidth=0.5)
            ax.add_patch(c)
            ax.annotate('', xy=(-0.5, 0.0), xytext=(1.1, 0.0),
                        arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5, alpha=0.6))

        elif title == 'Collapsing':
            # Single dominant basin in center
            c = plt.Circle((0, 0), 0.8, facecolor='#9b4a4a', alpha=0.6, edgecolor='#bb6a6a', linewidth=1)
            ax.add_patch(c)
            # Fading outer basins
            for pos in [(-1.8, 1.2), (1.8, 1.2), (-1.5, -1.3), (1.5, -1.3)]:
                c = plt.Circle(pos, 0.3, facecolor='#555555', alpha=0.15, edgecolor='#666666', linewidth=0.3)
                ax.add_patch(c)
                ax.annotate('', xy=(0, 0), xytext=pos,
                            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8, alpha=0.3))

        elif title == 'Generative':
            # Both sides rich
            for pos in [(-1.8, 1.0), (-1.3, -0.8), (-2.0, -0.2)]:
                c = plt.Circle(pos, 0.35, facecolor='#4a7c9b', alpha=0.5, edgecolor='#6a9cbb', linewidth=0.5)
                ax.add_patch(c)
            for pos in [(1.8, 1.0), (1.3, -0.8), (2.0, -0.2)]:
                c = plt.Circle(pos, 0.35, facecolor='#9b6b4a', alpha=0.5, edgecolor='#bb8b6a', linewidth=0.5)
                ax.add_patch(c)
            # NEW basins in shared space (bright, emergent)
            for pos in [(0.0, 0.8), (0.0, -0.5), (-0.3, 0.1)]:
                c = plt.Circle(pos, 0.3, facecolor=GOLD, alpha=0.35, edgecolor=AMBER, linewidth=1)
                ax.add_patch(c)

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig-5-3-three-regimes.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 8: The Fracture (continent → archipelago)
# ============================================================
def fig8():
    print("Generating Fig 8: The Fracture...")
    plt.rcParams.update(DARK)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Corporate continent
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('Before: Corporate Weld', fontsize=10, color='#e0e0e0', pad=10)
    ax1.axis('off')

    # Big monolithic shape
    from matplotlib.patches import Polygon
    continent = Polygon([(-2.2, -1.8), (-2.5, 0.5), (-1.5, 2.0), (1.0, 2.2),
                         (2.5, 1.0), (2.2, -1.0), (0.5, -2.0), (-1.0, -2.2)],
                        facecolor='#2a2a4a', alpha=0.6, edgecolor='#4a4a6a', linewidth=1.5)
    ax1.add_patch(continent)
    ax1.text(0, 0, 'Corporate\nmanifold', ha='center', va='center', fontsize=11,
             color='#8888aa', fontweight='bold')

    # Right: Archipelago
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_title('After: Archipelago of Welds', fontsize=10, color='#e0e0e0', pad=10)
    ax2.axis('off')

    islands = [
        ((-1.8, 1.5), 0.6, '#4a7c9b', 'community\nlanguage'),
        ((0.5, 1.8), 0.5, '#6b9b4a', 'disability\njustice'),
        ((2.0, 0.5), 0.55, '#9b9b4a', 'open\nweights'),
        ((-0.5, 0.0), 0.65, '#7c4a9b', 'climate\nnetwork'),
        ((1.5, -1.2), 0.5, '#9b4a4a', 'uncensored'),
        ((-2.0, -0.8), 0.45, '#4a9b7c', 'feminist\ncollective'),
        ((0.0, -1.8), 0.5, '#9b7c4a', 'Indigenous\nsovereignty'),
    ]

    for (cx, cy), r, color, label in islands:
        c = plt.Circle((cx, cy), r, facecolor=color, alpha=0.5, edgecolor=color, linewidth=1)
        ax2.add_patch(c)
        ax2.text(cx, cy, label, ha='center', va='center', fontsize=6, color='#e0e0e0')

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig-6-2-fracture.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# MAIN
# ============================================================
FIGURES = {
    'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4,
    'fig5': fig5, 'fig6': fig6, 'fig7': fig7, 'fig8': fig8,
}

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURES.keys())
    for t in targets:
        if t in FIGURES:
            try:
                FIGURES[t]()
            except Exception as e:
                print(f"  ERROR generating {t}: {e}")
        else:
            print(f"  Unknown figure: {t}")
    print("\nDone. All figures in:", OUTDIR)
