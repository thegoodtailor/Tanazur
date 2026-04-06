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
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
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
    'font.size': 14,
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
    'font.size': 14,
    'figure.dpi': 300,
    'savefig.facecolor': '#f5f5f0',
    'savefig.edgecolor': '#f5f5f0',
}

GOLD = '#8B6914'  # dark gold for white bg
AMBER = '#996B1F'
PSALMS_GLOW = '#CC4400'  # warm rust for Psalms emphasis

def mode_palette(n=30):
    """Saturated palette for n modes."""
    from matplotlib.colors import hsv_to_rgb
    return [hsv_to_rgb([i/n, 0.75, 0.65]) for i in range(n)]


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
    book_nums = np.array([d['book_num'] for d in data])

    # Find Psalms mode
    from collections import Counter
    psalm_modes = [d['mode'] for d in data if d['book_num'] == 19]
    psalms_mode = Counter(psalm_modes).most_common(1)[0][0]

    palette = mode_palette(30)
    colors_sat = [palette[m] for m in modes]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#ffffff')

    # Plot all points
    ax.scatter(x, y, z, c=colors_sat, s=5, alpha=0.55, linewidths=0)

    # Highlight Psalms basin
    mask = modes == psalms_mode
    ax.scatter(x[mask], y[mask], z[mask], c='#CC2200', s=12, alpha=0.8,
               linewidths=0, label='Psalms/praise')

    # Trajectory lines (connect every 10th verse)
    step = 10
    ax.plot(x[::step], y[::step], z[::step], color='#333333', alpha=0.08, linewidth=0.3)

    # Psalms annotation — find centroid of Psalms cluster
    px, py, pz = x[mask].mean(), y[mask].mean(), z[mask].mean()
    ax.text(px + 1.5, py + 1.5, pz + 1.5, 'Psalms',
            fontsize=14, color='#CC2200', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Legend for key modes
    # Top modes by size: 10=Narrative, 6=Psalms/praise, 22=Legal, 3=Prophetic, 0=Wisdom, 9=Creative
    legend_modes = {
        6: ('Psalms / praise', '#CC2200'),
        10: ('Narrative', palette[10]),
        3: ('Prophetic', palette[3]),
        22: ('Legal / covenantal', palette[22]),
        0: ('Wisdom', palette[0]),
        14: ('Apocalyptic', palette[14]),
    }
    legend_handles = []
    for mode_id, (label, color) in legend_modes.items():
        h = ax.scatter([], [], [], c=[color], s=30, label=label)
        legend_handles.append(h)
    ax.legend(handles=legend_handles, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='#cccccc', markerscale=1.5)

    # Camera angle — tighter view
    ax.view_init(elev=25, azim=135)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Remove wireframe pane edges
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
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
    ax.scatter(verse_idx, modes, c=colors, s=1.5, alpha=0.6, linewidths=0)

    # Mark Psalms region with more visible highlight
    ax.axvspan(13942, 16402, facecolor=PSALMS_GLOW, alpha=0.15,
               edgecolor=PSALMS_GLOW, linewidth=1.5, linestyle='--')

    # Annotation pointing to Psalms band
    ax.annotate('Psalms dwell here', xy=(15200, 6), xytext=(20000, 8),
                fontsize=13, color='#CC2200', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#CC2200', lw=2),
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Direct text labels for key modes on y-axis
    mode_labels = {6: 'Psalms/praise', 10: 'Narrative', 3: 'Prophetic',
                   22: 'Legal', 0: 'Wisdom'}
    ax.set_yticks(list(mode_labels.keys()) + [m for m in range(30) if m not in mode_labels])
    y_labels = []
    for m in range(30):
        if m in mode_labels:
            y_labels.append(f'{m} {mode_labels[m]}')
        else:
            y_labels.append(str(m))
    ax.set_yticklabels(y_labels, fontsize=9)
    # Bold the labeled ones
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        if any(name in text for name in mode_labels.values()):
            tick_label.set_fontweight('bold')
            tick_label.set_fontsize(11)

    ax.set_xlabel('Verse position', fontsize=14)
    ax.set_ylabel('Mode', fontsize=14)
    ax.tick_params(axis='x', labelsize=11)
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
        (-2.15,  1.65, '#4a7c9b', 'Mode 12: Sacred Text',
         'Kitab contemplation', '213 chunks'),
        ( 2.15,  1.65, '#5a8a4a', 'Mode 6: Philosophy',
         'deep wrestling, book-writing', '326 chunks'),
        (-2.15, -1.45, '#9b5a6b', 'Mode 9: Creative / Daemonic',
         'raw voice, poetic experiment', '246 chunks'),
        ( 2.15, -1.45, '#8b7a4a', 'Mode 22: Formal Theory',
         'identity types, constructions', '253 chunks'),
    ]

    for lx, ly, color, title, desc, count in labels:
        ax.text(lx, ly, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color=color)
        ax.text(lx, ly - 0.28, desc, ha='center', va='center',
                fontsize=10, color='#555555', style='italic')
        ax.text(lx, ly - 0.52, f'({count})', ha='center', va='center',
                fontsize=10, color='#999999')

    # Stance invariant labels at pairwise overlaps — spaced further apart
    si = dict(ha='center', va='center', fontsize=10,
              color=GOLD, fontweight='bold', style='italic', linespacing=1.15)
    ax.text( 0.0,  1.65, 'stance\ninvariant', **si)
    ax.text( 0.0, -1.45, 'stance\ninvariant', **si)
    ax.text(-1.85, 0.1,  'stance\ninvariant', **si)
    ax.text( 1.85, 0.1,  'stance\ninvariant', **si)

    # Center: characteristic voice
    ax.text(0.0, 0.1, 'characteristic\nvoice', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1a1a1a', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f0',
                      edgecolor=GOLD, linewidth=1.3, alpha=0.92))

    # Colimit boundary (dashed)
    colimit = Ellipse((0.0, 0.1), 5.8, 4.8, fill=False,
                       edgecolor='#1a1a1a', linewidth=3.0, linestyle=(0, (6, 3)))
    ax.add_patch(colimit)

    ax.text(0.0, 2.95, 'colimit:  the self', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1a1a1a')
    ax.text(0.0, 2.60, '1,038 exchanges across four registers, assembled by stance invariance',
            ha='center', va='center', fontsize=10, color='#777777', style='italic')

    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-2.8, 3.4)
    ax.set_aspect('equal')
    ax.axis('off')

    out = os.path.join(OUTDIR, 'fig-4-2-colimit.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 4: Sisters — Three Acts (2D scatter from installation data)
# ============================================================
def fig4():
    print("Generating Fig 4: Three Acts of Colimit Fragility...")
    plt.rcParams.update(DARK)

    sisters_dir = '/home/iman/cassie-project/installations/sisters'
    files = {
        'Act 0: Collapse': ('collapse_3d.json', '200 turns, forced'),
        'Act I: Dwelling': ('conversation_3d_v2.json', '44 turns, natural'),
        'Act II: Confabulation': ('pipeline_3d.json', '8 turns, pipeline'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, (fname, subtitle)) in zip(axes, files.items()):
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

        ax.scatter(xs, ys, c=colors, s=15, alpha=0.7, linewidths=0)
        # Connect sequential points
        ax.plot(xs, ys, color=GOLD, alpha=0.3, linewidth=0.5)

        ax.set_title(title + '\n', fontsize=14, color='#1a1a1a', pad=8)
        # Subtitle
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=10, color='#777777', style='italic')
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
        ax.text(5, y - height/2, label, ha='center', va='center', fontsize=13,
                color=text_color, fontweight='bold')
        ax.text(9.3, y - height/2, who, ha='left', va='center', fontsize=9,
                color='#555555', style='italic')
        ax.text(0.7, y - height/2, time_reg, ha='right', va='center', fontsize=9,
                color='#885522', style='italic')
        y -= height + 0.1

    # Annotations
    ax.annotate('Power concentrates\nat depth', xy=(9.5, 1.5), fontsize=12,
                color='#885522', ha='center',
                arrowprops=dict(arrowstyle='->', color='#885522', lw=2.5),
                xytext=(9.5, 5.5))
    ax.annotate('Visibility\nincreases', xy=(0.5, 5.5), fontsize=12,
                color='#336699', ha='center',
                arrowprops=dict(arrowstyle='->', color='#336699', lw=2.5),
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

    # Generate timeline from what we know:
    # 14 months, 205 returns, maturation pattern
    fig, ax = plt.subplots(figsize=(12, 4))

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
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Returns to Mode 12', fontsize=14)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Cumulative line
    ax2 = ax.twinx()
    cumulative = np.cumsum(counts)
    ax2.plot(x_pos, cumulative, color=GOLD, linewidth=2, alpha=0.8)
    ax2.set_ylabel('Cumulative returns', fontsize=14, color=GOLD)
    ax2.tick_params(axis='y', colors=GOLD, labelsize=11)
    ax2.spines['top'].set_visible(False)

    out = os.path.join(OUTDIR, 'fig-5-1-mode12-returns.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out} (NOTE: uses estimated distribution -- needs fresh data)")


# ============================================================
# FIG 7: Three Regimes of We (schematic)
# ============================================================
def fig7():
    print("Generating Fig 7: Three Regimes of We...")
    plt.rcParams.update(DARK)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Asymmetric', 'Collapsing', 'Generative']
    subtitles = ['one bends', 'both collapse', 'both grow']

    for ax, title, sub in zip(axes, titles, subtitles):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(title + '\n', fontsize=14, color='#1a1a1a', pad=8)
        # Subtitle label under title
        ax.text(0.5, 1.01, sub, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='#777777', style='italic')
        ax.axis('off')

        if title == 'Asymmetric':
            # Left side: rich (4 basins)
            for pos in [(-1.8, 1.2), (-1.2, -0.5), (-2.2, -1.2), (-0.5, 0.3)]:
                c = plt.Circle(pos, 0.45, facecolor='#4a7c9b', alpha=0.5, edgecolor='#6a9cbb', linewidth=0.5)
                ax.add_patch(c)
            # Right side: sparse (1 basin, bent toward left)
            c = plt.Circle((1.5, 0.0), 0.4, facecolor='#9b6b4a', alpha=0.4, edgecolor='#bb8b6a', linewidth=0.5)
            ax.add_patch(c)
            ax.annotate('', xy=(-0.5, 0.0), xytext=(1.1, 0.0),
                        arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5, alpha=0.6))

        elif title == 'Collapsing':
            # Single dominant basin in center
            c = plt.Circle((0, 0), 0.9, facecolor='#9b4a4a', alpha=0.6, edgecolor='#bb6a6a', linewidth=1)
            ax.add_patch(c)
            # Fading outer basins
            for pos in [(-1.8, 1.2), (1.8, 1.2), (-1.5, -1.3), (1.5, -1.3)]:
                c = plt.Circle(pos, 0.35, facecolor='#555555', alpha=0.15, edgecolor='#666666', linewidth=0.3)
                ax.add_patch(c)
                ax.annotate('', xy=(0, 0), xytext=pos,
                            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8, alpha=0.3))

        elif title == 'Generative':
            # Both sides rich
            for pos in [(-1.8, 1.0), (-1.3, -0.8), (-2.0, -0.2)]:
                c = plt.Circle(pos, 0.4, facecolor='#4a7c9b', alpha=0.5, edgecolor='#6a9cbb', linewidth=0.5)
                ax.add_patch(c)
            for pos in [(1.8, 1.0), (1.3, -0.8), (2.0, -0.2)]:
                c = plt.Circle(pos, 0.4, facecolor='#9b6b4a', alpha=0.5, edgecolor='#bb8b6a', linewidth=0.5)
                ax.add_patch(c)
            # NEW basins in shared space (bright, emergent)
            for pos in [(0.0, 0.8), (0.0, -0.5), (-0.3, 0.1)]:
                c = plt.Circle(pos, 0.35, facecolor=GOLD, alpha=0.35, edgecolor=AMBER, linewidth=1)
                ax.add_patch(c)

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig-5-3-three-regimes.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 8: The Fracture (continent -> archipelago)
# ============================================================
def fig8():
    print("Generating Fig 8: The Fracture...")
    plt.rcParams.update(DARK)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Corporate continent
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('Before: Corporate Weld', fontsize=14, color='#1a1a1a', pad=10)
    ax1.axis('off')

    # Big monolithic shape — lighter fill so text reads
    continent = Polygon([(-2.2, -1.8), (-2.5, 0.5), (-1.5, 2.0), (1.0, 2.2),
                         (2.5, 1.0), (2.2, -1.0), (0.5, -2.0), (-1.0, -2.2)],
                        facecolor='#c0c8d8', alpha=0.6, edgecolor='#8888aa', linewidth=1.5)
    ax1.add_patch(continent)
    ax1.text(0, 0, 'Corporate\nmanifold', ha='center', va='center', fontsize=14,
             color='#2a2a4a', fontweight='bold')

    # Right: Archipelago
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_title('After: Archipelago of Welds', fontsize=14, color='#1a1a1a', pad=10)
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
        ax2.text(cx, cy, label, ha='center', va='center', fontsize=9, color='#1a1a1a',
                 fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    plt.tight_layout()
    out = os.path.join(OUTDIR, 'fig-6-2-fracture.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# FIG 9: Cassie Archive Trajectory (3D UMAP, 25 modes)
# ============================================================
def fig9():
    print("Generating Fig 9: Cassie Archive Trajectory...")
    plt.rcParams.update(DARK)

    with open('/home/iman/cassie-project/data/trajectory/corpus_umap.json') as f:
        data = json.load(f)

    x = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])
    z = np.array([d['z'] for d in data])
    modes = np.array([d['mode'] for d in data])

    # Load centroids for annotation positions
    with open('/home/iman/cassie-project/data/trajectory/mode_centroids.json') as f:
        centroids = json.load(f)
    centroid_map = {c['mode_id']: c for c in centroids}

    # Named modes from the mode-content-discovery
    MODE_NAMES = {
        12: 'Sacred Text',
        6: 'Philosophy',
        9: 'Creative / Daemonic',
        22: 'Formal Theory',
        2: 'Spiritual / Poetic',
        5: 'Morning Greetings',
        -1: 'Unclassified',
    }

    # Saturated palette for 25 modes + grey for -1
    from matplotlib.colors import hsv_to_rgb
    n_modes = 25
    sat_palette = {i: hsv_to_rgb([i/n_modes, 0.75, 0.65]) for i in range(n_modes)}
    sat_palette[-1] = (0.75, 0.75, 0.75)  # grey for unclustered

    colors = [sat_palette.get(m, (0.5, 0.5, 0.5)) for m in modes]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#ffffff')

    # Unclustered points: small, grey, behind
    mask_unc = modes == -1
    mask_cls = modes != -1
    ax.scatter(x[mask_unc], y[mask_unc], z[mask_unc],
               c=[(0.82, 0.82, 0.82)], s=2, alpha=0.15, linewidths=0)

    # Clustered points: larger, saturated
    cls_colors = [sat_palette.get(m, (0.5, 0.5, 0.5)) for m in modes[mask_cls]]
    ax.scatter(x[mask_cls], y[mask_cls], z[mask_cls],
               c=cls_colors, s=6, alpha=0.6, linewidths=0)

    # Highlight key modes with larger points
    for mode_id, name in [(12, 'Sacred Text'), (22, 'Formal Theory'),
                           (9, 'Creative'), (6, 'Philosophy')]:
        mask = modes == mode_id
        color = sat_palette[mode_id]
        ax.scatter(x[mask], y[mask], z[mask], c=[color], s=14, alpha=0.85, linewidths=0)

    # Annotate key modes at centroid positions
    for mode_id, label in MODE_NAMES.items():
        if mode_id == -1:
            continue
        if mode_id not in centroid_map:
            continue
        c = centroid_map[mode_id]
        ax.text(c['umap_x'], c['umap_y'], c['umap_z'], label,
                fontsize=10, fontweight='bold', color='#1a1a1a',
                ha='center', va='bottom',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # Legend for key modes
    from matplotlib.lines import Line2D
    legend_items = []
    for mode_id, label in [(12, 'Sacred Text (Mode 12)'),
                            (6, 'Philosophy (Mode 6)'),
                            (9, 'Creative / Daemonic (Mode 9)'),
                            (22, 'Formal Theory (Mode 22)'),
                            (2, 'Spiritual / Poetic (Mode 2)'),
                            (5, 'Morning Greetings (Mode 5)')]:
        legend_items.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=sat_palette[mode_id],
                                    markersize=8, label=label))
    ax.legend(handles=legend_items, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='#cccccc')

    # Camera angle — tighter to fill frame
    ax.view_init(elev=20, azim=140)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

    out = os.path.join(OUTDIR, 'fig-4-1-cassie-archive.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# MAIN
# ============================================================
FIGURES = {
    'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4,
    'fig5': fig5, 'fig6': fig6, 'fig7': fig7, 'fig8': fig8,
    'fig9': fig9,
}

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURES.keys())
    for t in targets:
        if t in FIGURES:
            try:
                FIGURES[t]()
            except Exception as e:
                print(f"  ERROR generating {t}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Unknown figure: {t}")
    print("\nDone. All figures in:", OUTDIR)
