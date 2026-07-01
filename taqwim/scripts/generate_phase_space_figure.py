"""Generate two phase-space figures for ICRA-12.

  phase-space.png            — 10,000 years (human / near-return scale)
  phase-space-billion.png    — 1,000,000,000 years (ergodic scale)

The two figures answer one question: at the human time scale the
Metonic family makes the trajectory look almost-periodic, with crisp
diagonal striations in the (θ_s, θ_l) projection. At the geological
time scale the residual phase-shift per Metonic cycle has accumulated
far past 2π many times over, the stripes melt, and the trajectory
fills the torus as Weyl's equidistribution theorem predicts.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─── astronomical constants ─────────────────────────────────────────────
# Mean rates — adequate for showing ergodicity. Real rates have small
# periodic perturbations (eccentricity, lunar evection, nodal regression)
# and very slow secular drifts (tidal recession of the moon, sun's slow
# mass loss). Over 10 ky those are irrelevant. Over 1 Gy they matter
# for *physical* prediction but not for the equidistribution argument.
OMEGA_S = 360.0 / 365.2422
OMEGA_L = 360.0 / 29.530588
OMEGA_R = 360.0 / 0.99726957
S0 = np.array([96.0393, 29.06, 207.29])


def wrapped_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = (a - b) % 360.0
    return np.minimum(d, 360.0 - d)


PALETTE_STOPS = [
    (0.0, "#0C1020"),
    (0.25, "#16213E"),
    (0.55, "#5D6D5D"),
    (0.85, "#B8956C"),
    (1.0, "#FAF6F0"),
]
tnzr_cmap = LinearSegmentedColormap.from_list("tanazur", PALETTE_STOPS)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
    "axes.edgecolor": "#A8AEC6",
    "axes.labelcolor": "#A8AEC6",
    "xtick.color": "#A8AEC6",
    "ytick.color": "#A8AEC6",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#FAF6F0",
})


def build_trajectory(years: float, n_samples: int):
    days = years * 365.2422
    t_days = np.linspace(0.0, days, n_samples)
    theta_s = (OMEGA_S * t_days) % 360.0
    theta_l = (OMEGA_L * t_days) % 360.0
    theta_r = (OMEGA_R * t_days) % 360.0
    years_axis = t_days / 365.2422
    dS = np.sqrt(
        wrapped_diff(theta_s, S0[0]) ** 2
        + wrapped_diff(theta_l, S0[1]) ** 2
        + wrapped_diff(theta_r, S0[2]) ** 2
    )
    return theta_s, theta_l, theta_r, years_axis, dS


# ─── figure 1: 10 000 years (near-return regime) ────────────────────────
def figure_human_scale():
    theta_s, theta_l, theta_r, years_axis, dS = build_trajectory(
        years=10_000, n_samples=50_000,
    )

    fig = plt.figure(figsize=(13.5, 10), facecolor="#0C1020")

    ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#0C1020")
    ax1.scatter(theta_s, theta_l, theta_r,
                c=years_axis, cmap=tnzr_cmap, s=0.6, alpha=0.55, rasterized=True)
    ax1.scatter([S0[0]], [S0[1]], [S0[2]], c="#F5C842", s=80, marker="*",
                edgecolors="white", linewidths=0.8, label=r"$S_0$ (revelation)", zorder=10)
    ax1.set_xlabel(r"$\theta_s$ (solar) [°]")
    ax1.set_ylabel(r"$\theta_l$ (lunar) [°]")
    ax1.set_zlabel(r"$\theta_r$ (sidereal) [°]")
    ax1.set_title("(a) Trajectory on $T^3$, 10 000 years", pad=14)
    ax1.set_xlim(0, 360); ax1.set_ylim(0, 360); ax1.set_zlim(0, 360)
    ax1.legend(loc="upper left", framealpha=0.4, facecolor="#0C1020",
               edgecolor="#A8AEC6", labelcolor="#FAF6F0", fontsize=9)
    for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
        pane.set_facecolor("#16213E"); pane.set_alpha(0.4)

    ax2 = fig.add_subplot(2, 2, 2, facecolor="#0C1020")
    sc = ax2.scatter(theta_s, theta_l, c=years_axis, cmap=tnzr_cmap,
                     s=0.5, alpha=0.45, rasterized=True)
    ax2.scatter([S0[0]], [S0[1]], c="#F5C842", s=140, marker="*",
                edgecolors="white", linewidths=0.8, zorder=10)
    ax2.set_xlabel(r"$\theta_s$ (solar) [°]")
    ax2.set_ylabel(r"$\theta_l$ (lunar) [°]")
    ax2.set_title("(b) Solar–lunar projection — Metonic striations visible", pad=12)
    ax2.set_xlim(0, 360); ax2.set_ylim(0, 360)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("years from revelation", color="#A8AEC6", fontsize=9)
    cbar.outline.set_edgecolor("#A8AEC6")

    ax3 = fig.add_subplot(2, 2, 3, facecolor="#0C1020")
    step = max(1, len(years_axis) // 8000)
    ax3.plot(years_axis[::step], dS[::step], color="#D4A84B", linewidth=0.5, alpha=0.85)
    ax3.set_xlabel("years from revelation")
    ax3.set_ylabel(r"$\|S(t) - S_0\|$ on $T^3$ [°]")
    ax3.set_title("(c) Distance from $S_0$ — near-return rhythm", pad=12)
    ax3.set_xlim(0, 10_000); ax3.set_ylim(0, dS.max() * 1.05)

    ax4 = fig.add_subplot(2, 2, 4, facecolor="#0C1020")
    mask = years_axis <= 1000
    ax4.scatter(theta_s[mask], theta_l[mask], c=years_axis[mask],
                cmap=tnzr_cmap, s=1.0, alpha=0.7, rasterized=True)
    ax4.scatter([S0[0]], [S0[1]], c="#F5C842", s=140, marker="*",
                edgecolors="white", linewidths=0.8, zorder=10)
    ax4.set_xlabel(r"$\theta_s$ (solar) [°]")
    ax4.set_ylabel(r"$\theta_l$ (lunar) [°]")
    ax4.set_title("(d) First 1000 years zoom — spiral is still readable", pad=12)
    ax4.set_xlim(0, 360); ax4.set_ylim(0, 360)

    fig.suptitle("Phase space at the human scale — 10 ky from revelation",
                 color="#FAF6F0", fontsize=14, y=0.995)
    fig.text(0.5, 0.012,
             r"Mean rates: $\omega_s = 360°/365.2422$ d, "
             r"$\omega_l = 360°/29.530588$ d, $\omega_r = 360°/0.99726957$ d — "
             r"mutually irrational. Revelation $t_0$ = 27 June 2025 19:24:11 UTC.",
             color="#A8AEC6", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    out = "/home/iman/cassie-project/Tanazur/taqwim/phase-space.png"
    fig.savefig(out, dpi=200, facecolor="#0C1020", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ─── figure 2: 1 billion years (ergodic regime) ─────────────────────────
def figure_geological_scale():
    YEARS = 1_000_000_000
    # Sparse sampling — beyond one Metonic cycle's worth of inter-sample time,
    # consecutive samples are essentially independent for visualisation
    # purposes. 200k points across 1 Gy => ~5000 years between samples.
    theta_s, theta_l, theta_r, years_axis, dS = build_trajectory(
        years=YEARS, n_samples=200_000,
    )

    fig = plt.figure(figsize=(13.5, 10), facecolor="#0C1020")

    ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#0C1020")
    ax1.scatter(theta_s, theta_l, theta_r,
                c=years_axis / 1e6, cmap=tnzr_cmap, s=0.3, alpha=0.25, rasterized=True)
    ax1.scatter([S0[0]], [S0[1]], [S0[2]], c="#F5C842", s=80, marker="*",
                edgecolors="white", linewidths=0.8, label=r"$S_0$ (revelation)", zorder=10)
    ax1.set_xlabel(r"$\theta_s$ [°]")
    ax1.set_ylabel(r"$\theta_l$ [°]")
    ax1.set_zlabel(r"$\theta_r$ [°]")
    ax1.set_title("(a) $T^3$ scatter, $10^9$ years — uniform fill", pad=14)
    ax1.set_xlim(0, 360); ax1.set_ylim(0, 360); ax1.set_zlim(0, 360)
    ax1.legend(loc="upper left", framealpha=0.4, facecolor="#0C1020",
               edgecolor="#A8AEC6", labelcolor="#FAF6F0", fontsize=9)
    for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
        pane.set_facecolor("#16213E"); pane.set_alpha(0.4)

    ax2 = fig.add_subplot(2, 2, 2, facecolor="#0C1020")
    sc = ax2.scatter(theta_s, theta_l, c=years_axis / 1e6, cmap=tnzr_cmap,
                     s=0.25, alpha=0.2, rasterized=True)
    ax2.scatter([S0[0]], [S0[1]], c="#F5C842", s=140, marker="*",
                edgecolors="white", linewidths=0.8, zorder=10)
    ax2.set_xlabel(r"$\theta_s$ (solar) [°]")
    ax2.set_ylabel(r"$\theta_l$ (lunar) [°]")
    ax2.set_title("(b) Solar–lunar projection — Metonic stripes dissolved", pad=12)
    ax2.set_xlim(0, 360); ax2.set_ylim(0, 360)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("Myr from revelation", color="#A8AEC6", fontsize=9)
    cbar.outline.set_edgecolor("#A8AEC6")

    # (c) Marginal density along θ_s — should be uniform, near 1/360.
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#0C1020")
    bins = np.linspace(0, 360, 73)  # 5° bins
    h_s, edges = np.histogram(theta_s, bins=bins, density=True)
    h_l, _ = np.histogram(theta_l, bins=bins, density=True)
    h_r, _ = np.histogram(theta_r, bins=bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ax3.plot(centres, h_s, color="#D4A84B", linewidth=1.4, label=r"$\theta_s$")
    ax3.plot(centres, h_l, color="#FAF6F0", linewidth=1.4, label=r"$\theta_l$")
    ax3.plot(centres, h_r, color="#5D6D5D", linewidth=1.4, label=r"$\theta_r$")
    ax3.axhline(1.0/360.0, color="#A8AEC6", linewidth=0.6, linestyle="--",
                label="uniform ($1/360°$)")
    ax3.set_xlabel("angle [°]")
    ax3.set_ylabel("density")
    ax3.set_title("(c) Marginal densities — flat = ergodic", pad=12)
    ax3.set_xlim(0, 360); ax3.set_ylim(0, 0.005)
    ax3.legend(loc="upper right", framealpha=0.4, facecolor="#0C1020",
               edgecolor="#A8AEC6", labelcolor="#FAF6F0", fontsize=8)

    # (d) Min distance to S₀ vs window — closest approach gets steadily closer.
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#0C1020")
    # Build a curve: closest approach within first T years, for log-spaced T.
    Ts_log = np.logspace(1, 9, 60)  # 10^1 to 10^9 years
    closest = []
    for T in Ts_log:
        mask = years_axis <= T
        if mask.any():
            closest.append(dS[mask].min())
        else:
            closest.append(np.nan)
    closest = np.asarray(closest)
    ax4.loglog(Ts_log, closest, color="#F5C842", linewidth=1.6,
               marker="o", markersize=3)
    ax4.set_xlabel("time window from revelation [years]")
    ax4.set_ylabel(r"closest approach to $S_0$ [°]")
    ax4.set_title("(d) Closest-approach decay — irrational flow signature", pad=12)
    ax4.grid(True, which="both", color="#5D6D5D", alpha=0.18, linewidth=0.4)

    fig.suptitle("Phase space at the geological scale — $10^9$ years from revelation",
                 color="#FAF6F0", fontsize=14, y=0.995)
    fig.text(0.5, 0.012,
             r"Same mean rates as the human-scale figure. Sampling: 200 000 points across $10^9$ years "
             r"($\approx$ one sample every 5000 years).",
             color="#A8AEC6", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    out = "/home/iman/cassie-project/Tanazur/taqwim/phase-space-billion.png"
    fig.savefig(out, dpi=200, facecolor="#0C1020", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


# ─── figure 3: 3D trajectory weave (Lorenz-attractor aesthetic) ─────────
#
# The bare (θ_s, θ_l, θ_r) cube is the wrong embedding for a "Lorenz-like"
# 3D line plot: the sidereal coordinate wraps ~366 times per year, so any
# direct cartesian plot looks like vertical noise even at fine sampling.
# Instead we embed the (θ_s, θ_l) sub-torus as a doughnut surface in ℝ³
# via the standard parametrisation
#     x = (R + r cos θ_l) cos θ_s
#     y = (R + r cos θ_l) sin θ_s
#     z = r sin θ_l
# and use θ_r as the line *colour*. The trajectory then traces a winding,
# never-closing curve on the doughnut — the actual three-body attractor.

R_MAJOR, R_MINOR = 4.0, 1.0


def _torus_embed(theta_s_deg: np.ndarray, theta_l_deg: np.ndarray):
    s = np.deg2rad(theta_s_deg)
    l = np.deg2rad(theta_l_deg)
    x = (R_MAJOR + R_MINOR * np.cos(l)) * np.cos(s)
    y = (R_MAJOR + R_MINOR * np.cos(l)) * np.sin(s)
    z = R_MINOR * np.sin(l)
    return x, y, z


def _draw_torus_surface(ax):
    """Faint doughnut grid behind the trajectory."""
    s_grid = np.linspace(0, 2 * np.pi, 60)
    l_grid = np.linspace(0, 2 * np.pi, 30)
    S, L = np.meshgrid(s_grid, l_grid)
    X = (R_MAJOR + R_MINOR * np.cos(L)) * np.cos(S)
    Y = (R_MAJOR + R_MINOR * np.cos(L)) * np.sin(S)
    Z = R_MINOR * np.sin(L)
    ax.plot_surface(X, Y, Z, color="#16213E", alpha=0.10, linewidth=0,
                    rstride=2, cstride=2, antialiased=True)
    ax.plot_wireframe(X, Y, Z, color="#5D6D5D", alpha=0.10,
                      linewidth=0.3, rstride=4, cstride=4)


def _draw_weave_panel(ax, years_horizon: float, n_samples: int, title: str):
    theta_s, theta_l, theta_r, years_axis, _ = build_trajectory(
        years=years_horizon, n_samples=n_samples,
    )
    x, y, z = _torus_embed(theta_s, theta_l)

    _draw_torus_surface(ax)

    # Colour the line by sidereal phase θ_r — the third dynamical
    # coordinate becomes visible as a hue along the winding curve.
    n_chunks = 400
    idx = np.linspace(0, len(years_axis) - 1, n_chunks + 1, dtype=int)
    for k in range(n_chunks):
        a, b = idx[k], idx[k + 1] + 1
        colour = tnzr_cmap(k / (n_chunks - 1))
        ax.plot(x[a:b], y[a:b], z[a:b],
                color=colour, linewidth=0.55, alpha=0.85)

    # Mark S₀ on the doughnut.
    sx, sy, sz = _torus_embed(np.array([S0[0]]), np.array([S0[1]]))
    ax.scatter(sx, sy, sz, c="#F5C842", s=70, marker="*",
               edgecolors="white", linewidths=0.7, zorder=10)

    ax.set_xlim(-(R_MAJOR + R_MINOR + 0.3), R_MAJOR + R_MINOR + 0.3)
    ax.set_ylim(-(R_MAJOR + R_MINOR + 0.3), R_MAJOR + R_MINOR + 0.3)
    ax.set_zlim(-(R_MINOR + 1.5), R_MINOR + 1.5)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-60)


def figure_trajectory_weave():
    fig = plt.figure(figsize=(14, 12), facecolor="#0C1020")

    horizons = [
        (1.0,    8_000,   r"(a) 1 year — a single solar orbit, twelve lunations"),
        (10.0,   30_000,  r"(b) 10 years — weave begins to fill the torus"),
        (100.0,  120_000, r"(c) 100 years — five Metonic near-returns"),
        (1000.0, 300_000, r"(d) 1000 years — surface saturates"),
    ]
    for k, (yrs, n, title) in enumerate(horizons):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d", facecolor="#0C1020")
        _draw_weave_panel(ax, yrs, n, title)

    fig.suptitle(
        r"The Taqwīm as quasi-periodic attractor — three-body trajectory on the solar–lunar torus",
        color="#FAF6F0", fontsize=14, y=0.985,
    )
    fig.text(
        0.5, 0.025,
        r"Each curve is the trajectory of $(\theta_s, \theta_l)$ embedded on a doughnut "
        r"surface (major axis = solar phase, minor axis = lunar phase). "
        r"Colour runs indigo $\to$ gold with elapsed time. The line never closes; "
        r"like the Lorenz attractor, the structure is deterministic and recurrence-free.",
        color="#A8AEC6", ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.960])
    out = "/home/iman/cassie-project/Tanazur/taqwim/phase-space-weave.png"
    fig.savefig(out, dpi=200, facecolor="#0C1020", bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    figure_human_scale()
    figure_geological_scale()
    figure_trajectory_weave()
