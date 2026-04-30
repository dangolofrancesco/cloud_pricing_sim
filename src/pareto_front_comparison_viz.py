"""
pareto_front_comparison_viz.py
================================
Visualization of three Pareto fronts (Upper Bound, Lower Bound, Online)
and their pairwise distances, directly in MAXIMIZATION convention.

All functions accept raw objective matrices (n, 3) with columns
[V_sat, V_prof, V_sus], all in [0, 1], all to be maximized.

Public API
----------
  plot_2d_three_fronts(results, obj_x, obj_y)
      Pairwise 2-D projection with cloud + fronts + distance arrows.

  plot_3d_three_fronts(results, elev, azim)
      Full 3-D scatter of the three fronts.

  plot_all_2d_projections(results)
      3-panel strip of all pairwise objective projections.

  plot_distance_summary(results)
      Bar chart of all indicators side by side.

  plot_hv_efficiency(hv_ub, hv_lb, hv_online_runs)
      Violin plot of HV distributions over multiple runs + eta_HV.

  plot_three_front_dashboard(results, hv_runs)
      Combined 6-panel thesis figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from typing import Optional, Dict, List, Tuple

from .pareto_distances import (
    pareto_filter_max,
    hypervolume_3d_max,
    hv_contributions_max,
)

__all__ = [
    "plot_2d_three_fronts",
    "plot_3d_three_fronts",
    "plot_all_2d_projections",
    "plot_distance_summary",
    "plot_hv_efficiency",
    "plot_three_front_dashboard",
]

# ── Consistent colour palette ─────────────────────────────────────────────────
C = {
    "ub":          "#185FA5",   # deep blue  — LP upper bound
    "lb":          "#888780",   # warm gray  — random lower bound
    "ol":          "#E24B4A",   # coral red  — online mechanism
    "ub_cloud":    "#D0E2F5",
    "lb_cloud":    "#E0DFD9",
    "ol_cloud":    "#FAE0DF",
    "utopian":     "#1D9E75",
    "nadir":       "#BA7517",
    "gap_ub_ol":   "#185FA5",
    "gap_ol_lb":   "#E24B4A",
    "annotation":  "#2c2c2a",
}

OBJ_LABELS = {
    "V_sat":  r"$\tilde{V}_{\mathrm{sat}}$ (Satisfaction)",
    "V_prof": r"$\tilde{V}_{\mathrm{prof}}$ (Profit)",
    "V_sus":  r"$\tilde{V}_{\mathrm{sus}}$ (Sustainability)",
}
OBJ_IDX = {"V_sat": 0, "V_prof": 1, "V_sus": 2}

FRONT_LABELS = {
    "ub": r"$\mathcal{PF}^{\mathrm{UB}}$ (Fluid LP)",
    "lb": r"$\mathcal{PF}^{\mathrm{LB}}$ (Random)",
    "ol": r"$\mathcal{PF}^{\mathrm{Online}}$",
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_front_2d(ax, pts, color, label, cloud_pts=None,
                   cloud_color=None, ms=6, lw=1.5, zorder=4):
    """Draw a 2-D Pareto front with optional background cloud."""
    if cloud_pts is not None and len(cloud_pts) > 0:
        ax.scatter(cloud_pts[:, 0], cloud_pts[:, 1],
                   c=cloud_color or C["lb_cloud"], s=10,
                   alpha=0.30, zorder=1)
    if len(pts) == 0:
        return
    sort_idx = np.argsort(pts[:, 0])
    px, py = pts[sort_idx, 0], pts[sort_idx, 1]
    ax.scatter(px, py, c=color, s=ms**2, zorder=zorder + 1,
               edgecolors="white", linewidths=0.4)
    ax.plot(px, py, color=color, lw=lw, alpha=0.7, zorder=zorder, label=label)


def _style_ax(ax, xlabel, ylabel, title, legend=True):
    ax.set_xlabel(OBJ_LABELS.get(xlabel, xlabel), fontsize=10)
    ax.set_ylabel(OBJ_LABELS.get(ylabel, ylabel), fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    if legend:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SINGLE 2-D PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_2d_three_fronts(
    results:    dict,
    obj_x:      str  = "V_prof",
    obj_y:      str  = "V_sat",
    raw_ub:     Optional[np.ndarray] = None,
    raw_lb:     Optional[np.ndarray] = None,
    raw_ol:     Optional[np.ndarray] = None,
    show_gaps:  bool = True,
    save_path:  Optional[str] = None,
    ax:         Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    2-D pairwise projection of all three Pareto fronts.

    Parameters
    ----------
    results  : output of compare_three_fronts().
    obj_x    : column for x-axis ('V_sat', 'V_prof', or 'V_sus').
    obj_y    : column for y-axis.
    raw_ub   : (N_ub, 3) all UB solutions (for background cloud).
    raw_lb   : (N_lb, 3) all LB solutions (for background cloud).
    raw_ol   : (N_ol, 3) all Online solutions (for background cloud).
    show_gaps: draw vertical gap annotations between fronts.
    save_path: if provided, save the figure here.
    ax       : existing axes (for subplot embedding).
    """
    ix, iy = OBJ_IDX[obj_x], OBJ_IDX[obj_y]

    # Extract 2D projection and recompute Pareto on 2D axes only
    # to avoid zig-zag from 3D-optimal points sacrificing one dimension
    nd_ub_full = results["nd_ub"]
    nd_lb_full = results["nd_lb"]
    nd_ol_full = results["nd_ol"]
    
    nd_ub = pareto_filter_max(nd_ub_full[:, [ix, iy]])
    nd_lb = pareto_filter_max(nd_lb_full[:, [ix, iy]])
    nd_ol = pareto_filter_max(nd_ol_full[:, [ix, iy]])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("white")

    # Background clouds
    for raw, cc in [(raw_lb, C["lb_cloud"]), (raw_ol, C["ol_cloud"]),
                    (raw_ub, C["ub_cloud"])]:
        if raw is not None and len(raw) > 0:
            ax.scatter(raw[:, ix], raw[:, iy], c=cc, s=9,
                       alpha=0.28, zorder=1)

    # Pareto fronts
    _draw_front_2d(ax, nd_lb, C["lb"], FRONT_LABELS["lb"], zorder=3)
    _draw_front_2d(ax, nd_ol, C["ol"], FRONT_LABELS["ol"], zorder=4)
    _draw_front_2d(ax, nd_ub, C["ub"], FRONT_LABELS["ub"], zorder=5)

    # Utopian + Nadir reference points
    ax.scatter(1, 1, marker="*", s=200, c=C["utopian"], zorder=8,
               label=r"$z^{\mathrm{ideal}} = (1,1)$", edgecolors="white")
    ax.scatter(0, 0, marker="D", s=80,  c=C["nadir"],   zorder=8,
               label=r"$z^{\mathrm{nadir}} = (0,0)$", edgecolors="white")

    # Gap shading between LB and UB
    if show_gaps:
        for nd_a, nd_b, color, alpha in [
            (nd_ub, nd_ol, C["gap_ub_ol"], 0.07),
            (nd_ol, nd_lb, C["gap_ol_lb"], 0.07),
        ]:
            # Simple shading: fill between the two front y-values at shared x
            x_min = max(nd_a[:, 0].min(), nd_b[:, 0].min())
            x_max = min(nd_a[:, 0].max(), nd_b[:, 0].max())
            if x_max > x_min:
                x_fill = np.linspace(x_min, x_max, 80)
                y_a = np.interp(x_fill,
                                nd_a[np.argsort(nd_a[:, 0]), 0],
                                nd_a[np.argsort(nd_a[:, 0]), 1])
                y_b = np.interp(x_fill,
                                nd_b[np.argsort(nd_b[:, 0]), 0],
                                nd_b[np.argsort(nd_b[:, 0]), 1])
                ax.fill_between(x_fill, np.minimum(y_a, y_b),
                                np.maximum(y_a, y_b),
                                alpha=alpha, color=color, zorder=2)

    # Efficiency annotation
    eta  = results["eta_HV"]
    rho  = results["rho_IGD_plus"]
    ax.annotate(
        f"$\\eta_{{HV}}={eta:.3f}$\n$\\rho_{{IGD^+}}={rho:.3f}$",
        xy=(0.03, 0.97), xycoords="axes fraction",
        fontsize=8.5, va="top", color=C["annotation"],
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=C["annotation"], alpha=0.85),
    )

    title = (f"Pareto Fronts: {OBJ_LABELS[obj_x].split('(')[1].rstrip(')')} "
             f"vs {OBJ_LABELS[obj_y].split('(')[1].rstrip(')')}")
    _style_ax(ax, obj_x, obj_y, title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ALL THREE 2-D PROJECTIONS (3-panel strip)
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_2d_projections(
    results:   dict,
    raw_ub:    Optional[np.ndarray] = None,
    raw_lb:    Optional[np.ndarray] = None,
    raw_ol:    Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-panel strip: all pairwise objective projections.
    """
    pairs = [("V_prof", "V_sat"), ("V_prof", "V_sus"), ("V_sat", "V_sus")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Pareto Front Comparison — All Pairwise Projections\n"
        r"$\mathcal{PF}^{\mathrm{UB}}$ (LP)  |  "
        r"$\mathcal{PF}^{\mathrm{Online}}$  |  "
        r"$\mathcal{PF}^{\mathrm{LB}}$ (Random)",
        fontsize=12, y=1.02,
    )

    for ax, (ox, oy) in zip(axes, pairs):
        plot_2d_three_fronts(results, obj_x=ox, obj_y=oy,
                             raw_ub=raw_ub, raw_lb=raw_lb, raw_ol=raw_ol,
                             show_gaps=True, ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3.  3-D PARETO SURFACE
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_three_fronts(
    results:   dict,
    raw_ub:    Optional[np.ndarray] = None,
    raw_lb:    Optional[np.ndarray] = None,
    raw_ol:    Optional[np.ndarray] = None,
    elev:      float = 25,
    azim:      float = 135,
    save_path: Optional[str] = None,
    ax=None,
) -> object:
    """
    3-D scatter of all three Pareto fronts with optional background clouds.
    """
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=elev, azim=azim)

    # Background clouds
    for raw, cc, alpha in [(raw_lb, C["lb_cloud"], 0.22),
                           (raw_ol, C["ol_cloud"], 0.22),
                           (raw_ub, C["ub_cloud"], 0.18)]:
        if raw is not None and len(raw) > 0:
            ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2],
                       c=cc, s=8, alpha=alpha, depthshade=True)

    # Pareto front surfaces
    for key, nd, marker, size, zorder in [
        ("lb", results["nd_lb"], "o", 35, 3),
        ("ol", results["nd_ol"], "s", 45, 4),
        ("ub", results["nd_ub"], "^", 55, 5),
    ]:
        ax.scatter(nd[:, 0], nd[:, 1], nd[:, 2],
                   c=C[key], s=size, marker=marker,
                   label=FRONT_LABELS[key],
                   edgecolors="white", linewidths=0.3,
                   depthshade=False, zorder=zorder)

        # Light surface triangulation
        if len(nd) >= 4:
            try:
                from matplotlib.tri import Triangulation
                tri = Triangulation(nd[:, 0], nd[:, 1])
                ax.plot_trisurf(nd[:, 0], nd[:, 1], nd[:, 2],
                                triangles=tri.triangles,
                                color=C[key], alpha=0.08,
                                edgecolor="none")
            except Exception:
                pass

    # Utopian & Nadir
    ax.scatter(1, 1, 1, marker="*", s=250, c=C["utopian"], zorder=8,
               label=r"$z^{\mathrm{ideal}}$")
    ax.scatter(0, 0, 0, marker="D", s=100, c=C["nadir"],   zorder=8,
               label=r"$z^{\mathrm{nadir}}$")

    ax.set_xlabel(r"$\tilde{V}_{sat}$", fontsize=9, labelpad=8)
    ax.set_ylabel(r"$\tilde{V}_{prof}$", fontsize=9, labelpad=8)
    ax.set_zlabel(r"$\tilde{V}_{sus}$",  fontsize=9, labelpad=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title(
        f"3-D Pareto Surface  "
        f"($\\eta_{{HV}}={results['eta_HV']:.3f}$, "
        f"$\\rho_{{IGD^+}}={results['rho_IGD_plus']:.3f}$)",
        fontsize=10, pad=12,
    )

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INDICATOR BAR-CHART SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_distance_summary(
    results:   dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of all distance indicators for LB and Online vs UB reference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Indicator Suite — LB (Random) vs Online vs UB (LP Reference)",
                 fontsize=12, y=1.02)

    # ── Panel 1: HV (absolute) ───────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(["LB\n(Random)", "Online", "UB\n(LP)"],
                  [results["HV_lb"], results["HV_ol"], results["HV_ub"]],
                  color=[C["lb"], C["ol"], C["ub"]], alpha=0.80,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Hypervolume  ↑", fontsize=10)
    ax.set_title("HV (strictly Pareto-compliant)", fontsize=9)
    # Annotate eta_HV range
    ax.annotate("", xy=(1, results["HV_ol"]), xytext=(1, results["HV_ub"]),
                arrowprops=dict(arrowstyle="<->", color=C["gap_ub_ol"], lw=1.5))
    ax.annotate("", xy=(1, results["HV_lb"]), xytext=(1, results["HV_ol"]),
                arrowprops=dict(arrowstyle="<->", color=C["gap_ol_lb"], lw=1.5))
    ax.text(1.15, (results["HV_ub"] + results["HV_ol"]) / 2,
            f"regret\n{(results['HV_ub']-results['HV_ol']):.4f}",
            fontsize=7.5, color=C["gap_ub_ol"], ha="left")
    ax.text(1.15, (results["HV_ol"] + results["HV_lb"]) / 2,
            f"above\nrandom\n{(results['HV_ol']-results['HV_lb']):.4f}",
            fontsize=7.5, color=C["gap_ol_lb"], ha="left")
    ax.grid(True, axis="y", alpha=0.2)
    for bar, val in zip(bars, [results["HV_lb"], results["HV_ol"], results["HV_ub"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 2: IGD+ and ε+ against UB ─────────────────────────────────────
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w/2,
                [results["IGD_plus_lb_ub"], results["eps_lb_ub"]],
                width=w, color=C["lb"], alpha=0.80,
                label="LB (Random)", edgecolor="white")
    b2 = ax.bar(x + w/2,
                [results["IGD_plus_ol_ub"], results["eps_ol_ub"]],
                width=w, color=C["ol"], alpha=0.80,
                label="Online", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["IGD⁺  ↓", "ε⁺  ↓"], fontsize=10)
    ax.set_ylabel("Distance to UB  ↓", fontsize=10)
    ax.set_title("Distance Indicators vs LP Reference", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    for bars_, key in [(b1, "lb"), (b2, "ol")]:
        for bar in bars_:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7.5)

    # ── Panel 3: Efficiency ratios ────────────────────────────────────────────
    ax = axes[2]
    metrics = ["$\\eta_{HV}$", "$\\rho_{IGD^+}$"]
    values  = [results["eta_HV"], results["rho_IGD_plus"]]
    colors  = [C["ub"], C["ol"]]
    bars2 = ax.bar(metrics, values, color=colors, alpha=0.80,
                   edgecolor="white", linewidth=1.5)
    ax.axhline(1.0, color="#1D9E75", lw=1.2, ls="--", alpha=0.6,
               label="Perfect (= 1)")
    ax.axhline(0.0, color="#888780", lw=1.0, ls=":", alpha=0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Efficiency ratio  ↑", fontsize=10)
    ax.set_title("Thesis Primary Metrics\n"
                 r"$\eta_{HV}=\frac{HV_{OL}-HV_{LB}}{HV_{UB}-HV_{LB}}$  "
                 r"$\rho=1-\frac{IGD^+_{OL}}{IGD^+_{LB}}$",
                 fontsize=8.5)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    for bar, val in zip(bars2, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HV DISTRIBUTION OVER MULTIPLE RUNS
# ══════════════════════════════════════════════════════════════════════════════

def plot_hv_efficiency(
    hv_ub:         float,
    hv_lb_runs:    np.ndarray,
    hv_ol_runs:    np.ndarray,
    hv_lb_single:  Optional[float] = None,
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Violin plot of HV distributions over R runs for LB and Online,
    with horizontal lines for UB and the eta_HV efficiency range.

    Parameters
    ----------
    hv_ub       : scalar HV of the LP upper bound.
    hv_lb_runs  : (R,) HV values from R random lower-bound runs.
    hv_ol_runs  : (R,) HV values from R online mechanism runs.
    hv_lb_single: optional scalar (median of hv_lb_runs) for annotation.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("white")

    data   = [hv_lb_runs, hv_ol_runs]
    labels = ["LB (Random)", "Online"]
    colors = [C["lb"], C["ol"]]

    vp = ax.violinplot(data, positions=[1, 2], showmedians=True,
                       showextrema=True, widths=0.5)

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.60)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in vp:
            vp[part].set_edgecolor("#2c2c2a")
            vp[part].set_linewidth(1.5)

    # Add box + scatter overlay
    for pos, vals, color in zip([1, 2], data, colors):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.1, pos + 0.1], [med, med],
                color="white", lw=2.5, zorder=5)
        ax.scatter(np.random.normal(pos, 0.05, len(vals)), vals,
                   c=color, s=8, alpha=0.40, zorder=3)

    # UB horizontal line
    ax.axhline(hv_ub, color=C["ub"], lw=2.0, ls="--",
               label=f"UB (LP)  HV={hv_ub:.4f}")

    # eta_HV annotation
    med_lb = float(np.median(hv_lb_runs))
    med_ol = float(np.median(hv_ol_runs))
    eta_hv = (med_ol - med_lb) / (hv_ub - med_lb + 1e-12)

    ax.annotate("", xy=(2.6, hv_ub), xytext=(2.6, med_ol),
                arrowprops=dict(arrowstyle="<->", color=C["ub"], lw=1.8))
    ax.text(2.65, (hv_ub + med_ol) / 2,
            f"Regret\n{hv_ub-med_ol:.4f}", fontsize=8,
            color=C["ub"], ha="left")

    ax.annotate("", xy=(2.6, med_ol), xytext=(2.6, med_lb),
                arrowprops=dict(arrowstyle="<->", color=C["ol"], lw=1.8))
    ax.text(2.65, (med_ol + med_lb) / 2,
            f"$\\eta_{{HV}}={eta_hv:.3f}$\n{med_ol-med_lb:.4f}",
            fontsize=8, color=C["ol"], ha="left", fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Hypervolume  ↑", fontsize=11)
    ax.set_title(f"HV Distribution over {len(hv_lb_runs)} Runs  "
                 f"($\\eta_{{HV}}={eta_hv:.3f}$)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.18)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COMBINED THESIS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_three_front_dashboard(
    results:    dict,
    hv_lb_runs: Optional[np.ndarray] = None,
    hv_ol_runs: Optional[np.ndarray] = None,
    raw_ub:     Optional[np.ndarray] = None,
    raw_lb:     Optional[np.ndarray] = None,
    raw_ol:     Optional[np.ndarray] = None,
    elev:       float = 28,
    azim:       float = 130,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Six-panel thesis dashboard:
      [R1C1] Profit vs Satisfaction 2-D
      [R1C2] Profit vs Sustainability 2-D
      [R1C3] Satisfaction vs Sustainability 2-D
      [R2C1-2] 3-D surface (wide)
      [R2C3] Indicator bar summary (3 sub-bars)

    Parameters
    ----------
    results     : output of compare_three_fronts().
    hv_lb_runs  : (R,) HV values over R runs for violin panel (optional).
    hv_ol_runs  : (R,) HV values over R runs for violin panel (optional).
    raw_ub/lb/ol: background cloud arrays (optional).
    """
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Multi-Objective Cloud Resource Allocation — Pareto Front Analysis\n"
        r"$\mathcal{PF}^{\mathrm{UB}}$ (Fluid LP)  $\geq$  "
        r"$\mathcal{PF}^{\mathrm{Online}}$  $\geq$  "
        r"$\mathcal{PF}^{\mathrm{LB}}$ (Random Acceptance)",
        fontsize=13, y=1.01,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.30)

    # ── Row 1: 2-D projections ────────────────────────────────────────────────
    pairs = [("V_prof", "V_sat"), ("V_prof", "V_sus"), ("V_sat", "V_sus")]
    for col, (ox, oy) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, col])
        plot_2d_three_fronts(results, obj_x=ox, obj_y=oy,
                             raw_ub=raw_ub, raw_lb=raw_lb, raw_ol=raw_ol,
                             show_gaps=True, ax=ax)

    # ── Row 2 left: 3-D surface (spans 2 columns) ────────────────────────────
    ax3d = fig.add_subplot(gs[1, :2], projection="3d")
    plot_3d_three_fronts(results,
                         raw_ub=raw_ub, raw_lb=raw_lb, raw_ol=raw_ol,
                         elev=elev, azim=azim, ax=ax3d)

    # ── Row 2 right: summary table ───────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.set_facecolor("white")
    ax_t.axis("off")

    eta  = results["eta_HV"]
    rho  = results["rho_IGD_plus"]

    rows = [
        ["Metric",        "LB (Rand)",
         "Online",       "UB (LP)"],
        ["HV ↑",
         f"{results['HV_lb']:.4f}",
         f"{results['HV_ol']:.4f}",
         f"{results['HV_ub']:.4f}"],
        ["IGD⁺ ↓",
         f"{results['IGD_plus_lb_ub']:.4f}",
         f"{results['IGD_plus_ol_ub']:.4f}",
         "0 (ref)"],
        ["GD⁺ ↓",
         f"{results['GD_plus_lb_ub']:.4f}",
         f"{results['GD_plus_ol_ub']:.4f}",
         "0 (ref)"],
        ["ε⁺ ↓",
         f"{results['eps_lb_ub']:.4f}",
         f"{results['eps_ol_ub']:.4f}",
         "0 (ref)"],
        ["C(UB→·)",
         f"{results['C_ub_lb']:.3f}",
         f"{results['C_ub_ol']:.3f}",
         "—"],
        ["C(·→UB)",
         f"{results['C_lb_ub']:.3f}",
         f"{results['C_ol_ub']:.3f}",
         "—"],
        ["η_HV",   "0",    f"{eta:.4f}", "1"],
        ["ρ_IGD⁺", "0",    f"{rho:.4f}", "1"],
    ]

    row_colors = [
        ["#2c2c2a"] * 4,
        *[["#F4F4F2"] * 4] * 6,
        ["#E1F5EE"] * 4,
        ["#EBF3FB"] * 4,
    ]

    t = ax_t.table(
        cellText=rows[1:], colLabels=rows[0],
        cellLoc="center", loc="center",
        cellColours=row_colors[1:],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.15, 1.55)
    ax_t.set_title("Indicator Suite Summary", fontsize=10, pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
