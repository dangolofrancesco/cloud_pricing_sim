"""
ParetoFrontVisualizer  (fixed)
==============================
Visualization of Pareto frontiers for multi-objective
cloud resource allocation (Profit, Satisfaction, Sustainability).
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers projection)
from scipy.spatial import ConvexHull
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")


# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "dominated":   "#C4C1B8",   # warm gray
    "linear":      "#E24B4A",   # coral red
    "epsilon":     "#185FA5",   # deep blue
    "chebyshev":   "#1D9E75",   # teal green
    "utopian":     "#1D9E75",   # teal star
    "nadir":       "#BA7517",   # amber diamond
    "convex_hull": "#7F77DD",   # purple
    "hyperplane":  "#185FA5",
    "eps_line":    "#185FA5",
    "linf_ball":   "#1D9E75",
    "missed":      "#E24B4A",
}

# ─── Axis labels ──────────────────────────────────────────────────────────────
OBJECTIVE_LABELS = {
    "V_sat":  r"Customer Satisfaction  $\tilde{V}_{\mathrm{sat}}$",
    "V_prof": r"Provider Profit  $\tilde{V}_{\mathrm{prof}}$",
    "V_sus":  r"Sustainability  $\tilde{V}_{\mathrm{sus}}$",
}
OBJECTIVE_LABELS_RAW = {
    "V_sat":  r"Customer Satisfaction  $V_{\mathrm{sat}}$",
    "V_prof": r"Provider Profit  $V_{\mathrm{prof}}$",
    "V_sus":  r"Sustainability  $V_{\mathrm{sus}}$",
}
OBJECTIVE_SHORT = {
    "V_sat":  r"$\tilde{V}_{sat}$",
    "V_prof": r"$\tilde{V}_{prof}$",
    "V_sus":  r"$\tilde{V}_{sus}$",
}
METHOD_LABELS = {
    "linear":    "Linear Scalarization (Weighted-Sum)",
    "epsilon":   "ε-Constraint Method",
    "chebyshev": "Chebyshev (Minimax) Scalarization",
}

OBJ_KEYS = ["V_sat", "V_prof", "V_sus"]


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _dominates_max(a: np.ndarray, b: np.ndarray) -> bool:
    """True when a Pareto-dominates b under MAXIMIZATION."""
    return bool(np.all(a >= b) and np.any(a > b))


def _extract_objective_matrix(
    solutions: List[Dict],
    keys: List[str] = None,
) -> np.ndarray:
    """
    Return (N, k) matrix of objective values from a list of solution dicts.
    Reads from 'normalized_objectives'.
    """
    if keys is None:
        keys = OBJ_KEYS
    rows = []
    for sol in solutions:
        norm = sol["normalized_objectives"]
        rows.append([norm[k] for k in keys])
    return np.array(rows)


def _compute_pareto_indices_max(obj_matrix: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated rows under MAXIMIZATION.
    O(N^2 · k) — fine for thesis-scale experiments.
    """
    N = obj_matrix.shape[0]
    is_dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if is_dominated[i]:
            continue
        for j in range(N):
            if i == j or is_dominated[j]:
                continue
            if _dominates_max(obj_matrix[j], obj_matrix[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def _pool_matrix(
    solutions: List[Dict],
    dominated_pool: Optional[List[Dict]],
    keys: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge `solutions` and optional `dominated_pool` into one objective matrix.

    Returns
    -------
    pool_matrix : (N_total, k) — all points
    pareto_idx  : indices into pool_matrix that are Pareto-optimal
                  (computed only over `solutions` so pool noise does not
                   pollute the front)
    """
    if keys is None:
        keys = OBJ_KEYS
    sol_matrix = _extract_objective_matrix(solutions, keys)
    pareto_idx = _compute_pareto_indices_max(sol_matrix)

    if dominated_pool:
        dom_matrix = _extract_objective_matrix(dominated_pool, keys)
        full_matrix = np.vstack([sol_matrix, dom_matrix])
        # Shift pareto indices — they index into sol_matrix, same as full_matrix[:len(sol_matrix)]
        return full_matrix, pareto_idx, len(sol_matrix)
    return sol_matrix, pareto_idx, len(sol_matrix)


# ═════════════════════════════════════════════════════════════════════════════
class ParetoFrontVisualizer:
    """
    Thesis-quality Pareto front visualizer.

    Parameters
    ----------
    optimizer : StaticMultiObjectiveOptimizer  (optional)
        Provides z_ideal / z_nadir reference points.
        Must have attributes z_sat_max, z_prof_max, z_carb_max.
    figsize_2d, figsize_3d : tuple
    dpi : int
    """

    def __init__(
        self,
        optimizer=None,
        figsize_2d: Tuple[int, int] = (8, 6),
        figsize_3d: Tuple[int, int] = (10, 8),
        dpi: int = 150,
    ):
        self.optimizer = optimizer
        self.figsize_2d = figsize_2d
        self.figsize_3d = figsize_3d
        self.dpi = dpi

        # FIX-6: reference points stored in NORMALIZED space [0,1]^3
        # so they plot correctly on normalized axes.
        self._z_ideal_norm = None   # (1, 1, 1) in normalized space
        self._z_nadir_norm = None   # (0, 0, 0) in normalized space
        self._has_ref = False

        if optimizer is not None and hasattr(optimizer, "z_sat_max"):
            # After normalization:
            #   V_sat_norm  = V_sat  / z_sat_max   → ideal = 1, nadir = 0
            #   V_prof_norm = V_prof / z_prof_max  → ideal = 1, nadir = 0
            #   V_sus_norm  = -C_carbon / z_carb_max → ideal = 0 carbon → norm = 0? 
            # V_sus = -C_carbon; at zero emissions V_sus=0; max V_sus = 0 (no carbon accepted)
            # Nadir V_sus = -z_carb_max → normalized = 0
            # Ideal V_sus = 0 carbon cost → normalized = 1 (after fix in optimizer)
            self._z_ideal_norm = np.array([1.0, 1.0, 1.0])
            self._z_nadir_norm = np.array([0.0, 0.0, 0.0])
            self._has_ref = True

    # ══════════════════════════════════════════════════════════════════════════
    # 2-D  single plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_2d_front(
        self,
        solutions: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        method_name: str = "linear",
        obj_x: str = "V_prof",
        obj_y: str = "V_sat",
        show_dominated: bool = True,
        show_reference_points: bool = True,
        show_convex_hull: bool = False,
        annotate_method_geometry: bool = False,
        title: str = None,
        save_path: str = None,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        """
        2D Pareto front projection with optional dominated background cloud.

        Parameters
        ----------
        solutions : list[dict]
            Pareto-candidate solutions from the optimizer sweep.
        dominated_pool : list[dict] | None
            Extra solutions (e.g. from compute_dominated_pool) used ONLY
            for the gray background scatter.  Never affects front detection.
        obj_x, obj_y : str
            Keys into normalized_objectives: 'V_sat', 'V_prof', 'V_sus'.
        """
        # ── FIX-2 / FIX-3: define obj_keys FIRST, compute indices ONCE ────
        keys = OBJ_KEYS
        idx_x = keys.index(obj_x)
        idx_y = keys.index(obj_y)

        sol_matrix = _extract_objective_matrix(solutions, keys)
        pareto_idx = _compute_pareto_indices_max(sol_matrix)
        dominated_mask_sol = np.ones(len(solutions), dtype=bool)
        dominated_mask_sol[pareto_idx] = False

        # ── FIX-4: build the full pool for gray dots ───────────────────────
        if dominated_pool:
            dom_matrix = _extract_objective_matrix(dominated_pool, keys)
            # All pool points — we label them dominated by definition
            pool_x = np.concatenate([
                sol_matrix[dominated_mask_sol, idx_x],
                dom_matrix[:, idx_x],
            ])
            pool_y = np.concatenate([
                sol_matrix[dominated_mask_sol, idx_y],
                dom_matrix[:, idx_y],
            ])
        else:
            pool_x = sol_matrix[dominated_mask_sol, idx_x]
            pool_y = sol_matrix[dominated_mask_sol, idx_y]

        pareto_x = sol_matrix[pareto_idx, idx_x]
        pareto_y = sol_matrix[pareto_idx, idx_y]
        sort_order = np.argsort(pareto_x)
        pareto_x = pareto_x[sort_order]
        pareto_y = pareto_y[sort_order]

        # ── Figure ────────────────────────────────────────────────────────
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=self.figsize_2d)
            fig.patch.set_facecolor("white")

        method_color = COLORS.get(method_name, COLORS["linear"])

        # Gray dominated cloud
        if show_dominated and len(pool_x) > 0:
            ax.scatter(
                pool_x, pool_y,
                c=COLORS["dominated"], s=22, alpha=0.40,
                label="Dominated solutions", zorder=2,
            )

        # Pareto front
        ax.scatter(
            pareto_x, pareto_y,
            c=method_color, s=55, zorder=5,
            label=f"Pareto Front ({METHOD_LABELS.get(method_name, method_name)})",
            edgecolors="white", linewidths=0.5,
        )
        ax.plot(pareto_x, pareto_y, color=method_color, lw=1.2, alpha=0.6, zorder=4)

        # Reference points (FIX-6: use normalized coordinates)
        if show_reference_points and self._has_ref:
            ideal_xy = np.array([self._z_ideal_norm[idx_x], self._z_ideal_norm[idx_y]])
            nadir_xy = np.array([self._z_nadir_norm[idx_x], self._z_nadir_norm[idx_y]])
            ax.scatter(
                *ideal_xy, marker="*", s=300, c=COLORS["utopian"], zorder=7,
                label=r"$\mathbf{z}^{\mathrm{ideal}}$ (Utopian)",
                edgecolors="white", linewidths=0.5,
            )
            ax.scatter(
                *nadir_xy, marker="D", s=100, c=COLORS["nadir"], zorder=7,
                label=r"$\mathbf{z}^{\mathrm{nadir}}$ (Nadir)",
                edgecolors="white", linewidths=0.5,
            )

        # Convex hull
        if show_convex_hull and len(pareto_x) >= 3:
            hull_pts = np.column_stack([pareto_x, pareto_y])
            try:
                hull = ConvexHull(hull_pts)
                for simplex in hull.simplices:
                    ax.plot(
                        hull_pts[simplex, 0], hull_pts[simplex, 1],
                        color=COLORS["convex_hull"], lw=1.0, ls="--", alpha=0.55,
                    )
            except Exception:
                pass

        # Method geometry annotations
        if annotate_method_geometry:
            self._annotate_geometry(ax, method_name, pareto_x, pareto_y, idx_x, idx_y)

        # Labels
        ax.set_xlabel(OBJECTIVE_LABELS.get(obj_x, obj_x), fontsize=11)
        ax.set_ylabel(OBJECTIVE_LABELS.get(obj_y, obj_y), fontsize=11)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

        if title is None:
            title = f"Pareto Front — {METHOD_LABELS.get(method_name, method_name)}"
        ax.set_title(title, fontsize=11, pad=10)

        if own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return ax

    # ── geometry annotations ──────────────────────────────────────────────────

    def _annotate_geometry(self, ax, method, px, py, idx_x, idx_y):
        if method == "linear":
            cx, cy = np.mean(px), np.mean(py)
            xlim = ax.get_xlim()
            x_range = np.linspace(xlim[0], xlim[1], 120)
            for w in [0.15, 0.35, 0.55, 0.80]:
                w2 = 1 - w
                if abs(w2) < 1e-9:
                    continue
                c_val = w * cx + w2 * cy
                y_line = (c_val - w * x_range) / w2
                ylim = ax.get_ylim()
                valid = (y_line >= ylim[0]) & (y_line <= ylim[1])
                ax.plot(x_range[valid], y_line[valid],
                        color=COLORS["hyperplane"], lw=0.7, alpha=0.22)
            ax.annotate(
                "Weight hyperplanes\n$w^T z = c$",
                xy=(cx * 0.55, cy * 1.1),
                fontsize=7.5, color=COLORS["hyperplane"], ha="center", alpha=0.75,
            )

        elif method == "epsilon":
            y_vals = np.linspace(np.min(py), np.max(py), 7)
            for eps in y_vals:
                ax.axhline(eps, color=COLORS["eps_line"], lw=0.7, alpha=0.28, ls=":")
            ax.annotate(
                r"$\varepsilon$-cuts",
                xy=(np.max(px) * 0.82, np.max(py) * 0.94),
                fontsize=7.5, color=COLORS["eps_line"], ha="center", alpha=0.85,
            )

        elif method == "chebyshev" and self._has_ref:
            center = np.array([self._z_ideal_norm[idx_x], self._z_ideal_norm[idx_y]])
            r_range = max(np.ptp(px), np.ptp(py)) if len(px) > 1 else 0.5
            for r_frac in [0.25, 0.50, 0.80]:
                r = r_frac * r_range
                diamond = plt.Polygon(
                    [[center[0], center[1] - r],
                     [center[0] + r, center[1]],
                     [center[0], center[1] + r],
                     [center[0] - r, center[1]]],
                    fill=False, edgecolor=COLORS["linf_ball"],
                    lw=0.9, ls="--", alpha=0.42,
                )
                ax.add_patch(diamond)
            ax.annotate(
                r"$L_\infty$ balls from $z^{\mathrm{ideal}}$",
                xy=(center[0] * 0.75, center[1] * 0.88),
                fontsize=7.5, color=COLORS["linf_ball"], ha="center", alpha=0.85,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 2-D  all three pairwise projections (one method)
    # ══════════════════════════════════════════════════════════════════════════

    def plot_2d_all_projections(
        self,
        solutions: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        method_name: str = "linear",
        show_dominated: bool = True,
        annotate_method_geometry: bool = False,
        save_path: str = None,
    ) -> plt.Figure:
        """Three-panel strip: Profit×Sat | Profit×Sus | Sat×Sus."""
        pairs = [("V_prof", "V_sat"), ("V_prof", "V_sus"), ("V_sat", "V_sus")]
        pair_titles = ["Profit vs Satisfaction", "Profit vs Sustainability",
                       "Satisfaction vs Sustainability"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Pareto Front — All Pairwise Projections  |  "
            f"{METHOD_LABELS.get(method_name, method_name)}",
            fontsize=12, y=1.02,
        )

        for ax, (ox, oy), pt in zip(axes, pairs, pair_titles):
            # FIX-7: pass dominated_pool through
            self.plot_2d_front(
                solutions,
                dominated_pool=dominated_pool,
                method_name=method_name,
                obj_x=ox, obj_y=oy,
                show_dominated=show_dominated,
                annotate_method_geometry=annotate_method_geometry,
                ax=ax, title=pt,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # 3-D  single plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_3d_front(
        self,
        solutions: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        method_name: str = "linear",
        show_dominated: bool = True,
        show_reference_points: bool = True,
        show_surface: bool = True,
        elev: float = 25,
        azim: float = 135,
        title: str = None,
        save_path: str = None,
        ax=None,
    ):
        """
        3D Pareto surface with optional dominated background cloud.

        Parameters
        ----------
        dominated_pool : list[dict] | None
            Extra solutions used only for the gray 3D scatter.
        """
        keys = OBJ_KEYS
        sol_matrix = _extract_objective_matrix(solutions, keys)
        pareto_idx = _compute_pareto_indices_max(sol_matrix)
        dominated_mask_sol = np.ones(len(solutions), dtype=bool)
        dominated_mask_sol[pareto_idx] = False

        # FIX-5: build full dominated cloud
        if dominated_pool:
            dom_matrix = _extract_objective_matrix(dominated_pool, keys)
            dom_cloud = np.vstack([sol_matrix[dominated_mask_sol], dom_matrix])
        else:
            dom_cloud = sol_matrix[dominated_mask_sol]

        pareto_pts = sol_matrix[pareto_idx]
        method_color = COLORS.get(method_name, COLORS["linear"])

        own_fig = ax is None
        if own_fig:
            fig = plt.figure(figsize=self.figsize_3d)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111, projection="3d")

        ax.view_init(elev=elev, azim=azim)

        # Gray dominated cloud
        if show_dominated and len(dom_cloud) > 0:
            ax.scatter(
                dom_cloud[:, 0], dom_cloud[:, 1], dom_cloud[:, 2],
                c=COLORS["dominated"], s=14, alpha=0.28,
                label="Dominated", depthshade=True,
            )

        # Pareto surface scatter
        ax.scatter(
            pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2],
            c=method_color, s=50, zorder=5,
            label="Pareto Front",
            edgecolors="white", linewidths=0.3, depthshade=True,
        )

        # Triangulated surface
        if show_surface and len(pareto_pts) >= 4:
            try:
                from matplotlib.tri import Triangulation
                tri = Triangulation(pareto_pts[:, 0], pareto_pts[:, 1])
                ax.plot_trisurf(
                    pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2],
                    triangles=tri.triangles,
                    color=method_color, alpha=0.13,
                    edgecolor=method_color, linewidth=0.25,
                )
            except Exception:
                pass

        # Reference points (FIX-6: normalized coordinates)
        if show_reference_points and self._has_ref:
            ax.scatter(
                *self._z_ideal_norm, marker="*", s=300, c=COLORS["utopian"],
                zorder=7, label=r"$z^{\mathrm{ideal}}$",
                edgecolors="white", linewidths=0.5,
            )
            ax.scatter(
                *self._z_nadir_norm, marker="D", s=120, c=COLORS["nadir"],
                zorder=7, label=r"$z^{\mathrm{nadir}}$",
                edgecolors="white", linewidths=0.5,
            )

        ax.set_xlabel(OBJECTIVE_LABELS.get("V_sat", "V_sat"), fontsize=9, labelpad=9)
        ax.set_ylabel(OBJECTIVE_LABELS.get("V_prof", "V_prof"), fontsize=9, labelpad=9)
        ax.set_zlabel(OBJECTIVE_LABELS.get("V_sus", "V_sus"), fontsize=9, labelpad=9)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

        if title is None:
            title = f"3D Pareto Surface — {METHOD_LABELS.get(method_name, method_name)}"
        ax.set_title(title, fontsize=11, pad=14)

        if own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return ax

    # ══════════════════════════════════════════════════════════════════════════
    # Side-by-side method comparison  (2-D)
    # ══════════════════════════════════════════════════════════════════════════

    def plot_method_comparison_2d(
        self,
        solutions_linear: List[Dict],
        solutions_epsilon: List[Dict],
        solutions_chebyshev: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        obj_x: str = "V_prof",
        obj_y: str = "V_sat",
        annotate_method_geometry: bool = True,
        save_path: str = None,
    ) -> plt.Figure:
        """Three-panel side-by-side comparison of all scalarization methods."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Scalarization Method Comparison — 2D Pareto Front Projection",
            fontsize=13, y=1.02,
        )

        configs = [
            (solutions_linear,    "linear",    axes[0]),
            (solutions_epsilon,   "epsilon",   axes[1]),
            (solutions_chebyshev, "chebyshev", axes[2]),
        ]
        for sols, method, ax in configs:
            self.plot_2d_front(
                sols,
                dominated_pool=dominated_pool,
                method_name=method,
                obj_x=obj_x, obj_y=obj_y,
                show_dominated=True,
                show_reference_points=True,
                annotate_method_geometry=annotate_method_geometry,
                ax=ax,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Side-by-side method comparison  (3-D)
    # ══════════════════════════════════════════════════════════════════════════

    def plot_method_comparison_3d(
        self,
        solutions_linear: List[Dict],
        solutions_epsilon: List[Dict],
        solutions_chebyshev: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        elev: float = 25,
        azim: float = 135,
        save_path: str = None,
    ) -> plt.Figure:
        """Three-panel 3D comparison."""
        fig = plt.figure(figsize=(20, 6.5))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Scalarization Method Comparison — 3D Pareto Surface",
            fontsize=13, y=1.02,
        )

        configs = [
            (solutions_linear,    "linear",    131),
            (solutions_epsilon,   "epsilon",   132),
            (solutions_chebyshev, "chebyshev", 133),
        ]
        for sols, method, subplot_idx in configs:
            ax = fig.add_subplot(subplot_idx, projection="3d")
            self.plot_3d_front(
                sols,
                dominated_pool=dominated_pool,
                method_name=method,
                elev=elev, azim=azim,
                show_dominated=True,
                show_reference_points=True,
                show_surface=True,
                ax=ax,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Overlay: all methods on one 2-D plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_overlay_2d(
        self,
        solutions_dict: Dict[str, List[Dict]],
        dominated_pool: Optional[List[Dict]] = None,
        obj_x: str = "V_prof",
        obj_y: str = "V_sat",
        title: str = "Pareto Front Overlay — All Methods",
        save_path: str = None,
    ) -> plt.Figure:
        """Overlay Pareto fronts from multiple methods on one 2D plot."""
        fig, ax = plt.subplots(figsize=self.figsize_2d)
        fig.patch.set_facecolor("white")

        keys = OBJ_KEYS
        idx_x = keys.index(obj_x)
        idx_y = keys.index(obj_y)

        # Draw dominated background once (union of all pools)
        if dominated_pool:
            dom_matrix = _extract_objective_matrix(dominated_pool, keys)
            ax.scatter(
                dom_matrix[:, idx_x], dom_matrix[:, idx_y],
                c=COLORS["dominated"], s=18, alpha=0.35, zorder=1,
                label="Dominated solutions",
            )

        marker_styles = {"linear": ("o", 50), "epsilon": ("s", 45), "chebyshev": ("^", 50)}

        for method_name, solutions in solutions_dict.items():
            obj_matrix = _extract_objective_matrix(solutions, keys)
            pareto_idx = _compute_pareto_indices_max(obj_matrix)
            pareto_pts = obj_matrix[pareto_idx]
            sort_order = np.argsort(pareto_pts[:, idx_x])
            px = pareto_pts[sort_order, idx_x]
            py = pareto_pts[sort_order, idx_y]

            marker, size = marker_styles.get(method_name, ("o", 40))
            color = COLORS.get(method_name, "#333333")
            ax.scatter(px, py, c=color, s=size, marker=marker, zorder=5,
                       label=METHOD_LABELS.get(method_name, method_name),
                       edgecolors="white", linewidths=0.4)
            ax.plot(px, py, color=color, lw=1.0, alpha=0.45, zorder=4)

        if self._has_ref:
            ax.scatter(
                self._z_ideal_norm[idx_x], self._z_ideal_norm[idx_y],
                marker="*", s=300, c=COLORS["utopian"], zorder=7,
                label=r"$z^{\mathrm{ideal}}$", edgecolors="white", linewidths=0.5,
            )
            ax.scatter(
                self._z_nadir_norm[idx_x], self._z_nadir_norm[idx_y],
                marker="D", s=100, c=COLORS["nadir"], zorder=7,
                label=r"$z^{\mathrm{nadir}}$", edgecolors="white", linewidths=0.5,
            )

        ax.set_xlabel(OBJECTIVE_LABELS.get(obj_x, obj_x), fontsize=11)
        ax.set_ylabel(OBJECTIVE_LABELS.get(obj_y, obj_y), fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Overlay: all methods on one 3-D plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_overlay_3d(
        self,
        solutions_dict: Dict[str, List[Dict]],
        dominated_pool: Optional[List[Dict]] = None,
        elev: float = 25,
        azim: float = 135,
        title: str = "3D Pareto Front Overlay — All Methods",
        save_path: str = None,
    ) -> plt.Figure:
        """Overlay Pareto fronts from multiple methods on one 3D plot."""
        fig = plt.figure(figsize=self.figsize_3d)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        keys = OBJ_KEYS

        if dominated_pool:
            dom_matrix = _extract_objective_matrix(dominated_pool, keys)
            ax.scatter(
                dom_matrix[:, 0], dom_matrix[:, 1], dom_matrix[:, 2],
                c=COLORS["dominated"], s=12, alpha=0.22,
                label="Dominated", depthshade=True,
            )

        marker_styles = {"linear": ("o", 45), "epsilon": ("s", 40), "chebyshev": ("^", 45)}

        for method_name, solutions in solutions_dict.items():
            obj_matrix = _extract_objective_matrix(solutions, keys)
            pareto_idx = _compute_pareto_indices_max(obj_matrix)
            pareto_pts = obj_matrix[pareto_idx]

            marker, size = marker_styles.get(method_name, ("o", 40))
            color = COLORS.get(method_name, "#333333")
            ax.scatter(
                pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2],
                c=color, s=size, marker=marker, zorder=5,
                label=METHOD_LABELS.get(method_name, method_name),
                edgecolors="white", linewidths=0.3, depthshade=True,
            )

        if self._has_ref:
            ax.scatter(*self._z_ideal_norm, marker="*", s=300, c=COLORS["utopian"],
                       zorder=7, label=r"$z^{\mathrm{ideal}}$")
            ax.scatter(*self._z_nadir_norm, marker="D", s=120, c=COLORS["nadir"],
                       zorder=7, label=r"$z^{\mathrm{nadir}}$")

        ax.set_xlabel(OBJECTIVE_LABELS.get("V_sat"), fontsize=9, labelpad=9)
        ax.set_ylabel(OBJECTIVE_LABELS.get("V_prof"), fontsize=9, labelpad=9)
        ax.set_zlabel(OBJECTIVE_LABELS.get("V_sus"), fontsize=9, labelpad=9)
        ax.set_title(title, fontsize=12, pad=14)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Convexity diagnostic
    # ══════════════════════════════════════════════════════════════════════════

    def plot_convexity_diagnostic(
        self,
        solutions: List[Dict],
        dominated_pool: Optional[List[Dict]] = None,
        method_name: str = "linear",
        obj_x: str = "V_prof",
        obj_y: str = "V_sat",
        save_path: str = None,
    ) -> Tuple[plt.Figure, Dict]:
        """
        Convexity test: convex hull membership, gap detection, chord test.

        Returns (fig, diagnostic_dict).
        """
        keys = OBJ_KEYS
        idx_x = keys.index(obj_x)
        idx_y = keys.index(obj_y)

        sol_matrix = _extract_objective_matrix(solutions, keys)
        pareto_idx = _compute_pareto_indices_max(sol_matrix)
        pareto_pts_2d = sol_matrix[pareto_idx][:, [idx_x, idx_y]]
        sort_order = np.argsort(pareto_pts_2d[:, 0])
        pareto_pts_2d = pareto_pts_2d[sort_order]

        # Gap detection
        f1_gaps = np.diff(pareto_pts_2d[:, 0])
        mean_gap = f1_gaps.mean() if len(f1_gaps) > 0 else 0.0
        std_gap  = f1_gaps.std()  if len(f1_gaps) > 1 else 0.0
        threshold = mean_gap + 3 * std_gap if std_gap > 0 else mean_gap * 3
        gap_indices = np.where(f1_gaps > threshold)[0] if threshold > 0 else np.array([], dtype=int)

        # Convex hull
        all_on_hull = False
        if len(pareto_pts_2d) >= 3:
            try:
                hull = ConvexHull(pareto_pts_2d)
                all_on_hull = len(hull.vertices) == len(pareto_pts_2d)
            except Exception:
                pass

        has_gap = len(gap_indices) > 0
        if all_on_hull and not has_gap:
            verdict = "CONVEX (connected)"
        elif has_gap:
            verdict = "NON-CONVEX (disconnected front)"
        else:
            verdict = "NON-CONVEX (concave region, connected)"

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=self.figsize_2d)
        fig.patch.set_facecolor("white")

        # Dominated background
        if dominated_pool:
            dom_matrix = _extract_objective_matrix(dominated_pool, keys)
            ax.scatter(
                dom_matrix[:, idx_x], dom_matrix[:, idx_y],
                c=COLORS["dominated"], s=18, alpha=0.35, zorder=1,
                label="Dominated solutions",
            )

        ax.scatter(
            pareto_pts_2d[:, 0], pareto_pts_2d[:, 1],
            c=COLORS.get(method_name, "#E24B4A"), s=55, zorder=5,
            label="Pareto front", edgecolors="white", linewidths=0.5,
        )
        ax.plot(pareto_pts_2d[:, 0], pareto_pts_2d[:, 1],
                color=COLORS.get(method_name, "#E24B4A"), lw=1.2, alpha=0.55, zorder=4)

        # Convex hull overlay
        if len(pareto_pts_2d) >= 3:
            try:
                hull = ConvexHull(pareto_pts_2d)
                for simplex in hull.simplices:
                    ax.plot(
                        pareto_pts_2d[simplex, 0], pareto_pts_2d[simplex, 1],
                        color=COLORS["convex_hull"], lw=1.5, ls="--", alpha=0.5,
                    )
            except Exception:
                pass

        # Chord test
        if len(pareto_pts_2d) >= 6:
            n_pts = len(pareto_pts_2d)
            anchor_a = pareto_pts_2d[n_pts // 5]
            anchor_b = pareto_pts_2d[4 * n_pts // 5]
            lams = np.linspace(0, 1, 60)
            chord = np.outer(lams, anchor_a) + np.outer(1 - lams, anchor_b)
            ax.plot(chord[:, 0], chord[:, 1],
                    color="#185FA5", lw=2.0, ls="--", alpha=0.65,
                    label=r"Convex combination $\lambda A + (1-\lambda)B$")
            ax.scatter(*anchor_a, marker="s", s=110, c="#185FA5", zorder=7, label="Point A")
            ax.scatter(*anchor_b, marker="s", s=110, c="#185FA5", zorder=7, label="Point B")

        for gi in gap_indices:
            ax.axvspan(pareto_pts_2d[gi, 0], pareto_pts_2d[gi + 1, 0],
                       alpha=0.08, color=COLORS["missed"])

        verdict_color = "#1D9E75" if verdict.startswith("CONVEX") else "#E24B4A"
        ax.annotate(
            f"Verdict: {verdict}",
            xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=9, color=verdict_color, va="top", weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=verdict_color, alpha=0.9),
        )

        ax.set_xlabel(OBJECTIVE_LABELS.get(obj_x, obj_x), fontsize=11)
        ax.set_ylabel(OBJECTIVE_LABELS.get(obj_y, obj_y), fontsize=11)
        ax.set_title("Convexity Diagnostic", fontsize=12, pad=10)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=8.5, loc="best", framealpha=0.9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")

        diagnostic = {
            "verdict": verdict,
            "all_points_on_convex_hull": all_on_hull,
            "has_disconnection_gap": has_gap,
            "n_gaps": int(len(gap_indices)),
            "gap_at_indices": gap_indices.tolist(),
            "n_pareto_points": len(pareto_pts_2d),
        }
        return fig, diagnostic

    # ══════════════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════════════

    def print_method_summary(self, solutions_dict: Dict[str, List[Dict]]) -> None:
        keys = OBJ_KEYS
        print("\n" + "=" * 90)
        print(f"{'Method':<35} {'# Pareto':>9} {'Avg Ṽ_sat':>11} "
              f"{'Avg Ṽ_prof':>11} {'Avg Ṽ_sus':>11}")
        print("=" * 90)
        for method_name, solutions in solutions_dict.items():
            obj_matrix = _extract_objective_matrix(solutions, keys)
            pareto_idx = _compute_pareto_indices_max(obj_matrix)
            pareto_pts = obj_matrix[pareto_idx]
            n = len(pareto_pts)
            avg = pareto_pts.mean(axis=0) if n > 0 else np.zeros(3)
            label = METHOD_LABELS.get(method_name, method_name)
            print(f"{label:<35} {n:>9d} {avg[0]:>11.4f} {avg[1]:>11.4f} {avg[2]:>11.4f}")
        print("=" * 90 + "\n")