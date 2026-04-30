"""
pareto_distances.py
===================
General-purpose Pareto front distance and quality indicators,
implemented entirely in MAXIMIZATION convention.

No I/O transform (1 - f) is ever applied.
All functions operate directly on normalized objective vectors
V_sat, V_prof, V_sus ∈ [0, 1], all to be maximized.

Convention used throughout
--------------------------
  - Higher objective values are better.
  - The Utopian point is z_ideal = (1, 1, 1).
  - The Nadir point is z_nadir = (0, 0, 0).
  - The reference point for HV is placed BELOW the nadir:
        r_hv = (-offset, -offset, -offset)   (e.g., offset = 0.1)
    so the HV measures the volume ABOVE the front toward the Utopian.
  - d+(z, a) for maximization: distance in the DOMINATED direction.
        d+(z, a) = sqrt( sum_k max(z_k - a_k, 0)^2 )
    Zero iff a dominates z (a >= z component-wise).
  - IGD+(A, Z) = (1/|Z|) * sum_{z in Z} min_{a in A} d+(z, a)
  - eps_add(A, Z) = max_{z in Z}  min_{a in A}  max_k (z_k - a_k)
    Zero iff A dominates Z entirely.

Three-front comparison API
--------------------------
  Given:
    F_ub     — Upper-bound Pareto front  (Fluid LP)          shape (n_ub, 3)
    F_lb     — Lower-bound Pareto front  (Random acceptance)  shape (n_lb, 3)
    F_online — Online mechanism Pareto front                  shape (n_ol, 3)

  All in normalized maximization space [0,1]^3.

  The function `compare_three_fronts(F_ub, F_lb, F_online)` returns a
  structured dict with every indicator and the two efficiency ratios:

    eta_HV  = (HV_online - HV_lb) / (HV_ub - HV_lb)   ∈ [0, 1]
    rho_igd = 1 - IGD+(online, UB) / IGD+(lb, UB)      ∈ [0, 1]

References
----------
  Zitzler & Thiele (1998) PPSN V, DOI 10.1007/BFb0056872
  Zitzler et al.   (2003) IEEE TEVC, DOI 10.1109/TEVC.2003.810758
  Ishibuchi et al. (2015) EMO, DOI 10.1007/978-3-319-15892-1_8
  Beume et al.     (2009) IEEE TEVC, DOI 10.1109/TEVC.2009.2015575
  Vargha & Delaney (2000) JEBS, DOI 10.3102/10769986025002101
"""

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from typing import List, Dict, Optional, Tuple

__all__ = [
    "pareto_filter_max",
    "extract_norm_matrix",
    "hypervolume_3d_max",
    "hv_contributions_max",
    "d_plus_max",
    "igd_plus_max",
    "gd_plus_max",
    "epsilon_additive_max",
    "c_metric_max",
    "schott_spacing",
    "maximum_spread",
    "compare_two_fronts",
    "compare_three_fronts",
    "vargha_delaney_a12",
    "effect_size_label",
    "statistical_comparison",
]

# ── objective key order (used throughout) ────────────────────────────────────
OBJ_KEYS = ("V_sat", "V_prof", "V_sus")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_norm_matrix(solutions: List[Dict],
                        keys: Tuple[str, ...] = OBJ_KEYS) -> np.ndarray:
    """
    Extract (N, k) objective matrix from a list of solution dicts.
    Reads from 'normalized_objectives'.
    """
    return np.array([[s["normalized_objectives"][k] for k in keys]
                     for s in solutions], dtype=float)


def pareto_filter_max(F: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated subset of F under MAXIMIZATION.

    Parameters
    ----------
    F : (N, k) array of objective vectors.

    Returns
    -------
    nd : (M, k) non-dominated rows, M <= N.
    """
    N = len(F)
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]:
            continue
        for j in range(N):
            if i == j or dominated[j]:
                continue
            # j dominates i iff j >= i on all and j > i on some
            if np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                dominated[i] = True
                break
    return F[~dominated]


# ══════════════════════════════════════════════════════════════════════════════
# 2. HYPERVOLUME  (MAXIMIZATION CONVENTION)
#    Reference point r placed BELOW the nadir so HV measures volume ABOVE front.
# ══════════════════════════════════════════════════════════════════════════════

def _hv_2d_max(pts: np.ndarray, r: np.ndarray) -> float:
    """
    Exact 2-D hypervolume for maximization.
    r must satisfy r[k] <= pts[i,k] for all i, k.
    Sweep from left (low f1) to right (high f1).
    """
    # Keep only points that dominate r
    pts = pts[np.all(pts >= r, axis=1)]
    if len(pts) == 0:
        return 0.0
    # Sort by first objective descending
    pts = pts[np.argsort(-pts[:, 0])]
    area = 0.0
    prev_f2 = r[1]
    for p in pts:
        if p[1] > prev_f2:
            area += (p[0] - r[0]) * (p[1] - prev_f2)
            prev_f2 = p[1]
    return float(area)


def hypervolume_3d_max(F: np.ndarray,
                       r: Optional[np.ndarray] = None) -> float:
    """
    Exact 3-D Hypervolume Indicator (maximization convention).
    Implements the dimension-sweep algorithm: O(n log n) for m=3.

    HV(F, r) = lambda_3( union_{a in F} [r, a] )

    where [r, a] = { y : r <= y <= a } is the hyper-rectangle above r
    and below a.

    Parameters
    ----------
    F : (n, 3) array of non-dominated objective vectors in [0,1]^3.
        All values are to be MAXIMIZED.
    r : (3,) reference point placed below the nadir.
        Default: (-0.1, -0.1, -0.1).

    Returns
    -------
    float — hypervolume value. Larger is better.
    """
    if r is None:
        r = np.array([-0.1, -0.1, -0.1])
    F = np.asarray(F, float)
    # Keep only points that dominate r
    F = F[np.all(F >= r, axis=1)]
    if len(F) == 0:
        return 0.0

    # Sort by third objective ascending (process slabs bottom → top)
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    prev_f3 = r[2]

    for i in range(len(F)):
        slab_height = F[i, 2] - prev_f3
        if slab_height > 0:
            # For all points at or above F[i,2], compute 2-D HV in (f1, f2)
            hv += slab_height * _hv_2d_max(F[i:, :2], r[:2])
        prev_f3 = F[i, 2]

    return float(hv)


def hv_contributions_max(F: np.ndarray,
                         r: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Exclusive hypervolume contribution of each point in F.

    HVC[i] = HV(F, r) - HV(F without row i, r)

    Parameters
    ----------
    F : (n, 3) non-dominated objective vectors (maximization).
    r : reference point (default (-0.1, -0.1, -0.1)).

    Returns
    -------
    contributions : (n,) array, same order as F.
    """
    hv_total = hypervolume_3d_max(F, r)
    contribs = np.zeros(len(F))
    for i in range(len(F)):
        F_minus_i = np.delete(F, i, axis=0)
        contribs[i] = hv_total - hypervolume_3d_max(F_minus_i, r)
    return contribs


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODIFIED DISTANCE d+  (MAXIMIZATION CONVENTION)
# ══════════════════════════════════════════════════════════════════════════════

def d_plus_max(z: np.ndarray, a: np.ndarray) -> float:
    """
    Modified distance d+(z, a) in MAXIMIZATION convention.

    d+(z, a) = sqrt( sum_k max(z_k - a_k, 0)^2 )

    Geometric interpretation:
      - Zero iff a dominates z (a[k] >= z[k] for all k).
      - Equals the Euclidean distance projected onto the non-dominated
        quadrant: only counts dimensions where a FAILS to reach z.
      - Ensures IGD+ and GD+ are weakly Pareto-compliant.

    Parameters
    ----------
    z : reference point (e.g., from the LP front).
    a : candidate point (e.g., from the online mechanism).

    Returns
    -------
    float >= 0.
    """
    return float(np.sqrt(np.sum(np.maximum(z - a, 0.0) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# 4. IGD+ and GD+  (MAXIMIZATION CONVENTION, WEAKLY PARETO-COMPLIANT)
# ══════════════════════════════════════════════════════════════════════════════

def igd_plus_max(A: np.ndarray, Z: np.ndarray) -> float:
    """
    Inverted Generational Distance Plus — IGD+ (maximization).

    IGD+(A, Z) = (1/|Z|) * sum_{z in Z} min_{a in A} d+(z, a)

    Captures BOTH convergence (are A points close to Z?) and
    diversity (does A cover all of Z?).

    Weakly Pareto-compliant: if A dominates B then IGD+(A,Z) <= IGD+(B,Z).

    Parameters
    ----------
    A : (n_A, m) — candidate front (e.g., online mechanism). Maximization.
    Z : (n_Z, m) — reference front (e.g., LP upper bound). Maximization.

    Returns
    -------
    float >= 0. Zero iff A = Z (or A entirely dominates Z).
    """
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)
    total = 0.0
    for z in Z:
        # d+ from this reference point to the nearest candidate
        dists = np.sqrt(np.sum(np.maximum(z - A, 0.0) ** 2, axis=1))
        total += dists.min()
    return total / len(Z)


def gd_plus_max(A: np.ndarray, Z: np.ndarray) -> float:
    """
    Generational Distance Plus — GD+ (maximization).
    Convergence-only indicator (no diversity component).

    GD+(A, Z) = (1/|A|) * sum_{a in A} min_{z in Z} d+(z, a)

    Weakly Pareto-compliant (resolves GD's counter-examples).
    Use as a secondary convergence diagnostic alongside IGD+.

    Parameters
    ----------
    A : (n_A, m) candidate front. Maximization.
    Z : (n_Z, m) reference front. Maximization.
    """
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)
    total = 0.0
    for a in A:
        dists = np.sqrt(np.sum(np.maximum(Z - a, 0.0) ** 2, axis=1))
        total += dists.min()
    return total / len(A)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EPSILON INDICATOR  (MAXIMIZATION CONVENTION, WEAKLY PARETO-COMPLIANT)
# ══════════════════════════════════════════════════════════════════════════════

def epsilon_additive_max(A: np.ndarray, Z: np.ndarray) -> float:
    """
    Additive unary epsilon-indicator I_eps+(A, Z) — maximization convention.

    I_eps+(A, Z) = max_{z in Z}  min_{a in A}  max_k (z_k - a_k)

    Geometric meaning: the smallest uniform shift eps such that every
    point in Z is weakly dominated by some point in A + eps*1.
    Equivalently: the worst-case Chebyshev gap between A and Z.

      eps <= 0 iff A weakly dominates Z entirely.
      eps > 0  means the worst LP-front point that A fails to dominate,
               and by how much.

    Parameters
    ----------
    A : (n_A, m) candidate front. Maximization.
    Z : (n_Z, m) reference front. Maximization.

    Returns
    -------
    float. Ideally <= 0. Lower is better.
    """
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)
    return float(max(
        min(np.max(z - a) for a in A)
        for z in Z
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SET COVERAGE  (DESCRIPTIVE ONLY — NOT PARETO-COMPLIANT)
# ══════════════════════════════════════════════════════════════════════════════

def c_metric_max(A: np.ndarray, B: np.ndarray) -> float:
    """
    Set coverage C(A, B) — maximization convention.

    C(A, B) = |{ b in B : exists a in A, a dominates b }| / |B|

    Fraction of B's points that are weakly dominated by at least one
    point in A. Ranges in [0, 1]. Asymmetric: always report both
    C(A,B) and C(B,A).

    NOT Pareto-compliant as a unary indicator. Use for descriptive
    reporting only, never as the primary ranking criterion.

    Parameters
    ----------
    A : (n_A, m) first front. Maximization.
    B : (n_B, m) second front. Maximization.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    count = 0
    for b in B:
        # a dominates b iff a >= b on all and a > b on some
        if np.any(np.all(A >= b, axis=1) & np.any(A > b, axis=1)):
            count += 1
    return count / len(B)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SPREAD / DIVERSITY  (DESCRIPTIVE)
# ══════════════════════════════════════════════════════════════════════════════

def schott_spacing(F: np.ndarray) -> float:
    """
    Schott's Spacing metric (1995).
    Measures uniformity of distribution along the front.
    Lower = more uniform. NOT a convergence measure.

    S = sqrt( (1/(n-1)) * sum_i (d_i - d_bar)^2 )
    d_i = min_{j != i} sum_k |F_ik - F_jk|   (L1 nearest-neighbour)
    """
    F = np.asarray(F, float)
    if len(F) < 2:
        return 0.0
    dists = np.array([
        min(np.sum(np.abs(F[i] - F[j]))
            for j in range(len(F)) if j != i)
        for i in range(len(F))
    ])
    d_bar = dists.mean()
    return float(np.sqrt(np.mean((dists - d_bar) ** 2)))


def maximum_spread(F: np.ndarray) -> float:
    """
    Maximum Spread: Euclidean diagonal of the bounding box of F.
    Higher = broader coverage of objective space.
    """
    F = np.asarray(F, float)
    return float(np.sqrt(np.sum((F.max(axis=0) - F.min(axis=0)) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# 8. PAIRWISE FRONT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_two_fronts(F_candidate: np.ndarray,
                       F_reference: np.ndarray,
                       r_hv: Optional[np.ndarray] = None,
                       label_cand: str = "Candidate",
                       label_ref:  str = "Reference") -> dict:
    """
    Compute the full indicator suite comparing F_candidate against F_reference.

    Both fronts must be in MAXIMIZATION convention, normalized to [0,1]^3.
    Non-dominated filtering is applied internally — pass raw solution sets
    or pre-filtered fronts, both work.

    Parameters
    ----------
    F_candidate : (n, 3) candidate front (maximization, [0,1]^3).
    F_reference : (m, 3) reference front (maximization, [0,1]^3).
    r_hv        : HV reference point. Default (-0.1, -0.1, -0.1).
    label_cand  : Name for the candidate (used in printed output).
    label_ref   : Name for the reference (used in printed output).

    Returns
    -------
    dict with keys:
        HV_cand, HV_ref,
        IGD_plus, GD_plus,
        eps_add,
        C_cand_ref, C_ref_cand,
        Spacing, MaxSpread,
        n_nd_cand, n_nd_ref
    """
    if r_hv is None:
        r_hv = np.array([-0.1, -0.1, -0.1])

    nd_c = pareto_filter_max(np.asarray(F_candidate, float))
    nd_r = pareto_filter_max(np.asarray(F_reference, float))

    return {
        "label_cand":  label_cand,
        "label_ref":   label_ref,
        "HV_cand":     hypervolume_3d_max(nd_c, r_hv),
        "HV_ref":      hypervolume_3d_max(nd_r, r_hv),
        "IGD_plus":    igd_plus_max(nd_c, nd_r),
        "GD_plus":     gd_plus_max(nd_c, nd_r),
        "eps_add":     epsilon_additive_max(nd_c, nd_r),
        "C_cand_ref":  c_metric_max(nd_c, nd_r),
        "C_ref_cand":  c_metric_max(nd_r, nd_c),
        "Spacing":     schott_spacing(nd_c),
        "MaxSpread":   maximum_spread(nd_c),
        "n_nd_cand":   len(nd_c),
        "n_nd_ref":    len(nd_r),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. THREE-FRONT COMPARISON — MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def compare_three_fronts(F_ub:     np.ndarray,
                         F_lb:     np.ndarray,
                         F_online: np.ndarray,
                         r_hv:     Optional[np.ndarray] = None) -> dict:
    """
    Complete three-front comparison: Upper Bound, Lower Bound, Online.

    Computes all pairwise indicators and the two thesis efficiency ratios:

        eta_HV  = (HV_online - HV_lb) / (HV_ub - HV_lb)   ∈ [0, 1]

            Fraction of available hypervolume range captured by the
            online mechanism above the random baseline.

        rho_IGD = 1 - IGD+(online, UB) / IGD+(lb, UB)     ∈ [0, 1]

            Fractional improvement in average LP-front proximity of
            the online mechanism over the random baseline.

    Parameters
    ----------
    F_ub     : (n, 3) Fluid LP upper-bound front.  Maximization, [0,1]^3.
    F_lb     : (m, 3) Random lower-bound front.    Maximization, [0,1]^3.
    F_online : (p, 3) Online mechanism front.      Maximization, [0,1]^3.
    r_hv     : HV reference point. Default (-0.1, -0.1, -0.1).

    Returns
    -------
    dict with full indicator suite and efficiency ratios.
    """
    if r_hv is None:
        r_hv = np.array([-0.1, -0.1, -0.1])

    nd_ub = pareto_filter_max(np.asarray(F_ub,     float))
    nd_lb = pareto_filter_max(np.asarray(F_lb,     float))
    nd_ol = pareto_filter_max(np.asarray(F_online, float))

    # ── Hypervolumes ─────────────────────────────────────────────────────────
    hv_ub = hypervolume_3d_max(nd_ub, r_hv)
    hv_lb = hypervolume_3d_max(nd_lb, r_hv)
    hv_ol = hypervolume_3d_max(nd_ol, r_hv)

    # ── IGD+ against UB as reference set ─────────────────────────────────────
    igd_lb_ub = igd_plus_max(nd_lb, nd_ub)
    igd_ol_ub = igd_plus_max(nd_ol, nd_ub)

    # ── GD+ against UB ───────────────────────────────────────────────────────
    gd_lb_ub  = gd_plus_max(nd_lb, nd_ub)
    gd_ol_ub  = gd_plus_max(nd_ol, nd_ub)

    # ── epsilon-indicator against UB ─────────────────────────────────────────
    eps_lb_ub = epsilon_additive_max(nd_lb, nd_ub)
    eps_ol_ub = epsilon_additive_max(nd_ol, nd_ub)

    # ── C-metric (all six directions) ────────────────────────────────────────
    c_ub_ol = c_metric_max(nd_ub, nd_ol)   # frac of OL dominated by UB
    c_ub_lb = c_metric_max(nd_ub, nd_lb)   # frac of LB dominated by UB
    c_ol_ub = c_metric_max(nd_ol, nd_ub)   # frac of UB dominated by OL
    c_ol_lb = c_metric_max(nd_ol, nd_lb)   # frac of LB dominated by OL
    c_lb_ub = c_metric_max(nd_lb, nd_ub)   # frac of UB dominated by LB
    c_lb_ol = c_metric_max(nd_lb, nd_ol)   # frac of OL dominated by LB

    # ── Efficiency ratios ─────────────────────────────────────────────────────
    hv_range = hv_ub - hv_lb
    eta_hv   = (hv_ol - hv_lb) / hv_range if abs(hv_range) > 1e-12 else 0.0

    rho_igd  = (1.0 - igd_ol_ub / igd_lb_ub
                if abs(igd_lb_ub) > 1e-12 else 0.0)

    # ── Spread metrics ───────────────────────────────────────────────────────
    spread_ub = maximum_spread(nd_ub)
    spread_lb = maximum_spread(nd_lb)
    spread_ol = maximum_spread(nd_ol)

    spacing_ol = schott_spacing(nd_ol)

    return {
        # ── Front sizes ──────────────────────────────────────────────────────
        "n_nd_ub":     len(nd_ub),
        "n_nd_lb":     len(nd_lb),
        "n_nd_ol":     len(nd_ol),

        # ── Non-dominated arrays (for downstream use) ─────────────────────────
        "nd_ub":       nd_ub,
        "nd_lb":       nd_lb,
        "nd_ol":       nd_ol,

        # ── Hypervolumes ─────────────────────────────────────────────────────
        "HV_ub":       hv_ub,
        "HV_lb":       hv_lb,
        "HV_ol":       hv_ol,

        # ── IGD+ ─────────────────────────────────────────────────────────────
        "IGD_plus_lb_ub":  igd_lb_ub,
        "IGD_plus_ol_ub":  igd_ol_ub,

        # ── GD+ ──────────────────────────────────────────────────────────────
        "GD_plus_lb_ub":   gd_lb_ub,
        "GD_plus_ol_ub":   gd_ol_ub,

        # ── Epsilon ──────────────────────────────────────────────────────────
        "eps_lb_ub":   eps_lb_ub,
        "eps_ol_ub":   eps_ol_ub,

        # ── C-metric ─────────────────────────────────────────────────────────
        "C_ub_ol":     c_ub_ol,
        "C_ub_lb":     c_ub_lb,
        "C_ol_ub":     c_ol_ub,
        "C_ol_lb":     c_ol_lb,
        "C_lb_ub":     c_lb_ub,
        "C_lb_ol":     c_lb_ol,

        # ── Spread ───────────────────────────────────────────────────────────
        "MaxSpread_ub":  spread_ub,
        "MaxSpread_lb":  spread_lb,
        "MaxSpread_ol":  spread_ol,
        "Spacing_ol":    spacing_ol,

        # ── PRIMARY THESIS METRICS ────────────────────────────────────────────
        "eta_HV":      float(eta_hv),
        "rho_IGD_plus": float(rho_igd),
    }


def print_three_front_report(results: dict) -> None:
    """
    Pretty-print the output of compare_three_fronts().
    """
    print("\n" + "=" * 70)
    print("THREE-FRONT COMPARISON REPORT")
    print("=" * 70)

    print(f"\n  Front sizes:")
    print(f"    UB (Fluid LP)  : {results['n_nd_ub']:>5d} non-dominated points")
    print(f"    LB (Random)    : {results['n_nd_lb']:>5d} non-dominated points")
    print(f"    Online         : {results['n_nd_ol']:>5d} non-dominated points")

    print(f"\n{'  Metric':<30} {'LB (Random)':>14} {'Online':>14} {'UB (LP)':>14}")
    print("  " + "-" * 66)
    print(f"  {'HV ↑':<28} {results['HV_lb']:>14.6f} "
          f"{results['HV_ol']:>14.6f} {results['HV_ub']:>14.6f}")
    print(f"  {'IGD+ ↓  (vs UB)':<28} {results['IGD_plus_lb_ub']:>14.6f} "
          f"{results['IGD_plus_ol_ub']:>14.6f} {'0 (ref)':>14}")
    print(f"  {'GD+  ↓  (vs UB)':<28} {results['GD_plus_lb_ub']:>14.6f} "
          f"{results['GD_plus_ol_ub']:>14.6f} {'0 (ref)':>14}")
    print(f"  {'ε+   ↓  (vs UB)':<28} {results['eps_lb_ub']:>14.6f} "
          f"{results['eps_ol_ub']:>14.6f} {'0 (ref)':>14}")
    print(f"  {'MaxSpread ↑':<28} {results['MaxSpread_lb']:>14.6f} "
          f"{results['MaxSpread_ol']:>14.6f} {results['MaxSpread_ub']:>14.6f}")

    print(f"\n  Coverage matrix C(row dominates column):")
    print(f"    C(UB → Online) = {results['C_ub_ol']:.4f}  "
          "(frac of Online dominated by UB)")
    print(f"    C(UB → LB)    = {results['C_ub_lb']:.4f}  "
          "(frac of LB dominated by UB)")
    print(f"    C(Online → UB) = {results['C_ol_ub']:.4f}  "
          "(frac of UB dominated by Online — ideally > 0)")
    print(f"    C(Online → LB) = {results['C_ol_lb']:.4f}  "
          "(frac of LB dominated by Online)")

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  PRIMARY THESIS METRICS                             │")
    print(f"  │                                                     │")
    print(f"  │  η_HV  = {results['eta_HV']:>6.4f}   "
          f"({results['eta_HV']*100:>5.1f}% of HV range captured)  │")
    print(f"  │  ρ_IGD+ = {results['rho_IGD_plus']:>6.4f}   "
          f"({results['rho_IGD_plus']*100:>5.1f}% IGD+ improvement)   │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 10. STATISTICAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vargha-Delaney A12 effect size.
    A12 = P(X > Y) + 0.5 * P(X = Y).
    A12 > 0.5 means X tends to produce larger values.
    For minimized indicators, A12 < 0.5 means x is better.
    Reference: Vargha & Delaney (2000) JEBS, DOI 10.3102/10769986025002101.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, n = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]), method="average")
    R1 = ranks[:m].sum()
    return float((R1 / m - (m + 1) / 2.0) / n)


def effect_size_label(a12: float) -> str:
    """Magnitude label for A12 (Vargha & Delaney 2000, Table 1)."""
    d = abs(a12 - 0.5)
    if d < 0.06:   return "negligible"
    if d < 0.147:  return "small"
    if d < 0.21:   return "medium"
    return "large"


def statistical_comparison(x: np.ndarray, y: np.ndarray,
                            label_x: str = "X",
                            label_y: str = "Y",
                            maximize: bool = True) -> dict:
    """
    Full pairwise statistical comparison between two indicator distributions.

    Parameters
    ----------
    x, y     : 1-D arrays of indicator values (one per run).
    label_x  : Name for x (e.g., 'Online').
    label_y  : Name for y (e.g., 'Random').
    maximize : True if higher indicator is better (HV); False if lower (IGD+).

    Returns
    -------
    dict with p_value, A12, effect, medians, IQRs, and 'better' label.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    _, p = mannwhitneyu(x, y, alternative="two-sided")
    a12 = vargha_delaney_a12(x, y)
    # A12 > 0.5: x produces larger values
    x_is_better = (a12 > 0.5) if maximize else (a12 < 0.5)
    better = label_x if x_is_better else label_y

    return {
        "p_value":   float(p),
        "A12":       float(a12),
        "effect":    effect_size_label(a12),
        "median_x":  float(np.median(x)),
        "iqr_x":     float(np.percentile(x, 75) - np.percentile(x, 25)),
        "median_y":  float(np.median(y)),
        "iqr_y":     float(np.percentile(y, 75) - np.percentile(y, 25)),
        "better":    better,
        "label_x":   label_x,
        "label_y":   label_y,
    }


def print_statistical_report(stat: dict) -> None:
    """Pretty-print output of statistical_comparison()."""
    print(f"\n  {stat['label_x']} vs {stat['label_y']}:")
    print(f"    {stat['label_x']:12s}: median={stat['median_x']:.4f}  "
          f"IQR={stat['iqr_x']:.4f}")
    print(f"    {stat['label_y']:12s}: median={stat['median_y']:.4f}  "
          f"IQR={stat['iqr_y']:.4f}")
    print(f"    Mann-Whitney p = {stat['p_value']:.3e}")
    print(f"    A12 = {stat['A12']:.3f}  ({stat['effect']} effect)  "
          f"→ {stat['better']} wins")
