"""
discretizer_simulator.py
========================
Evaluates Uniform, Geometric, and DP Optimal discretization baselines
by measuring their loss through the StaticCloudSimulator environment.

Allocation Policy — Fluid Volume LP (OPT_LP)
---------------------------------------------
All jobs are known at t=0 (static/offline setting).  The oracle solves a
Linear Program over fractional decisions x_j ∈ [0, 1]:

    max   ∑_j r_j · x_j
    s.t.  ∑_j x_j · resource_j · duration_j  ≤  C_max · T   (Fluid Volume)
          x_j = 0   if phi_j ≤ 0                              (Individual Rationality)
          x_j ∈ [0, 1]

where the scalarized per-job reward is:

    r_j = lambda1 · q_j · v_j  +  lambda2 · (phi_j − c_elec_j)
          − lambda3 · c_carbon_j

with equal lambdas lambda1 = lambda2 = lambda3 = 1/3.

The Fluid Volume constraint replaces instantaneous capacity tracking:
instead of checking point-in-time usage it bounds the total resource
volume (core·hours) consumed over the horizon T, matching the OPT_LP
benchmark used in DLENT regret bounds.

Loss Metric (Global Objective Space)
--------------------------------------
Two LP solves per baseline:
  - Continuous  x*_cont  →  R_cont = ∑_j x*_cont_j · r_cont_j
  - Discrete    x*_disc  →  R_disc = ∑_j x*_disc_j · r_disc_j

    Global Objective Loss  =  R_cont − R_disc

Three discretization baselines (K bins each)
--------------------------------------------
  1. Uniform   — arithmetic grid over [phi_min, phi_max]
  2. Geometric — geometric (log-spaced) grid
  3. DP Optimal — dynamic-programming grid minimising revenue loss

Both v (valuation) and phi (virtual value) are discretized independently.
"""

import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Literal

warnings.filterwarnings("ignore")

class Discretizer:
    """
    Converts continuous virtual values / valuations into a discrete price menu
    using one of three binning strategies.
    """

    def __init__(self, K_bins: int):
        self.K = K_bins

    def _enforce_ir(self, phi_array: np.ndarray) -> np.ndarray:
        """Return only the individually-rational values (phi > 0)."""
        return phi_array[phi_array > 0]

    def _map_to_bins(
        self,
        values: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """
        Given sorted bin edges, assign each value to its lower-bound bin price.
        Values below the first edge fall into bin 0; values above the last edge
        fall into bin K-1.
        """
        bin_indices = np.digitize(values, edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.K - 1)
        return edges[bin_indices]

    # ------------------------------------------------------------------
    # Public grid methods — each accepts the *full* array (including
    # phi <= 0) and returns a discretized array of the *same length*,
    # with non-IR values left at their original (negative) values so
    # the simulator's IR filter still works correctly.
    # ------------------------------------------------------------------

    def uniform_grid(self, values: np.ndarray) -> np.ndarray:
        """
        Baseline 1 — Uniform / Arithmetic Grid.
        Divides [min_valid, max_valid] into K equal segments.
        Non-positive values are passed through unchanged.
        """
        result = values.copy().astype(float)
        mask = values > 0
        valid = values[mask]
        if len(valid) == 0:
            return result

        phi_min, phi_max = valid.min(), valid.max()
        edges = np.linspace(phi_min, phi_max, self.K + 1)
        result[mask] = self._map_to_bins(valid, edges)
        return result

    def geometric_grid(self, values: np.ndarray) -> np.ndarray:
        """
        Baseline 2 — Geometric / Log-spaced Grid.
        Multiplicative spacing; bins narrow at the bottom, wide at the top.
        Non-positive values are passed through unchanged.
        """
        result = values.copy().astype(float)
        mask = values > 0
        valid = values[mask]
        if len(valid) == 0:
            return result

        phi_min = max(valid.min(), 1e-4)   # geomspace requires > 0
        phi_max = valid.max()
        edges = np.geomspace(phi_min, phi_max, self.K + 1)
        result[mask] = self._map_to_bins(valid, edges)
        return result

    def dp_optimal_grid(self, values: np.ndarray) -> np.ndarray:
        """
        Baseline 3 — DP Optimal Grid.
        Finds the K bin boundaries that minimise revenue loss for this dataset.
        Complexity: O(K * N^2) with numpy vectorisation.
        Non-positive values are passed through unchanged.
        """
        result = values.copy().astype(float)
        mask = values > 0
        valid = values[mask]
        if len(valid) == 0:
            return result

        V = np.sort(valid)
        N = len(V)

        # If fewer jobs than bins, everyone keeps their exact value
        if N <= self.K:
            return result

        # --- Precompute cost matrix ---
        cum_V = np.insert(np.cumsum(V), 0, 0)
        cost_matrix = np.zeros((N, N))
        for i in range(N):
            j_idx = np.arange(i, N)
            seg_sums = cum_V[j_idx + 1] - cum_V[i]
            bin_rev  = (j_idx - i + 1) * V[i]
            cost_matrix[i, i:] = seg_sums - bin_rev

        # --- DP table ---
        dp      = np.full((self.K + 1, N), np.inf)
        tracker = np.zeros((self.K + 1, N), dtype=int)

        for j in range(N):
            dp[1, j]      = cost_matrix[0, j]
            tracker[1, j] = 0

        for k in range(2, self.K + 1):
            for j in range(k - 1, N):
                i_cands    = np.arange(k - 1, j + 1)
                prev_costs = dp[k - 1, i_cands - 1]
                curr_costs = cost_matrix[i_cands, j]
                total      = prev_costs + curr_costs
                best       = np.argmin(total)
                dp[k, j]      = total[best]
                tracker[k, j] = i_cands[best]

        # --- Backtrack to find bin lower-bounds ---
        bin_starts = []
        curr_j = N - 1
        for k in range(self.K, 0, -1):
            start_i = tracker[k, curr_j]
            bin_starts.append(V[start_i])
            curr_j = start_i - 1
        bin_starts.reverse()

        edges = np.array(bin_starts)
        edges = np.append(edges, np.inf)

        result[mask] = self._map_to_bins(valid, edges)
        return result

    # ------------------------------------------------------------------
    # Phase-lagged train / apply interface
    # ------------------------------------------------------------------

    def get_dp_boundaries(self, train_data: np.ndarray) -> np.ndarray:
        """
        TRAIN (Phase M-1): Run the DP algorithm on historical data and return
        exactly K bin lower-bound boundary points.

        Only IR-valid values (> 0) participate in the DP; the returned
        boundaries are therefore always positive, which guarantees that
        apply_boundaries will never accidentally map a positive test value
        to 0 due to a missing lower bin.

        Parameters
        ----------
        train_data : 1-D array of continuous values from Phase M-1.
                     May contain non-positive values (IR violations); these
                     are filtered out before fitting.

        Returns
        -------
        boundaries : sorted array of at most K positive lower-bound values.
        """
        # Filter to IR-valid training values only
        valid = train_data[train_data > 0]
        if len(valid) == 0:
            return np.array([0.0])

        V = np.sort(valid)
        N = len(V)

        # Fewer points than bins → every unique value is its own boundary
        if N <= self.K:
            return np.unique(V)

        # Precompute cost matrix (revenue loss for any segment V[i..j])
        cum_V = np.insert(np.cumsum(V), 0, 0)
        cost_matrix = np.zeros((N, N))
        for i in range(N):
            j_idx    = np.arange(i, N)
            seg_sums = cum_V[j_idx + 1] - cum_V[i]
            bin_rev  = (j_idx - i + 1) * V[i]
            cost_matrix[i, i:] = seg_sums - bin_rev

        # DP table: dp[k, j] = min loss to cover V[0..j] using exactly k bins
        dp      = np.full((self.K + 1, N), np.inf)
        tracker = np.zeros((self.K + 1, N), dtype=int)

        for j in range(N):
            dp[1, j]      = cost_matrix[0, j]
            tracker[1, j] = 0

        for k in range(2, self.K + 1):
            for j in range(k - 1, N):
                i_cands    = np.arange(k - 1, j + 1)
                prev_costs = dp[k - 1, i_cands - 1]
                curr_costs = cost_matrix[i_cands, j]
                total      = prev_costs + curr_costs
                best       = np.argmin(total)
                dp[k, j]      = total[best]
                tracker[k, j] = i_cands[best]

        # Backtrack to recover the K lower-bound values
        bin_starts = []
        curr_j = N - 1
        for k in range(self.K, 0, -1):
            start_i = tracker[k, curr_j]
            bin_starts.append(V[start_i])
            curr_j = start_i - 1
        bin_starts.reverse()

        return np.array(bin_starts)  # length ≤ K, strictly positive, sorted

    def apply_boundaries(
        self, test_data: np.ndarray, boundaries: np.ndarray
    ) -> np.ndarray:
        """
        TEST (Phase M): Map unseen values to the nearest historical lower bound.

        Rules
        -----
        * Positive test values below the lowest boundary are mapped to
          boundaries[0] (the smallest known positive price) — NOT to 0.
          This avoids spurious IR violations caused purely by out-of-range
          test values, which would be a data-leakage artefact, not a true
          discretization error.
        * Non-positive test values (IR violations in continuous space) are
          passed through unchanged at their original value.  The LP oracle's
          IR filter (phi > 0 bound) handles them correctly.

        Parameters
        ----------
        test_data  : 1-D array of continuous values from Phase M.
        boundaries : sorted positive boundary array from get_dp_boundaries.

        Returns
        -------
        discrete_vals : array of the same length as test_data.
        """
        result = test_data.copy().astype(float)

        pos_mask = test_data > 0
        pos_vals = test_data[pos_mask]

        if len(pos_vals) == 0 or len(boundaries) == 0:
            return result

        # searchsorted(..., side='right') - 1 gives the index of the largest
        # boundary that is ≤ the test value (i.e. the lower-bound bin).
        idx = np.searchsorted(boundaries, pos_vals, side="right") - 1

        # Values below boundaries[0] → clamp to bin 0 (smallest known price).
        # This prevents a positive phi from being discretized to 0 merely
        # because it falls outside the training range.
        idx = np.clip(idx, 0, len(boundaries) - 1)

        result[pos_mask] = boundaries[idx]
        return result

    # ------------------------------------------------------------------
    # Theoretical K and calibration (static helpers)
    # ------------------------------------------------------------------

    # Default calibration constant C = (4β/α)^{2/5}.
    # Empirical default: K≈30 observed at N^{1/5}≈3.6  →  C ≈ 8.3
    _C_DEFAULT: float = 8.3
    _C_MAX_K:   int   = 30

    @staticmethod
    def _theoretical_k(w: int, C: float | None = None) -> int:
        """
        Theoretical upper-bound bin count for a phase with W jobs.

        Derived by minimising Total Regret = α√(TK) + βT/K²:
            K* = C · T^{1/5},  where C = (4β/α)^{2/5}

        Parameters
        ----------
        w : batch size (number of jobs in the phase).
        C : calibrated constant (default: Discretizer._C_DEFAULT).

        Returns
        -------
        K : int ≥ 2
        """
        if C is None:
            C = Discretizer._C_DEFAULT
        return max(2, int(np.ceil(C * (w ** 0.2))))

    @staticmethod
    def calibrate_C(
        v_continuous:   np.ndarray,
        phi_continuous: np.ndarray | None = None,
        q_continuous:   np.ndarray | None = None,
        warmup_n:       int   = 500,
        k_search_max:   int | None = None,
        k_search_threshold_pct: float = 1.0,
        lambda_1: float = 1 / 3,
        lambda_2: float = 1 / 3,
        seed:     int   = 42,
    ) -> float:
        """
        Phase-0 warm-up: calibrate the constant C = K*_empirical / N^{1/5}.

        Uses a strict 50/50 train/test split on the warm-up batch to prevent
        in-sample overfitting.  The DP boundaries are fitted on the train half
        and the scalarized multi-objective loss is evaluated on the test half,
        so no data leaks between the two splits.

        Parameters
        ----------
        v_continuous   : full continuous valuation array.
        phi_continuous : full virtual-value array (optional; zeros if None).
        q_continuous   : priority weights (optional; ones if None).
        warmup_n       : warm-up sub-sample size (default 500).
        k_search_max   : K ceiling for the sweep (default: _C_MAX_K = 30).
        k_search_threshold_pct : early-stop: stop when marginal improvement
                                 on the TEST split < this % (default 1 %).
        lambda_1, lambda_2     : scalarization weights for L_sat and L_prof.
        seed           : RNG seed for reproducible sub-sampling.

        Returns
        -------
        C : float — calibrated constant, ready to pass into _theoretical_k.
        """
        if k_search_max is None:
            k_search_max = Discretizer._C_MAX_K

        rng      = np.random.default_rng(seed=seed)
        actual_n = min(warmup_n, len(v_continuous))
        indices  = rng.choice(len(v_continuous), size=actual_n, replace=False)

        batch_v   = v_continuous[indices]
        batch_phi = phi_continuous[indices] if phi_continuous is not None \
                    else np.zeros(actual_n)
        batch_q   = q_continuous[indices]   if q_continuous   is not None \
                    else np.ones(actual_n)

        # ── Strict 50/50 train / test split ──────────────────────────────────
        # Splitting is done on the SHUFFLED batch (already random) so the two
        # halves are drawn from the same distribution without any ordering bias.
        split     = actual_n // 2
        train_v,   test_v   = batch_v[:split],   batch_v[split:]
        train_phi, test_phi = batch_phi[:split],  batch_phi[split:]
        _,         test_q   = batch_q[:split],    batch_q[split:]

        best_k_star = 2
        best_err    = float("inf")
        prev_err    = float("inf")

        for k in range(2, k_search_max + 1):
            disc = Discretizer(K_bins=k)

            # TRAIN: fit boundaries on Phase M-1 data
            bounds_v   = disc.get_dp_boundaries(train_v)
            bounds_phi = disc.get_dp_boundaries(train_phi)

            # TEST: apply historical boundaries to Phase M data
            v_disc_test   = disc.apply_boundaries(test_v,   bounds_v)
            phi_disc_test = disc.apply_boundaries(test_phi, bounds_phi)

            # Scalarized multi-objective loss on the test split
            loss_v   = float(np.sum(test_q * (test_v   - v_disc_test)))
            loss_phi = float(np.sum(          test_phi  - phi_disc_test))
            err      = lambda_1 * loss_v + lambda_2 * loss_phi

            if err < best_err:
                best_err    = err
                best_k_star = k

            # Early stop: marginal improvement on test set below threshold
            if prev_err < float("inf") and prev_err > 0:
                if ((prev_err - err) / prev_err) * 100 < k_search_threshold_pct:
                    break
            prev_err = err

        # C is calibrated against the full warm-up size (actual_n), not just
        # the train half, because _theoretical_k will be called with full
        # phase batch sizes in production.
        C = best_k_star / (actual_n ** 0.2)
        print(
            f"  [Phase-0 Calibration]  N={actual_n:,}  "
            f"N^{{1/5}}={actual_n**0.2:.3f}  "
            f"K*_empirical={best_k_star}  →  C = {C:.4f}"
        )
        return C


# ──────────────────────────────────────────────────────────────────────────────
# 2.  StaticCloudSimulator  (Fluid LP oracle + loss measurement)
# ──────────────────────────────────────────────────────────────────────────────

class StaticCloudSimulator:
    """
    Static offline allocation oracle based on the Fluid Volume LP (OPT_LP).

    All jobs are known at t=0.  The oracle solves:

        max   ∑_j r_j · x_j
        s.t.  ∑_j x_j · resource_j · duration_j  ≤  C_max · T   (Fluid Volume)
              x_j = 0  if phi_j ≤ 0                               (Individual Rationality)
              x_j ∈ [0, 1]                                        (fractional relaxation)

    The Fluid Volume constraint bounds total resource-volume (unit·time) over
    the full horizon T, rather than tracking instantaneous occupancy.  This is
    the OPT_LP benchmark required for DLENT regret bounds.

    Per-job scalarized reward (weighted linear scalarization):

        r_j = lambda1 · q_j · v_j
            + lambda2 · (phi_j − c_elec_j)
            − lambda3 · c_carbon_j

    Parameters
    ----------
    capacity       : instantaneous cluster capacity C_max (resource units).
    horizon        : time horizon T (same unit as duration).
    lambdas        : (lambda1, lambda2, lambda3), default equal weights 1/3.
    """

    def __init__(
        self,
        capacity,
        horizon: float,
        lambdas: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        # Accept either a scalar capacity (backwards-compatible) or a
        # dict with per-resource capacities, e.g. {'cpu': 230.0, 'ram':130.0}
        if isinstance(capacity, dict):
            self.C_max = {k: float(v) for k, v in capacity.items()}
            # ensure both keys exist
            if 'cpu' not in self.C_max:
                self.C_max['cpu'] = 0.0
            if 'ram' not in self.C_max:
                self.C_max['ram'] = 0.0
        else:
            cap = float(capacity)
            self.C_max = {'cpu': cap, 'ram': cap}

        self.T = horizon
        self.l1, self.l2, self.l3 = lambdas

    # ------------------------------------------------------------------
    # Per-job reward vector
    # ------------------------------------------------------------------

    def _reward_vector(
        self,
        v:        np.ndarray,
        phi:      np.ndarray,
        q:        np.ndarray,
        c_elec:   np.ndarray,
        c_carbon: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the scalarized reward coefficient r_j for every job.

        r_j = lambda1 · q_j · v_j
            + lambda2 · (phi_j − c_elec_j)
            − lambda3 · c_carbon_j
        """
        return (
            self.l1 * q * v
            + self.l2 * (phi - c_elec)
            - self.l3 * c_carbon
        )

    # ------------------------------------------------------------------
    # Fluid LP oracle
    # ------------------------------------------------------------------

    def run_allocation(
        self,
        df: pd.DataFrame,
        v_col:   str,
        phi_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the Fluid Volume LP and return fractional decisions + rewards.

        Individual Rationality is enforced by fixing x_j = 0 (upper bound 0)
        for all jobs with phi_j ≤ 0, before the LP is even constructed.

        Parameters
        ----------
        df      : DataFrame with columns: duration, resource_req, priority,
                  c_elec, c_carbon, and the caller-supplied v_col / phi_col.
        v_col   : column for valuation  (continuous or discrete).
        phi_col : column for virtual value (continuous or discrete).

        Returns
        -------
        x        : fractional decision array x_j ∈ [0, 1]
        rewards  : per-job contribution x_j · r_j
        """
        n = len(df)

        v        = df[v_col].values.astype(float)
        phi      = df[phi_col].values.astype(float)
        q        = df["priority"].values.astype(float)
        c_elec   = df["c_elec"].values.astype(float)
        c_carbon = df["c_carbon"].values.astype(float)
        duration = df["duration"].values.astype(float)
        # Detect resource columns — prefer explicit per-resource columns
        if "A_cpu" in df.columns and "A_ram" in df.columns:
            resource_cpu = df["A_cpu"].values.astype(float)
            resource_ram = df["A_ram"].values.astype(float)
        elif "resource_cpu" in df.columns and "resource_ram" in df.columns:
            resource_cpu = df["resource_cpu"].values.astype(float)
            resource_ram = df["resource_ram"].values.astype(float)
        elif "resource_req" in df.columns:
            # Backwards compatible single-resource column → treat as CPU
            resource_cpu = df["resource_req"].values.astype(float)
            resource_ram = np.zeros_like(resource_cpu)
        else:
            raise KeyError(
                "No resource columns found. Expected 'A_cpu' and 'A_ram', or 'resource_req'."
            )

        # Per-job reward coefficient
        r_j = self._reward_vector(v, phi, q, c_elec, c_carbon)

        # Fluid volume demand per job: resource_j · duration_j
        volume_cpu = resource_cpu * duration
        volume_ram = resource_ram * duration

        # Fluid Volume constraints (one per resource):  ∑_j x_j · vol_{r,j} ≤ C_max[r] · T
        A_ub = np.vstack([volume_cpu.reshape(1, n), volume_ram.reshape(1, n)])
        b_ub = np.array([self.C_max['cpu'] * self.T, self.C_max['ram'] * self.T])

        # Individual Rationality: x_j ∈ [0, 0] for phi_j ≤ 0
        ir_valid = phi > 0
        bounds = [(0.0, 1.0) if ir_valid[j] else (0.0, 0.0) for j in range(n)]

        # linprog minimises, so negate the reward vector
        res = linprog(-r_j, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.status != 0:
            x = np.zeros(n, dtype=float)
        else:
            x = res.x

        rewards = x * r_j
        return x, rewards

    # ------------------------------------------------------------------
    # Loss evaluation
    # ------------------------------------------------------------------

    def evaluate_loss(
        self,
        df: pd.DataFrame,
        v_cont_col:   str = "v_cont",
        phi_cont_col: str = "phi_cont",
        v_disc_col:   str = "v_disc",
        phi_disc_col: str = "phi_disc",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Solve the LP twice (continuous and discrete) and compute the
        Global Objective Loss.

        Returns
        -------
        results_df : job-level DataFrame with
                     x_cont, x_disc, r_cont, r_disc, loss_per_job
        metrics    : dict of aggregate scalar metrics
        """
        if verbose:
            print("  [Continuous] Solving Fluid LP with continuous (v, phi) …")
        x_cont, r_cont = self.run_allocation(df, v_cont_col, phi_cont_col)

        if verbose:
            print("  [Discrete]   Solving Fluid LP with discrete  (v, phi) …")
        x_disc, r_disc = self.run_allocation(df, v_disc_col, phi_disc_col)

        # Per-job contribution loss
        loss_per_job = r_cont - r_disc   # = x_cont·r_cont_j  − x_disc·r_disc_j

        results_df = df.copy().reset_index(drop=True)
        results_df["x_cont"]       = x_cont
        results_df["x_disc"]       = x_disc
        results_df["r_cont"]       = r_cont
        results_df["r_disc"]       = r_disc
        results_df["loss_per_job"] = loss_per_job

        total_cont = float(np.sum(r_cont))
        total_disc = float(np.sum(r_disc))
        abs_loss   = total_cont - total_disc
        pct_loss   = (abs_loss / total_cont * 100) if total_cont > 0 else 0.0

        # IR-downgrade: jobs where continuous assigned x>0 but discrete forced x=0
        # (phi_disc ≤ 0 due to binning rounding a previously positive phi down)
        ir_downgraded = int(np.sum((x_cont > 0) & (x_disc == 0)))

        # Acceptance / rejection counts for quick interpretability
        accepted_cont = int(np.sum(x_cont > 0))
        accepted_disc = int(np.sum(x_disc > 0))
        ir_rejected_cont = int(np.sum(df[phi_cont_col].values.astype(float) <= 0))
        ir_rejected_disc = int(np.sum(df[phi_disc_col].values.astype(float) <= 0))

        metrics = {
            "Total Continuous Reward": total_cont,
            "Total Discrete Reward":   total_disc,
            "Absolute Global Loss":    abs_loss,
            "Percentage Loss (%)":     pct_loss,
            "Accepted Jobs (Continuous)": accepted_cont,
            "Accepted Jobs (Discrete)":   accepted_disc,
            "IR Rejected Jobs (Continuous)": ir_rejected_cont,
            "IR Rejected Jobs (Discrete)":   ir_rejected_disc,
            "IR Downgrades":           ir_downgraded,
        }

        return results_df, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SimulatorDiscretizer  — wires the two together for all three baselines
# ──────────────────────────────────────────────────────────────────────────────

class SimulatorDiscretizer:
    """
    Evaluates Uniform, Geometric, and DP Optimal discretization baselines
    by running the StaticCloudSimulator (Fluid LP) and measuring the
    Global Objective Loss.

    Parameters
    ----------
    K_bins   : number of discrete price levels.
    capacity : instantaneous cluster capacity C_max (resource units).
    horizon  : time horizon T (same unit as df["duration"]).
               The Fluid Volume budget is  C_max * T.
    lambdas  : (lambda1, lambda2, lambda3) for the scalarized reward.
               Defaults to equal weighting (1/3, 1/3, 1/3).
    """

    METHODS: dict[str, str] = {
        "Uniform":    "uniform_grid",
        "Geometric":  "geometric_grid",
        "DP Optimal": "dp_optimal_grid",
    }

    def __init__(
        self,
        K_bins: int,
        capacity: float,
        horizon: float,
        lambdas: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        self.K         = K_bins
        self.disc      = Discretizer(K_bins)
        self.simulator = StaticCloudSimulator(capacity, horizon, lambdas)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _build_sim_df(
        self,
        df: pd.DataFrame,
        method_name: Literal["Uniform", "Geometric", "DP Optimal"],
        v_col:   str = "v",
        phi_col: str = "phi_v",
    ) -> pd.DataFrame:
        """
        Discretize v and phi columns and attach them to a simulator-ready df.
        Continuous columns are copied as-is; discrete columns are added.
        """
        grid_fn = getattr(self.disc, self.METHODS[method_name])

        v_values   = df[v_col].values.astype(float)
        phi_values = df[phi_col].values.astype(float)

        sim_df = df.copy()
        sim_df["v_cont"]   = v_values
        sim_df["phi_cont"] = phi_values
        sim_df["v_disc"]   = grid_fn(v_values)
        sim_df["phi_disc"] = grid_fn(phi_values)

        return sim_df

    def evaluate_method(
        self,
        df: pd.DataFrame,
        method_name: Literal["Uniform", "Geometric", "DP Optimal"],
        v_col:   str = "v",
        phi_col: str = "phi_v",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run the full simulation pipeline for a single discretization method.

        Required columns in df
        ----------------------
        v_col, phi_col, arrival_time, duration, resource_req,
        priority, c_elec, c_carbon

        Returns
        -------
        results_df : job-level results (x_cont, x_disc, r_cont, r_disc, loss)
        metrics    : aggregate scalar metrics dict
        """
        if method_name not in self.METHODS:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                f"Choose from: {list(self.METHODS)}"
            )

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Method: {method_name}  (K={self.K})")
            print(f"{'─'*60}")

        t0     = time.perf_counter()
        sim_df = self._build_sim_df(df, method_name, v_col, phi_col)
        results_df, metrics = self.simulator.evaluate_loss(
            sim_df, verbose=verbose
        )
        elapsed = time.perf_counter() - t0
        metrics["Execution Time (s)"] = round(elapsed, 4)

        if verbose:
            self._print_metrics(method_name, metrics)

        return results_df, metrics

    # ------------------------------------------------------------------
    # Compare all three baselines
    # ------------------------------------------------------------------

    def compare_all(
        self,
        df: pd.DataFrame,
        v_col:   str = "v",
        phi_col: str = "phi_v",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate all three methods and return a side-by-side comparison table.

        Returns
        -------
        summary_df : DataFrame indexed by method name with one row per baseline
        """
        rows = {}
        for method in self.METHODS:
            _, metrics = self.evaluate_method(
                df, method, v_col, phi_col, verbose=verbose
            )
            rows[method] = metrics

        summary_df = pd.DataFrame(rows).T
        summary_df.index.name = "Method"

        if verbose:
            sep = "=" * 78
            print(f"\n{sep}")
            print(f"  COMPARISON SUMMARY  (K={self.K}, C_max={self.simulator.C_max})")
            print(f"{sep}")
            print(summary_df.to_string())
            print(f"{sep}\n")

        return summary_df

    # ------------------------------------------------------------------
    # Sweep over K values for a single method
    # ------------------------------------------------------------------

    def sweep_k(
        self,
        df: pd.DataFrame,
        method_name: Literal["Uniform", "Geometric", "DP Optimal"],
        k_values: list[int],
        v_col:   str = "v",
        phi_col: str = "phi_v",
        verbose: bool = True,
        early_stop_threshold_pct: float | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate one method at multiple values of K and return the loss curve.
        Useful for understanding how granularity affects Global Objective Loss.

        Returns
        -------
        sweep_df : DataFrame with K, metrics per row.
        """
        rows = []
        prev_pct = float('nan')
        for k in k_values:
            sd = SimulatorDiscretizer(
                K_bins=k,
                capacity=self.simulator.C_max,
                horizon=self.simulator.T,
                lambdas=(self.simulator.l1, self.simulator.l2, self.simulator.l3),
            )
            _, metrics = sd.evaluate_method(
                df, method_name, v_col, phi_col, verbose=False
            )
            pct = metrics["Percentage Loss (%)"]
            metrics["K"] = k
            rows.append(metrics)
            if verbose:
                print(
                    f"  {method_name:12s}  K={k:3d} → "
                    f"Loss={metrics['Absolute Global Loss']:10.4f}  "
                    f"({pct:5.2f}%)"
                )
            # If early-stop threshold supplied, compute marginal improvement
            # relative to previous K and stop this method's sweep when the
            # improvement falls below the threshold.
            if early_stop_threshold_pct is not None:
                if not np.isnan(prev_pct) and prev_pct > 0:
                    improvement = (prev_pct - pct) / abs(prev_pct) * 100
                    if improvement < early_stop_threshold_pct:
                        if verbose:
                            print(
                                f"    Early-stop triggered for {method_name} at K={k} "
                                f"(improvement={improvement:.3f}% < {early_stop_threshold_pct}%)"
                            )
                        break
                prev_pct = pct

        return pd.DataFrame(rows).set_index("K")

    # ------------------------------------------------------------------
    # Phase-lagged DP scaling test
    # ------------------------------------------------------------------

    def run_phase_lagged_scaling_test(
        self,
        df:              pd.DataFrame,
        v_col:           str   = "v_rate",
        phi_col:         str   = "phi_rate",
        initial_batch_size: int = 500,
        n_phases:        int | None = None,
        K_fixed:         int   = 32,
        k_search_max:    int   = 64,
        k_search_threshold_pct: float = 1.0,
        C:               float | None = None,
        lambda_1:        float = 1/3,
        lambda_2:        float = 1/3,
        seed:            int   = 42,
    ) -> pd.DataFrame:
        """
        Phase-lagged DP scaling test using the Fluid LP as the evaluation oracle.

        Data pipeline (strict no-leakage):
        ───────────────────────────────────
        Phase 0 (warm-up):
          • Draws ``initial_batch_size`` jobs, splits 50/50.
          • Fits DP boundaries on the train half; evaluates scalarized loss on
            the test half to calibrate C (if not supplied) and to produce the
            first set of boundaries for Phase 1.

        Phase M  (M = 1, 2, …):
          • Draws the next ``batch_size`` jobs (batch doubles every phase).
          • Evaluates three DP strategies using boundaries trained on Phase M-1:
              1. Fixed K  (K = K_fixed, user-specified ceiling)
              2. Theoretical K*  (K_m = ⌈C · N^{1/5}⌉, where N is Phase M-1 batch size)
              3. Optimal K search  (sweeps K = 2..k_search_max on Phase M-1
                                    data; picks argmin loss on Phase M-1 test
                                    split; always ≤ K*)
          • For each strategy, ``get_dp_boundaries`` is called on Phase M-1
            data; ``apply_boundaries`` maps Phase M (unseen) data to those
            boundaries.  The resulting discrete arrays are fed into the Fluid
            LP oracle to compute Global Objective Loss vs. the continuous LP.
          • Reports per-phase: K chosen, Global Loss (absolute + %), IR
            downgrades, accepted jobs, and wall-clock search time.

        No data from Phase M is ever used to fit boundaries evaluated on
        Phase M — no leakage in either direction.

        Parameters
        ----------
        df                      : full job DataFrame (all phases pooled).
        v_col, phi_col          : column names for valuation and virtual value.
        initial_batch_size      : Phase-0 / Phase-1 batch size (doubles each phase).
        n_phases                : cap on phases (None = run until data exhausted).
        K_fixed                 : fixed-K baseline ceiling.
        k_search_max            : upper limit for the optimal-K sweep.
        k_search_threshold_pct  : early-stop: stop sweep when marginal loss
                                  improvement < this % (default 1 %).
        C                       : pre-calibrated constant for _theoretical_k.
                                  If None, Phase-0 calibration is run first.
        lambda_1, lambda_2      : scalarization weights (matching the simulator).
        seed                    : RNG seed for the shuffle.

        Returns
        -------
        pd.DataFrame with one row per phase and columns:
            Phase, Batch_Size,
            K_Fixed,       Loss_Fixed (abs + %),   IR_Down_Fixed,
            K_Theo,        Loss_Theo  (abs + %),   IR_Down_Theo,
            K_Star,        Loss_Optimal (abs + %), IR_Down_Optimal,
            Time_Search_sec,
            Cont_Reward  (continuous LP baseline for the phase batch)
        """
        sep = "=" * 78

        # ── Shuffle once; consume sequentially so phases never overlap ────────
        rng          = np.random.default_rng(seed=seed)
        shuffle_idx  = rng.permutation(len(df))
        df_shuffled  = df.iloc[shuffle_idx].reset_index(drop=True)

        v_all   = df_shuffled[v_col].values.astype(float)
        phi_all = df_shuffled[phi_col].values.astype(float)

        # ── Phase 0 (warm-up): fit initial boundaries, calibrate C ───────────
        warmup_n     = initial_batch_size
        warmup_v     = v_all[:warmup_n]
        warmup_phi   = phi_all[:warmup_n]
        offset       = warmup_n

        # 50/50 split on warm-up for calibration — no leakage
        split        = warmup_n // 2
        train_v_prev = warmup_v[:split]
        train_phi_prev = warmup_phi[:split]
        # (test half used only inside calibrate_C; not exposed here)

        if C is None:
            print(f"\n{sep}")
            print("  Phase-0 calibration (50/50 train/test split on warm-up batch)…")
            C = Discretizer.calibrate_C(
                v_continuous=warmup_v,
                phi_continuous=warmup_phi,
                warmup_n=warmup_n,
                k_search_max=k_search_max,
                k_search_threshold_pct=k_search_threshold_pct,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                seed=seed,
            )

        print(f"\n{sep}")
        print("  Phase-Lagged DP Scaling Test — three K strategies")
        print(f"  Fixed K={K_fixed}  |  Theoretical K*=⌈C·N^{{1/5}}⌉  C={C:.4f}")
        print(f"  Optimal K search (2..{k_search_max}, "
              f"stop if improvement < {k_search_threshold_pct}%)")
        print(f"  Fluid LP: C_max={self.simulator.C_max}  T={self.simulator.T}h")
        print(f"  λ1={lambda_1:.3f}  λ2={lambda_2:.3f}  λ3={self.simulator.l3:.3f}")
        print(f"{sep}")

        rows       = []
        batch_size = initial_batch_size
        phase      = 0

        while True:
            phase += 1
            if n_phases is not None and phase > n_phases:
                break
            if offset >= len(df_shuffled):
                print(f"\n  Phase {phase}: no data remaining — stopping.")
                break

            # ── Slice Phase-M batch (unseen test data) ────────────────────────
            batch_df = df_shuffled.iloc[offset: offset + batch_size].copy()
            actual_n = len(batch_df)
            offset  += actual_n

            if actual_n == 0:
                break
            if actual_n < batch_size:
                print(f"\n  Phase {phase}: only {actual_n:,} jobs remain "
                      f"(expected {batch_size:,}) — running reduced batch.")

            batch_v   = batch_df[v_col].values.astype(float)
            batch_phi = batch_df[phi_col].values.astype(float)

            # ── Continuous LP baseline for this phase (upper bound) ───────────
            # Run LP with raw continuous v/phi to get R_cont for this batch.
            batch_df["v_cont"]   = batch_v
            batch_df["phi_cont"] = batch_phi

            _, r_cont_arr = self.simulator.run_allocation(
                batch_df, v_col="v_cont", phi_col="phi_cont"
            )
            cont_reward = float(np.sum(r_cont_arr))

            # ── Evaluate one K strategy on Phase M ────────────────────
            def _eval_strategy(k: int, train_v: np.ndarray, train_phi: np.ndarray):
                """
                Fit boundaries on train data (Phase M-1), apply to Phase M,
                solve discrete LP, return (loss_abs, loss_pct, ir_down, accepted).
                No Phase-M data touches the boundary fitting.
                """
                disc         = Discretizer(K_bins=k)
                bounds_v     = disc.get_dp_boundaries(train_v)
                bounds_phi   = disc.get_dp_boundaries(train_phi)

                v_disc       = disc.apply_boundaries(batch_v,   bounds_v)
                phi_disc     = disc.apply_boundaries(batch_phi, bounds_phi)

                test_df = batch_df.copy()
                test_df["v_disc"]   = v_disc
                test_df["phi_disc"] = phi_disc

                _, r_disc_arr = self.simulator.run_allocation(
                    test_df, v_col="v_disc", phi_col="phi_disc"
                )
                disc_reward = float(np.sum(r_disc_arr))

                loss_abs  = cont_reward - disc_reward
                loss_pct  = (loss_abs / cont_reward * 100) if cont_reward > 0 else 0.0

                # IR downgrades: LP excluded a job that continuous LP accepted
                x_cont_phase, _ = self.simulator.run_allocation(
                    test_df, v_col="v_cont", phi_col="phi_cont"
                )
                x_disc_phase, _ = self.simulator.run_allocation(
                    test_df, v_col="v_disc", phi_col="phi_disc"
                )
                ir_down   = int(np.sum((x_cont_phase > 0) & (x_disc_phase == 0)))
                accepted  = int(np.sum(x_disc_phase > 0))

                return loss_abs, loss_pct, ir_down, accepted

            # ── Strategy 1: Fixed K ───────────────────────────────────────────
            t0 = time.perf_counter()
            loss_fixed, pct_fixed, ir_fixed, acc_fixed = _eval_strategy(
                K_fixed, train_v_prev, train_phi_prev
            )
            time_fixed = time.perf_counter() - t0

            # ── Strategy 2: Theoretical K* ────────────────────────────────────
            K_theo = min(
                Discretizer._theoretical_k(len(train_v_prev), C=C),
                K_fixed,   # theoretical bound is always ≤ fixed ceiling
            )
            t0 = time.perf_counter()
            loss_theo, pct_theo, ir_theo, acc_theo = _eval_strategy(
                K_theo, train_v_prev, train_phi_prev
            )
            time_theo = time.perf_counter() - t0

            # ── Strategy 3: Optimal K* search ────────────────────────────────
            # Sweep K on Phase M-1 data (50/50 split to avoid in-sample overfit).
            # The search uses only train_v_prev / train_phi_prev — no Phase-M
            # data is touched.
            t0_search = time.perf_counter()

            # 50/50 internal split on Phase M-1 training data
            prev_n     = len(train_v_prev)
            prev_split = prev_n // 2
            fit_v,   val_v   = train_v_prev[:prev_split],   train_v_prev[prev_split:]
            fit_phi, val_phi = train_phi_prev[:prev_split], train_phi_prev[prev_split:]

            best_k_star  = 2
            best_val_err = float("inf")
            prev_val_err = float("inf")

            for k in range(2, K_theo + 1):
                disc_k     = Discretizer(K_bins=k)
                bv         = disc_k.get_dp_boundaries(fit_v)
                bp         = disc_k.get_dp_boundaries(fit_phi)
                vd         = disc_k.apply_boundaries(val_v,   bv)
                pd_        = disc_k.apply_boundaries(val_phi, bp)
                err_v      = float(np.sum(val_v   - vd))
                err_phi    = float(np.sum(val_phi  - pd_))
                val_err    = lambda_1 * err_v + lambda_2 * err_phi

                if val_err < best_val_err:
                    best_val_err = val_err
                    best_k_star  = k

                if prev_val_err < float("inf") and prev_val_err > 0:
                    if ((prev_val_err - val_err) / prev_val_err) * 100 \
                            < k_search_threshold_pct:
                        break
                prev_val_err = val_err

            # Clamp: optimal K should never exceed the fixed ceiling
            best_k_star = min(best_k_star, K_fixed)
            time_search = time.perf_counter() - t0_search

            # Now evaluate on Phase M with the found K*
            loss_opt, pct_opt, ir_opt, acc_opt = _eval_strategy(
                best_k_star, train_v_prev, train_phi_prev
            )

            # ── Print phase summary ───────────────────────────────────────────
            print(f"\n  Phase {phase}  |  N={actual_n:,}  "
                  f"|  Cont. Reward = {cont_reward:.4f}")
            print(f"    Fixed       (K={K_fixed:3d}):  "
                  f"Loss={loss_fixed:10.4f} ({pct_fixed:5.2f}%)  "
                  f"|  IR↓={ir_fixed}  acc={acc_fixed}  "
                  f"|  Time={time_fixed:.4f}s")
            print(f"    Theoretical (K={K_theo:3d}):  "
                  f"Loss={loss_theo:10.4f} ({pct_theo:5.2f}%)  "
                  f"|  IR↓={ir_theo}  acc={acc_theo}  "
                  f"|  Time={time_theo:.4f}s")
            print(f"    Optimal K*  (K={best_k_star:3d}):  "
                  f"Loss={loss_opt:10.4f} ({pct_opt:5.2f}%)  "
                  f"|  IR↓={ir_opt}  acc={acc_opt}  "
                  f"|  Search={time_search:.4f}s "
                  f"(converged K={best_k_star}, max={k_search_max})")

            rows.append({
                "Phase":             phase,
                "Batch_Size":        actual_n,
                "Cont_Reward":       cont_reward,
                # Fixed K
                "K_Fixed":           K_fixed,
                "Loss_Fixed":        loss_fixed,
                "Loss_Fixed_pct":    pct_fixed,
                "IR_Down_Fixed":     ir_fixed,
                "Accepted_Fixed":    acc_fixed,
                "Time_Fixed_sec":    time_fixed,
                # Theoretical K
                "K_Theo":            K_theo,
                "Loss_Theo":         loss_theo,
                "Loss_Theo_pct":     pct_theo,
                "IR_Down_Theo":      ir_theo,
                "Accepted_Theo":     acc_theo,
                "Time_Theo_sec":     time_theo,
                # Optimal K*
                "K_Star":            best_k_star,
                "Loss_Optimal":      loss_opt,
                "Loss_Optimal_pct":  pct_opt,
                "IR_Down_Optimal":   ir_opt,
                "Accepted_Optimal":  acc_opt,
                "Time_Search_sec":   time_search,
            })

            # ── Slide the window: Phase M becomes Phase M-1 ──────────────────
            train_v_prev   = batch_v
            train_phi_prev = batch_phi
            batch_size    *= 2

        # ── Summary ──────────────────────────────────────────────────────────
        results = pd.DataFrame(rows)
        if len(results) > 0:
            print(f"\n{sep}")
            print(f"  TOTALS ACROSS {len(results)} PHASES")
            print(f"{sep}")
            for label, loss_col in [
                (f"Fixed        K={K_fixed}", "Loss_Fixed"),
                ("Theoretical  K=⌈C·N^1/5⌉", "Loss_Theo"),
                (f"Optimal K*   (search 2..{k_search_max})", "Loss_Optimal"),
            ]:
                total_loss = results[loss_col].sum()
                total_cont = results["Cont_Reward"].sum()
                overall_pct = (total_loss / total_cont * 100) if total_cont > 0 else 0.0
                print(f"  {label}:  "
                      f"Total Loss={total_loss:12.4f}  "
                      f"({overall_pct:.2f}% of total continuous reward)")
            print(f"{sep}\n")

        return results

    # ------------------------------------------------------------------
    # Pretty print helper
    # ------------------------------------------------------------------

    @staticmethod
    def _print_metrics(method_name: str, metrics: dict) -> None:
        print(f"\n  Results for '{method_name}':")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {key:<35s}: {val:.4f}")
            else:
                print(f"    {key:<35s}: {val}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Synthetic data generator  (for standalone testing)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_jobs(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic cloud job dataset for testing.

    Duration is in hours; resource_req in resource units.
    phi_v approximates the Myerson virtual value with ~20% IR-violating jobs.
    """
    rng = np.random.default_rng(seed)

    v     = rng.exponential(scale=10.0, size=n) + rng.uniform(0.5, 5.0, size=n)
    noise = rng.normal(0, 0.5 * v.std(), size=n)
    phi_v = v * rng.uniform(0.3, 1.2, size=n) + noise - 2.0

    df = pd.DataFrame(
        {
            "duration":     rng.exponential(scale=5.0, size=n).clip(1, 50),
            "resource_req": rng.integers(1, 20, size=n).astype(float),
            "priority":     rng.uniform(0.5, 2.0, size=n),
            "c_elec":       rng.uniform(0.1, 2.0, size=n),
            "c_carbon":     rng.uniform(0.0, 1.0, size=n),
            "v":            v,
            "phi_v":        phi_v,
        }
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # ── Configuration ────────────────────────────────────────────────────────
    K_BINS   = 16           # Number of discrete bins
    # Per-resource instantaneous capacities
    CLUSTER_CAPACITY = {"cpu": 230.0, "ram": 130.0}
    HORIZON  = 744.0        # Time horizon T in hours (31-day month)
    LAMBDAS  = (1/3, 1/3, 1/3)   # Equal-weight scalarization

    # ── Dataset ──────────────────────────────────────────────────────────────
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "batch_may2019_2k.csv"
    )

    print("=" * 78)
    print("  Discretization Loss via StaticCloudSimulator (Fluid LP / OPT_LP)")
    print(f"  K={K_BINS}  |  C_max={CLUSTER_CAPACITY}  |  T={HORIZON}h  |  lambdas={LAMBDAS}")
    budgets = {r: CLUSTER_CAPACITY[r] * HORIZON for r in CLUSTER_CAPACITY}
    print(f"  Fluid Volume Budgets (resource·hours): {budgets}")
    print("=" * 78)

    if os.path.exists(DATA_PATH):
        print(f"\nLoading dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

        # Normalize column names from the dataset to the simulator's expected
        # canonical names: 'duration', 'priority', 'c_elec', 'c_carbon', 'v', 'phi_v'
        col_map = {}
        if 'q_j' in df.columns:
            col_map['q_j'] = 'priority'
        if 'D (hours)' in df.columns:
            col_map['D (hours)'] = 'duration'
        if 'C_elec' in df.columns:
            col_map['C_elec'] = 'c_elec'
        if 'C_carbon' in df.columns:
            col_map['C_carbon'] = 'c_carbon'
        if 'v_total' in df.columns:
            col_map['v_total'] = 'v'
        elif 'v' in df.columns and 'v' not in col_map.values():
            col_map['v'] = 'v'
        if 'phi_total' in df.columns:
            col_map['phi_total'] = 'phi_v'
        elif 'phi_v' in df.columns and 'phi_v' not in col_map.values():
            col_map['phi_v'] = 'phi_v'

        if col_map:
            df = df.rename(columns=col_map)

        # Choose v/phi columns for the simulation
        v_col = 'v' if 'v' in df.columns else 'v'
        phi_col = 'phi_v' if 'phi_v' in df.columns else 'phi_v'

        print(f"  Total jobs: {len(df):,}")
    else:
        print("\nDataset not found — using synthetic data (n=500, seed=42).")
        df = generate_synthetic_jobs(n=500, seed=42)
        v_col, phi_col = "v", "phi_v"
        print(f"  Total jobs: {len(df):,}")
        ir_jobs = int((df[phi_col] > 0).sum())
        print(f"  IR-valid jobs (phi_v > 0): {ir_jobs:,}")

    # ── 1. Full comparison at fixed K ────────────────────────────────────────
    sd = SimulatorDiscretizer(
        K_bins=K_BINS, capacity=CLUSTER_CAPACITY, horizon=HORIZON, lambdas=LAMBDAS
    )
    summary = sd.compare_all(df, v_col=v_col, phi_col=phi_col, verbose=True)

    # ── 2. K-sweep for all three methods ─────────────────────────────────────
    K_SWEEP = [2, 4, 8, 16, 32, 64]
    print("=" * 78)
    print(f"  K-SWEEP  ({K_SWEEP})")
    print("=" * 78)

    sweep_results = {}
    for method in SimulatorDiscretizer.METHODS:
        print(f"\n  → {method}")
        sweep_results[method] = sd.sweep_k(
            df, method_name=method, k_values=K_SWEEP,
            v_col=v_col, phi_col=phi_col, verbose=True,
        )

    # Consolidated K-sweep table (Percentage Loss only)
    sweep_pct = pd.DataFrame(
        {m: sweep_results[m]["Percentage Loss (%)"] for m in sweep_results}
    )
    sweep_pct.index.name = "K"
    print("\n  Percentage Loss (%) by K and Method:")
    print(sweep_pct.to_string(float_format="{:.2f}".format))

    # ── 3. Phase-lagged DP scaling test ──────────────────────────────────────
    print("=" * 78)
    print("  PHASE-LAGGED DP SCALING TEST")
    print("=" * 78)

    phase_results = sd.run_phase_lagged_scaling_test(
        df,
        v_col=v_col,
        phi_col=phi_col,
        initial_batch_size=200,   # Phase-0 / Phase-1 batch size
        n_phases=5,               # cap for demo; remove for full dataset run
        K_fixed=16,
        k_search_max=30,
        k_search_threshold_pct=1.0,
        C=None,                   # will auto-calibrate in Phase-0
        lambda_1=1/3,
        lambda_2=1/3,
    )

    print("\nPhase-lagged results table:")
    cols_to_show = [
        "Phase", "Batch_Size", "Cont_Reward",
        "K_Fixed",  "Loss_Fixed",   "Loss_Fixed_pct",
        "K_Theo",   "Loss_Theo",    "Loss_Theo_pct",
        "K_Star",   "Loss_Optimal", "Loss_Optimal_pct",
        "Time_Search_sec",
    ]
    print(phase_results[cols_to_show].to_string(index=False, float_format="{:.3f}".format))

    print("\nDone.")