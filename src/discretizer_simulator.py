"""
discretizer_simulator.py
========================
Evaluates Uniform, Geometric, and DP Grid discretization baselines
by measuring their Grid Loss through the StaticCloudSimulator environment.

Allocation Policy — Fluid Volume LP (Fluid Upper Bound)
--------------------------------------------------------
All jobs are known at t=0 (static/offline setting). The oracle solves a
Linear Program over fractional decisions x_j ∈ [0, 1]:

    max   ∑_j r_j · x_j
    s.t.  ∑_j x_j · resource_j · duration_j  ≤  C_max · T   (Fluid Volume)
          x_j = 0   if r_j < 0                                (Scalarized Reward Filter)
          x_j ∈ [0, 1]

where the scalarized per-job reward is:

    r_j = lambda1 · q_j · v_j  +  lambda2 · (phi_j − c_elec_j)
          − lambda3 · c_carbon_j

with equal lambdas lambda1 = lambda2 = lambda3 = 1/3.

The Fluid Volume constraint replaces instantaneous capacity tracking:
instead of checking point-in-time usage it bounds the total resource
volume (core·hours) consumed over the horizon T, matching the
Fluid Upper Bound benchmark used in DLENT regret bounds.

Loss Metric (Grid Loss in Objective Space)
-------------------------------------------
Two LP solves per baseline:
  - Continuous  x*_cont  →  R_cont = ∑_j x*_cont_j · r_cont_j
  - Discrete    x*_disc  →  R_disc = ∑_j x*_disc_j · r_disc_j

    Grid Loss  =  R_cont − R_disc

Three discretization baselines (K bins each)
--------------------------------------------
  1. Uniform   — arithmetic grid over [phi_min, phi_max]
  2. Geometric — geometric (log-spaced) grid
  3. DP Grid — dynamic-programming grid minimising revenue loss

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

    @staticmethod
    def enforce_positive_reward(
        df: pd.DataFrame,
        v_col: str = "v_rate",
        phi_col: str = "phi_rate",
        lambdas: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> pd.DataFrame:
        """Return only jobs whose scalarized reward r_j >= 0.
        
        Jobs with r_j < 0 are Grid-Induced Rejections (GIR), NOT
        Individual Rationality violations. A job may have negative profit
        (Φ < 0) but still be accepted if it delivers high green energy usage
        and priority. The scalarized reward is the proper filter:

        r_j = λ₁·q_j·L_j + λ₂·(Φ(L_j) − C_elec_j) − λ₃·C_carbon_j
        """
        l1, l2, l3 = lambdas
        r = (
            l1 * df["priority"] * df[v_col]
            + l2 * (df[phi_col] - df["c_elec"])
            - l3 * df["c_carbon"]
        )
        return df[r > 0]

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
    # non-positive values) and returns a discretized array of the same
    # length, leaving non-positive values unchanged so downstream reward-
    # based GIR (Grid-Induced Rejection) filtering can still reject toxic
    # jobs correctly.
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
        Baseline 3 — DP Grid Grid.
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

        Only scalarized-reward-valid values (r_j >= 0) participate in the DP;
        the returned boundaries may not cover the full range of continuous
        boundaries are therefore always positive, which guarantees that
        apply_boundaries will never accidentally map a positive test value
        to 0 due to a missing lower bin.

        Parameters
        ----------
        train_data : 1-D array of continuous values from Phase M-1.
                     May contain non-positive values (Grid-Induced Rejections);
                     these are filtered out before fitting.

        Returns
        -------
        boundaries : sorted array of at most K positive lower-bound values.
        """
        # Filter to scalarized-reward-valid training values only (r_j >= 0)
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
          This avoids spurious Grid-Induced Rejections caused purely by
          out-of-range test values, which would be a data-leakage artefact,
          not a true discretization error.
        * Non-positive test values (jobs with r_j < 0 in continuous space)
          are passed through unchanged at their original value. The LP oracle's
          reward-based GIR filter handles them correctly.

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
    Static offline allocation oracle based on the Fluid Volume LP
    (Fluid Upper Bound).

    All jobs are known at t=0. The oracle solves:

        max   ∑_j r_j · x_j
        s.t.  ∑_j x_j · resource_j · duration_j  ≤  C_max · T   (Fluid Volume)
              x_j = 0  if r_j < 0                                 (Grid-Induced Rejection filter)
              x_j ∈ [0, 1]                                        (fractional relaxation)

    The Fluid Volume constraint bounds total resource-volume (unit·time) over
    the full horizon T, rather than tracking instantaneous occupancy. This is
    the Fluid Upper Bound benchmark required for DLENT regret bounds.

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
        rho: float | None = None,
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

        self.T   = horizon
        self.rho = rho
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

        Grid-Induced Rejection (GIR) filter follows the offline optimizer's
        scalarized reward filter: jobs with strictly negative scalarized reward
        (r_j < 0) are forced to x_j = 0 via variable bounds before solving.
        This is NOT Individual Rationality — a job may lose money (Φ < 0) but
        still be accepted if it uses 100% green energy and has high priority.

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

        # Fluid Volume constraints (one per resource):  ∑_j x_j · vol_{r,j} ≤ budget[r]
        # Adaptive mode (rho): budget = rho · Σ vol_{r,j}  (mirrors FluidLPOptimizer)
        # Legacy mode:         budget = C_max[r] · T
        A_ub = np.vstack([volume_cpu.reshape(1, n), volume_ram.reshape(1, n)])
        if self.rho is not None:
            b_ub = np.array([self.rho * volume_cpu.sum(), self.rho * volume_ram.sum()])
        else:
            b_ub = np.array([self.C_max['cpu'] * self.T, self.C_max['ram'] * self.T])

        # Grid-Induced Rejection (GIR) filter (offline_optimizer parity):
        # x_j ∈ [0, 0] for strictly negative scalarized rewards.
        gir_valid = r_j >= 0
        bounds = [(0.0, 1.0) if gir_valid[j] else (0.0, 0.0) for j in range(n)]

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
            print("  [Theoretical Upper Bound] Solving Fluid LP with continuous (v, phi) …")
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

        # Grid-Induced Rejections (GIR): jobs accepted in continuous but forced
        # to zero in discrete. With reward-based filtering, this often happens
        # when discretization pushes a job's scalarized reward from r_j >= 0 to r_j < 0.
        gir_count = int(np.sum((x_cont > 0) & (x_disc == 0)))

        # Acceptance / rejection counts for quick interpretability
        accepted_cont = int(np.sum(x_cont > 0))
        accepted_disc = int(np.sum(x_disc > 0))
        q = df["priority"].values.astype(float)
        c_elec = df["c_elec"].values.astype(float)
        c_carbon = df["c_carbon"].values.astype(float)
        r_cont_coeff = self._reward_vector(
            df[v_cont_col].values.astype(float),
            df[phi_cont_col].values.astype(float),
            q,
            c_elec,
            c_carbon,
        )
        r_disc_coeff = self._reward_vector(
            df[v_disc_col].values.astype(float),
            df[phi_disc_col].values.astype(float),
            q,
            c_elec,
            c_carbon,
        )
        gir_rejected_cont = int(np.sum(r_cont_coeff < 0))
        gir_rejected_disc = int(np.sum(r_disc_coeff < 0))

        metrics = {
            "Fluid Upper Bound Reward":       total_cont,
            "Total Discrete Reward":          total_disc,
            "Absolute Grid Loss":             abs_loss,
            "Percentage Loss (%)":            pct_loss,
            "Accepted Jobs (Fluid Upper Bound)": accepted_cont,
            "Accepted Jobs (Discrete)":          accepted_disc,
            "GIR Rejected Jobs (Fluid Upper Bound)": gir_rejected_cont,
            "GIR Rejected Jobs (Discrete)":          gir_rejected_disc,
            "Grid-Induced Rejections (GIR)":         gir_count,
            # Legacy aliases (backward compatibility)
            "Total Continuous Reward":    total_cont,
            "Accepted Jobs (Continuous)": accepted_cont,
            "IR Rejected Jobs (Theoretical Upper Bound)": gir_rejected_cont,
            "IR Rejected Jobs (Discrete)":                gir_rejected_disc,
            "IR Downgrades":              gir_count,
            "Theoretical Upper Bound Reward": total_cont,
            "IR Rejected Jobs (Continuous)":  gir_rejected_cont,
        }

        return results_df, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SimulatorDiscretizer  — wires the two together for all three baselines
# ──────────────────────────────────────────────────────────────────────────────

class SimulatorDiscretizer:
    """
    Evaluates Uniform, Geometric, and DP Grid discretization baselines
    by running the StaticCloudSimulator (Fluid LP) and measuring the
    Global Objective Loss.

    Parameters
    ----------
    K_bins   : number of discrete price levels.
    capacity : instantaneous cluster capacity C_max (resource units).
               Ignored when rho is set.
    horizon  : time horizon T (same unit as df["duration"]).
               Ignored when rho is set.
    lambdas  : (lambda1, lambda2, lambda3) for the scalarized reward.
               Defaults to equal weighting (1/3, 1/3, 1/3).
    rho      : load factor for adaptive fluid budget (default 0.6).
               When set, budget[r] = rho · Σ_j A_{r,j} · D_j  (mirrors
               FluidLPOptimizer), overriding the fixed C_max · T budget.
               Set to None to fall back to the legacy C_max · T mode.
    """

    METHODS: dict[str, str] = {
        "Uniform":    "uniform_grid",
        "Geometric":  "geometric_grid",
        "DP Grid": "dp_optimal_grid",
    }

    def __init__(
        self,
        K_bins: int,
        capacity: float = 0.0,
        horizon: float = 1.0,
        lambdas: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        rho: float | None = 0.6,
    ):
        self.K         = K_bins
        self.disc      = Discretizer(K_bins)
        self.simulator = StaticCloudSimulator(capacity, horizon, lambdas, rho=rho)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _build_sim_df(
        self,
        df: pd.DataFrame,
        method_name: Literal["Uniform", "Geometric", "DP Grid"],
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
        method_name: Literal["Uniform", "Geometric", "DP Grid"],
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
            if self.simulator.rho is not None:
                budget_str = f"rho={self.simulator.rho}"
            else:
                budget_str = f"C_max={self.simulator.C_max}"
            print(f"  COMPARISON SUMMARY  (K={self.K}, {budget_str})")
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
        method_name: Literal["Uniform", "Geometric", "DP Grid"],
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
                rho=self.simulator.rho,
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
                    f"Loss={metrics['Absolute Grid Loss']:10.4f}  "
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
              1. Static Baseline Grid  (e.g., K = K_fixed)
              2. Theoretical K Bound   (K_m = ⌈C · N^{1/5}⌉, where N is Phase M-1 batch size)
              3. Heuristic DP Grid     (sweeps K = 2..k_search_max on Phase M-1
                                        data; picks argmin loss on Phase M-1 test
                                        split; converges when marginal improvement < 1%)
          • For each strategy, ``get_dp_boundaries`` is called on Phase M-1
            data; ``apply_boundaries`` maps Phase M (unseen) data to those
            boundaries. The resulting discrete arrays are fed into the Fluid
            LP oracle to compute Grid Loss vs. the Fluid Upper Bound.
          • Reports per-phase: K chosen, Grid Loss (absolute + %), Grid-Induced
            Rejections (GIR), accepted jobs, and wall-clock search time.

        No data from Phase M is ever used to fit boundaries evaluated on
        Phase M — no leakage in either direction.

        Parameters
        ----------
        df                      : full job DataFrame (all phases pooled).
        v_col, phi_col          : column names for valuation and virtual value.
        initial_batch_size      : Phase-0 / Phase-1 batch size (doubles each phase).
        n_phases                : cap on phases (None = run until data exhausted).
        K_fixed                 : K for the Static Baseline Grid.
        k_search_max            : upper limit for the Heuristic DP Grid sweep.
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
            K_Static,       Loss_Static (abs + %),   GIR_Static,
            K_Theo_Bound,   Loss_Theo_Bound (abs + %), GIR_Theo_Bound,
            K_Heuristic_DP, Loss_Heuristic_DP (abs + %), GIR_Heuristic_DP,
            Time_Heuristic_DP_sec,
            Fluid_Upper_Bound_Reward (phase-level Fluid Upper Bound)
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
        print(f"  Static Baseline Grid K={K_fixed}  |  Theoretical K Bound=⌈C·N^{{1/5}}⌉  C={C:.4f}")
        print(f"  Heuristic DP Grid sweep (2..{k_search_max}, "
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

            # ── Theoretical Upper Bound for this phase ─────────────────────────
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
                solve discrete LP, return (loss_abs, loss_pct, gir_count, accepted).
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

                # Grid-Induced Rejections (GIR): LP excluded a job that the Fluid
                # Upper Bound accepted.
                x_cont_phase, _ = self.simulator.run_allocation(
                    test_df, v_col="v_cont", phi_col="phi_cont"
                )
                x_disc_phase, _ = self.simulator.run_allocation(
                    test_df, v_col="v_disc", phi_col="phi_disc"
                )
                gir_count = int(np.sum((x_cont_phase > 0) & (x_disc_phase == 0)))
                accepted  = int(np.sum(x_disc_phase > 0))

                return loss_abs, loss_pct, gir_count, accepted

            # ── Strategy 1: Static Baseline Grid ──────────────────────────────
            t0 = time.perf_counter()
            loss_fixed, pct_fixed, gir_fixed, acc_fixed = _eval_strategy(
                K_fixed, train_v_prev, train_phi_prev
            )
            time_fixed = time.perf_counter() - t0

            # ── Strategy 2: Theoretical K Bound ───────────────────────────────
            K_theo = min(
                Discretizer._theoretical_k(len(train_v_prev), C=C),
                K_fixed,   # theoretical bound is always ≤ fixed ceiling
            )
            t0 = time.perf_counter()
            loss_theo, pct_theo, gir_theo, acc_theo = _eval_strategy(
                K_theo, train_v_prev, train_phi_prev
            )
            time_theo = time.perf_counter() - t0

            # ── Strategy 3: Heuristic DP Grid ─────────────────────────────────
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

            # Clamp: heuristic K should never exceed the static baseline ceiling
            best_k_star = min(best_k_star, K_fixed)
            time_search = time.perf_counter() - t0_search

            # Now evaluate on Phase M with the selected heuristic K.
            loss_opt, pct_opt, gir_opt, acc_opt = _eval_strategy(
                best_k_star, train_v_prev, train_phi_prev
            )

            # ── Print phase summary ───────────────────────────────────────────
            print(f"\n  Phase {phase}  |  N={actual_n:,}  "
                  f"|  Fluid Upper Bound Reward = {cont_reward:.4f}")
            print(f"    Static Baseline Grid (K={K_fixed:3d}):  "
                  f"Loss={loss_fixed:10.4f} ({pct_fixed:5.2f}%)  "
                  f"|  GIR={gir_fixed}  acc={acc_fixed}  "
                  f"|  Time={time_fixed:.4f}s")
            print(f"    Theoretical K Bound (K={K_theo:3d}):  "
                  f"Loss={loss_theo:10.4f} ({pct_theo:5.2f}%)  "
                  f"|  GIR={gir_theo}  acc={acc_theo}  "
                  f"|  Time={time_theo:.4f}s")
            print(f"    Heuristic DP Grid (K={best_k_star:3d}):  "
                  f"Loss={loss_opt:10.4f} ({pct_opt:5.2f}%)  "
                  f"|  GIR={gir_opt}  acc={acc_opt}  "
                  f"|  Search={time_search:.4f}s "
                  f"(converged K={best_k_star}, max={k_search_max})")

            rows.append({
                "Phase":             phase,
                "Batch_Size":        actual_n,
                "Fluid_Upper_Bound_Reward": cont_reward,
                # Static Baseline Grid
                "K_Static":              K_fixed,
                "Loss_Static":           loss_fixed,
                "Loss_Static_pct":       pct_fixed,
                "GIR_Static":            gir_fixed,
                "Accepted_Static":       acc_fixed,
                "Time_Static_sec":       time_fixed,
                # Theoretical K Bound
                "K_Theo_Bound":          K_theo,
                "Loss_Theo_Bound":       loss_theo,
                "Loss_Theo_Bound_pct":   pct_theo,
                "GIR_Theo_Bound":        gir_theo,
                "Accepted_Theo_Bound":   acc_theo,
                "Time_Theo_Bound_sec":   time_theo,
                # Heuristic DP Grid
                "K_Heuristic_DP":        best_k_star,
                "Loss_Heuristic_DP":     loss_opt,
                "Loss_Heuristic_DP_pct": pct_opt,
                "GIR_Heuristic_DP":      gir_opt,
                "Accepted_Heuristic_DP": acc_opt,
                "Time_Heuristic_DP_sec": time_search,
            })

            # ── Slide the window: Phase M becomes Phase M-1 ──────────────────
            train_v_prev   = batch_v
            train_phi_prev = batch_phi
            batch_size    *= 2

        # ── Summary ──────────────────────────────────────────────────────────
        results = pd.DataFrame(rows)
        if len(results) > 0:
            # Legacy aliases for backward compatibility with previous notebooks
            # and scripts that still rely on the old notation.
            results["Theo_Upper_Bound_Reward"] = results["Fluid_Upper_Bound_Reward"]
            results["Cont_Reward"] = results["Fluid_Upper_Bound_Reward"]

            results["K_Fixed"] = results["K_Static"]
            results["Loss_Fixed"] = results["Loss_Static"]
            results["Loss_Fixed_pct"] = results["Loss_Static_pct"]
            results["IR_Down_Fixed"] = results["GIR_Static"]
            results["IR_Down_Static"] = results["GIR_Static"]
            results["Accepted_Fixed"] = results["Accepted_Static"]
            results["Time_Fixed_sec"] = results["Time_Static_sec"]

            results["K_Theo"] = results["K_Theo_Bound"]
            results["Loss_Theo"] = results["Loss_Theo_Bound"]
            results["Loss_Theo_pct"] = results["Loss_Theo_Bound_pct"]
            results["IR_Down_Theo"] = results["GIR_Theo_Bound"]
            results["IR_Down_Theo_Bound"] = results["GIR_Theo_Bound"]
            results["Accepted_Theo"] = results["Accepted_Theo_Bound"]
            results["Time_Theo_sec"] = results["Time_Theo_Bound_sec"]

            results["K_Star"] = results["K_Heuristic_DP"]
            results["Loss_Optimal"] = results["Loss_Heuristic_DP"]
            results["Loss_Optimal_pct"] = results["Loss_Heuristic_DP_pct"]
            results["IR_Down_Optimal"] = results["GIR_Heuristic_DP"]
            results["IR_Down_Heuristic_DP"] = results["GIR_Heuristic_DP"]
            results["Accepted_Optimal"] = results["Accepted_Heuristic_DP"]
            results["Time_Search_sec"] = results["Time_Heuristic_DP_sec"]

        if len(results) > 0:
            print(f"\n{sep}")
            print(f"  TOTALS ACROSS {len(results)} PHASES")
            print(f"{sep}")
            for label, loss_col in [
                (f"Static Baseline Grid  K={K_fixed}", "Loss_Static"),
                ("Theoretical K Bound   K=⌈C·N^1/5⌉", "Loss_Theo_Bound"),
                (f"Heuristic DP Grid     (search 2..{k_search_max})", "Loss_Heuristic_DP"),
            ]:
                total_loss = results[loss_col].sum()
                total_cont = results["Fluid_Upper_Bound_Reward"].sum()
                overall_pct = (total_loss / total_cont * 100) if total_cont > 0 else 0.0
                print(f"  {label}:  "
                      f"Total Loss={total_loss:12.4f}  "
                      f"({overall_pct:.2f}% of total Fluid Upper Bound reward)")
            print(f"{sep}\n")

        return results

    # ------------------------------------------------------------------
    # 2 × 5 Matrix: Geometric vs DP  ×  {K=32, K=64, K=128, Theo (DP), Heuristic}
    # ------------------------------------------------------------------

    def run_2x3_matrix_test(
        self,
        df:               pd.DataFrame,
        v_col:            str   = "v_rate",
        phi_col:          str   = "phi_rate",
        phase_size:       int   = 50,
        static_ks:        list[int] = None,
        k_heuristic_max:  int   = 256,
        k_search_threshold_pct: float = 1.0,
        C:                float | None = None,
        lambda_1:         float = 1 / 3,
        lambda_2:         float = 1 / 3,
        n_phases:         int | None = None,
        seed:             int   = 42,
        verbose:          bool  = True,
    ) -> pd.DataFrame:
        """
        Full 2 × 5 discretization cost-benefit matrix.

        Spacing algorithms (rows)
        ─────────────────────────
        • Geometric  — O(N log N) geomspace edges, no fitting required.
        • DP Grid    — O(K · N²) optimal bin boundaries fitted on Phase M-1.

        K-selection strategies (columns)
        ─────────────────────────────────
        • Static K=32 / K=64 / K=128  — fixed, no search.
        • Theoretical K  (DP only)    — K* = ⌈C · N^{1/5}⌉, calibrated in Phase 0.
        • Heuristic K                 — independent sweep for each spacing:
            - Geometric heuristic: sweeps K=2..k_heuristic_max on Phase M-1;
              loss = Σ(v - v_disc) + Σ(phi - phi_disc) on a 50/50 val split;
              stops when marginal improvement < k_search_threshold_pct %.
            - DP heuristic:        same sweep but uses DP boundaries on the
              same val split; no cap other than k_heuristic_max.

        Timing (isolated)
        ─────────────────
        Each cell records TWO timings:
          t_spacing_ms — time to compute the discrete v/phi arrays only
                         (geomspace+digitize  OR  DP fitting + apply).
                         This is the "algorithm cost" that differs between rows.
          t_lp_ms      — time to solve the two linprog calls (cont + disc).
                         This is the same oracle cost for both rows and should
                         be approximately equal; reported for completeness.

        Phase pipeline (strict no-leakage)
        ────────────────────────────────────
        • Phase 0 (warm-up): draws ``phase_size`` jobs; calibrates C for the
          Theoretical K column; produces initial boundaries for Phase 1.
        • Phase M (M ≥ 1): draws the NEXT ``phase_size`` jobs (fixed size).
          Boundaries / heuristic K are always fitted on Phase M-1 data only.
          Phase-M data is never seen during boundary fitting.

        Parameters
        ──────────
        df                    : full job DataFrame.
        v_col, phi_col        : valuation and virtual-value column names.
        phase_size            : fixed number of jobs per phase (default 50).
        static_ks             : list of fixed K values (default [32, 64, 128]).
        k_heuristic_max       : ceiling for heuristic K sweep (default 256).
        k_search_threshold_pct: early-stop threshold in % (default 1.0).
        C                     : pre-calibrated constant; if None, Phase-0
                                calibration is run automatically.
        lambda_1, lambda_2    : scalarization weights.
        n_phases              : cap on phases (None = run until data exhausted).
        seed                  : RNG seed for the shuffle.
        verbose               : print per-phase summaries.

        Returns
        ───────
        pd.DataFrame with one row per phase.  Columns follow the pattern:

            {Geo|DP}_{strategy}_Loss_abs
            {Geo|DP}_{strategy}_Loss_pct
            {Geo|DP}_{strategy}_GIR
            {Geo|DP}_{strategy}_K
            {Geo|DP}_{strategy}_t_spacing_ms
            {Geo|DP}_{strategy}_t_lp_ms

        where strategy ∈ {K32, K64, K128, Theo (DP only), Heuristic}.
        Plus: Phase, Batch_Size, Fluid_Upper_Bound_Reward, t_cont_lp_ms.
        """
        if static_ks is None:
            static_ks = [32, 64, 128]

        sep = "=" * 90

        # ── Shuffle once; consume in fixed-size windows ───────────────────────
        rng         = np.random.default_rng(seed=seed)
        shuffle_idx = rng.permutation(len(df))
        df_sh       = df.iloc[shuffle_idx].reset_index(drop=True)

        v_all   = df_sh[v_col].values.astype(float)
        phi_all = df_sh[phi_col].values.astype(float)

        # ── Phase 0: calibrate C on the first batch ───────────────────────────
        warmup_v   = v_all[:phase_size]
        warmup_phi = phi_all[:phase_size]

        if C is None:
            if verbose:
                print(f"\n{sep}")
                print("  Phase-0 calibration (50/50 train/test split) …")
            C = Discretizer.calibrate_C(
                v_continuous   = warmup_v,
                phi_continuous = warmup_phi,
                warmup_n       = phase_size,
                k_search_max   = k_heuristic_max,
                k_search_threshold_pct = k_search_threshold_pct,
                lambda_1 = lambda_1,
                lambda_2 = lambda_2,
                seed     = seed,
            )

        if verbose:
            print(f"\n{sep}")
            print(f"  2×5 DISCRETIZATION MATRIX — phase_size={phase_size}  "
                  f"K_static={static_ks}  K_heuristic_max={k_heuristic_max}")
            print(f"  Theoretical K formula: K* = {C:.4f} · N^(1/5)")
            print(f"  Early-stop threshold: {k_search_threshold_pct}% marginal improvement")
            print(f"{sep}")

        # ── Helper: apply Geometric spacing and time it separately from LP ────
        def _geo_discretize(k: int, train_v: np.ndarray, train_phi: np.ndarray,
                            test_v: np.ndarray, test_phi: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, float]:
            """
            Apply Geometric grid at resolution K to test arrays.
            Edges are derived from the training data range so no leakage occurs.
            Returns (v_disc, phi_disc, t_spacing_ms).
            """
            disc = Discretizer(K_bins=k)
            t0   = time.perf_counter()

            # Build edges from Phase M-1 range; apply to Phase M values.
            # _map_to_bins is an instance method, so we call geometric_grid
            # directly but on the test data (geomspace only needs min/max of
            # the training range, not the test values themselves).
            # We reconstruct the edge computation explicitly to time only the
            # spacing step, not the LP.
            def _geo_edges(ref_vals: np.ndarray, K: int) -> np.ndarray:
                valid = ref_vals[ref_vals > 0]
                if len(valid) == 0:
                    return np.array([1e-4, 1.0])
                phi_min = max(valid.min(), 1e-4)
                phi_max = valid.max()
                return np.geomspace(phi_min, phi_max, K + 1)

            edges_v   = _geo_edges(train_v,   k)
            edges_phi = _geo_edges(train_phi, k)

            # Apply: map each positive test value to its lower-bound bin
            def _apply_geo(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
                result = values.copy().astype(float)
                mask   = values > 0
                if mask.any():
                    idx = np.digitize(values[mask], edges) - 1
                    idx = np.clip(idx, 0, k - 1)
                    result[mask] = edges[idx]
                return result

            v_disc   = _apply_geo(test_v,   edges_v)
            phi_disc = _apply_geo(test_phi, edges_phi)
            t_ms     = (time.perf_counter() - t0) * 1_000
            return v_disc, phi_disc, t_ms

        # ── Helper: apply DP spacing and time it separately from LP ──────────
        def _dp_discretize(k: int, train_v: np.ndarray, train_phi: np.ndarray,
                           test_v: np.ndarray, test_phi: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, float]:
            """
            Fit DP boundaries on Phase M-1 and apply to Phase M.
            Returns (v_disc, phi_disc, t_spacing_ms).
            """
            disc = Discretizer(K_bins=k)
            t0   = time.perf_counter()
            bv   = disc.get_dp_boundaries(train_v)
            bp   = disc.get_dp_boundaries(train_phi)
            v_disc   = disc.apply_boundaries(test_v,   bv)
            phi_disc = disc.apply_boundaries(test_phi, bp)
            t_ms     = (time.perf_counter() - t0) * 1_000
            return v_disc, phi_disc, t_ms

        # ── Helper: run LP on a pre-discretized batch; time the LP only ───────
        def _run_lp(batch_df: pd.DataFrame,
                    v_disc: np.ndarray, phi_disc: np.ndarray,
                    cont_v: np.ndarray, cont_phi: np.ndarray,
                    ) -> tuple[float, float, int, int, float]:
            """
            Solve the discrete LP for one (spacing, K) cell.
            The continuous LP has already been solved outside this helper
            (cont_reward and x_cont are passed in via closure).
            Returns (loss_abs, loss_pct, gir, accepted, t_lp_ms).
            """
            tdf = batch_df.copy()
            tdf["v_disc"]   = v_disc
            tdf["phi_disc"] = phi_disc

            t0 = time.perf_counter()
            x_disc, r_disc_arr = self.simulator.run_allocation(
                tdf, v_col="v_disc", phi_col="phi_disc"
            )
            t_lp_ms = (time.perf_counter() - t0) * 1_000

            disc_reward = float(np.sum(r_disc_arr))
            loss_abs    = cont_reward - disc_reward
            loss_pct    = (loss_abs / cont_reward * 100) if cont_reward > 0 else 0.0
            gir         = int(np.sum((x_cont > 0) & (x_disc == 0)))
            accepted    = int(np.sum(x_disc > 0))
            return loss_abs, loss_pct, gir, accepted, t_lp_ms

        # ── Helper: Geometric heuristic K search on Phase M-1 val split ──────
        def _geo_heuristic_k(train_v: np.ndarray, train_phi: np.ndarray,
                             ) -> tuple[int, float]:
            """
            Sweep K=2..k_heuristic_max on a 50/50 split of Phase M-1 data.
            Loss = Σ(v_val - v_disc) + Σ(phi_val - phi_disc) on the val half.
            Stops when marginal improvement < k_search_threshold_pct %.
            Returns (best_k, t_search_ms).
            """
            t0      = time.perf_counter()
            n       = len(train_v)
            split   = n // 2
            fit_v,  val_v   = train_v[:split],   train_v[split:]
            fit_phi, val_phi = train_phi[:split], train_phi[split:]

            best_k   = 2
            best_err = float("inf")
            prev_err = float("inf")

            for k in range(2, k_heuristic_max + 1):

                def _geo_edges_inner(ref_vals: np.ndarray) -> np.ndarray:
                    valid = ref_vals[ref_vals > 0]
                    if len(valid) == 0:
                        return np.array([1e-4, 1.0])
                    return np.geomspace(max(valid.min(), 1e-4), valid.max(), k + 1)

                def _apply_inner(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
                    res  = values.copy().astype(float)
                    mask = values > 0
                    if mask.any():
                        idx = np.clip(np.digitize(values[mask], edges) - 1, 0, k - 1)
                        res[mask] = edges[idx]
                    return res

                ev  = _geo_edges_inner(fit_v)
                ep  = _geo_edges_inner(fit_phi)
                vd  = _apply_inner(val_v,   ev)
                pd_ = _apply_inner(val_phi, ep)

                err = (float(np.sum(val_v[val_v > 0]   - vd[val_v > 0]))   * lambda_1
                     + float(np.sum(val_phi[val_phi > 0] - pd_[val_phi > 0])) * lambda_2)
                err = max(err, 0.0)   # guard against tiny negatives from clipping

                if err < best_err:
                    best_err = err
                    best_k   = k

                if prev_err < float("inf") and prev_err > 0:
                    improvement = (prev_err - err) / prev_err * 100
                    if improvement < k_search_threshold_pct:
                        break
                prev_err = err

            return best_k, (time.perf_counter() - t0) * 1_000

        # ── Helper: DP heuristic K search on Phase M-1 val split ─────────────
        def _dp_heuristic_k(train_v: np.ndarray, train_phi: np.ndarray,
                            ) -> tuple[int, float]:
            """
            Sweep K=2..k_heuristic_max on a 50/50 split of Phase M-1 data.
            Loss = Σ(v_val - v_disc) + Σ(phi_val - phi_disc) using DP boundaries.
            Stops when marginal improvement < k_search_threshold_pct %.
            No cap other than k_heuristic_max.
            Returns (best_k, t_search_ms).
            """
            t0      = time.perf_counter()
            n       = len(train_v)
            split   = n // 2
            fit_v,  val_v   = train_v[:split],   train_v[split:]
            fit_phi, val_phi = train_phi[:split], train_phi[split:]

            best_k   = 2
            best_err = float("inf")
            prev_err = float("inf")

            for k in range(2, k_heuristic_max + 1):
                disc_k = Discretizer(K_bins=k)
                bv     = disc_k.get_dp_boundaries(fit_v)
                bp     = disc_k.get_dp_boundaries(fit_phi)
                vd     = disc_k.apply_boundaries(val_v,   bv)
                pd_    = disc_k.apply_boundaries(val_phi, bp)

                err = (float(np.sum(val_v[val_v > 0]   - vd[val_v > 0]))   * lambda_1
                     + float(np.sum(val_phi[val_phi > 0] - pd_[val_phi > 0])) * lambda_2)
                err = max(err, 0.0)

                if err < best_err:
                    best_err = err
                    best_k   = k

                if prev_err < float("inf") and prev_err > 0:
                    improvement = (prev_err - err) / prev_err * 100
                    if improvement < k_search_threshold_pct:
                        break
                prev_err = err

            return best_k, (time.perf_counter() - t0) * 1_000

        # ── Main phase loop ───────────────────────────────────────────────────
        # Phase 0 training data = first batch (warm-up); Phase 1 test = second
        # batch, etc.  We keep a rolling pointer so every phase has the same
        # fixed size and never re-uses data.
        train_v_prev   = v_all[:phase_size]
        train_phi_prev = phi_all[:phase_size]
        offset         = phase_size    # Phase 1 starts here

        rows   = []
        phase  = 0

        while True:
            phase += 1
            if n_phases is not None and phase > n_phases:
                break
            if offset >= len(df_sh):
                if verbose:
                    print(f"\n  Phase {phase}: no data remaining — stopping.")
                break

            batch_df = df_sh.iloc[offset: offset + phase_size].copy()
            actual_n = len(batch_df)
            offset  += actual_n

            if actual_n == 0:
                break

            batch_v   = batch_df[v_col].values.astype(float)
            batch_phi = batch_df[phi_col].values.astype(float)

            # ── Continuous LP (Fluid Upper Bound) — timed separately ──────────
            batch_df["v_cont"]   = batch_v
            batch_df["phi_cont"] = batch_phi

            t0_cont  = time.perf_counter()
            x_cont, r_cont_arr = self.simulator.run_allocation(
                batch_df, v_col="v_cont", phi_col="phi_cont"
            )
            t_cont_lp_ms = (time.perf_counter() - t0_cont) * 1_000
            cont_reward  = float(np.sum(r_cont_arr))

            # ── Heuristic K searches on Phase M-1 (independent per spacing) ──
            k_geo_h, t_geo_search_ms = _geo_heuristic_k(train_v_prev, train_phi_prev)
            k_dp_h,  t_dp_search_ms  = _dp_heuristic_k( train_v_prev, train_phi_prev)

            # Theoretical K (DP column only) — no cap beyond k_heuristic_max
            k_theo = min(
                Discretizer._theoretical_k(len(train_v_prev), C=C),
                k_heuristic_max,
            )

            # ── Evaluate every cell ───────────────────────────────────────────
            row: dict = {
                "Phase":                      phase,
                "Batch_Size":                 actual_n,
                "Fluid_Upper_Bound_Reward":   cont_reward,
                "t_cont_lp_ms":               t_cont_lp_ms,
                # Heuristic K chosen
                "K_Geo_Heuristic":            k_geo_h,
                "K_DP_Heuristic":             k_dp_h,
                "K_Theo":                     k_theo,
                # Search times (pure spacing search cost, no LP)
                "Geo_Heuristic_t_search_ms":  t_geo_search_ms,
                "DP_Heuristic_t_search_ms":   t_dp_search_ms,
            }

            # Iterate over all (spacing, K) combinations
            for k_label, k_val, spacing_fn, spacing_name in (
                # Static columns — Geometric
                *[(f"K{k}", k, _geo_discretize, "Geo") for k in static_ks],
                # Static columns — DP
                *[(f"K{k}", k, _dp_discretize,  "DP")  for k in static_ks],
                # Theoretical K — DP only
                ("Theo", k_theo, _dp_discretize, "DP"),
                # Heuristic K — Geometric
                ("Heuristic", k_geo_h, _geo_discretize, "Geo"),
                # Heuristic K — DP
                ("Heuristic", k_dp_h,  _dp_discretize,  "DP"),
            ):
                prefix = f"{spacing_name}_{k_label}"

                v_disc, phi_disc, t_sp = spacing_fn(
                    k_val, train_v_prev, train_phi_prev, batch_v, batch_phi
                )
                loss_abs, loss_pct, gir, acc, t_lp = _run_lp(
                    batch_df, v_disc, phi_disc, batch_v, batch_phi
                )

                row[f"{prefix}_Loss_abs"]       = loss_abs
                row[f"{prefix}_Loss_pct"]       = loss_pct
                row[f"{prefix}_GIR"]            = gir
                row[f"{prefix}_Accepted"]        = acc
                row[f"{prefix}_t_spacing_ms"]   = t_sp
                row[f"{prefix}_t_lp_ms"]        = t_lp

            rows.append(row)

            if verbose:
                # Compact per-phase summary
                print(f"\n  Phase {phase:>4d}  N={actual_n}  "
                      f"Fluid UB={cont_reward:.4f}  t_cont_LP={t_cont_lp_ms:.1f}ms")
                print(f"  {'Strategy':<26s}  {'Loss%':>7}  {'GIR':>5}  "
                      f"{'t_spacing':>10}  {'t_lp':>8}  {'K':>5}")
                print(f"  {'-'*70}")
                for k in static_ks:
                    for sp in ("Geo", "DP"):
                        pf  = f"{sp}_K{k}"
                        lbl = f"{sp} Static K={k}"
                        print(f"  {lbl:<26s}  "
                              f"{row[f'{pf}_Loss_pct']:>6.2f}%  "
                              f"{row[f'{pf}_GIR']:>5d}  "
                              f"{row[f'{pf}_t_spacing_ms']:>9.3f}ms  "
                              f"{row[f'{pf}_t_lp_ms']:>7.1f}ms  "
                              f"{k:>5d}")
                # Theo (DP only)
                pf = "DP_Theo"
                print(f"  {'DP  Theoretical K':<26s}  "
                      f"{row[f'{pf}_Loss_pct']:>6.2f}%  "
                      f"{row[f'{pf}_GIR']:>5d}  "
                      f"{row[f'{pf}_t_spacing_ms']:>9.3f}ms  "
                      f"{row[f'{pf}_t_lp_ms']:>7.1f}ms  "
                      f"{k_theo:>5d}")
                # Heuristics
                for sp, k_h, t_srch in (
                    ("Geo", k_geo_h, t_geo_search_ms),
                    ("DP",  k_dp_h,  t_dp_search_ms),
                ):
                    pf  = f"{sp}_Heuristic"
                    lbl = f"{sp}  Heuristic K={k_h}"
                    print(f"  {lbl:<26s}  "
                          f"{row[f'{pf}_Loss_pct']:>6.2f}%  "
                          f"{row[f'{pf}_GIR']:>5d}  "
                          f"{row[f'{pf}_t_spacing_ms']:>9.3f}ms  "
                          f"{row[f'{pf}_t_lp_ms']:>7.1f}ms  "
                          f"{'srch='+str(round(t_srch,1))+'ms':>9s}")

            # ── Advance training window and double batch size ─────────────────
            train_v_prev   = batch_v
            train_phi_prev = batch_phi
            phase_size    *= 2

        # ── Aggregate summary ─────────────────────────────────────────────────
        results = pd.DataFrame(rows)

        if verbose and len(results) > 0:
            total_ub = results["Fluid_Upper_Bound_Reward"].sum()
            print(f"\n{sep}")
            print(f"  AGGREGATE ACROSS {len(results)} PHASES  "
                  f"(total Fluid UB reward = {total_ub:.4f})")
            print(f"  {'Strategy':<30s}  {'TotalLoss':>12}  {'OverallLoss%':>13}  "
                  f"{'MeanSpacing(ms)':>16}  {'MeanLP(ms)':>11}")
            print(f"  {'-'*90}")

            report_cols = (
                [(f"Geo Static K={k}", f"Geo_K{k}") for k in static_ks]
                + [(f"DP  Static K={k}", f"DP_K{k}")  for k in static_ks]
                + [("DP  Theoretical K",  "DP_Theo")]
                + [("Geo Heuristic K",    "Geo_Heuristic")]
                + [("DP  Heuristic K",    "DP_Heuristic")]
            )
            for label, pf in report_cols:
                lc = f"{pf}_Loss_abs"
                sc = f"{pf}_t_spacing_ms"
                lpc = f"{pf}_t_lp_ms"
                if lc not in results.columns:
                    continue
                total_loss   = results[lc].sum()
                overall_pct  = total_loss / total_ub * 100 if total_ub > 0 else 0.0
                mean_spacing = results[sc].mean()
                mean_lp      = results[lpc].mean()
                print(f"  {label:<30s}  {total_loss:>12.4f}  {overall_pct:>12.2f}%  "
                      f"{mean_spacing:>15.3f}ms  {mean_lp:>10.1f}ms")
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
    phi_v approximates the Myerson virtual value with ~20% jobs having r_j < 0
    (Grid-Induced Rejections under equal-weight scalarization).
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
    K_BINS  = 32                  # Number of discrete bins
    RHO     = 0.6                 # Load factor: budget = rho · Σ A_j · D_j
    LAMBDAS = (1/3, 1/3, 1/3)    # Equal-weight scalarization

    # ── Dataset ──────────────────────────────────────────────────────────────
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "batch_may2019_30k.csv"
    )

    print("=" * 78)
    print("  Discretization Grid Loss via StaticCloudSimulator (Fluid Upper Bound)")
    print(f"  K={K_BINS}  |  rho={RHO}  |  lambdas={LAMBDAS}")
    print(f"  Fluid budget: rho · Σ_j A_j · D_j  (adaptive per dataset)")
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
        # Count jobs with r_j >= 0 under equal-weight scalarization
        r_j = (1/3) * df["priority"] * df[phi_col] + (1/3) * (df[phi_col] - df["c_elec"]) - (1/3) * df["c_carbon"]
        valid_jobs = int((r_j >= 0).sum())
        print(f"  Scalarized reward valid jobs (r_j >= 0): {valid_jobs:,}")

    # ── 1. Full comparison at fixed K ────────────────────────────────────────
    sd = SimulatorDiscretizer(K_bins=K_BINS, lambdas=LAMBDAS, rho=RHO)
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
        "Phase", "Batch_Size", "Fluid_Upper_Bound_Reward",
        "K_Static",       "Loss_Static",       "Loss_Static_pct", "GIR_Static",
        "K_Theo_Bound",   "Loss_Theo_Bound",   "Loss_Theo_Bound_pct", "GIR_Theo_Bound",
        "K_Heuristic_DP", "Loss_Heuristic_DP", "Loss_Heuristic_DP_pct", "GIR_Heuristic_DP",
        "Time_Heuristic_DP_sec",
    ]
    print(phase_results[cols_to_show].to_string(index=False, float_format="{:.3f}".format))

    print("\nDone.")