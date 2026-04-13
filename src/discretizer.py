import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

class Discretizer:
    """
    Evaluates different discretization algorithms to convert continuous virtual 
    values into a discrete 'menu of prices'.
    """
    
    def __init__(self, K_bins: int):
        self.K = K_bins
        
    def _enforce_bounds(self, phi_array: np.ndarray) -> np.ndarray:
        """Filters out negative virtual values (rejected by the mechanism)."""
        return phi_array[phi_array > 0]

    def uniform_grid(self, phi_array: np.ndarray) -> np.ndarray:
        """
        Baseline 1: Uniform/Arithmetic Grid.
        Slices the domain [min, max] into K equally sized segments.
        Returns the lower bound (L_k) assigned to each value in the array.
        """
        phi_valid = self._enforce_bounds(phi_array)
        if len(phi_valid) == 0:
            return np.array([])
            
        phi_min, phi_max = np.min(phi_valid), np.max(phi_valid)
        
        edges = np.linspace(phi_min, phi_max, self.K + 1)
        
        bin_indices = np.digitize(phi_valid, edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.K - 1)
        
        return edges[bin_indices]

    def geometric_grid(self, phi_array: np.ndarray) -> np.ndarray:
        """
        Baseline 2: Geometric/Power of 2 Grid.
        Multiplicative scaling. Bins are narrow at the bottom and wide at the top.
        """
        phi_valid = self._enforce_bounds(phi_array)
        if len(phi_valid) == 0:
            return np.array([])
            
        # Geometric grids require strictly positive bounds > 0
        phi_min = np.maximum(np.min(phi_valid), 1e-4) 
        phi_max = np.max(phi_valid)
        
        edges = np.geomspace(phi_min, phi_max, self.K + 1)
        
        bin_indices = np.digitize(phi_valid, edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.K - 1)
        
        return edges[bin_indices]

    def dp_optimal_grid(self, phi_array: np.ndarray) -> np.ndarray:
        """
        Baseline 3: Dynamic Programming Optimal Grid.
        Finds the exact K bin boundaries that minimize Revenue Loss for the given dataset.
        Complexity: O(K * N^2) - Optimized with Numpy Vectorization
        """
        phi_valid = self._enforce_bounds(phi_array)
        if len(phi_valid) == 0:
            return np.array([])

        # 1. Sort the virtual values to create a monotonic sequence
        V = np.sort(phi_valid)
        N = len(V)

        # If we have fewer jobs than available bins, everyone gets their exact continuous price
        if N <= self.K:
            return phi_valid

        # 2. Precompute the Cost Matrix (Revenue Loss for any segment V[i...j])
        # cost_matrix[i, j] = sum(V[i...j]) - (j - i + 1) * V[i]
        cost_matrix = np.zeros((N, N))

        # We optimize this by precomputing cumulative sums
        cum_V = np.insert(np.cumsum(V), 0, 0) # insert 0 for easy indexing

        for i in range(N):
            # For a fixed 'i', calculate cost for all 'j >= i' simultaneously (Vectorized)
            # Sum of V[i...j] is cum_V[j+1] - cum_V[i]
            # Number of elements is (j - i + 1)
            j_indices = np.arange(i, N)
            segment_sums = cum_V[j_indices + 1] - cum_V[i]
            bin_revenues = (j_indices - i + 1) * V[i]
            cost_matrix[i, i:] = segment_sums - bin_revenues

        # 3. Initialize DP Tables
        # dp[k][j] = min loss to cover the first 'j' elements using exactly 'k' bins
        dp = np.full((self.K + 1, N), np.inf)

        # tracker[k][j] = the starting index 'i' of the optimal last bin
        tracker = np.zeros((self.K + 1, N), dtype=int)

        # Base case: k=1 (Using exactly 1 bin for the first j elements)
        for j in range(N):
            dp[1, j] = cost_matrix[0, j]
            tracker[1, j] = 0

        # 4. Fill DP Table (The Bellman Equation)
        for k in range(2, self.K + 1):
            for j in range(k - 1, N):
                # We want to find the split point 'i' that minimizes:
                # dp[k-1][i-1] (cost of previous bins) + cost_matrix[i][j] (cost of current bin)

                # Possible split points: k-1 <= i <= j
                i_candidates = np.arange(k - 1, j + 1)

                # Calculate total costs for all candidates instantly (Vectorized)
                prev_costs = dp[k - 1, i_candidates - 1]
                curr_costs = cost_matrix[i_candidates, j]
                total_costs = prev_costs + curr_costs

                # Find the index that gives the absolute minimum loss
                best_idx = np.argmin(total_costs)
                dp[k, j] = total_costs[best_idx]
                tracker[k, j] = i_candidates[best_idx]

        # 5. Backtrack to find the optimal lower bounds (L_k)
        bin_starts = []
        curr_j = N - 1
        for k in range(self.K, 0, -1):
            start_i = tracker[k, curr_j]
            bin_starts.append(V[start_i])  # The price of the bin is the value at the start index
            curr_j = start_i - 1

        bin_starts.reverse() # We collected them backwards, so reverse them

        # 6. Map the original UNSORTED array to these optimal bins
        edges = np.array(bin_starts)
        edges = np.append(edges, np.inf) # Add infinity to catch the top edge

        # Use digitize to drop jobs into the exact optimal bins
        bin_indices = np.digitize(phi_valid, edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.K - 1)

        return edges[bin_indices]

    def calculate_financial_loss(
        self,
        v_continuous: np.ndarray,
        v_discrete: np.ndarray,
        return_details: bool = False,
    ):
        """
        Metric 1: Pure Financial Revenue Loss (The "Accounting" Metric)
        Calculates exactly how much cash the discretization strategy lost the business.
        Formula: Sum( V_j - \hat{V}_j )

        If return_details=True, also returns aggregate sums for both valuation
        arrays as (loss, sum_continuous, sum_discrete).
        """
        # Ensure we don't have negative losses (Individual Rationality check)
        # The discrete price should never exceed the continuous willingness to pay.
        loss_array = v_continuous - v_discrete

        loss = float(np.sum(loss_array))
        if not return_details:
            return loss

        sum_continuous = float(np.sum(v_continuous))
        sum_discrete = float(np.sum(v_discrete))
        return loss, sum_continuous, sum_discrete

    def calculate_objective_loss(self, phi_continuous: np.ndarray, phi_discrete: np.ndarray, 
                                 q_j: np.ndarray, lambda_2: float = 1.0) -> float:
        """
        Metric 2: The Oracle's Objective Loss (The "Algorithmic" Metric)
        Calculates how much the discretization hurt the DLENT algorithm's internal objective.
        Formula: Sum( lambda_2 * q_j * ( \Phi_j - \hat{\Phi}_j ) )
        """
        # Element-wise multiplication of the algorithmic loss
        algorithmic_loss_array = lambda_2 * q_j * (phi_continuous - phi_discrete)
        
        return np.sum(algorithmic_loss_array)

    # ------------------------------------------------------------------
    # Convergence & scaling analysis (static methods — no instance state)
    # ------------------------------------------------------------------

    _METHOD_MAP = {
        "Uniform":    lambda d: d.uniform_grid,
        "Geometric":  lambda d: d.geometric_grid,
        "DP Optimal": lambda d: d.dp_optimal_grid,
    }

    @staticmethod
    def _theoretical_k(w: int) -> int:
        """
        Theoretical upper bound on K for a phase with W jobs:

            K_m = ceil( (W^{(m-1)} / ln W^{(m-1)})^{1/5} )

        Enforces K >= 2 so the DP always has at least two bins.
        """
        return max(2, int(np.ceil((w / np.log(w)) ** 0.2)))

    @staticmethod
    def run_convergence_test(
        method_name: str,
        v_values: np.ndarray,
        phi_values: np.ndarray,
        q_values: np.ndarray,
        max_financial_continuous: float,
        max_objective_continuous: float,
        lambda_2: float = 1.0,
        start_k: int = 2,
        step_k: int = 2,
        max_k: int = 500,
        target_threshold_pct: float = 0.1,
    ) -> pd.DataFrame:
        """
        Sweeps K bins from start_k to max_k, evaluating financial and objective loss.
        Stops early when the marginal improvement in financial loss drops below
        target_threshold_pct between two consecutive steps.

        Note: for 'DP Optimal', complexity is O(K * N^2). Keep max_k small (<=50)
        for datasets with N > 1000 to maintain reasonable runtimes.
        """
        if method_name not in Discretizer._METHOD_MAP:
            raise ValueError(
                f"Unknown method '{method_name}'. Choose from: {list(Discretizer._METHOD_MAP.keys())}"
            )

        rows = []
        prev_financial_loss = None

        for k in range(start_k, max_k + 1, step_k):
            disc = Discretizer(K_bins=k)
            grid_func = Discretizer._METHOD_MAP[method_name](disc)

            start = time.perf_counter()
            v_discrete = grid_func(v_values)
            phi_discrete = grid_func(phi_values)
            elapsed = time.perf_counter() - start

            financial_loss = float(disc.calculate_financial_loss(v_values, v_discrete))
            objective_loss = float(
                disc.calculate_objective_loss(phi_values, phi_discrete, q_values, lambda_2=lambda_2)
            )
            financial_loss_pct = (
                (financial_loss / max_financial_continuous) * 100
                if max_financial_continuous else np.nan
            )
            objective_loss_pct = (
                (objective_loss / max_objective_continuous) * 100
                if max_objective_continuous else np.nan
            )
            improvement_pct = (
                np.nan
                if prev_financial_loss is None
                else ((prev_financial_loss - financial_loss) / prev_financial_loss) * 100
                if prev_financial_loss != 0
                else 0.0
            )

            rows.append({
                "Method": method_name,
                "K": k,
                "Financial_Loss": financial_loss,
                "Financial_Loss_pct": financial_loss_pct,
                "Objective_Loss": objective_loss,
                "Objective_Loss_pct": objective_loss_pct,
                "Execution_Time_sec": elapsed,
                "Improvement_pct": improvement_pct,
            })

            if (
                prev_financial_loss is not None
                and not np.isnan(improvement_pct)
                and improvement_pct < target_threshold_pct
            ):
                break
            prev_financial_loss = financial_loss

        return pd.DataFrame(rows)

    @staticmethod
    def run_dp_scaling_test(
        v_continuous: np.ndarray,
        initial_batch_size: int = 500,
        n_phases: int = None,
        K_fixed: int = 16,
        k_search_max: int = 30,
        k_search_threshold_pct: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compares three DP grid strategies across phases where the batch doubles each round.

        Phase batch sizes: initial_batch_size * 2^(phase-1).
        Samples with replacement when batch_size > len(v_continuous).

        Models
        ------
        1. Fixed K        : K = K_fixed every phase.
        2. Theoretical K  : K_m = ceil( (W^{(m-1)} / ln W^{(m-1)})^{1/5} )
                            where W^{(m-1)} is the previous phase batch size.
        3. Optimal K search: sweep K from 2..k_search_max, run DP at each K,
                            pick K* = argmin(error). The reported time includes
                            the entire sweep (cost of finding the best K).

        Returns a DataFrame with per-phase metrics and prints a full summary.
        """
        # Shuffle once, then consume sequentially so phases are non-overlapping.
        rng = np.random.default_rng(seed=42)
        shuffled = rng.permutation(v_continuous)

        rows = []
        totals = {m: {"error": 0.0, "time": 0.0}
                  for m in ("fixed", "theoretical", "optimal")}

        phase_label = f"up to {n_phases}" if n_phases is not None else "until data exhausted"
        sep = "=" * 76
        print(f"\n{sep}")
        print(f"  DP Scaling Test — three strategies")
        print(f"  Fixed K={K_fixed}  |  Theoretical K (formula)  |  "
              f"Optimal K search (2..{k_search_max}, stop if improvement < {k_search_threshold_pct}%)")
        print(f"  Phases: {phase_label}  |  Initial batch: {initial_batch_size:,} jobs  "
              f"|  Total available: {len(shuffled):,}")
        print(f"{sep}")

        offset = 0
        prev_batch_size = initial_batch_size
        batch_size = initial_batch_size
        phase = 0

        while True:
            phase += 1
            if n_phases is not None and phase > n_phases:
                break
            remaining = len(shuffled) - offset
            if remaining == 0:
                print(f"\n  Phase {phase}: no data remaining — stopping.")
                break

            # Use all remaining jobs if fewer than the full batch size
            batch = shuffled[offset: offset + batch_size]
            actual_n = len(batch)
            offset += actual_n
            if actual_n < batch_size:
                print(f"\n  Phase {phase}: only {actual_n:,} jobs remain "
                      f"(expected {batch_size:,}) — running with reduced batch.")

            # Total continuous revenue of this batch (denominator for % error)
            batch_total = float(np.sum(batch))

            # ── 1. Fixed-K DP ─────────────────────────────────────────────────
            disc = Discretizer(K_bins=K_fixed)
            t0 = time.perf_counter()
            v_disc = disc.dp_optimal_grid(batch)
            time_fixed = time.perf_counter() - t0
            error_fixed = float(disc.calculate_financial_loss(batch, v_disc))

            # ── 2. Theoretical-K DP ───────────────────────────────────────────
            K_theoretical = Discretizer._theoretical_k(prev_batch_size)
            disc = Discretizer(K_bins=K_theoretical)
            t0 = time.perf_counter()
            v_disc = disc.dp_optimal_grid(batch)
            time_theoretical = time.perf_counter() - t0
            error_theoretical = float(disc.calculate_financial_loss(batch, v_disc))

            # ── 3. Optimal K search ───────────────────────────────────────────
            # Sweep K=2..k_search_max. Stop early when the marginal improvement
            # drops below k_search_threshold_pct to find the "knee" of the curve.
            best_k_star = 2
            best_error_optimal = float("inf")
            prev_err_k = float("inf")

            t0_search = time.perf_counter()
            for k in range(2, k_search_max + 1):
                disc_k = Discretizer(K_bins=k)
                v_disc_k = disc_k.dp_optimal_grid(batch)
                err_k = float(disc_k.calculate_financial_loss(batch, v_disc_k))

                if err_k < best_error_optimal:
                    best_error_optimal = err_k
                    best_k_star = k

                # Early stop: improvement from previous K is below threshold
                if prev_err_k < float("inf") and prev_err_k > 0:
                    improvement_pct = (prev_err_k - err_k) / prev_err_k * 100
                    if improvement_pct < k_search_threshold_pct:
                        break

                prev_err_k = err_k
            time_optimal_search = time.perf_counter() - t0_search

            # ── % of batch revenue lost (always in [0, 100]) ──────────────────
            error_fixed_pct       = (error_fixed       / batch_total) * 100
            error_theoretical_pct = (error_theoretical / batch_total) * 100
            error_optimal_pct     = (best_error_optimal / batch_total) * 100

            # ── Accumulate totals ─────────────────────────────────────────────
            totals["fixed"]["error"]       += error_fixed
            totals["fixed"]["time"]        += time_fixed
            totals["theoretical"]["error"] += error_theoretical
            totals["theoretical"]["time"]  += time_theoretical
            totals["optimal"]["error"]     += best_error_optimal
            totals["optimal"]["time"]      += time_optimal_search

            rows.append({
                "Phase":                   phase,
                "Batch_Size":              actual_n,
                "Batch_Total":             batch_total,
                "K_Fixed":                 K_fixed,
                "Error_Fixed":             error_fixed,
                "Error_Fixed_pct":         error_fixed_pct,
                "Time_Fixed_sec":          time_fixed,
                "K_Theoretical":           K_theoretical,
                "Error_Theoretical":       error_theoretical,
                "Error_Theoretical_pct":   error_theoretical_pct,
                "Time_Theoretical_sec":    time_theoretical,
                "K_Star":                  best_k_star,
                "Error_Optimal":           best_error_optimal,
                "Error_Optimal_pct":       error_optimal_pct,
                "Time_Search_sec":         time_optimal_search,
            })

            print(f"\n  Phase {phase}  |  N={actual_n:,}  |  Batch revenue={batch_total:.2f}")
            print(f"    Fixed        (K={K_fixed:3d}):  "
                  f"Error={error_fixed:10.4f} ({error_fixed_pct:5.2f}% of batch)"
                  f"  |  Time={time_fixed:.4f}s")
            print(f"    Theoretical  (K={K_theoretical:3d}):  "
                  f"Error={error_theoretical:10.4f} ({error_theoretical_pct:5.2f}% of batch)"
                  f"  |  Time={time_theoretical:.4f}s")
            print(f"    Optimal K*   (K={best_k_star:3d}):  "
                  f"Error={best_error_optimal:10.4f} ({error_optimal_pct:5.2f}% of batch)"
                  f"  |  Search time={time_optimal_search:.4f}s "
                  f"(converged at K={best_k_star}, max={k_search_max})")

            prev_batch_size = actual_n
            batch_size *= 2

        # ── Summary ───────────────────────────────────────────────────────────
        n_ran = len(rows)
        print(f"\n{sep}")
        print(f"  TOTALS ACROSS {n_ran} PHASES")
        print(f"{sep}")
        for label, key in [
            (f"Fixed       K={K_fixed}", "fixed"),
            ("Theoretical K (formula)", "theoretical"),
            (f"Optimal K*  (search 2..{k_search_max})", "optimal"),
        ]:
            print(f"  {label}:  "
                  f"Total Error={totals[key]['error']:12.4f}  |  "
                  f"Total Time={totals[key]['time']:.4f}s")
        print(f"{sep}\n")

        return pd.DataFrame(rows)


# --- Local Test Execution Block ---
if __name__ == "__main__":
    import os
    import time

    # 1. Define paths assuming the script is run from the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "static_batch_may2019.csv")

    print("=== Discretizer Local Test: Two-Metric Loss Evaluation ===")
    print(f"Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print("ERROR: Processed dataset not found. Please run build_processed_dataset.py first.")
    else:
        # 2. Load dataset and enforce Individual Rationality (Mechanism Rejection)
        df = pd.read_csv(data_path)
        valid_df = df[df['phi_v'] > 0].copy()
        
        print(f"Total jobs requested: {len(df)}")
        print(f"Jobs accepted by Mechanism (phi_v > 0): {len(valid_df)}\n")

        # 3. Extract continuous arrays and parameters
        v_continuous = valid_df['v'].values
        phi_continuous = valid_df['phi_v'].values
        q_j = valid_df['q_j'].values
        
        # Oracle Weighting Parameter (Lambda_2)
        # Represents the strategic weight the provider places on Profit vs. Sustainability
        LAMBDA_2 = 1.0  
        K_BINS = 16
        
        discretizer = Discretizer(K_bins=K_BINS)

        # 4. Define a helper function to evaluate and print metrics for any grid method
        def evaluate_grid(name: str, grid_func):
            start_time = time.perf_counter()
            
            # Discretize both spaces independently as per the theoretical mapping
            phi_discrete = grid_func(phi_continuous)
            v_discrete = grid_func(v_continuous)
            
            execution_time = time.perf_counter() - start_time
            
            # Calculate the two distinct metrics
            financial_loss, v_cont_sum, v_disc_sum = discretizer.calculate_financial_loss(
                v_continuous,
                v_discrete,
                return_details=True,
            )
            objective_loss = discretizer.calculate_objective_loss(phi_continuous, phi_discrete, q_j, lambda_2=LAMBDA_2)
            
            print(f"--- {name} Grid (K={K_BINS}) ---")
            print(f"  Total Continuous Valuation : ${v_cont_sum:.2f}")
            print(f"  Total Discrete Valuation   : ${v_disc_sum:.2f}")
            print(f"  Pure Financial Cash Loss : ${financial_loss:.2f}")
            print(f"  Oracle's Objective Loss  : {objective_loss:.4f}")
            print(f"  Execution Time           : {execution_time:.4f} seconds\n")

        # 5. Run the evaluations
        evaluate_grid("Uniform (Arithmetic)", discretizer.uniform_grid)
        evaluate_grid("Geometric (Multiplicative)", discretizer.geometric_grid)
        evaluate_grid("DP Optimal", discretizer.dp_optimal_grid)

        print("Test completed successfully!")