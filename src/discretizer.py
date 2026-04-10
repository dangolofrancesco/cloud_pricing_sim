import numpy as np
import pandas as pd
from pathlib import Path

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