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

    def calculate_discretization_error(self, continuous_rewards: np.ndarray, discrete_rewards: np.ndarray) -> float:
        """
        Calculates the Total Revenue Loss strictly due to discretization.
        """
        return np.sum(continuous_rewards) - np.sum(discrete_rewards)


def _run_local_test() -> None:
    """
    Local smoke test for discretization quality on the processed static batch CSV.
    """
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "data" / "processed" / "static_batch_may2019.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Build it first with scripts/build_processed_dataset.py"
        )

    df = pd.read_csv(dataset_path)
    if "phi_v" not in df.columns:
        raise ValueError("Column 'phi_v' not found in processed dataset.")

    sample_n = min(2000, len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    phi = sample_df["phi_v"].to_numpy(dtype=float)

    K = 16
    discretizer = Discretizer(K_bins=K)

    phi_valid = discretizer._enforce_bounds(phi)
    if len(phi_valid) == 0:
        raise ValueError("No positive phi_v values available in sampled data.")

    uniform_disc = discretizer.uniform_grid(phi_valid)
    geometric_disc = discretizer.geometric_grid(phi_valid)

    uniform_loss = discretizer.calculate_discretization_error(phi_valid, uniform_disc)
    geometric_loss = discretizer.calculate_discretization_error(phi_valid, geometric_disc)

    def summarize(name: str, disc_values: np.ndarray, loss: float) -> None:
        used_bins = len(np.unique(disc_values))
        non_monotone_count = int(np.sum(disc_values > phi_valid))
        print(f"\n{name} results")
        print(f"- used bins: {used_bins}/{K}")
        print(f"- total loss: {loss:.8f}")
        print(f"- avg loss per job: {loss / len(phi_valid):.8f}")
        print(f"- values above original (should be ~0 for lower-bound discretization): {non_monotone_count}")
        print(
            f"- discretized min/median/max: "
            f"{np.min(disc_values):.8f} / {np.median(disc_values):.8f} / {np.max(disc_values):.8f}"
        )

    print("=== Discretizer Local Test ===")
    print(f"dataset: {dataset_path}")
    print(f"rows in dataset: {len(df)}")
    print(f"sample size: {sample_n}")
    print(f"positive phi_v in sample: {len(phi_valid)}")

    summarize("Uniform grid", uniform_disc, uniform_loss)
    summarize("Geometric grid", geometric_disc, geometric_loss)


if __name__ == "__main__":
    _run_local_test()