import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.discretizer import Discretizer

_METHOD_MAP = {
    "Uniform": lambda d: d.uniform_grid,
    "Geometric": lambda d: d.geometric_grid,
    "DP Optimal": lambda d: d.dp_optimal_grid,
}


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
    if method_name not in _METHOD_MAP:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: {list(_METHOD_MAP.keys())}")

    rows = []
    prev_financial_loss = None

    for k in range(start_k, max_k + 1, step_k):
        discretizer = Discretizer(K_bins=k)
        grid_func = _METHOD_MAP[method_name](discretizer)

        start = time.perf_counter()
        v_discrete = grid_func(v_values)
        phi_discrete = grid_func(phi_values)
        elapsed = time.perf_counter() - start

        financial_loss = float(discretizer.calculate_financial_loss(v_values, v_discrete))
        objective_loss = float(
            discretizer.calculate_objective_loss(phi_values, phi_discrete, q_values, lambda_2=lambda_2)
        )
        financial_loss_pct = (
            (financial_loss / max_financial_continuous) * 100 if max_financial_continuous else np.nan
        )
        objective_loss_pct = (
            (objective_loss / max_objective_continuous) * 100 if max_objective_continuous else np.nan
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


# --- Local Test Execution Block ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "static_batch_may2019.csv")

    print("=== Convergence Analysis: Three-Method Comparison ===")
    print(f"Loading dataset from: {data_path}\n")

    if not os.path.exists(data_path):
        print("ERROR: Processed dataset not found. Please run build_processed_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    valid_df = df[df['phi_v'] > 0].copy()

    v_continuous = valid_df['v'].values
    phi_continuous = valid_df['phi_v'].values
    q_j = valid_df['q_j'].values
    LAMBDA_2 = 1.0

    max_financial = float(np.sum(v_continuous))
    max_objective = float(np.sum(LAMBDA_2 * q_j * phi_continuous))

    print(f"Total jobs requested     : {len(df)}")
    print(f"Jobs accepted (phi_v > 0): {len(valid_df)}")
    print(f"Max Continuous Revenue   : ${max_financial:.2f}")
    print(f"Max Continuous Objective : {max_objective:.4f}\n")

    common_kwargs = dict(
        v_values=v_continuous,
        phi_values=phi_continuous,
        q_values=q_j,
        max_financial_continuous=max_financial,
        max_objective_continuous=max_objective,
        lambda_2=LAMBDA_2,
        start_k=2,
        step_k=2,
        target_threshold_pct=0.1,
    )

    print("Running Uniform convergence test...")
    df_uniform = run_convergence_test("Uniform", max_k=500, **common_kwargs)

    print("Running Geometric convergence test...")
    df_geometric = run_convergence_test("Geometric", max_k=500, **common_kwargs)

    # DP is O(K * N^2) — cap K to keep runtime manageable for N~1400
    print("Running DP Optimal convergence test (max_k=30)...")
    df_dp = run_convergence_test("DP Optimal", max_k=30, **common_kwargs)

    df_all = pd.concat([df_uniform, df_geometric, df_dp], ignore_index=True)

    # --- Summary at convergence point per method ---
    print("\n=== Convergence Summary (last K reached per method) ===")
    summary = df_all.groupby("Method").last()[
        ["K", "Financial_Loss", "Financial_Loss_pct", "Objective_Loss", "Objective_Loss_pct"]
    ]
    print(summary.to_string())

    # --- Head-to-head at K=16 ---
    K_REF = 16
    comparison = df_all[df_all["K"] == K_REF][
        ["Method", "Financial_Loss", "Financial_Loss_pct", "Objective_Loss", "Objective_Loss_pct", "Execution_Time_sec"]
    ]
    if not comparison.empty:
        print(f"\n=== Head-to-Head Comparison at K={K_REF} ===")
        print(comparison.to_string(index=False))

    print("\nAnalysis completed successfully!")
