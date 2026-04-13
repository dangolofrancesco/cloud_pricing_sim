import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.discretizer import Discretizer

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
    df_uniform = Discretizer.run_convergence_test("Uniform", max_k=500, **common_kwargs)

    print("Running Geometric convergence test...")
    df_geometric = Discretizer.run_convergence_test("Geometric", max_k=500, **common_kwargs)

    print("Running DP Optimal convergence test (max_k=30)...")
    df_dp = Discretizer.run_convergence_test("DP Optimal", max_k=30, **common_kwargs)

    df_all = pd.concat([df_uniform, df_geometric, df_dp], ignore_index=True)

    print("\n=== Convergence Summary (last K reached per method) ===")
    summary = df_all.groupby("Method").last()[
        ["K", "Financial_Loss", "Financial_Loss_pct", "Objective_Loss", "Objective_Loss_pct"]
    ]
    print(summary.to_string())

    K_REF = 16
    comparison = df_all[df_all["K"] == K_REF][
        ["Method", "Financial_Loss", "Financial_Loss_pct", "Objective_Loss", "Objective_Loss_pct", "Execution_Time_sec"]
    ]
    if not comparison.empty:
        print(f"\n=== Head-to-Head Comparison at K={K_REF} ===")
        print(comparison.to_string(index=False))

    print("\nAnalysis completed successfully!")
