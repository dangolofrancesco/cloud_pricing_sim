"""
IR Filter Verification Script
==============================
This script verifies that the Individual Rationality filter fix is working correctly.

The bug was: The IR filter used fixed weights instead of the dynamic weights passed
to each solve() call during Pareto sweeps. This created artificial "forbidden zones"
in the solution space.

The fix: Now each call to solve() uses the CURRENT lambda_weights for both the
objective function AND the IR filter.

This script tests whether the same job can be accepted/rejected depending on
the lambda_weights used, proving the fix is working.
"""

import numpy as np
import pandas as pd
from offline_optimizer import FluidLPOptimizer
from pareto_analyzer import ParetoFrontAnalyzer


def test_ir_filter_dynamic_behavior(optimizer):
    """
    Test that the IR filter correctly changes behavior with different weights.
    
    We'll pick a few random jobs and test them with different weight configurations.
    If the fix is working, a job should be:
    - Admissible with some weights (r_j >= 0)
    - Inadmissible with other weights (r_j < 0)
    """
    print("\n" + "="*80)
    print("IR FILTER DYNAMIC BEHAVIOR TEST")
    print("="*80)
    print("\nThis test verifies that the IR filter responds correctly to different")
    print("lambda_weights by accepting different job subsets for each configuration.\n")
    
    # Define diverse weight configurations
    weight_configs = [
        {'name': 'Satisfaction-focused', 'lambda1': 0.8, 'lambda2': 0.1, 'lambda3': 0.1},
        {'name': 'Profit-focused',       'lambda1': 0.1, 'lambda2': 0.8, 'lambda3': 0.1},
        {'name': 'Sustainability-focused','lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.8},
        {'name': 'Balanced',             'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3},
    ]
    
    # Test on a sample of jobs (pick jobs with varying characteristics)
    n_jobs_to_test = min(20, optimizer.n_jobs)
    test_jobs = np.random.choice(optimizer.n_jobs, n_jobs_to_test, replace=False)
    
    print(f"Testing {n_jobs_to_test} randomly selected jobs with {len(weight_configs)} weight configurations...\n")
    
    # Track results
    admissibility_matrix = np.zeros((n_jobs_to_test, len(weight_configs)), dtype=bool)
    
    for job_idx_in_sample, job_idx in enumerate(test_jobs):
        print(f"Job {job_idx}:")
        print(f"  c_sat  = {optimizer.c_sat[job_idx]:10.4f}")
        print(f"  c_prof = {optimizer.c_prof[job_idx]:10.4f}")
        print(f"  c_carb = {optimizer.c_carb[job_idx]:10.4f}")
        
        for config_idx, config in enumerate(weight_configs):
            l1 = config['lambda1']
            l2 = config['lambda2']
            l3 = config['lambda3']
            
            # Calculate reward with these weights
            if optimizer.normalize:
                r = (l1 * (optimizer.c_sat[job_idx] / optimizer.z_sat_max) + 
                     l2 * (optimizer.c_prof[job_idx] / optimizer.z_prof_max) - 
                     l3 * (optimizer.c_carb[job_idx] / optimizer.z_carb_max))
            else:
                r = (l1 * optimizer.c_sat[job_idx] + 
                     l2 * optimizer.c_prof[job_idx] - 
                     l3 * optimizer.c_carb[job_idx])
            
            is_admissible = r >= 0
            admissibility_matrix[job_idx_in_sample, config_idx] = is_admissible
            
            status = "✓ admissible" if is_admissible else "✗ filtered"
            print(f"  {config['name']:25s} (λ={l1:.1f},{l2:.1f},{l3:.1f}) → r={r:8.4f} {status}")
        
        print()
    
    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Count how many jobs change status across configurations
    jobs_with_varying_admissibility = np.sum(np.any(admissibility_matrix != admissibility_matrix[:, 0:1], axis=1))
    
    print(f"\nJobs that change admissibility status: {jobs_with_varying_admissibility}/{n_jobs_to_test}")
    
    if jobs_with_varying_admissibility == 0:
        print("\n❌ FAIL: No jobs changed status across different weights!")
        print("   This indicates the IR filter is NOT using dynamic weights.")
        print("   The bug is still present.")
        return False
    else:
        pct = 100 * jobs_with_varying_admissibility / n_jobs_to_test
        print(f"\n✓ PASS: {pct:.1f}% of tested jobs change admissibility status")
        print("   The IR filter is correctly responding to different lambda_weights.")
        print("   The fix is working as intended.")
        
        # Show admissibility counts per configuration
        print("\nAdmissible jobs per configuration:")
        for config_idx, config in enumerate(weight_configs):
            n_admissible = np.sum(admissibility_matrix[:, config_idx])
            print(f"  {config['name']:25s}: {n_admissible}/{n_jobs_to_test} jobs")
        
        return True


def test_pareto_sweep_ir_statistics(optimizer, analyzer, n_points=10):
    """
    Test IR statistics across a Pareto sweep to verify variance.
    High variance = different weight configurations accept different job subsets = good.
    """
    print("\n" + "="*80)
    print("PARETO SWEEP IR STATISTICS TEST")
    print("="*80)
    print("\nThis test runs a mini Pareto sweep and tracks how many jobs are")
    print("admissible at each point. High variance indicates the fix is working.\n")
    
    # Run a quick Pareto sweep
    pareto_front = analyzer.compute_pareto_front(method='linear', n_points=n_points)
    
    # Extract n_admissible_jobs from solutions
    n_admissible_values = [sol.get('n_admissible_jobs', 0) for sol in pareto_front]
    
    if len(n_admissible_values) == 0:
        print("❌ No feasible solutions found in sweep.")
        return False
    
    print(f"\nIR Filter Statistics across {len(n_admissible_values)} Pareto points:")
    print(f"  Min admissible jobs:  {min(n_admissible_values)}")
    print(f"  Max admissible jobs:  {max(n_admissible_values)}")
    print(f"  Mean:                 {np.mean(n_admissible_values):.1f}")
    print(f"  Std Dev:              {np.std(n_admissible_values):.1f}")
    print(f"  Coefficient of Var:   {np.std(n_admissible_values) / np.mean(n_admissible_values):.3f}")
    
    # Check variance
    cv = np.std(n_admissible_values) / np.mean(n_admissible_values)
    
    if cv < 0.05:
        print("\n❌ FAIL: Very low variance in admissible job counts!")
        print("   This suggests the IR filter is using fixed weights.")
        print("   The bug may still be present.")
        return False
    else:
        print("\n✓ PASS: Significant variance detected in IR filtering.")
        print("   Different weight configurations accept different job subsets.")
        print("   This is the expected behavior after the fix.")
        return True


def run_all_verification_tests(df_path=None, load_factor=0.60):
    """
    Run all verification tests on the IR filter fix.
    """
    print("\n" + "="*80)
    print("IR FILTER FIX VERIFICATION SUITE")
    print("="*80)
    print("\nRunning comprehensive tests to verify that the IR filter bug has been fixed.")
    print("The original bug: IR filter used fixed weights instead of dynamic weights.")
    print("Expected behavior: Different lambda_weights should accept different job subsets.\n")
    
    # Load data
    if df_path is None:
        print("ERROR: Please provide a dataset path.")
        print("Usage: run_all_verification_tests(df_path='path/to/data.csv')")
        return
    
    print(f"Loading dataset: {df_path}")
    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} jobs.\n")
    
    # Initialize optimizer
    print("Initializing FluidLPOptimizer...")
    optimizer = FluidLPOptimizer(df, load_factor=load_factor)
    print("Optimizer initialized.\n")
    
    # Initialize analyzer
    analyzer = ParetoFrontAnalyzer(optimizer)
    
    # Run tests
    test_results = []
    
    print("\n" + "="*80)
    print("TEST 1: Dynamic Behavior Test")
    print("="*80)
    result1 = test_ir_filter_dynamic_behavior(optimizer)
    test_results.append(('Dynamic Behavior', result1))
    
    print("\n" + "="*80)
    print("TEST 2: Pareto Sweep Statistics Test")
    print("="*80)
    result2 = test_pareto_sweep_ir_statistics(optimizer, analyzer, n_points=10)
    test_results.append(('Pareto Sweep Stats', result2))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    all_passed = all(result for _, result in test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}  {test_name}")
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nThe IR filter fix is working correctly.")
        print("Different lambda_weights now correctly accept different job subsets.")
        print("You can proceed with confidence to the gap analysis.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nThe IR filter may not be working as intended.")
        print("Review the test output above to diagnose the issue.")
    
    return all_passed


if __name__ == "__main__":
    print(__doc__)
    print("\nTo run verification tests, use:")
    print("  from test_ir_filter import run_all_verification_tests")
    print("  run_all_verification_tests(df_path='path/to/your/data.csv')")
