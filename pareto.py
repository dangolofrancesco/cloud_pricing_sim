import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Core dominance logic — the mathematical heart of everything
# ---------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Returns True if objective vector 'a' Pareto-dominates 'b'.
    
    Formally: a ≺ b iff
        (1) ∀ i: a[i] <= b[i]   (no worse on any objective)
        (2) ∃ j: a[j] <  b[j]   (strictly better on at least one)
    
    Both conditions must hold simultaneously (minimization assumed).
    """
    # Condition 1: a is no worse than b on ALL objectives
    no_worse_on_all = np.all(a <= b)
    # Condition 2: a is strictly better on AT LEAST ONE objective
    strictly_better_on_one = np.any(a < b)
    return bool(no_worse_on_all and strictly_better_on_one)


def compute_pareto_front(objective_matrix: np.ndarray) -> np.ndarray:
    """
    Given an (N x k) matrix of objective vectors (rows = solutions,
    cols = objectives), returns the indices of Pareto-optimal solutions.
    
    Time complexity: O(N² · k) — fine for thesis experiments,
    but use R-tree / NSGA-II for N > 10,000.
    """
    N = objective_matrix.shape[0]
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(objective_matrix[j], objective_matrix[i]):
                is_dominated[i] = True
                break  # once dominated, no need to check further

    pareto_indices = np.where(~is_dominated)[0]
    return pareto_indices


# ---------------------------------------------------------------
# Simulate a cloud allocation scenario (2 objectives for visual clarity)
# f1 = -Profit (minimize), f2 = CarbonEmissions (minimize)
# ---------------------------------------------------------------
np.random.seed(42)
N = 200

# Simulate a NON-CONVEX front by mixing two clusters
cluster_a = np.random.randn(100, 2) * 0.3 + np.array([1.0, 3.5])
cluster_b = np.random.randn(100, 2) * 0.3 + np.array([3.5, 1.0])
# Add a "gap" — solutions between clusters are dominated, creating non-convexity
solutions = np.vstack([cluster_a, cluster_b])

pareto_idx = compute_pareto_front(solutions)
pareto_points = solutions[pareto_idx]

# ---------------------------------------------------------------
# Utopian and Nadir points
# ---------------------------------------------------------------
# Ideal: best achievable per-objective (over Pareto front, not all solutions!)
z_ideal = pareto_points.min(axis=0)

# Nadir: worst value achieved on the Pareto front
z_nadir = pareto_points.max(axis=0)

print(f"Utopian Point z_ideal: f1={z_ideal[0]:.3f}, f2={z_ideal[1]:.3f}")
print(f"Nadir   Point z_nadir: f1={z_nadir[0]:.3f}, f2={z_nadir[1]:.3f}")

# ---------------------------------------------------------------
# Plot: decision vs objective space distinction made explicit
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

dominated = np.ones(N, dtype=bool)
dominated[pareto_idx] = False

ax.scatter(solutions[dominated, 0], solutions[dominated, 1],
           c='#adb5bd', s=30, label='Dominated solutions', alpha=0.5)
ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
           c='#e63946', s=60, label='Pareto Front $\\mathcal{PF}^*$', zorder=5)
ax.scatter(*z_ideal, marker='*', s=300, c='#2a9d8f',
           label='Utopian $\\mathbf{z}^{\\mathrm{ideal}}$', zorder=6)
ax.scatter(*z_nadir, marker='D', s=100, c='#e9c46a',
           label='Nadir $\\mathbf{z}^{\\mathrm{nadir}}$', zorder=6)

ax.set_xlabel('$f_1$: Negative Profit (minimize)', fontsize=12)
ax.set_ylabel('$f_2$: Carbon Emissions (minimize)', fontsize=12)
ax.set_title('Pareto Front with Utopian & Nadir Points\n(Non-Convex Example)', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('module1_pareto_foundation.png', dpi=150)
plt.show()

