"""
Static Multi-Objective Resource Allocation Optimization
========================================================
Tests three scalarization methods to analyze Pareto-frontier convexity:

  1. Linear Scalarization    — weighted sum via LP (fast; misses non-convex regions)
  2. ε-Constraint Method     — optimize one objective, bound the others
  3. Chebyshev Scalarization — L∞ distance to ideal point (covers non-convex regions)

Mathematical Benchmark:
  Uses the Fluid Volume Approximation (OPT_LP) for capacity constraints, matching
  the theoretical baseline required to calculate DLENT Regret Bounds.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Core Fluid Constraint Builder
# ---------------------------------------------------------------------------

def build_fluid_volume_constraints(
    jobs_df: pd.DataFrame,
    cluster_capacity: dict,
    horizon_hours: float,
) -> tuple:
    """
    Build Fluid Volume constraints for the static LP benchmark.

    Instead of tracking instantaneous capacity, the Fluid LP treats the total
    hardware capacity over the entire time horizon T as a 'pool of volume'
    (e.g., Total Core-Hours).

    For each resource i in {cpu, ram}:
        sum_j x_j * A_ij * D_j <= c_i * T
    """
    # Calculate resource volumes demanded by each job
    A_cpu = jobs_df['A_cpu'].values.astype(float)
    A_ram = jobs_df['A_ram'].values.astype(float)
    D_hours = jobs_df['D (hours)'].values.astype(float)

    cpu_volume_demand = A_cpu * D_hours
    ram_volume_demand = A_ram * D_hours

    # Create the A_ub matrix (2 rows: CPU and RAM)
    A_ub = np.vstack([cpu_volume_demand, ram_volume_demand])

    # Create the b_ub vector (Total capacity volume = instantaneous capacity * horizon)
    b_ub = np.array([
        float(cluster_capacity['cpu']) * horizon_hours,
        float(cluster_capacity['ram']) * horizon_hours
    ])

    return A_ub, b_ub


# ---------------------------------------------------------------------------
# Optimizer Class
# ---------------------------------------------------------------------------

class StaticMultiObjectiveOptimizer:
    """
    Static multi-objective optimizer for cloud resource allocation.
    Assumes perfect hindsight to solve for the globally optimal accept/reject
    fractional decision vector subject to Fluid Volume constraints.
    """

    def __init__(
        self,
        jobs_df: pd.DataFrame,
        resource_capacities: dict,
        horizon_hours: float = 744.0, # Default for a 31-day month (e.g. May 2019)
        scc_value: float = 0.05
    ):
        """
        Parameters
        ----------
        jobs_df : pd.DataFrame
            Output of DataGenerator.
        resource_capacities : dict
            {'cpu': float, 'ram': float} representing the INSTANTANEOUS physical capacity.
        horizon_hours : float
            The total time horizon T of the simulation (e.g., 744 hours for 1 month).
        scc_value : float
            Social cost of carbon multiplier.
        """
        self.jobs_df = jobs_df.copy().reset_index(drop=True)
        self.n_jobs = len(self.jobs_df)
        self.scc_value = scc_value
        self.resource_capacities = resource_capacities
        self.horizon_hours = horizon_hours

        # Build Fluid Volume Constraints
        self.A_ub, self.b_ub = build_fluid_volume_constraints(
            self.jobs_df,
            self.resource_capacities,
            self.horizon_hours
        )

        self._extract_job_features()

    def _extract_job_features(self):
        """Cache per-job NumPy arrays from the DataFrame for fast vectorized math."""
        df = self.jobs_df
        self.q_j        = df['q_j'].values                           # Priority class
        self.v_total    = df['v_total'].values                       # Customer valuation ($)

        # Handle Discrete vs Continuous Valuation (Crucial for Discretization Loss)
        if 'phi_total' in df.columns:
            self.phi_total = df['phi_total'].values.astype(float)
        else:
            self.phi_total = (df['q_j'].values.astype(float) * df['v_total'].values.astype(float))

        self.A_cpu      = df['A_cpu'].values                         # CPU demand (cores)
        self.A_ram      = df['A_ram'].values                         # RAM demand (GB)
        self.w_j_kw     = df['w_j_kw'].values                        # Power draw (kW)
        self.D_hours    = df['D (hours)'].values                     # Duration (h)

        # Electricity Cost Array
        if 'C_elec' in df.columns:
            self.C_elec = df['C_elec'].values.astype(float)
        else:
            self.C_elec = (self.w_j_kw * self.D_hours * df['elec_price_per_kWh'].values.astype(float))

        # Carbon Cost Array
        if 'C_carbon' in df.columns:
            self.C_carbon = df['C_carbon'].values.astype(float)
        else:
            ci = df['carbon_intensity_gCO2_per_kWh'].values.astype(float)
            self.C_carbon = (self.w_j_kw * self.D_hours * ci / 1e6) * self.scc_value

    def compute_objectives(self, x: np.ndarray) -> dict:
        """
        Compute the three raw mechanism design objectives.
        f_sat = q_j * v
        f_prof = phi - C_elec
        f_sus = -C_carbon
        """
        satisfaction   = float(np.sum(x * self.q_j * self.v_total))
        profit         = float(np.sum(x * (self.phi_total - self.C_elec)))
        sustainability = float(-np.sum(x * self.C_carbon))

        return {
            'satisfaction':   satisfaction,
            'profit':         profit,
            'sustainability': sustainability,
            'carbon_cost':    -sustainability,
        }

    def normalize_objectives(self, objectives: dict) -> dict:
        return {
            'V_sat':    objectives['satisfaction'],
            'V_prof':   objectives['profit'],
            'V_sus':    objectives['sustainability'],
            'C_carbon': objectives['carbon_cost'],
        }

    def check_resource_constraints(self, x: np.ndarray) -> tuple:
        """Verify the fluid volume constraints satisfy A_ub @ x <= b_ub."""
        used = self.A_ub @ np.asarray(x, dtype=float)
        violations = np.maximum(used - self.b_ub, 0.0)
        max_viol = float(violations.max()) if len(violations) else 0.0

        return (max_viol <= 1e-6), {
            'cpu_volume_used': float(used[0]),
            'cpu_volume_cap': float(self.b_ub[0]),
            'ram_volume_used': float(used[1]),
            'ram_volume_cap': float(self.b_ub[1]),
            'max_violation': max_viol
        }

    def _per_job_reward(self, l1: float, l2: float, l3: float) -> np.ndarray:
        return (l1 * self.q_j * self.v_total
                + l2 * (self.phi_total - self.C_elec)
                - l3 * self.C_carbon)

    def _build_solution(self, x: np.ndarray, extra: dict) -> dict:
        objectives = self.compute_objectives(x)
        normalized = self.normalize_objectives(objectives)
        feasible, constraint_info = self.check_resource_constraints(x)
        n_accepted = int(np.sum(x > 0))
        return {
            'x':                    x,
            'objectives':           objectives,
            'normalized_objectives': normalized,
            'feasible':             feasible,
            'constraint_info':      constraint_info,
            'n_accepted':           n_accepted,
            'acceptance_rate':      n_accepted / max(1, self.n_jobs),
            **extra,
        }

    # ── Method 1: Linear Scalarization ───────────────────────────────────────
    def linear_scalarization(self, lambda_weights: dict) -> dict:
        l1, l2, l3 = (lambda_weights['lambda1'], lambda_weights['lambda2'], lambda_weights['lambda3'])
        c = -self._per_job_reward(l1, l2, l3)

        res = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, bounds=[(0, 1)] * self.n_jobs, method='highs')

        x_solution = np.round(res.x).astype(int) if res.status == 0 else np.zeros(self.n_jobs)
        scalarized_val = float(-res.fun) if res.status == 0 else 0.0

        return self._build_solution(x_solution, {
            'scalarized_value': scalarized_val,
            'lambda_weights':   lambda_weights,
        })

    # ── Method 2: Epsilon Constraint ─────────────────────────────────────────
    def epsilon_constraint(self, primary_objective: str = 'profit', epsilon_values: dict = None) -> dict:
        if epsilon_values is None: epsilon_values = {}

        sat_coef  =  self.q_j * self.v_total
        prof_coef =  self.phi_total - self.C_elec
        sus_coef  = -self.C_carbon

        coef_map = {'satisfaction': sat_coef, 'profit': prof_coef, 'sustainability': sus_coef}
        if primary_objective not in coef_map:
            raise ValueError(f"Unknown primary_objective: '{primary_objective}'")

        c = -coef_map[primary_objective]
        rows, rhs = [self.A_ub], [self.b_ub]

        if 'satisfaction' in epsilon_values and primary_objective != 'satisfaction':
            rows.append(np.array([-sat_coef]))
            rhs.append(np.array([-epsilon_values['satisfaction']]))

        if 'profit' in epsilon_values and primary_objective != 'profit':
            rows.append(np.array([-prof_coef]))
            rhs.append(np.array([-epsilon_values['profit']]))

        if 'carbon_cost' in epsilon_values and primary_objective != 'sustainability':
            rows.append(np.array([self.C_carbon])) # Carbon cost is positive
            rhs.append(np.array([epsilon_values['carbon_cost']]))

        A_ub_ext = np.vstack(rows)
        b_ub_ext = np.concatenate(rhs)

        res = linprog(c, A_ub=A_ub_ext, b_ub=b_ub_ext, bounds=[(0, 1)] * self.n_jobs, method='highs')
        x_sol = res.x if res.status == 0 else np.zeros(self.n_jobs)

        return self._build_solution(np.round(x_sol).astype(int), {
            'primary_objective': primary_objective,
            'epsilon_values':    epsilon_values,
        })

    # ── Method 3: Chebyshev Scalarization ────────────────────────────────────
    def chebyshev_scalarization(self, reference_point: dict, lambda_weights: dict = None) -> dict:
        if lambda_weights is None:
            lambda_weights = {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3}

        l1, l2, l3 = lambda_weights['lambda1'], lambda_weights['lambda2'], lambda_weights['lambda3']
        sat_coef  =  self.q_j * self.v_total
        prof_coef =  self.phi_total - self.C_elec
        sus_coef  = -self.C_carbon

        z_sat  = reference_point['satisfaction']
        z_prof = reference_point['profit']
        z_sus  = reference_point['sustainability']

        N = self.n_jobs
        c = np.append(np.zeros(N), 1.0) # Objective is to minimize t

        def cheb_row(lam, coef, z_star):
            return np.append(-lam * coef, -1.0), -lam * z_star

        rows, rhs = [], []
        for lam, coef, z_star in [(l1, sat_coef, z_sat), (l2, prof_coef, z_prof), (l3, sus_coef, z_sus)]:
            r, b = cheb_row(lam, coef, z_star)
            rows.append(r)
            rhs.append(b)

        # Apply Fluid constraints to x_j (the N auxiliary column for t is 0.0)
        for i in range(self.A_ub.shape[0]):
            rows.append(np.append(self.A_ub[i], 0.0))
            rhs.append(self.b_ub[i])

        A_ub_ext = np.vstack(rows)
        b_ub_ext = np.array(rhs)
        bounds = [(0, 1)] * N + [(0, None)]

        res = linprog(c, A_ub=A_ub_ext, b_ub=b_ub_ext, bounds=bounds, method='highs')

        if res.status != 0:
            x_solution = np.zeros(N, dtype=int)
            t_val = float('inf')
        else:
            x_solution = np.round(res.x[:N]).astype(int)
            t_val = float(res.x[N])

        return self._build_solution(x_solution, {
            'chebyshev_distance': t_val,
            'reference_point':    reference_point,
            'lambda_weights':     lambda_weights,
        })

    def _compute_ideal_point(self) -> dict:
        sol_sat  = self.linear_scalarization({'lambda1': 1.0, 'lambda2': 0.0, 'lambda3': 0.0})
        sol_prof = self.linear_scalarization({'lambda1': 0.0, 'lambda2': 1.0, 'lambda3': 0.0})
        sol_sus  = self.linear_scalarization({'lambda1': 0.0, 'lambda2': 0.0, 'lambda3': 1.0})
        return {
            'satisfaction':  sol_sat['normalized_objectives']['V_sat'],
            'profit':        sol_prof['normalized_objectives']['V_prof'],
            'sustainability': sol_sus['normalized_objectives']['V_sus'],
        }

    def compute_pareto_front(self, n_points: int = 20, method: str = 'linear') -> list:
        step = 1.0 / max(n_points - 1, 1)
        weight_grid = [
            (i * step, j * step, max(1.0 - i * step - j * step, 0.0))
            for i in range(n_points) for j in range(n_points - i)
            if 1.0 - i * step - j * step >= -1e-9
        ]
        ideal_point = self._compute_ideal_point() if method == 'chebyshev' else None
        solutions = []

        for l1, l2, l3 in weight_grid:
            weights = {'lambda1': l1, 'lambda2': l2, 'lambda3': l3}
            try:
                sol = self.linear_scalarization(weights) if method == 'linear' else self.chebyshev_scalarization(ideal_point, weights)
                if sol['feasible']: solutions.append(sol)
            except Exception: pass
        return solutions


# ---------------------------------------------------------------------------
# Visualizer Class
# ---------------------------------------------------------------------------

class ParetoFrontVisualizer:
    """Plot and analyze the Pareto front produced by StaticMultiObjectiveOptimizer."""

    @staticmethod
    def plot_3d_pareto_front(solutions: list, title: str = "3D Pareto Front", method_name: str = ""):
        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection='3d')

        V_sat  = [s['normalized_objectives']['V_sat']  for s in solutions]
        V_prof = [s['normalized_objectives']['V_prof'] for s in solutions]
        V_sus  = [s['normalized_objectives']['V_sus']  for s in solutions]

        if 'scalarized_value' in solutions[0]:
            colors = [s['scalarized_value'] for s in solutions]
            clabel = 'Scalarized Value ($)'
        else:
            colors = [s['chebyshev_distance'] for s in solutions]
            clabel = 'Chebyshev Distance'

        sc = ax.scatter(V_sat, V_prof, V_sus, c=colors, cmap='viridis', s=80, alpha=0.65)
        ax.set_xlabel('Customer Satisfaction ($)', fontsize=10)
        ax.set_ylabel('Provider Profit ($)', fontsize=10)
        ax.set_zlabel('Sustainability ($, higher = less carbon)', fontsize=10)
        ax.set_title(f'{title}\n{method_name}', fontsize=12, fontweight='bold')
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.5)
        return fig, ax

    @staticmethod
    def plot_2d_projections(solutions: list, title: str = "2D Pareto Projections"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        V_sat  = [s['normalized_objectives']['V_sat']  for s in solutions]
        V_prof = [s['normalized_objectives']['V_prof'] for s in solutions]
        V_sus  = [s['normalized_objectives']['V_sus']  for s in solutions]

        axes[0].scatter(V_sat, V_prof, alpha=0.6, s=40)
        axes[0].set_xlabel('Customer Satisfaction ($)')
        axes[0].set_ylabel('Provider Profit ($)')
        axes[0].set_title('Satisfaction vs Profit')
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(V_sat, V_sus, alpha=0.6, s=40, color='green')
        axes[1].set_xlabel('Customer Satisfaction ($)')
        axes[1].set_ylabel('Sustainability ($ saved)')
        axes[1].set_title('Satisfaction vs Sustainability')
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(V_prof, V_sus, alpha=0.6, s=40, color='red')
        axes[2].set_xlabel('Provider Profit ($)')
        axes[2].set_ylabel('Sustainability ($ saved)')
        axes[2].set_title('Profit vs Sustainability')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig, axes
