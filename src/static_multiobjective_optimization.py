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

__all__ = [
    'dominates',
    'compute_pareto_front',
    'apply_tiered_priority',
    'build_fluid_volume_constraints',
    'StaticMultiObjectiveOptimizer',
    'ParetoFrontVisualizer',
]


# ---------------------------------------------------------------------------
# Pareto Dominance — direct implementation of the formal ≺ relation
# (minimization convention; negate objectives for maximization problems)
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Returns True if objective vector 'a' Pareto-dominates 'b'.

    Formally: a ≺ b  iff
        (1) ∀ i: a[i] <= b[i]   (no worse on any objective)
        (2) ∃ j: a[j] <  b[j]   (strictly better on at least one)

    Minimization assumed. For maximization pass negated vectors.
    """
    no_worse_on_all = np.all(a <= b)
    strictly_better_on_one = np.any(a < b)
    return bool(no_worse_on_all and strictly_better_on_one)


def compute_pareto_front(objective_matrix: np.ndarray) -> np.ndarray:
    """
    Given an (N × k) matrix of objective vectors (rows = solutions,
    cols = objectives), returns the indices of Pareto-optimal solutions.

    Time complexity: O(N² · k).  Minimization assumed.
    """
    N = objective_matrix.shape[0]
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(objective_matrix[j], objective_matrix[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


def apply_tiered_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps raw Google trace priorities [0, 360] to 5 standard Service Tiers.
    This prevents the Customer Satisfaction objective (q * v) from exploding 
    and overshadowing Profit and Sustainability.
    """
    df_mapped = df.copy()
    if 'q_j' in df_mapped.columns:
        bins = [-1, 99, 115, 119, 350, float('inf')]
        labels = [1, 2, 3, 4, 5] # 1: Best Effort -> 5: Latency Critical
        df_mapped['q_j'] = pd.cut(df_mapped['q_j'], bins=bins, labels=labels).astype(float)
    return df_mapped

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
    (e.g., Total Core-Hours). This is the required benchmark for DLENT regret bounds.

    For each resource i in {cpu, ram}:
        sum_j x_j * A_ij * D_j <= c_i * T
    """
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


class StaticMultiObjectiveOptimizer:
    """
    Static multi-objective optimizer for cloud resource allocation.
    Assumes perfect hindsight to solve for the globally optimal accept/reject
    fractional decision vector subject to Fluid Volume constraints.
    Includes Min-Max normalization to prevent scalarization bias.
    """

    def __init__(
        self,
        jobs_df: pd.DataFrame,
        resource_capacities: dict,
        horizon_hours: float = 744.0, # Default for a 31-day month (e.g. May 2019)
        scc_value: float = 0.05,
        normalize: bool = True
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
        self.normalize = normalize

        # Build Fluid Volume Constraints
        self.A_ub, self.b_ub = build_fluid_volume_constraints(
            self.jobs_df,
            self.resource_capacities,
            self.horizon_hours
        )

        self._extract_job_features()

        if self.normalize:
            self._compute_ideal_and_nadir()

    def _extract_job_features(self):
        """Extract arrays and pre-compute objective coefficients."""
        df = self.jobs_df
        self.q_j = df['q_j'].values                           
        self.v_total = df['v_total'].values                       
        self.phi_total = df['phi_total'].values.astype(float)
        self.A_cpu = df['A_cpu'].values                         
        self.A_ram = df['A_ram'].values                         
        self.w_j_kw = df['w_j_kw'].values                       
        self.D_hours = df['D (hours)'].values   
        self.C_elec = df['C_elec'].values.astype(float)     
        self.C_carbon = df['C_carbon'].values.astype(float)     
        self.c_sat  = self.q_j * self.v_total
        self.c_prof = self.phi_total - self.C_elec
        self.c_carb = self.C_carbon

    def _compute_ideal_and_nadir(self):
        """
        Runs independent LPs to find the theoretical limits of the system 
        for use as Normalization Denominators.
        """
        bounds = [(0, 1)] * self.n_jobs
        
        # 1. Max Satisfaction
        res_sat = linprog(-self.c_sat, A_ub=self.A_ub, b_ub=self.b_ub, bounds=bounds, method='highs')
        self.z_sat_max = -res_sat.fun if res_sat.status == 0 else 1.0
        
        # 2. Max Profit
        res_prof = linprog(-self.c_prof, A_ub=self.A_ub, b_ub=self.b_ub, bounds=bounds, method='highs')
        self.z_prof_max = -res_prof.fun if res_prof.status == 0 else 1.0
        
        # 3. Nadir Carbon (Max possible carbon cost)
        res_carb = linprog(-self.c_carb, A_ub=self.A_ub, b_ub=self.b_ub, bounds=bounds, method='highs')
        self.z_carb_max = -res_carb.fun if res_carb.status == 0 else 1.0
        
        # Protect against division by zero in heavily constrained environments
        self.z_sat_max = max(self.z_sat_max, 1e-9)
        self.z_prof_max = max(self.z_prof_max, 1e-9)
        self.z_carb_max = max(self.z_carb_max, 1e-9)      

    def compute_objectives(self, x: np.ndarray) -> dict:
        """
        Compute the three raw mechanism design objectives.
        f_sat = q_j * v
        f_prof = phi - C_elec
        f_sus = -C_carbon
        """
        satisfaction = float(np.sum(x * self.q_j * self.v_total))
        profit = float(np.sum(x * (self.phi_total - self.C_elec)))
        sustainability = float(-np.sum(x * self.C_carbon))

        return {
            'satisfaction':   satisfaction,
            'profit':         profit,
            'sustainability': sustainability,
            'carbon_cost':    -sustainability,
        }

    def normalize_objectives(self, objectives: dict) -> dict:
        """
        Min-Max normalize objectives into [0, 1] using Utopian anchors.
        Nadir is implicitly 0 (accept-nothing solution) for sat and prof.
        Sustainability nadir is z_carb_max (maximum possible carbon cost).

        Only applied when self.normalize=True and z_*_max are computed.
        """
        if not self.normalize:
            # Passthrough — return raw values with renamed keys
            return {
                'V_sat':    objectives['satisfaction'],
                'V_prof':   objectives['profit'],
                'V_sus':    objectives['sustainability'],
                'C_carbon': objectives['carbon_cost'],
            }

        return {
            'V_sat':    objectives['satisfaction']  / self.z_sat_max,
            'V_prof':   objectives['profit']        / self.z_prof_max,
            # V_sus: negate carbon cost, then normalize by max carbon
            # Result in [0,1]: 0 = worst sustainability, 1 = best
            'V_sus':    1.0 - (objectives['carbon_cost'] / self.z_carb_max),
            'C_carbon':  objectives['carbon_cost'],   # keep raw for reporting
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

        if self.normalize:
            # Optimize relative percentages rather than raw magnitudes
            r_j = (l1 * (self.c_sat / self.z_sat_max) + 
                   l2 * (self.c_prof / self.z_prof_max) - 
                   l3 * (self.c_carb / self.z_carb_max))
        else:
            r_j = (l1 * self.c_sat + l2 * self.c_prof - l3 * self.c_carb)
            
        res = linprog(-r_j, A_ub=self.A_ub, b_ub=self.b_ub, bounds=[(0, 1)] * self.n_jobs, method='highs')
        
        x_solution = res.x if res.status == 0 else np.zeros(self.n_jobs)
        scalarized_val = float(-res.fun) if res.status == 0 else 0.0
        
        return self._build_solution(x_solution, {
            'scalarized_value': scalarized_val,
            'lambda_weights':   lambda_weights,
        })

    # ── Method 2: Epsilon Constraint ─────────────────────────────────────────
    def epsilon_constraint(self, primary_objective: str = 'profit', epsilon_values: dict = None) -> dict:
        if epsilon_values is None:
            epsilon_values = {}

        coef_map = {
            'satisfaction':  self.c_sat,
            'profit':        self.c_prof,
            'sustainability': -self.c_carb,
        }
        if primary_objective not in coef_map:
            raise ValueError(f"Unknown primary_objective: '{primary_objective}'")

        c = -coef_map[primary_objective]
        rows, rhs = [self.A_ub], [self.b_ub]

        # Bounds are still applied in RAW dollars
        if 'satisfaction' in epsilon_values and primary_objective != 'satisfaction':
            rows.append(np.array([-self.c_sat]))
            rhs.append(np.array([-epsilon_values['satisfaction']]))

        if 'profit' in epsilon_values and primary_objective != 'profit':
            rows.append(np.array([-self.c_prof]))
            rhs.append(np.array([-epsilon_values['profit']]))

        if 'carbon_cost' in epsilon_values and primary_objective != 'sustainability':
            rows.append(np.array([self.c_carb]))
            rhs.append(np.array([epsilon_values['carbon_cost']]))

        A_ub_ext = np.vstack(rows)
        b_ub_ext = np.concatenate(rhs)

        res = linprog(c, A_ub=A_ub_ext, b_ub=b_ub_ext, bounds=[(0, 1)] * self.n_jobs, method='highs')
        x_sol = res.x if res.status == 0 else np.zeros(self.n_jobs)

        return self._build_solution(x_sol, {
            'primary_objective': primary_objective,
            'epsilon_values':    epsilon_values,
        })

    # ── Method 3: Chebyshev Scalarization ────────────────────────────────────
    def chebyshev_scalarization(self, reference_point: dict = None, lambda_weights: dict = None) -> dict:
        if lambda_weights is None:
            lambda_weights = {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3}

        l1 = lambda_weights['lambda1']
        l2 = lambda_weights['lambda2']
        l3 = lambda_weights['lambda3']

        N = self.n_jobs
        c = np.append(np.zeros(N), 1.0) # Objective: minimize auxiliary variable t
        rows, rhs = [], []
        
        if self.normalize:
            # Distance from Utopian (100% or 1.0) and Nadir constraints
            # Sat: l1 * (1 - c_sat/z_sat_max) <= t  ==> -l1*(c_sat/z_sat_max) - t <= -l1
            rows.append(np.append(-l1 * (self.c_sat / self.z_sat_max), -1.0))
            rhs.append(-l1)
            
            # Prof: l2 * (1 - c_prof/z_prof_max) <= t ==> -l2*(c_prof/z_prof_max) - t <= -l2
            rows.append(np.append(-l2 * (self.c_prof / self.z_prof_max), -1.0))
            rhs.append(-l2)
            
            # Carb: l3 * (c_carb/z_carb_max - 0) <= t ==> l3*(c_carb/z_carb_max) - t <= 0
            rows.append(np.append(l3 * (self.c_carb / self.z_carb_max), -1.0))
            rhs.append(0.0)
        else:
            if reference_point is None:
                reference_point = self._compute_ideal_point()
            z_sat = reference_point['satisfaction']
            z_prof = reference_point['profit']
            
            rows.append(np.append(-l1 * self.c_sat, -1.0)); rhs.append(-l1 * z_sat)
            rows.append(np.append(-l2 * self.c_prof, -1.0)); rhs.append(-l2 * z_prof)
            rows.append(np.append(l3 * self.c_carb, -1.0)); rhs.append(0.0)

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
            x_solution = res.x[:N]
            t_val = float(res.x[N])

        return self._build_solution(x_solution, {
            'chebyshev_distance': t_val,
            'lambda_weights':     lambda_weights,
        })

    def _compute_ideal_point(self) -> dict:
        """Fallback ideal point computer for the Unnormalized scenario."""
        sol_sat  = self.linear_scalarization({'lambda1': 1.0, 'lambda2': 0.0, 'lambda3': 0.0})
        sol_prof = self.linear_scalarization({'lambda1': 0.0, 'lambda2': 1.0, 'lambda3': 0.0})
        return {
            'satisfaction':  sol_sat['normalized_objectives']['V_sat'],
            'profit':        sol_prof['normalized_objectives']['V_prof'],
            'sustainability': 0.0
        }

    def compute_pareto_front(self, n_points: int = 20, method: str = 'linear') -> list:
        step = 1.0 / max(n_points - 1, 1)
        weight_grid = [
            (i * step, j * step, max(1.0 - i * step - j * step, 0.0))
            for i in range(n_points) for j in range(n_points - i)
            if 1.0 - i * step - j * step >= -1e-9
        ]
        
        ideal_point = None
        if method == 'chebyshev' and not self.normalize:
            ideal_point = self._compute_ideal_point()

        solutions = []
        for l1, l2, l3 in weight_grid:
            weights = {'lambda1': l1, 'lambda2': l2, 'lambda3': l3}
            try:
                if method == 'linear':
                    sol = self.linear_scalarization(weights)
                else:
                    sol = self.chebyshev_scalarization(reference_point=ideal_point, lambda_weights=weights)
                if sol['feasible']: 
                    solutions.append(sol)
            except Exception: pass
        return solutions
    
    def compute_dominated_pool(self, n_random: int = 200) -> list:
        """
        Generate a pool of feasible but likely-dominated solutions by
        solving the LP with random, unbalanced, or extreme weight vectors.
        Used exclusively for visualization of the dominated region.

        Strategy: random Dirichlet samples from the weight simplex — these
        produce a wide spread of solutions across the objective space,
        including many that will be dominated by the Pareto front.
        """
        dominated_pool = []
        # Dirichlet(alpha < 1) concentrates weight on one objective,
        # generating solutions that over-optimize one objective at the
        # expense of others — prime candidates for dominated solutions.
        for _ in range(n_random):
            # alpha = 0.3: strongly skewed weights (dominated region)
            w = np.random.dirichlet([0.3, 0.3, 0.3])
            try:
                sol = self.linear_scalarization({
                    'lambda1': w[0],
                    'lambda2': w[1],
                    'lambda3': w[2],
                })
                if sol['feasible']:
                    dominated_pool.append(sol)
            except Exception:
                pass
        return dominated_pool
