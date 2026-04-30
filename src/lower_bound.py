"""
lower_bound.py
==============
Generates the lower-bound Pareto front for the tri-objective cloud
resource allocation problem via a Random Acceptance Policy.

Mathematical model
------------------
The Random Acceptance Policy is the worst-case causal feasible policy:
each job j is accepted independently with probability p*, where p* is
the maximum uniform acceptance probability that satisfies the Fluid Volume
constraints in expectation:

    p* = min(
            b_ub[0] / sum(A_cpu_j * D_j),   # CPU constraint
            b_ub[1] / sum(A_ram_j * D_j),   # RAM constraint
            1.0
         )

This gives the system load factor: the fraction of total arriving volume
that cluster capacity can serve if acceptance were perfectly uniform.

Under this policy, the expected objectives are:
    E[f_sat]  = p* * sum_j (q_j * v_j)
    E[f_prof] = p* * sum_j (phi_j - C_elec_j)
    E[f_carb] = p* * sum_j C_carbon_j

All scale identically with p*, so in expectation the random policy traces
a single ray in objective space — it cannot trade objectives against each
other. The stochastic envelope of many runs forms the lower-bound front.

Two sampling strategies are provided:
  1. Fixed-p Monte Carlo  — R iid Bernoulli(p*) runs
  2. p-sweep              — vary p in [0, p*] to reveal the full
                            stochastic envelope (richer front shape)

All output solution dicts are schema-compatible with
StaticMultiObjectiveOptimizer._build_solution(), so they plug directly
into ParetoFrontVisualizer without modification.

No I/O transform is applied. All objectives are in MAXIMIZATION convention,
normalized to [0, 1] via the optimizer's z_*_max anchors.
"""

import numpy as np
from typing import Optional

__all__ = ["RandomLowerBound"]


class RandomLowerBound:
    """
    Random Acceptance Policy lower-bound generator.

    Parameters
    ----------
    optimizer : StaticMultiObjectiveOptimizer
        Must be initialised with normalize=True so that z_sat_max,
        z_prof_max and z_carb_max are available.

    Attributes
    ----------
    p_star : float
        Maximum feasible uniform acceptance probability (load factor).
    E_sat, E_prof, E_carb : float
        Expected raw objective values under the random policy.
    std_sat, std_prof, std_carb : float
        Standard deviations of the raw objective values.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self, optimizer):
        self.opt = optimizer
        self._compute_p_star()
        self._compute_moments()

    def _compute_p_star(self) -> None:
        """
        Derive p* from the Fluid Volume capacity constraints.

            p* = min(b_cpu / V_cpu_total,  b_ram / V_ram_total,  1)

        where V_cpu_total = sum_j A_cpu_j * D_j  (total CPU volume demanded).
        """
        opt = self.opt
        V_cpu = float(np.dot(opt.A_cpu, opt.D_hours))
        V_ram = float(np.dot(opt.A_ram, opt.D_hours))

        p_cpu = opt.b_ub[0] / V_cpu if V_cpu > 0 else 1.0
        p_ram = opt.b_ub[1] / V_ram if V_ram > 0 else 1.0

        self.p_star = float(min(p_cpu, p_ram, 1.0))

        self._p_cpu_max = float(p_cpu)
        self._p_ram_max = float(p_ram)
        self._V_cpu     = V_cpu
        self._V_ram     = V_ram

    def _compute_moments(self) -> None:
        """
        Compute E[f] and Var[f] for the three objectives under Bernoulli(p*).

        Since x_j ~ iid Bernoulli(p*):
            E[f]   = p* * sum_j c_j
            Var[f] = p*(1-p*) * sum_j c_j^2
        """
        opt = self.opt
        p, q = self.p_star, 1.0 - self.p_star

        self.E_sat  = p * opt.c_sat.sum()
        self.E_prof = p * opt.c_prof.sum()
        self.E_carb = p * opt.c_carb.sum()

        self.std_sat  = float(np.sqrt(p * q * (opt.c_sat  ** 2).sum()))
        self.std_prof = float(np.sqrt(p * q * (opt.c_prof ** 2).sum()))
        self.std_carb = float(np.sqrt(p * q * (opt.c_carb ** 2).sum()))

        # Pairwise covariances (reveals objective correlation structure)
        self.cov_sat_prof  = p * q * float(np.dot(opt.c_sat,  opt.c_prof))
        self.cov_sat_carb  = p * q * float(np.dot(opt.c_sat,  opt.c_carb))
        self.cov_prof_carb = p * q * float(np.dot(opt.c_prof, opt.c_carb))

    # ── public API ─────────────────────────────────────────────────────────────

    def describe(self) -> None:
        """Print a diagnostic summary of the lower-bound setup."""
        print("=" * 65)
        print("RANDOM LOWER BOUND — Setup Summary")
        print("=" * 65)
        print(f"  Total CPU volume demand  : {self._V_cpu:>12,.1f} core-hours")
        print(f"  CPU capacity (b_ub[0])   : {self.opt.b_ub[0]:>12,.1f} core-hours")
        print(f"  p_cpu_max                : {self._p_cpu_max:>12.4f}")
        print(f"  Total RAM volume demand  : {self._V_ram:>12,.1f} GB-hours")
        print(f"  RAM capacity (b_ub[1])   : {self.opt.b_ub[1]:>12,.1f} GB-hours")
        print(f"  p_ram_max                : {self._p_ram_max:>12.4f}")
        print(f"\n  *** p* (load factor)     : {self.p_star:>12.4f} ***")
        print(f"\n  Expected raw objectives under Bernoulli(p*):")
        print(f"    E[f_sat]  = {self.E_sat:>12,.2f}  ±  {self.std_sat:>10,.2f}")
        print(f"    E[f_prof] = {self.E_prof:>12,.2f}  ±  {self.std_prof:>10,.2f}")
        print(f"    E[f_carb] = {self.E_carb:>12,.2f}  ±  {self.std_carb:>10,.2f}")
        print(f"\n  Objective correlations (cov / p*q*):")
        pq = self.p_star * (1 - self.p_star)
        if pq > 0:
            print(f"    corr(sat, prof) = {self.cov_sat_prof  / pq / (self.std_sat  * self.std_prof  / pq + 1e-12):>8.4f}")
            print(f"    corr(sat, carb) = {self.cov_sat_carb  / pq / (self.std_sat  * self.std_carb  / pq + 1e-12):>8.4f}")
            print(f"    corr(prof,carb) = {self.cov_prof_carb / pq / (self.std_prof * self.std_carb  / pq + 1e-12):>8.4f}")
        print("=" * 65)

    def single_run(self, seed: Optional[int] = None,
                   p_override: Optional[float] = None) -> dict:
        """
        One realisation of the random acceptance policy.

        Parameters
        ----------
        seed       : RNG seed for reproducibility.
        p_override : Use a different acceptance probability (for p-sweep).

        Returns
        -------
        dict — schema-compatible with StaticMultiObjectiveOptimizer output.
        """
        p = p_override if p_override is not None else self.p_star
        rng = np.random.default_rng(seed)

        # Binary accept/reject decision — causally feasible (no future knowledge)
        x = rng.binomial(1, p, self.opt.n_jobs).astype(float)

        objectives = self.opt.compute_objectives(x)
        normalized = self.opt.normalize_objectives(objectives)
        feasible, constraint_info = self.opt.check_resource_constraints(x)

        return {
            "x":                     x,
            "objectives":            objectives,
            "normalized_objectives": normalized,
            "feasible":              feasible,
            "constraint_info":       constraint_info,
            "n_accepted":            int(x.sum()),
            "acceptance_rate":       float(x.mean()),
            "policy":                "random",
            "p_used":                float(p),
            "p_star":                self.p_star,
        }

    def run_monte_carlo(self, R: int = 500, seed: int = 0,
                        verbose: bool = True) -> list:
        """
        Generate R iid realisations of Bernoulli(p*) acceptance.

        Parameters
        ----------
        R       : Number of Monte Carlo runs (≥ 500 recommended).
        seed    : Base random seed; run r uses seed + r.
        verbose : Print progress summary.

        Returns
        -------
        list[dict] — R solution dicts.
        """
        solutions = [self.single_run(seed=seed + r) for r in range(R)]
        if verbose:
            n_feas = sum(s["feasible"] for s in solutions)
            print(f"[LB Monte Carlo] R={R}  feasible={n_feas}  "
                  f"p*={self.p_star:.4f}")
        return solutions

    def run_p_sweep(self, K: int = 20, R_per_p: int = 50,
                    seed: int = 0, verbose: bool = True) -> list:
        """
        Stratified p-sweep: vary p uniformly across [p_min, p*].

        This generates a richer lower-bound front because different p
        values trade acceptance rate against resource utilisation,
        revealing the full shape of the stochastic objective envelope.

        Parameters
        ----------
        K        : Number of p levels (uniformly spaced in [p_min, p*]).
        R_per_p  : Realisations per p level.
        seed     : Base random seed.
        verbose  : Print progress summary.

        Returns
        -------
        list[dict] — K * R_per_p solution dicts.
        """
        rng = np.random.default_rng(seed)
        p_grid = np.linspace(max(0.01, self.p_star * 0.05), self.p_star, K)
        solutions = []
        for k_idx, p_k in enumerate(p_grid):
            for r in range(R_per_p):
                local_seed = int(rng.integers(0, 2**31))
                solutions.append(self.single_run(seed=local_seed,
                                                  p_override=float(p_k)))
        if verbose:
            n_feas = sum(s["feasible"] for s in solutions)
            print(f"[LB p-sweep] K={K}  R_per_p={R_per_p}  "
                  f"total={len(solutions)}  feasible={n_feas}  "
                  f"p* range=[{p_grid[0]:.3f}, {p_grid[-1]:.3f}]")
        return solutions

    def get_expected_solution(self) -> dict:
        """
        Return the deterministic solution at E[x_j] = p* (analytical mean).

        This is NOT a valid Pareto-front point (it is the mean of the
        stochastic distribution), but it is useful as an anchor for
        sanity-checking the normalization.
        """
        x_mean = np.full(self.opt.n_jobs, self.p_star)
        objectives = self.opt.compute_objectives(x_mean)
        normalized = self.opt.normalize_objectives(objectives)
        return {
            "x": x_mean,
            "objectives": objectives,
            "normalized_objectives": normalized,
            "feasible": True,
            "policy": "random_expected",
            "p_used": self.p_star,
            "p_star": self.p_star,
        }
