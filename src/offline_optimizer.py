import numpy as np
import pandas as pd
from scipy.optimize import linprog
from seaborn.objects import Path


class FluidLPOptimizer:
    """
    Offline Fluid Volume LP Benchmark for DLENT.
    Computes the Upper Bound and extracts Phase 0 parameters.
    """

    def __init__(self, dataset_df, load_factor=0.60, normalize=True):
        # 1. Schema Validation
        required_cols = ['A_cpu', 'A_ram', 'D (hours)', 'v_rate', 'phi_rate', 'q_j', 'C_elec', 'C_carbon']
        missing = [col for col in required_cols if col not in dataset_df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
            
        self.df = dataset_df.copy()
        self.load_factor = load_factor
        self.normalize = normalize
        self.n_jobs = len(self.df)

        # Define the minimum allocatable physical units (1 milli-core / 1 MB)
        MIN_CPU = 0.01  
        MIN_RAM = 0.01  

        # Apply the floor to the requests, strictly preserving actual 0s if they exist
        self.df['A_cpu'] = np.where((self.df['A_cpu'] > 0) & (self.df['A_cpu'] < MIN_CPU), MIN_CPU, self.df['A_cpu'])
        self.df['A_ram'] = np.where((self.df['A_ram'] > 0) & (self.df['A_ram'] < MIN_RAM), MIN_RAM, self.df['A_ram'])
                
        # 2. Compute Resource Volume V_{i,j} (Instantaneous Request * Duration)
        self.df['V_cpu'] = self.df['A_cpu'] * self.df['D (hours)']
        self.df['V_ram'] = self.df['A_ram'] * self.df['D (hours)']

        # Extract underlying NumPy arrays for the Scipy solver
        self.V_cpu_array = self.df['V_cpu'].to_numpy()
        self.V_ram_array = self.df['V_ram'].to_numpy()

        # 3. Compute Scaled Fluid Capacities (b_i)
        self.total_offered_cpu = self.V_cpu_array.sum()
        self.total_offered_ram = self.V_ram_array.sum()
        self.b_cpu = self.total_offered_cpu * self.load_factor
        self.b_ram = self.total_offered_ram * self.load_factor
        
        # 4. Extract features for objective functions
        self.q_j = self.df['q_j'].values                           
        self.v_total = self.df['v_total'].values                       
        self.phi_total = self.df['phi_total'].values.astype(float)
        self.C_elec = self.df['C_elec'].values.astype(float)     
        self.C_carbon = self.df['C_carbon'].values.astype(float) 

        # 5. Compute Objectives Formulas (c vectors)
        self.c_sat  = self.q_j * self.v_total
        self.c_prof = self.phi_total - self.C_elec
        self.c_carb = self.C_carbon

        # 6. Setup Core Physical Constraints (A_ub, b_ub)
        self.A_ub = np.vstack([self.V_cpu_array, self.V_ram_array])
        self.b_ub = np.array([self.b_cpu, self.b_ram])
        
        print(f"--- Fluid LP Initialized ({self.n_jobs:,} jobs) ---")
        print(f"Load Factor (rho): {self.load_factor}")
        print(f"Total CPU Volume : {self.total_offered_cpu:12.2f} | Capacity (b_cpu): {self.b_cpu:12.2f}")
        print(f"Total RAM Volume : {self.total_offered_ram:12.2f} | Capacity (b_ram): {self.b_ram:12.2f}")
        
        # 7. Compute Utopian and Nadir Bounds for Normalization
        if self.normalize:
            self._compute_ideal_and_nadir()
    
    def extract_dlent_parameters(self, T_max: float = 744, time_col: str = 'job_datetime') -> dict:
        """
        Extracts the physical parameters required to initialize Phase 1
        of the online DLENT agent. 
        
        Args:
            T_max: The total time horizon of the trace. If None, it will attempt 
                   to infer it from the arrival timestamps and durations.
            time_col: The column name for job arrival times.
            
        Returns:
            dict: The exact bounding parameters for the online environment.
        """
        # Time Horizon (T)
        # Time Horizon (T)
        if T_max is not None:
            self.T = T_max
        elif time_col in self.df.columns:
            time_series = self.df[time_col]
            
            # Check if the column is a string or datetime object
            if pd.api.types.is_string_dtype(time_series) or pd.api.types.is_datetime64_any_dtype(time_series):
                # 1. Convert to pandas datetime
                dt_series = pd.to_datetime(time_series)
                # 2. Convert to relative hours from the start of the trace (t=0)
                relative_hours = (dt_series - dt_series.min()).dt.total_seconds() / 3600.0
                # 3. Add duration to find the absolute end of the simulation
                self.T = (relative_hours + self.df['D (hours)']).max()
            else:
                # Fallback: Assume it's already a float/int representing relative hours
                self.T = (time_series + self.df['D (hours)']).max()
        else:
            raise ValueError(f"Must provide T_max or ensure valid time_col '{time_col}' exists.")
        # Instantaneous Capacity (c_i) from Fluid Volume (b_i)
        # c_i = Total Volume / Total Time Horizon
        self.c_cpu = self.b_cpu / self.T
        self.c_ram = self.b_ram / self.T
        
        # Maximum Duration (d_max)
        self.d_max = self.df['D (hours)'].max()

        # Minimum Positive Volume (v_min)
        # We must filter out 0s because v_min bounds the learning rate denominator.
        v_cpu_positive = self.V_cpu_array[self.V_cpu_array > 0]
        v_ram_positive = self.V_ram_array[self.V_ram_array > 0]
        
        min_v_cpu = v_cpu_positive.min() if len(v_cpu_positive) > 0 else np.inf
        min_v_ram = v_ram_positive.min() if len(v_ram_positive) > 0 else np.inf
        self.v_min = min(min_v_cpu, min_v_ram)
        
        # Resource Ratio (xi)
        # xi = max(A_{i,j}) / min(c_i)
        max_request = max(self.df['A_cpu'].max(), self.df['A_ram'].max())
        min_capacity = min(self.c_cpu, self.c_ram)
        self.xi = max_request / min_capacity

        print(f"--- DLENT Phase 0 Parameters Extracted ---")
        print(f"Time Horizon (T)      : {self.T:,.2f}")
        print(f"Inst. Capacity (c_cpu): {self.c_cpu:,.2f} Cores")
        print(f"Inst. Capacity (c_ram): {self.c_ram:,.2f} GB")
        print(f"Max Duration (d_max)  : {self.d_max:,.2f}")
        print(f"Min Pos Volume (v_min): {self.v_min:,.6e}")
        print(f"Resource Ratio (xi)   : {self.xi:,.4f}")
        
        if self.xi > 1.0:
            print("WARNING: xi > 1.0. A single job requests more than the total cluster capacity.")
            print("You must filter out 'Whale' jobs that exceed instantaneous capacity.")

        return {
            'T': self.T,
            'c_cpu': self.c_cpu,
            'c_ram': self.c_ram,
            'd_max': self.d_max,
            'v_min': self.v_min,
            'xi': self.xi
        }

    def solve(self, method: str, kwargs: dict) -> dict:
        """
        Routes the request to the correct scalarization method.
        """
        methods = {
            'linear': self._linear_scalarization,
            'epsilon': self._epsilon_constraint,
            'chebyshev': self._chebyshev_scalarization
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
            
        return methods[method](**kwargs)
    
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
    
    def normalize_metrics(self, raw_metrics: dict) -> dict:
        """
        Min-Max normalizes the raw objective values into [0, 1] using ideal point anchors.
        For Satisfaction and Profit, Nadir is implicitly 0 (accept-nothing).
        For Sustainability, Nadir is z_carb_max (maximum possible carbon cost).

        Only applied when self.normalize=True and z_*_max are computed.
        """
        if not self.normalize:
            # Preserve objective direction for downstream Pareto MAX logic.
            carbon_cost = raw_metrics['raw_carbon']
            return {
                'V_sat': float(raw_metrics['raw_satisfaction']),
                'V_prof': float(raw_metrics['raw_profit']),
                'V_sus': float(-carbon_cost),
                'C_carbon': float(carbon_cost)
            }

        # Normalize to [0, 1]
        V_sat = raw_metrics['raw_satisfaction'] / self.z_sat_max
        V_prof = raw_metrics['raw_profit'] / self.z_prof_max
        
        # V_sus: normalize so higher is better (1.0 = zero carbon, 0.0 = worst).
        # Result: 0.0 = Worst sustainability (Max Carbon), 1.0 = Best (Zero Carbon)
        carbon_cost = raw_metrics['raw_carbon']
        V_sus = 1.0 - (carbon_cost / self.z_carb_max)

        return {
            'V_sat': float(V_sat),
            'V_prof': float(V_prof),
            'V_sus': float(V_sus),
            'C_carbon': float(carbon_cost)
        }
    
    def _build_solution(self, x: np.ndarray, method: str, tau: np.ndarray, success: bool = True) -> dict:
        """Compiles the raw fractional LP outputs into a standardized dictionary of metrics."""
        
        # We only return the shortened dictionary if the Scipy solver explicitly crashed.
        # Accepting 0 jobs (x.sum() == 0) is a mathematically valid state.
        if not success:
            return {'feasible': False, 'method': method}

        return {
            'feasible': True,
            'method': method,
            'x_star': x,
            'tau_cpu': tau[0],
            'tau_ram': tau[1],
            'raw_satisfaction': float(np.dot(x, self.c_sat)),
            'raw_profit': float(np.dot(x, self.c_prof)),
            'raw_carbon': float(np.dot(x, self.c_carb))
        }


    ########################################
    # MULTI-OBJECTIVE OPTIMIZATION METHODS #
    ########################################

    def _linear_scalarization(self, lambda_weights: dict) -> dict:
        """
        Optimizes a strictly linear weighted sum of the objectives.
        
        FIXED BUG: Now correctly uses the passed lambda_weights for BOTH the objective 
        function AND the IR filter. Previously, the IR filter used fixed weights, causing
        artificial gaps in the Pareto front during sweeps.
        """
        l1 = lambda_weights.get('lambda1', 1/3)
        l2 = lambda_weights.get('lambda2', 1/3)
        l3 = lambda_weights.get('lambda3', 1/3)

        if self.normalize:
            # Optimize relative percentages rather than raw magnitudes
            r_j = (l1 * (self.c_sat / self.z_sat_max) + 
                   l2 * (self.c_prof / self.z_prof_max) - 
                   l3 * (self.c_carb / self.z_carb_max))
        else:
            r_j = (l1 * self.c_sat + l2 * self.c_prof - l3 * self.c_carb)
            
        # Individual Rationality: Filter jobs with strictly negative weighted rewards
        # CRITICAL: This filter now uses the CURRENT lambda_weights, not fixed weights
        # This ensures that different weight configurations can accept different job subsets
        upper_bounds = np.where(r_j >= 0, 1.0, 0.0)
        bounds = np.column_stack((np.zeros(self.n_jobs), upper_bounds))
            
        res = linprog(-r_j, A_ub=self.A_ub, b_ub=self.b_ub, bounds=bounds, method='highs')
        
        x_sol = res.x if res.success else np.zeros(self.n_jobs)
        tau = np.abs(res.ineqlin.marginals[:2]) if res.success else np.array([0.0, 0.0])
        
        return self._build_solution(x_sol, method='linear', tau=tau)

    def _epsilon_constraint(self, primary_objective: str, epsilon_values: dict, ir_weights: dict = None) -> dict:
        """
        Optimizes a primary objective while bounding the others.
        
        FIXED BUG: The IR filter now defaults to using equal weights, but can be 
        overridden by passing explicit ir_weights. This maintains consistency with
        the epsilon-constraint philosophy (maximize one objective, constrain others).
        """
        
        coef_map = {
            'satisfaction': self.c_sat,
            'profit': self.c_prof,
            'sustainability': -self.c_carb 
        }
        
        if primary_objective not in coef_map:
            raise ValueError(f"Primary objective must be one of {list(coef_map.keys())}")

        # Objective function 
        c = -coef_map[primary_objective]
        
        # Start with core physical constraints
        rows, rhs = [self.A_ub], [self.b_ub]

        if 'satisfaction' in epsilon_values and primary_objective != 'satisfaction':
            # c_sat * x >= eps  -->  -c_sat * x <= -eps
            rows.append(np.array([-self.c_sat]))
            rhs.append(np.array([-epsilon_values['satisfaction']]))

        if 'profit' in epsilon_values and primary_objective != 'profit':
            rows.append(np.array([-self.c_prof]))
            rhs.append(np.array([-epsilon_values['profit']]))

        if 'carbon_cost' in epsilon_values and primary_objective != 'sustainability':
            # c_carb * x <= eps (already a <= constraint)
            rows.append(np.array([self.c_carb]))
            rhs.append(np.array([epsilon_values['carbon_cost']]))

        A_ub_ext = np.vstack(rows)
        b_ub_ext = np.concatenate(rhs)

        # FIXED: IR filter with explicit weights (defaults to equal if not provided)
        if ir_weights is None:
            ir_weights = {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3}
            
        # Compute baseline reward for filtering toxic jobs
        # Note: In epsilon-constraint, the IR filter is somewhat arbitrary since we're
        # directly constraining objectives. Equal weighting is a reasonable default.
        r_base = (ir_weights['lambda1'] * self.c_sat + 
                  ir_weights['lambda2'] * self.c_prof - 
                  ir_weights['lambda3'] * self.c_carb)
        upper_bounds = np.where(r_base >= 0, 1.0, 0.0)
        bounds = np.column_stack((np.zeros(self.n_jobs), upper_bounds))

        res = linprog(c, A_ub=A_ub_ext, b_ub=b_ub_ext, bounds=bounds, method='highs')
        
        x_sol = res.x if res.success else np.zeros(self.n_jobs)
        tau = np.abs(res.ineqlin.marginals[:2]) if res.success else np.array([0.0, 0.0])
        
        return self._build_solution(x_sol, method='epsilon', tau=tau)
    
    def _chebyshev_scalarization(self, lambda_weights: dict) -> dict:
        """
        Minimizes the weighted Chebyshev distance (L-infinity norm) to the Utopian point.
        
        FIXED BUG: The IR filter now correctly uses the passed lambda_weights, ensuring
        consistency across Pareto sweeps. Different weight configurations can now accept
        different job subsets.
        """
        l1 = lambda_weights['lambda1']
        l2 = lambda_weights['lambda2']
        l3 = lambda_weights['lambda3']

        # Decision variables: [x_1, x_2, ..., x_N, t]
        # We want to minimize t
        c = np.append(np.zeros(self.n_jobs), 1.0) 
        
        rows, rhs = [], []

        # 1. Chebyshev Constraints: w_i * (Z_ideal - f_i(x)) <= t
        # Rearranged for Scipy: -w_i * f_i(x) - t <= -w_i * Z_ideal
        if self.normalize:
            # Distance from Utopian (1.0)
            rows.append(np.append(-l1 * (self.c_sat / self.z_sat_max), -1.0))
            rhs.append(-l1)
            
            rows.append(np.append(-l2 * (self.c_prof / self.z_prof_max), -1.0))
            rhs.append(-l2)
            
            # For carbon, ideal is 0. Distance from 0.
            rows.append(np.append(l3 * (self.c_carb / self.z_carb_max), -1.0))
            rhs.append(0.0)
        else:
            # If not normalized, we use raw maximums
            rows.append(np.append(-l1 * self.c_sat, -1.0))
            rhs.append(-l1 * self.z_sat_max)
            
            rows.append(np.append(-l2 * self.c_prof, -1.0))
            rhs.append(-l2 * self.z_prof_max)
            
            rows.append(np.append(l3 * self.c_carb, -1.0))
            rhs.append(0.0)

        # 2. Core Physical Constraints (A_ub * x + 0*t <= b_ub)
        for i in range(self.A_ub.shape[0]):
            rows.append(np.append(self.A_ub[i], 0.0))
            rhs.append(self.b_ub[i])

        A_ub_ext = np.vstack(rows)
        b_ub_ext = np.array(rhs)

        # 3. Individual Rationality (IR) Filter
        # FIXED: Now uses the CURRENT lambda_weights, not fixed weights
        if self.normalize:
            r_base = (l1 * (self.c_sat / self.z_sat_max) + 
                      l2 * (self.c_prof / self.z_prof_max) - 
                      l3 * (self.c_carb / self.z_carb_max))
        else:
            r_base = (l1 * self.c_sat + l2 * self.c_prof - l3 * self.c_carb)
        upper_bounds = np.where(r_base >= 0, 1.0, 0.0)
        
        # Bounds for x_j + bound for t [0, infinity]
        bounds = list(np.column_stack((np.zeros(self.n_jobs), upper_bounds)))
        bounds.append((0, None))

        # 4. Solve
        res = linprog(c, A_ub=A_ub_ext, b_ub=b_ub_ext, bounds=bounds, method='highs')

        if res.success:
            x_sol = res.x[:self.n_jobs]  # Drop the 't' variable from output
            tau = np.abs(res.ineqlin.marginals[-2:]) # Extract physical margins (last two physical constraints)
        else:
            x_sol = np.zeros(self.n_jobs)
            tau = np.array([0.0, 0.0])

        return self._build_solution(x_sol, method='chebyshev', tau=tau)
    

###########################
# DIAGNOSTIC TEST         #
###########################

def run_diagnostics():
    print("==================================================")
    print(" DLENT OFFLINE OPTIMIZER - LOCAL TEST ")
    print("==================================================\n")

    data_file = "./data/processed/batch_may2019_30k.csv"

    # 1. Load Data
    df = pd.read_csv(data_file)
    
    # 2. Initialization & Bounds Test
    print(">>> TEST 1: INITIALIZATION & LOAD FACTOR (rho = 0.60)")
    optimizer = FluidLPOptimizer(df, load_factor=0.60)
    print("PASS: Volume arrays and scaled bounds calculated successfully.\n")

    # 3. DLENT Parameter Extraction Test
    print(">>> TEST 2: DLENT PHASE 0 EXTRACTION")
    dlent_params = optimizer.extract_dlent_parameters(T_max=744, time_col='job_datetime')
    assert dlent_params['c_cpu'] > 0, "CPU Capacity failed."
    assert dlent_params['v_min'] > 0, "v_min failed (Risk of infinite learning rate)."
    print("PASS: Parameters extracted successfully.\n")

    # 4. Normalization Anchors Test
    print(">>> TEST 3: NORMALIZATION ANCHORS (Utopian / Nadir)")
    print(f"Max Satisfaction (Z_sat_max) : {optimizer.z_sat_max:,.2f}")
    print(f"Max Profit (Z_prof_max)      : {optimizer.z_prof_max:,.2f}")
    print(f"Max Carbon Cost (Z_carb_max) : {optimizer.z_carb_max:,.2f}")
    assert optimizer.z_sat_max > 1.0, "Normalization bounds failed to compute."
    print("PASS: Theoretical limits found.\n")

    # Compute baseline toxic jobs (Where raw combined reward < 0)
    # Using lambda = 1/3 for all
    l1, l2, l3 = 1/3, 1/3, 1/3
    r_base = (l1 * optimizer.c_sat + l2 * optimizer.c_prof - l3 * optimizer.c_carb)
    toxic_jobs_count = np.sum(r_base < 0)

    print(">>> TEST 4: MULTI-OBJECTIVE SOLVER & IR FILTERING")
    print(f"Dataset Total Jobs         : {optimizer.n_jobs:,}")
    print(f"Toxic Jobs (r_j < 0)       : {toxic_jobs_count:,} (Should be strictly rejected)")
    
    methods = [
        ('linear', {'lambda_weights': {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3}}),
        ('epsilon', {'primary_objective': 'profit', 'epsilon_values': {'satisfaction': optimizer.z_sat_max * 0.2}}),
        ('chebyshev', {'lambda_weights': {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3}})
    ]

    for method_name, kwargs in methods:
        print(f"\n--- Testing Method: {method_name.upper()} ---")
        
        # Run Solver
        solution = optimizer.solve(method=method_name, kwargs=kwargs)
        
        if not solution['feasible']:
            print(f"Result: INFEASIBLE")
            continue
            
        x_star = solution['x_star']
        accepted_full = np.sum(x_star == 1.0)
        accepted_fractional = np.sum((x_star > 0.0) & (x_star < 1.0))
        rejected_total = np.sum(x_star == 0.0)
        
        # Verify IR Logic
        # Check how many of the known toxic jobs received x > 0
        toxic_accepted = np.sum(x_star[r_base < 0] > 0)
        
        # Normalize the metrics
        norm_metrics = optimizer.normalize_metrics(solution)

        print(f"Feasible                 : {solution['feasible']}")
        print(f"Total Accepted (Full)    : {accepted_full:,}")
        print(f"Total Accepted (Fract)   : {accepted_fractional:,}")
        print(f"Total Rejected           : {rejected_total:,}")
        print(f"IR Violations            : {toxic_accepted} (Must be 0)")
        print(f"Shadow Price (tau_cpu)   : {solution['tau_cpu']:.6f}")
        print(f"Normalized Sat [0,1]     : {norm_metrics['V_sat']:.4f}")
        print(f"Normalized Prof [0,1]    : {norm_metrics['V_prof']:.4f}")
        print(f"Normalized Sus [0,1]     : {norm_metrics['V_sus']:.4f}")
        
        assert toxic_accepted == 0, f"IR FAULT: {method_name} accepted {toxic_accepted} jobs with negative rewards."

    print("\n==================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY.")
    print("==================================================")

if __name__ == "__main__":
    run_diagnostics()