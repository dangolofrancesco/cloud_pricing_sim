import numpy as np
import pandas as pd
import heapq
import os 

class TemporalResourceManager:
    """
    Manages instantaneous hardware capacity by tracking job start and end times.
    Ensures the baseline respects physical constraints at every time step.
    """
    def __init__(self, initial_capacities: dict):
        self.initial_capacities = initial_capacities
        self.current_capacity = initial_capacities.copy()
        # Stores (end_time, sequence_id, resource_requirements) for active jobs
        self.active_jobs = []
        self._sequence_id = 0

    def release_finished_jobs(self, current_time: float):
        """Reclaims resources from jobs whose end_time <= current_time."""
        while self.active_jobs and self.active_jobs[0][0] <= current_time:
            _, _, resources = heapq.heappop(self.active_jobs)
            for res_type, amount in resources.items():
                self.current_capacity[res_type] += amount

    def try_allocate(self, current_time: float, duration: float, requirements: dict) -> bool:
        """Attempts to allocate a job based on current physical availability."""
        self.release_finished_jobs(current_time)
        
        can_fit = all(self.current_capacity[res_type] >= amount 
                      for res_type, amount in requirements.items())
        
        if can_fit:
            for res_type, amount in requirements.items():
                self.current_capacity[res_type] -= amount
            
            end_time = current_time + duration
            self._sequence_id += 1
            heapq.heappush(self.active_jobs, (end_time, self._sequence_id, requirements))
            return True
        return False

class StochasticGreedyBaseline:
    """
    Stochastic Greedy Baseline for Cloud Resource Allocation.
    Generates the Lower Bound Pareto Front via a weight sweep.
    """
    def __init__(self, jobs_df: pd.DataFrame, capacities: dict, z_star: dict):
        jobs = jobs_df.copy()
        jobs['job_datetime'] = pd.to_datetime(jobs['job_datetime'], errors='coerce')
        jobs['D (hours)'] = pd.to_numeric(jobs['D (hours)'], errors='coerce')

        if jobs['job_datetime'].isna().any():
            raise ValueError("Invalid values found in 'job_datetime'.")
        if jobs['D (hours)'].isna().any():
            raise ValueError("Invalid values found in 'D (hours)'.")
        if (jobs['D (hours)'] < 0).any():
            raise ValueError("Negative values found in 'D (hours)'.")

        # Ensure chronological order for arrival processing.
        jobs = jobs.sort_values('job_datetime').reset_index(drop=True)

        # Keep all scheduling arithmetic in float hours to match duration units.
        if jobs.empty:
            jobs['arrival_hour'] = np.array([], dtype=np.float64)
        else:
            t0 = jobs.loc[0, 'job_datetime']
            jobs['arrival_hour'] = (
                (jobs['job_datetime'] - t0).dt.total_seconds() / 3600.0
            ).astype(np.float64)

        self.jobs_df = jobs
        self.capacities = capacities
        self.z_star = z_star 

    def run_simulation(self, lambdas: dict) -> dict:
        """Executes a sequential simulation pass for a specific lambda weight vector."""
        l1, l2, l3 = lambdas['lambda1'], lambdas['lambda2'], lambdas['lambda3']
        manager = TemporalResourceManager(self.capacities)
        
        cumulative_sat = 0.0
        cumulative_prof = 0.0
        cumulative_carb = 0.0
        n_accepted = 0

        # Feature extraction
        q_j = self.jobs_df['q_j'].values
        v_cont = self.jobs_df['v_total'].values
        phi_L = self.jobs_df['phi_total'].values # Conservative virtual value [cite: 907]
        c_elec = self.jobs_df['C_elec'].values
        c_carb = self.jobs_df['C_carbon'].values
        
        cpu_reqs = self.jobs_df['A_cpu'].values
        ram_reqs = self.jobs_df['A_ram'].values
        durations = self.jobs_df['D (hours)'].to_numpy(dtype=np.float64, copy=False)
        arrivals = self.jobs_df['arrival_hour'].to_numpy(dtype=np.float64, copy=False)

        for i in range(len(self.jobs_df)):
            f_sat = q_j[i] * v_cont[i]
            f_prof = phi_L[i] - c_elec[i]
            f_sus_cost = c_carb[i]

            # Scalarized Reward using Ideal Point Normalization 
            norm_reward = (
                l1 * (f_sat / self.z_star['sat']) + 
                l2 * (f_prof / self.z_star['prof']) - 
                l3 * (f_sus_cost / self.z_star['carb'])
            )

            # Decision Logic - Accept if scalarized reward is positive and resources are available
            if norm_reward > 0:
                reqs = {'cpu': cpu_reqs[i], 'ram': ram_reqs[i]}
                if manager.try_allocate(arrivals[i], durations[i], reqs):
                    cumulative_sat += f_sat
                    cumulative_prof += f_prof
                    cumulative_carb += f_sus_cost
                    n_accepted += 1

        return {
            'satisfaction': cumulative_sat,
            'profit': cumulative_prof,
            'sustainability': -cumulative_carb,
            'n_accepted': n_accepted
        }
    
# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = 'data/processed/batch_may2019_2k.csv'
    
    if os.path.exists(DATA_PATH):
        # 1. Load Data
        df = pd.read_csv(DATA_PATH)
        
        # 2. Priority Mapping [cite: 333, 556]
        bins = [-1, 99, 115, 119, 350, float('inf')]
        labels = [1, 2, 3, 4, 5]
        df['q_j'] = pd.cut(df['q_j'], bins=bins, labels=labels).astype(float)

        # 3. Define Baseline Constants

        total_cpu_volume = (df['A_cpu'] * df['D (hours)']).sum()
        total_ram_volume = (df['A_ram'] * df['D (hours)']).sum()

        t_start = pd.to_datetime(df['job_datetime']).min()
        t_end = pd.to_datetime(df['job_datetime']).max()
        horizon_hours = (t_end - t_start).total_seconds() / 3600.0

        # 3. Define capacity as 60% of average required density
        # This represents the instantaneous capacity needed to fulfill 60% of demand over T
        cluster_capacities = {
            'cpu': (0.60 * total_cpu_volume) / horizon_hours,
            'ram': (0.60 * total_ram_volume) / horizon_hours
        }

        print(f"Calculated 60% Capacity -> CPU: {cluster_capacities['cpu']:.2f} cores, RAM: {cluster_capacities['ram']:.2f} GB")
                
        # Approximate z_star for internal scaling (ideally from Fluid LP Optimizer)
        z_star_approx = {
            'sat': (df['q_j'] * df['v_total']).sum(),
            'prof': (df['phi_total'] - df['C_elec']).sum(),
            'carb': df['C_carbon'].sum()
        }

        # 4. Run Baseline
        baseline = StochasticGreedyBaseline(df, cluster_capacities, z_star_approx)
        test_weights = {'lambda1': 1/3, 'lambda2': 1/3, 'lambda3': 1/3} # Profit/Sat focus
        
        print(f"--- Running Lower Bound Baseline (Greedy) ---")
        results = baseline.run_simulation(test_weights)
        
        print(f"Results for weights {test_weights}:")
        print(f" - Acceptance Rate: {results['n_accepted']/len(df)*100:.2f}%")
        print(f" - Cumulative Profit: ${results['profit']:.2f}")
        print(f" - Cumulative Sat:    ${results['satisfaction']:.2f}")
        print(f" - Carbon Externality: ${-results['sustainability']:.2f}")
    else:
        print(f"File not found: {DATA_PATH}. Please check your directory structure.")
