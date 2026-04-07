import os
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from typing import Tuple, Dict, Optional, Union


def calculate_wj(cpu_req, ram_req, cpu_watts=4.0, ram_watts=0.5, pue=1.10):
    server_watts = (cpu_req * cpu_watts) + (ram_req * ram_watts)
    facility_watts = server_watts * pue
    return facility_watts / 1000.0

class DataGenerator:
    """
    Handles the ingestion and synthesis of cloud cluster workloads and energy grid data.
    Fuses Google Cluster traces with Argonne National Lab energy profiles to create a 
    Multi-Objective Static Batch.
    """
    
    def __init__(self, scc_value: float = 0.05):
        """
        Args:
            scc_value: Social Cost of Carbon ($ per gram of CO2). 
        """
        self.scc_value = scc_value
        self.raw_jobs = None
        self.energy_data: Union[pd.DataFrame, dict, None] = None
        self.processed_jobs = None

    @staticmethod
    def _normalize_collection_id(series: pd.Series) -> pd.Series:
        """Normalize collection_id values to nullable integer for consistent joins."""
        return pd.to_numeric(series, errors='coerce').astype('Int64')

    def load_google_traces(self, events_path: str, usage_path: str, sample_frac: float = 1.0) -> pd.DataFrame:
        print("Loading instance events...")
        events_df = pd.read_csv(
            events_path,
            usecols=['collection_id', 'priority', 'scheduling_class', 'resource_request_cpus', 'resource_request_ram', 'machine_id']
        )

        events_df['collection_id'] = pd.to_numeric(events_df['collection_id'], errors='coerce').astype('Int64')

        # FIX 1: Count the number of tasks in the job so we can scale actual usage properly later
        events_agg = events_df.groupby('collection_id').agg({
            'priority': 'first',             
            'scheduling_class': 'first',     
            'resource_request_cpus': 'sum',  
            'resource_request_ram': 'sum',
            'machine_id': 'count' # We use this column just to count the number of Tasks
        }).rename(columns={'machine_id': 'task_count'}).reset_index()
        
        if sample_frac < 1.0:
            events_agg = events_agg.sample(frac=sample_frac, random_state=42)
            
        print("Loading instance usage...")
        usage_df = pd.read_csv(
            usage_path,
            usecols=['collection_id', 'start_time', 'end_time', 'average_usage_cpus', 'average_usage_memory']
        )
        usage_df['collection_id'] = pd.to_numeric(usage_df['collection_id'], errors='coerce').astype('Int64')
        
        # FIX 2: Calculate Wall-Clock Duration boundaries instead of summing individual task windows
        usage_agg = usage_df.groupby('collection_id').agg({
            'start_time': 'min',           # Earliest task start
            'end_time': 'max',             # Latest task end
            'average_usage_cpus': 'mean',  # Average usage per single task
            'average_usage_memory': 'mean' 
        }).reset_index()
        
        # Calculate actual Wall-Clock Duration (D) in hours
        usage_agg['duration_hours'] = (usage_agg['end_time'] - usage_agg['start_time']) / (1e6 * 3600)
        
        print("Merging datasets...")
        merged_df = pd.merge(events_agg, usage_agg, on='collection_id', how='inner')
        
        merged_df['actual_cpu_usage'] = merged_df['average_usage_cpus'] 
        merged_df['actual_ram_usage'] = merged_df['average_usage_memory'] 
        
        merged_df = merged_df.rename(columns={
            'priority': 'q_j',
            'resource_request_cpus': 'A_cpu',
            'resource_request_ram': 'A_ram',
            'duration_hours': 'D (hours)'
        })
        
        merged_df['job_datetime'] = pd.to_datetime('2019-05-01') + pd.to_timedelta(merged_df['start_time'], unit='us')
        
        merged_df = merged_df.dropna(subset=['A_cpu', 'q_j', 'D (hours)'])
        
        # Clean up temporary columns
        merged_df = merged_df.drop(columns=['task_count', 'average_usage_cpus', 'average_usage_memory', 'end_time'])
        
        self.raw_jobs = merged_df
        print(f"Successfully loaded and merged {len(self.raw_jobs)} jobs.")
        
        return self.raw_jobs

    def load_energy_data(self, energy_filepath: Optional[str] = None) -> Union[pd.DataFrame, dict]:
        """
        Loads local Chicago/Argonne physical carbon intensity data and electricity pricing.
        """
        if energy_filepath and os.path.exists(energy_filepath):
            # ADDED: Parse the timestamp column properly
            self.energy_data = pd.read_csv(energy_filepath, parse_dates=['timestamp'])
            print(f"Loaded dynamic grid data: {len(self.energy_data)} hourly records.")
        else:
            print("No energy file provided. Using default Argonne/Chicago static parameters.")
            self.energy_data = {
                'avg_carbon_intensity': 380.0, 
                'avg_elec_price': 0.04         
            }
        return self.energy_data

    def synthesize_valuations(self, df: pd.DataFrame, base_val_multiplier: float = 100.0, sigma: float = 0.5) -> pd.DataFrame:
        df = df.copy()
        demand_score = df['A_cpu'] + df['A_ram']
        expected_v = np.maximum(demand_score * base_val_multiplier, 1e-4) 
        mu_array = np.log(expected_v) - (sigma**2) / 2
        
        df['v_mu'] = mu_array
        df['v_sigma'] = sigma
        df['v'] = np.random.lognormal(mean=mu_array, sigma=sigma)
        return df

    def apply_myerson_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        s = df['v_sigma'].values
        scale = np.exp(df['v_mu'].values)
        v = df['v'].values
        
        F_v = lognorm.cdf(v, s=s, scale=scale)
        f_v = lognorm.pdf(v, s=s, scale=scale)
        
        inv_hazard = np.where(f_v > 1e-8, (1.0 - F_v) / f_v, 0.0)
        df['phi_v'] = v - inv_hazard
        return df

    def generate_static_batch(
        self,
        N: int,
        cpu_watts: float = 4.0,
        ram_watts: float = 0.5,
        pue: float = 1.10,
    ) -> pd.DataFrame:
        if self.raw_jobs is None:
            raise ValueError("Must call load_google_traces() before generating a batch.")
        if self.energy_data is None:
            raise ValueError("Must call load_energy_data() before generating a batch.")
        if N <= 0:
            raise ValueError("N must be greater than 0.")
            
        print(f"Generating static batch of {N} jobs...")
        eligible_jobs = self.raw_jobs[self.raw_jobs['A_cpu'] >= 0.0]
        available_jobs = len(eligible_jobs)
        if available_jobs == 0:
            raise ValueError(
                "No eligible jobs available to sample from. "
                "Verify merged trace data before generating a batch."
            )
        if N > available_jobs:
            print(f" -> Requested N={N} but only {available_jobs} jobs are available. Sampling all available jobs.")
            N = available_jobs

        batch = eligible_jobs.sample(n=N, random_state=42).copy()

        # Sort chronologically 
        batch = batch.sort_values(by='job_datetime').reset_index(drop=True)
        
        batch = self.synthesize_valuations(batch)
        batch = self.apply_myerson_transformation(batch)
        batch['w_j_kw'] = calculate_wj(batch['A_cpu'], batch['A_ram'], cpu_watts, ram_watts, pue)
        
        # DYNAMIC GRID MAPPING LOGIC 
        if isinstance(self.energy_data, pd.DataFrame):
            # 1. Round job start time down to the nearest hour to match the grid dataset
            batch['grid_hour'] = batch['job_datetime'].dt.floor('h')
            
            # 2. Merge the dynamic grid rates
            batch = pd.merge(batch, self.energy_data, left_on='grid_hour', right_on='timestamp', how='left')
            
            # 3. Apply costs (fill missing values with medians just in case a job falls out of bounds)
            elec_price = batch['elec_price_per_kWh'].fillna(0.035)
            grid_intensity = batch['carbon_intensity_gCO2_per_kWh'].fillna(350.0)
            
            # 4. Cleanup merging artifacts
            batch = batch.drop(columns=['grid_hour', 'timestamp'])
        else:
            # Fallback to static
            elec_price = self.energy_data.get('avg_elec_price', 0.04)
            grid_intensity = self.energy_data.get('avg_carbon_intensity', 380.0)
            batch['elec_price_per_kWh'] = elec_price
            batch['carbon_intensity_gCO2_per_kWh'] = grid_intensity

        # Calculate Final Costs
        batch['C_elec'] = batch['w_j_kw'] * batch['D (hours)'] * elec_price
        batch['C_carbon'] = batch['w_j_kw'] * batch['D (hours)'] * grid_intensity * self.scc_value
        
        self.processed_jobs = batch
        print("Static batch generation complete.")
        
        columns_to_show = [
            'collection_id', 'job_datetime', 'q_j', 'scheduling_class',
            'A_cpu', 'A_ram', 'actual_cpu_usage', 'actual_ram_usage', 'D (hours)', 
            'v', 'phi_v', 'w_j_kw', 
            'elec_price_per_kWh', 'carbon_intensity_gCO2_per_kWh', 
            'C_elec', 'C_carbon'
        ]
        return self.processed_jobs[columns_to_show]

# --- Local Test Execution Block ---
if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_path = os.path.join(base_dir, "data", "raw", "sample_instance_events_forced.csv")
    usage_path = os.path.join(base_dir, "data", "raw", "sample_instance_usage_forced.csv")
    grid_path = os.path.join(base_dir, "data", "raw", "chicago_grid_may2019.csv")

    print(f"Checking for files...")
    if not os.path.exists(events_path) or not os.path.exists(usage_path) or not os.path.exists(grid_path):
        print("\nERROR: CSV files not found.")
    else:
        print("\n--- Initializing DataGenerator ---")
        generator = DataGenerator(scc_value=0.05)
        
        # Load Traces
        generator.load_google_traces(events_path, usage_path, sample_frac=1.0)

        # --- NEW: Let's analyze the raw actual_cpu_usage distribution ---
        print("\n--- Raw Dataset Analysis: actual_cpu_usage ---")
        print(generator.raw_jobs['actual_cpu_usage'].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99]))
        
        # Load Energy Grid
        generator.load_energy_data(grid_path)
        
        # Generate Batch
        print("\n--- Generating Static Batch (N=10) ---")
        batch_df = generator.generate_static_batch(N=10)
        
        # Display Results
        print("\n--- Test Results: Full Mechanism Pipeline ---")
        
        # Retained all mechanism variables PLUS the new dynamic grid variables
        columns_to_show = [
            'collection_id', 'job_datetime', 'q_j', 'scheduling_class',
            'A_cpu', 'A_ram', 'actual_cpu_usage', 'actual_ram_usage', 'D (hours)', 
            'v', 'phi_v', 'w_j_kw', 
            'elec_price_per_kWh', 'carbon_intensity_gCO2_per_kWh', 
            'C_elec', 'C_carbon'
        ]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: f'{x:.8f}')
        print(batch_df[columns_to_show].head(10))
        
        print("\n--- Validation Checks ---")
        print(f"Total jobs in batch: {len(batch_df)}")
        print(f"Any null values in virtual value (phi_v)? : {batch_df['phi_v'].isnull().any()}")
        print(f"Any null values in dynamic pricing? : {batch_df['elec_price_per_kWh'].isnull().any()}")
        print("\nTest completed successfully!")