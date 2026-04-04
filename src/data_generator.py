import os 
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class DataGenerator:
    """
    Handles the ingestion and synthesis of cloud cluster workloads and energy grid data.
    """
    
    def __init__(self, trace_filepath: str, energy_filepath: str, scc_value: float):
        self.trace_filepath = trace_filepath
        self.energy_filepath = energy_filepath
        self.scc_value = scc_value # Social Cost of Carbon (SCC)
        self.raw_jobs = None
        self.processed_jobs = None

    def load_google_traces(self, events_path: str, usage_path: str, sample_frac: float = 0.01) -> pd.DataFrame:
        """
        Loads and merges the Google Cluster Trace (2019) event and usage data.
        
        Args:
            events_path: Path to the instance_events CSV (contains requests and priority).
            usage_path: Path to the instance_usage CSV (contains actual CPU/RAM usage).
            sample_frac: Fraction of data to sample (useful for memory management during dev).
            
        Returns:
            pd.DataFrame: A unified DataFrame containing the extracted features.
        """
        print("Loading instance events...")
        
        # 1. Load the request features
        events_df = pd.read_csv(
            events_path,
            usecols=['collection_id', 'priority', 'scheduling_class', 'resource_request_cpus', 'resource_request_ram', 'machine_id']
        )
        
        # Sample the data to maintain tractability in the static simulator
        if sample_frac < 1.0:
            events_df = events_df.sample(frac=sample_frac, random_state=42)
            
        print("Loading instance usage...")

        # 2. Load the ground truth usage features
        usage_df = pd.read_csv(
            usage_path,
            usecols=['collection_id', 'start_time', 'end_time', 'average_usage_cpus', 'average_usage_memory']
        )
        
        # Calculate actual duration (D) based on 5-minute histograms
        # Assuming timestamps are in microseconds as per Google trace spec
        usage_df['duration_hours'] = (usage_df['end_time'] - usage_df['start_time']) / (1e6 * 3600)
        
        # Group usage by collection_id to get the mean actual usage and total duration
        usage_agg = usage_df.groupby('collection_id').agg({
            'duration_hours': 'sum',
            'average_usage_cpus': 'mean',
            'average_usage_memory': 'mean'
        }).reset_index()
        
        print("Merging datasets...")
        # 3. Merge the request data with the actual usage outcome
        merged_df = pd.merge(events_df, usage_agg, on='collection_id', how='inner')
        
        # Rename columns to match our mathematical notation for clarity
        merged_df = merged_df.rename(columns={
            'priority': 'q_j',
            'resource_request_cpus': 'A_cpu',
            'resource_request_ram': 'A_ram',
            'average_usage_cpus': 'actual_cpu_usage', # Will be mapped to energy w_j later
            'duration_hours': 'D'
        })
        
        # Drop rows with missing crucial data
        merged_df = merged_df.dropna(subset=['A_cpu', 'q_j', 'D'])
        
        self.raw_jobs = merged_df
        print(f"Successfully loaded and merged {len(self.raw_jobs)} jobs.")
        
        return self.raw_jobs

    def load_energy_data(self) -> pd.DataFrame:
        """
        Loads the local Chicago/Argonne physical carbon intensity data (gCO2/kWh) 
        and electricity pricing data.
        """
        pass

    def synthesize_valuations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesizes a continuous latent valuation 'v' using a Lognormal distribution 
        whose mean is proportional to the job's resource demand (A).
        """
        pass

    def apply_myerson_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Myerson transformation to ensure Individual Rationality 
        and revenue maximization: \Phi(v) = v - (1 - F(v)) / f(v).
        """
        pass

    def generate_static_batch(self, N: int) -> pd.DataFrame:
        """
        Fuses trace and energy data, synthesizes valuations, and returns a 
        static batch of N jobs with perfect a priori knowledge for the static simulator.
        """
        pass


# ==========================================
# LOCAL TEST EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Running DataGenerator ingestion test...\n")
    
    # 1. Define paths for our temporary mock files
    mock_events_path = "mock_instance_events.csv"
    mock_usage_path = "mock_instance_usage.csv"
    
    # 2. Create mock instance_events.csv
    events_data = pd.DataFrame({
        'collection_id': [1001, 1002, 1003],
        'priority': [0, 2, 4],  
        'scheduling_class': [1, 2, 3],
        'resource_request_cpus': [0.05, 0.25, 0.80], 
        'resource_request_ram': [0.01, 0.10, 0.50],
        'machine_id': [10, 10, 20] 
    })
    events_data.to_csv(mock_events_path, index=False)
    
    # 3. Create mock instance_usage.csv
    # Features: ground truth stochastic outcomes rolled up into histograms [cite: 309, 310]
    usage_data = pd.DataFrame({
        'collection_id': [1001, 1001, 1002, 1003],
        'start_time': [0, 300000000, 0, 0],              # Microseconds
        'end_time': [300000000, 600000000, 3600000000, 7200000000], 
        'average_usage_cpus': [0.04, 0.05, 0.20, 0.75],
        'average_usage_memory': [0.01, 0.01, 0.09, 0.48]
    })
    usage_data.to_csv(mock_usage_path, index=False)
    
    # 4. Instantiate the generator and run the loader
    try:
        # We pass dummy strings for energy_filepath and scc_value for now
        gen = DataGenerator(trace_filepath="dummy", energy_filepath="dummy", scc_value=51.0)
        
        result_df = gen.load_google_traces(
            events_path=mock_events_path, 
            usage_path=mock_usage_path, 
            sample_frac=1.0
        )
        
        print("\n--- Test Results: Merged DataFrame ---")
        print(result_df.to_string())
        
        # Simple assertions to validate logic
        assert 'q_j' in result_df.columns, "Renaming failed: 'q_j' column missing."
        assert 'D' in result_df.columns, "Duration calculation failed: 'D' column missing."
        assert len(result_df) == 3, "Merge logic failed: expected 3 unique jobs."
        
        print("\n All ingestion tests passed successfully!")
        
    finally:
        # 5. Clean up the mock files so we don't clutter the directory
        if os.path.exists(mock_events_path):
            os.remove(mock_events_path)
        if os.path.exists(mock_usage_path):
            os.remove(mock_usage_path)