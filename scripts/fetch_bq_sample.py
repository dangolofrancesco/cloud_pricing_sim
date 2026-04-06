from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

os.makedirs("data/raw", exist_ok=True)
project_id = os.getenv("GCP_PROJECT_ID")

if not project_id:
    raise EnvironmentError("GCP_PROJECT_ID environment variable is not set.")

client = bigquery.Client(project=project_id)

print("Step 1: Sampling 500 exact unique Job IDs...")
# We run this ONCE to lock in our exact sample of jobs
query_jobs = """
    SELECT DISTINCT collection_id
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE type = 0 
    AND RAND() < 0.05
    LIMIT 500
"""
jobs_df = client.query(query_jobs).to_dataframe()
sampled_job_ids = jobs_df['collection_id'].tolist()

print(f"Successfully locked in {len(sampled_job_ids)} specific jobs.")

# We pass this exact list of IDs as a parameter to guarantee 100% intersection
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("job_ids", "INT64", sampled_job_ids)
    ]
)

print("\nStep 2: Fetching Instance Events for these specific jobs...")
query_events = """
    SELECT 
        collection_id, 
        priority, 
        scheduling_class, 
        resource_request.cpus AS resource_request_cpus, 
        resource_request.memory AS resource_request_ram, 
        machine_id
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE type = 0 
    AND collection_id IN UNNEST(@job_ids)
"""
events_df = client.query(query_events, job_config=job_config).to_dataframe()
events_df.to_csv("data/raw/sample_instance_events.csv", index=False)
print(f" -> Downloaded {len(events_df)} event tasks.")

print("\nStep 3: Fetching Instance Usage for these exact same jobs...")
query_usage = """
    SELECT 
        collection_id, 
        start_time, 
        end_time, 
        average_usage.cpus AS average_usage_cpus, 
        average_usage.memory AS average_usage_memory
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    WHERE collection_id IN UNNEST(@job_ids)
"""
usage_df = client.query(query_usage, job_config=job_config).to_dataframe()
usage_df.to_csv("data/raw/sample_instance_usage.csv", index=False)
print(f" -> Downloaded {len(usage_df)} usage histograms.")

print("\nSamples successfully synchronized and saved to data/raw/!")