from google.cloud import bigquery
import pandas as pd
import os

# Ensure your local data directory exists
os.makedirs("data/raw", exist_ok=True)

# Initialize the BigQuery client 
# (Make sure you have authenticated your environment using `gcloud auth application-default login`)
client = bigquery.Client()

print("Fetching real Instance Events sample...")
# Query the instance_events table from Cell A
# type = 0 refers to the 'SUBMIT' event in the trace schema
query_events = """
    SELECT collection_id, priority, scheduling_class, resource_request_cpus, resource_request_ram, machine_id
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
    WHERE type = 0 
    LIMIT 10000
"""
events_df = client.query(query_events).to_dataframe()
events_df.to_csv("data/raw/sample_instance_events.csv", index=False)

print("Fetching real Instance Usage sample...")
# Query the instance_usage table from Cell A
query_usage = """
    SELECT collection_id, start_time, end_time, average_usage_cpus, average_usage_memory
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
    LIMIT 50000
"""
usage_df = client.query(query_usage).to_dataframe()
usage_df.to_csv("data/raw/sample_instance_usage.csv", index=False)

print("Samples saved successfully to data/raw/!")