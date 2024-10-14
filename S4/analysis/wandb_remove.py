import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from environment import WANDB_INFO

import wandb
from datetime import datetime

# Initialize wandb API
api = wandb.Api()


# Fetch all runs of the project
runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")


# Define the cutoff time
cutoff_time = datetime.strptime("2024-5-3 00:00:00", "%Y-%m-%d %H:%M:%S")

counter = 0
# Loop through the runs and delete if they were created before the cutoff_time
for run in runs:
    created_at = run.created_at
    created_at = datetime.fromisoformat(created_at)
    if created_at < cutoff_time:
        print(f"Run created at {created_at} has been deleted!")
        run.delete()
        counter += 1
print(f"Deleted {counter} runs!")