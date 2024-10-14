import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from environment import WANDB_INFO
import pandas as pd 
import wandb
api = wandb.Api(timeout=300)

# Project is specified by <entity/project-name>
runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")

summary_list, config_list, name_list, id_list, create_time_list, state_list = [], [], [], [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})
    
    # .name is the human-readable name of the run.
    name_list.append(run.name)
    id_list.append(run.id)
    create_time_list.append(run.created_at)
    state_list.append(run.state)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list,
#     'id': id_list,
#     })


# runs_df.to_csv(f"project.csv")

df = pd.concat([
    pd.DataFrame(summary_list),
    pd.DataFrame(config_list),
    pd.DataFrame({'run_name':name_list}),
    pd.DataFrame({'run_id':id_list}),
    pd.DataFrame({'create_time':create_time_list}),
    pd.DataFrame({'state':state_list}),
], axis = 1)

save_path = f"{root_dir}/analysis/results.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_pickle(save_path)