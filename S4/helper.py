import os, sys
root_dir = os.getcwd()
sys.path.append(root_dir)

import numpy as np
import random, torch, transformers, wandb, pickle
import torch.nn.functional as F
import torch.nn as nn

class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)
        
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    transformers.set_seed(seed)
    
def complex_l1_penalty(real_param, image_param):
    return torch.sum(torch.abs(real_param)) + torch.sum(torch.abs(image_param))

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
def init_wandb(
    wandb_entity,
    wandb_proj,
    wandb_config,
    overwrite,
):
    if overwrite:
        wandb.init(
            project = wandb_proj,
            entity = wandb_entity,
            config = wandb_config,
        )
        return None
        
    # first check whether there exists a run with the same configuration
    api = wandb.Api(timeout=300)
    runs = api.runs(f"{wandb_entity}/{wandb_proj}")
    find_existing_run = None
    for run in runs:
        run_config_list = {k: v for k,v in run.config.items() if not k.startswith('_')}
        this_run = True
        for key in wandb_config:
            if key == 'overwrite':
                continue
            if (not key in run_config_list) or (run_config_list[key] != wandb_config[key]): 
                this_run = False
                break
        if this_run: 
            if run.state != 'finished' or find_existing_run is not None:
                # remove crashed one or duplicated one
                if run.state != 'finished':
                    print(f"Remove crashed run: {run.name}")
                if run.state == 'finished':
                    print(f"Remove duplicated run: {run.name}")
                run.delete()
            else:
                find_existing_run = run

                print("########"*3)
                print(f"Find existing run in wandb: {run.name}")
                print("########"*3)
        
    # initialize wandb
    if find_existing_run is None:
            
        wandb.init(
            project = wandb_proj,
            entity = wandb_entity,
            config = wandb_config,
        )
    else:
        print('Not overwrite, and the job has been done! Exit!')
        exit(0)
        
    return find_existing_run
    
def classification_loss(pred_Y, target_Y):
    num_classes = pred_Y.shape[1]
    target_Y = target_Y.permute(0,2,1)
    target_labels = target_Y.argmax(dim=2).reshape(-1)
    
    # # apply softmax to pred_Y
    # pred_Y = F.softmax(pred_Y, dim=1)
    pred_Y = pred_Y.permute(0,2,1)
    logits = pred_Y.reshape(-1, num_classes)
    return F.cross_entropy(logits, target_labels)

class adapterLinear(nn.Linear):
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True,
        **kwargs,
    ):
        super(adapterLinear, self).__init__(
            in_features, 
            out_features, 
            bias,
            **kwargs,
        )
        
        self.adapter = None
        self.update_channels = None 
        
    def forward(
        self, 
        input,
        method,
    ):
        output = super().forward(input)
        if not self.adapter is None:
            if method == 'lora':
                # lora
                output += self.adapter(input)
            elif method == 'ours':
                # ours
                adapter_params = torch.zeros(self.out_features, self.in_features, device=input.device)
                for i, channel_idx in enumerate(self.update_channels):
                    adapter_params[:,channel_idx] = self.adapter[:,i]
                output += F.linear(input, adapter_params)
            
        return output
    
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

