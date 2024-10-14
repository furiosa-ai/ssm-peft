import os
root_dir = os.getcwd()

from copy import deepcopy
from torch import nn
import wandb, torch, transformers
from dataclasses import dataclass, field, asdict
from typing import Literal

from environment import WANDB_INFO
from helper import set_seed, init_wandb, complex_l1_penalty, classification_loss
from load_model import load_model
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from helper import split_train_val, load_pickle, save_pickle
import pandas as pd
from copy import deepcopy

import pdb

@dataclass
class TrainingConfigs:
    n_epochs: int = field(default=500, metadata={"help": "Number of epochs to train."})
    batch_size: int = field(default=4000, metadata={"help": "Batch size."})
    seq_len: int = field(default=200, metadata={"help": "Sequence length."})
    lr: float = field(default=0.005, metadata={"help": "Learning rate."})
    wandb: bool = field(default=True, metadata={"help": "Log to wandb."})
    warmup_epochs: int = field(default=20, metadata={"help": "Number of warmup epochs."})
    warmup_lr: float = field(default=0.01, metadata={"help": "Warmup learning rate."})
    overwrite: bool = field(default=False, metadata={"help": "Overwrite the existing results."})
    device: Literal['cuda', 'cpu'] = field(default='cuda', metadata={"help": "Device."})
    pretrain_epochs: int = field(default=10, metadata={"help": "Number of pretrain epochs."})
    pretrain_lr: float = field(default=0.01, metadata={"help": "Pretrain learning rate."})
    pretrain: bool = field(default=False, metadata={"help": "Pretrain the model."})
    
@dataclass 
class ModelConfigs:
    d_model: int = field(default=64, metadata={"help": "Model dimension."})
    frozen_n_layers: int = 4
    frozen_d_state: int = 16
    dropout: float = 0.0
    task_type: Literal['regression', 'classification'] = field(default='regression', metadata={"help": "Task type."})
    model_mode: Literal['theory', 'default'] = field(default='theory', metadata={"help": "S4 mode."})
    
@dataclass
class PeftConfigs:
    lp_method: Literal['lora', 'full', 'freeze', 'ours'] = field(default='full', metadata={"help": "Linear projection method."})
    lp_lora_rank: int = 16
    D_method: Literal['full', 'freeze'] = field(default='full', metadata={"help": "D method."})
    ssm_method: Literal['ours', 'full', 'freeze', 'lora'] = field(default='full', metadata={"help": "SSM method."})
    ssm_lora_rank: int = 4
    our_mode: Literal['hard', 'soft'] = field(default='hard', metadata={"help": "Our method mode."})
    select_states_dim: int = 8
    updatable_states_dim: int = 8
    select_channels_dim: int = 16
    updatable_channels_dim: int = 16
    select_states_penalty: float = field(default=0.0, metadata={"help": "Negative penalty."})
    updatable_states_penalty: float = field(default=0.0, metadata={"help": "Unuse penalty."})
    select_channels_penalty: float = field(default=0.0, metadata={"help": "Negative penalty."})
    updatable_channels_penalty: float = field(default=0.0, metadata={"help": "Unuse penalty."})
    ssm_warmup_method: Literal['full', 'ssm'] = field(default='full', metadata={"help": "SSM warmup method."})
    load_ours_config: int = field(default=-1, metadata={"help": "Load our config. -1: no load. non-negative: load the corresponding row."})
    select_dim_only: bool = field(default=False, metadata={"help": "Select dim only."})
    update_dim_only: bool = field(default=False, metadata={"help": "Update dim only."})
    ssm_warmup_mode: Literal['magnitude', 'loss', 'grad', 'hybrid'] = field(default='magnitude', metadata={"help": "Hard select dim mode."})

@dataclass
class DataConfigs:
    data: Literal['synthetic', 'mnist', 'cifar10'] = field(default='synthetic', metadata={"help": "Dataset."})
    target_n_layers: int = 1
    target_d_state: int = 8

class simulation_s4d:
    def __init__(
        self,
        training_configs,
        model_configs,
        peft_configs,
        data_configs,
    ):
        self.random_seed = 123
        set_seed(self.random_seed)
        
        self.training_configs = training_configs
        self.model_configs = model_configs 
        self.peft_configs = peft_configs
        self.data_configs = data_configs
        
        self.get_model = load_model(model = 's4')
        
        if self.model_configs.task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif self.model_configs.task_type == 'classification':
            self.criterion = classification_loss
        else:
            raise NotImplementedError(f"Unknown task type {self.model_configs.task_type}!")
        
        self.log_wandb = training_configs.wandb
        
        self.usable_states, self.update_states = None, None
        self.usable_channels, self.update_channels = None, None
        
        self.get_dataset()
        
        self.frozen_model = self.get_model(
            n_layers = model_configs.frozen_n_layers,
            d_model = model_configs.d_model,
            d_state = model_configs.frozen_d_state,
            dropout = model_configs.dropout,
            A_init = 'random',
        ).to(self.training_configs.device)
        
        # pretrain
        if self.training_configs.pretrain:
            self.pretrain_path = f'{root_dir}/checkpoints/pretrained_synthetic_layer_{self.model_configs.frozen_n_layers}_epoch_{self.training_configs.pretrain_epochs}_lr_{self.training_configs.pretrain_lr}_synthetic_s4.pth'
            if os.path.exists(self.pretrain_path):
                checkpoint = torch.load(
                    self.pretrain_path,
                    map_location=torch.device(self.training_configs.device),
                )
                print(f"Loading pretrained model from {self.pretrain_path}")
                if self.model_configs.task_type == 'regression':
                    print(f"Loss: {checkpoint['loss']}")
                else:
                    print(f"Accuracy: {checkpoint['acc']}")
                self.frozen_model.load_state_dict(checkpoint['model_state_dict'])
                print("Pretrained model loaded!")
            else:
                print("Pretraining the model...")
                self.optimizer = torch.optim.AdamW(
                    self.frozen_model.parameters(), 
                    lr = self.training_configs.pretrain_lr
                )
                self.train(
                    n_epochs = self.training_configs.pretrain_epochs,
                    model = self.frozen_model,
                    lp_method = 'full',
                    ssm_method = 'full',
                    pretrain = True,
                )

        self.frozen_model.eval()
        
        self.update_model = deepcopy(self.frozen_model)
        self.get_optimizer()
        
        self.update_model = self.update_model.to(self.training_configs.device)
        
    def get_dataset(self):
        set_seed(self.random_seed)
        if self.data_configs.data == 'synthetic':
            self.target_model = self.get_model(
                n_layers = self.data_configs.target_n_layers,
                d_model = self.model_configs.d_model,
                d_state = self.data_configs.target_d_state,
                dropout = self.model_configs.dropout,
                A_init = 'random',
            ).to(self.training_configs.device)
            self.target_model.eval()
        elif self.data_configs.data == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(1, 784).t())
            ])
            transform_train = transform_test = transform

            trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform_train)
            trainset, _ = split_train_val(trainset, val_split=0.1)

            valset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform_test)
            _, valset = split_train_val(valset, val_split=0.1)

            testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform_test)

            d_input = 1
            d_output = 10
        
    def get_train_params(self):
        num_tunable_params = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)
        self.train_params = num_tunable_params
        
        self.total_params = sum(p.numel() for p in self.frozen_model.parameters())
        
        self.train_params_percent = num_tunable_params / self.total_params * 100
        
        if self.log_wandb:
            wandb.config.train_params = self.train_params
            wandb.config.total_params = self.total_params
            wandb.config.train_params_percent = self.train_params_percent
            wandb.config.model = 's4'
            
        print("########"*3)
        print('## Trainable Params:')
        print("########"*3)
        print(f'| Trainable Params: {self.train_params}')
        print(f'| Total Params: {self.total_params}')
        print(f'| Trainable Params Percent: {self.train_params_percent}')
        print("########"*3)
            
            # else:
            #     existing_wandb_run.config['train_params'] = self.train_params
            #     existing_wandb_run.config['total_params'] = self.total_params
            #     existing_wandb_run.config['train_params_percent'] = self.train_params_percent
            #     existing_wandb_run.config['overwrite'] = self.training_configs.overwrite
            #     existing_wandb_run.update()
            #     print('Configs updated!')
        
    def get_optimizer(
        self,
    ): 
        params_ft = []         
            
        if self.peft_configs.ssm_method == 'ours':
            params_ft.extend(self.get_ssm_our_adapter())
            
        param_dict = dict(self.update_model.named_parameters())
        for name, param in param_dict.items():
            if 'output_linear.weight' in name:
                if self.peft_configs.lp_method == 'lora':
                    params_ft.extend(
                        self.get_lp_lora_adapter(
                            param_name = name,
                            lora_rank = self.peft_configs.lp_lora_rank,
                        )
                    )
                elif self.peft_configs.lp_method == 'ours':
                    if self.peft_configs.ssm_method != 'ours': 
                        raise ValueError(f"Our method for linear projection requires our method for SSM, not {self.peft_configs.ssm_method}! ")
                    
                    params_ft.append(
                        self.get_lp_our_adapter(
                            param_name = name,
                        )
                    )
                elif self.peft_configs.lp_method == 'freeze':
                    continue
                elif self.peft_configs.lp_method == 'full':
                    params_ft.append(param)
                else:
                    raise NotImplementedError(f"Unknown peft method for linear projection {self.peft_configs.lp_method}!")
                    
            elif 'output_linear.bias' in name:
                if self.peft_configs.lp_method == 'freeze':
                    continue
                elif self.peft_configs.lp_method in ['full', 'lora', 'ours']:
                    params_ft.append(param)
                else:
                    raise NotImplementedError(f"Unknown peft method for linear projection {self.peft_configs.lp_method}!")
                    
            elif 'D' in name: 
                if self.peft_configs.D_method == 'freeze':
                    continue
                elif self.peft_configs.D_method == 'full':
                    params_ft.append(param)
                else:
                    raise NotImplementedError(f"Unknown peft method for D {self.peft_configs.D_method}!")
            elif 'kernel.log_dt' in name:
                if self.peft_configs.ssm_method == 'freeze':
                    continue
                elif self.peft_configs.ssm_method in ['full', 'ours', 'lora']:
                    params_ft.append(param)
                else:
                    raise NotImplementedError(f"Unknown peft method for kernel {self.peft_configs.ssm_method}!")
            elif 'kernel' in name:
                if self.peft_configs.ssm_method in ['freeze', 'ours', 'lora']:
                    continue
                elif self.peft_configs.ssm_method == 'full':
                    params_ft.append(param)
                else:
                    raise NotImplementedError(f"Unknown peft method for kernel {self.peft_configs.ssm_method}!")
            else:
                raise ValueError(f"Unknown parameter {name}!")
            
        self.optimizer = torch.optim.AdamW(params_ft, lr = self.training_configs.lr)
        
        self.get_train_params()
                                       
    def get_lp_lora_adapter(
        self,
        param_name,
        lora_rank,
    ):
        _, module_layer, module_name, _ = param_name.split('.')
        module = getattr(self.update_model.layers[int(module_layer)], module_name)
        module.adapter = nn.Sequential(
            nn.Linear(module.in_features, lora_rank, bias = False),
            nn.Linear(lora_rank, module.out_features, bias = False),
        )
        nn.init.zeros_(module.adapter[1].weight)
        
        ft_params = list(module.adapter.parameters())
            
        return ft_params
    
    def get_lp_our_adapter(
        self,
        param_name,
    ):
        _, module_layer, module_name, _ = param_name.split('.')
        module_layer = int(module_layer)
        module = getattr(self.update_model.layers[module_layer], module_name)

        if self.model_configs.model_mode == 'theory':
            adapter_params = torch.zeros(
                self.model_configs.d_model,
                len(self.update_channels[module_layer]),
            )
            

        elif self.model_configs.model_mode == 'default':
            adapter_params = torch.zeros(
                self.model_configs.d_model,
                len(self.update_channels[module_layer]) * 2,
            )
        
        else:
            raise ValueError(f"Invalid model_mode {self.model_configs.model_mode}!")
        
        module.adapter = nn.Parameter(adapter_params)
        module.update_channels = self.update_channels[module_layer]
        
        return module.adapter
    
    def get_ssm_lora_adapter(self):
        params = []
        for layer_idx, layer in enumerate(self.update_model.layers):
            layer.kernel.A_imag_adapter = [
                nn.Parameter(torch.zeros(
                    (self.model_configs.d_model, self.peft_configs.ssm_lora_rank),
                    dtype = torch.float32,
                    device=self.training_configs.device,
                )),
                nn.Parameter(torch.randn(
                    (self.peft_configs.ssm_lora_rank, self.model_configs.frozen_d_state),
                    dtype = torch.float32,
                    device=self.training_configs.device,
                )),
            ]
            layer.kernel.log_A_real_adapter = [
                nn.Parameter(torch.zeros(
                    (self.model_configs.d_model, self.peft_configs.ssm_lora_rank),
                    dtype = torch.float32,
                    device=self.training_configs.device,
                )),
                nn.Parameter(torch.randn(
                    (self.peft_configs.ssm_lora_rank, self.model_configs.frozen_d_state),
                    dtype = torch.float32,
                    device=self.training_configs.device,
                )),
            ]
            layer.kernel.C_adapter = [
                nn.Parameter(torch.view_as_real(torch.zeros(
                    (self.model_configs.d_model, self.peft_configs.ssm_lora_rank),
                    dtype = torch.cfloat,
                    device=self.training_configs.device,
                ))),
                nn.Parameter(torch.view_as_real(torch.randn(
                    (self.peft_configs.ssm_lora_rank, self.model_configs.frozen_d_state),
                    dtype = torch.cfloat,
                    device=self.training_configs.device,
                ))),
            ]
            
            params.extend(layer.kernel.A_imag_adapter)
            params.extend(layer.kernel.log_A_real_adapter)
            params.extend(layer.kernel.C_adapter)
            
            # layer.kernel.C.requires_grad = False
            # layer.kernel.log_A_real.requires_grad = False
            # layer.kernel.A_imag.requires_grad = False  
            
        return params
        
    def get_ssm_our_adapter(self):
        
        # find nonzero dims and update dims
        if self.peft_configs.our_mode == 'hard':
            self.get_ssm_our_adapter_hard()
        elif self.peft_configs.our_mode == 'soft':
            self.get_ssm_our_adapter_soft()
        else:
            raise NotImplementedError(f"Unknown our method {self.peft_configs.our_method}!")
        
        # reset the model and add adapters
        self.update_model = deepcopy(self.frozen_model).to(self.training_configs.device)
        
        params = []
        for layer_idx, layer in enumerate(self.update_model.layers):
            not_usable_channels = list(set(list(range(self.model_configs.d_model))) - set(self.usable_channels[layer_idx]))
            with torch.no_grad():
                # set the unusable dims of C to be zero
                layer.kernel.C[not_usable_channels] = torch.zeros_like(layer.kernel.C[not_usable_channels])
                
            layer.kernel.C_adapter, layer.kernel.log_A_real_adapter, layer.kernel.A_imag_adapter = [], [], []

            for channel_idx in self.update_channels[layer_idx]:
                
                not_usable_states = list(set(list(range(self.model_configs.frozen_d_state))) - set(self.usable_states[layer_idx][channel_idx]))
                with torch.no_grad():
                    # set the unusable dims of C to be zero
                    layer.kernel.C[channel_idx][not_usable_states] = torch.zeros_like(layer.kernel.C[channel_idx][not_usable_states])
                        
                C_adapter_params = torch.zeros(
                    len(self.update_states[layer_idx][channel_idx]), 
                    dtype = torch.cfloat, 
                    device = self.training_configs.device,
                )
                log_A_real_adapter_params = torch.zeros(
                    len(self.update_states[layer_idx][channel_idx]), 
                    device = self.training_configs.device,
                )
                A_imag_adapter_params = torch.zeros(
                    len(self.update_states[layer_idx][channel_idx]), 
                    device = self.training_configs.device,
                )
                
                layer.kernel.C_adapter.append(nn.Parameter(torch.view_as_real(C_adapter_params)))
                params.append(layer.kernel.C_adapter[-1])
                
                layer.kernel.log_A_real_adapter.append(nn.Parameter(log_A_real_adapter_params))
                params.append(layer.kernel.log_A_real_adapter[-1])
                
                layer.kernel.A_imag_adapter.append(nn.Parameter(A_imag_adapter_params))
                params.append(layer.kernel.A_imag_adapter[-1])
                
            layer.kernel.C.requires_grad = False
            layer.kernel.log_A_real.requires_grad = False
            layer.kernel.A_imag.requires_grad = False  
            
        return params
    
    def get_ssm_params(self, keep_A = False):
        ssm_params = []
        for layer in self.update_model.layers:
            for name, param in layer.named_parameters():
                if not keep_A:
                    if ('kernel.A' in name) and (not 'adapter' in name):
                        continue
                if 'kernel' in name:
                    ssm_params.append(param)
                    
            ssm_params.append(layer.D)
            
        return ssm_params
    
    def warmup(self):
        if self.peft_configs.our_mode == 'hard':
            loss_func = self.criterion
            keep_A = True
        elif self.peft_configs.our_mode == 'soft':
            loss_func = lambda pred_y, target_y: self.criterion(pred_y,target_y) + self.get_l1_penalty()
            keep_A = False
        
        if self.peft_configs.ssm_warmup_method == 'full':
            optimizer = torch.optim.AdamW(self.update_model.parameters(), lr = self.training_configs.warmup_lr)
        elif self.peft_configs.ssm_warmup_method == 'ssm':
            optimizer = torch.optim.AdamW(self.get_ssm_params(keep_A=keep_A), lr = self.training_configs.warmup_lr)
            
        tqdm_obj = tqdm(range(self.training_configs.warmup_epochs))
        for _ in tqdm_obj:
            X = self.generate_input()
            
            target_Y = self.target_model(X).detach()
            pred_Y = self.update_model(
                x = X, 
                updatable_channels_dim = None, 
                updatable_states_dim = None, 
                lp_method = 'full',
                ssm_method = 'full',
            )
            
            loss = loss_func(pred_Y, target_Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            tqdm_obj.set_description(f"| Warmup | Loss: {loss.item():.4f}")
            
    def select_dims_hard(
        self, 
        layer,
        layer_idx,
        channel_indices,
        select_dims,
        updatable_dims,
        dim_size,
    ):

        A_real = -torch.exp(layer.kernel.log_A_real)[channel_indices]
        A_imag = layer.kernel.A_imag[channel_indices]
        
        # choose non-zero dims
        C = torch.view_as_complex(layer.kernel.C[channel_indices])
        A = A_real + A_imag
        dt = torch.exp(layer.kernel.log_dt[channel_indices])
        dtA = A * dt.unsqueeze(-1)
        C_final = C * (torch.exp(dtA) - 1.) / A
        AC = C_final * torch.exp(dtA) ** (self.training_configs.seq_len//2) # (d_model, d_state)
        
        if self.peft_configs.our_mode == 'hard':
            # torch.norm(AC[i]) -> scalar
            usable_dims = sorted(list(range(dim_size)), key = lambda i: torch.norm(AC[i]), reverse = True)[:select_dims]
        elif self.peft_configs.our_mode == 'soft':
            # TODO: update the penalty loss, etc
            usable_dims = list(range(dim_size))
        else:
            raise NotImplementedError(f"Unknown mode of our method {self.peft_configs.our_mode}!")
            
        # choose the updatable dimensions based on the update of A
        A = torch.stack((A_real, A_imag), dim=-1) # (d_model, d_state, 2)
        
        A_real_frozen = torch.exp(self.frozen_model.layers[layer_idx].kernel.log_A_real)[channel_indices]
        A_imag_frozen = self.frozen_model.layers[layer_idx].kernel.A_imag[channel_indices]
        A_frozen = torch.stack((A_real_frozen, A_imag_frozen), dim=-1)
        
        A_diff = A - A_frozen # (d_model, d_state, 2)
        update_dims = sorted(usable_dims, key = lambda i: torch.norm(A_diff[i]), reverse = True)[:updatable_dims]
            
        return usable_dims, update_dims
    
    def select_all_dims(self):
        usable_states, usable_channels = {}, {}
        for layer_idx, layer in enumerate(self.update_model.layers):
            usable_states[layer_idx] = {}
            usable_channels[layer_idx] = list(range(self.model_configs.d_model))
            for channel_idx in range(self.model_configs.d_model):
                usable_states[layer_idx][channel_idx] = list(range(self.model_configs.frozen_d_state))
                
        return usable_states, usable_channels
    
    def get_ssm_our_adapter_hard_magnitude(
        self,
        updatable_states_dim = None,
        updatable_channels_dim = None,
    ):
        self.warmup()
                
        usable_states, update_states = {}, {}
        usable_channels, update_channels = {}, {}
        for layer_idx, layer in enumerate(self.update_model.layers):
            # find the important rows: usable dims from C & dims need update from A
            usable_states[layer_idx], update_states[layer_idx] = {}, {}
            usable_channels[layer_idx], update_channels[layer_idx] = self.select_dims_hard(
                layer,
                layer_idx,
                list(range(self.model_configs.d_model)),
                self.peft_configs.select_channels_dim,
                updatable_channels_dim,
                dim_size = self.model_configs.d_model,
            )
            
            for channel_idx in usable_channels[layer_idx]:
                usable_states[layer_idx][channel_idx], _ = self.select_dims_hard(
                    layer,
                    layer_idx,
                    channel_idx,
                    self.peft_configs.select_states_dim,
                    0,
                    dim_size = self.model_configs.frozen_d_state,
                )
            
            for channel_idx in update_channels[layer_idx]:
                
                usable_states[layer_idx][channel_idx], update_states[layer_idx][channel_idx] = self.select_dims_hard(
                    layer,
                    layer_idx,
                    channel_idx,
                    self.peft_configs.select_states_dim,
                    updatable_states_dim,
                    dim_size = self.model_configs.frozen_d_state,
                )
        self.usable_states, self.update_states = usable_states, update_states
        self.usable_channels, self.update_channels = usable_channels, update_channels
    
    def get_ssm_our_adapter_hard_loss(
        self,
        update_only = False,
        update_states_default = None,
        update_channels_default = None,
        usable_states_default = None,
        usable_channels_default = None,
    ):
        # try all possible configurations
        usable_states, update_states = {}, {}
        usable_channels, update_channels = {}, {}
        
        if not update_only:
            # choose usable channels and states
            for layer_idx, layer in enumerate(self.update_model.layers):
                usable_states[layer_idx] = {}
                
                log_A_real = layer.kernel.log_A_real.clone()
                A_imag = layer.kernel.A_imag.clone()
                C = layer.kernel.C.clone()
            
                losses = {}
                for channel_idx in range(self.model_configs.d_model):
                    with torch.no_grad():
                        layer.kernel.log_A_real[channel_idx] = torch.zeros_like(layer.kernel.log_A_real[channel_idx])
                        layer.kernel.A_imag[channel_idx] = torch.zeros_like(layer.kernel.A_imag[channel_idx])
                        layer.kernel.C[channel_idx] = torch.zeros_like(layer.kernel.C[channel_idx])
                        
                    X = self.generate_input()
                    target_Y = self.target_model(X).detach()
                    pred_Y = self.update_model(
                        x = X, 
                        updatable_channels_dim = None, 
                        updatable_states_dim = None, 
                        lp_method = 'full',
                        ssm_method = 'full',
                    )
                    loss = self.criterion(pred_Y, target_Y)
                    losses[channel_idx] = loss.cpu().item()
                    
                    with torch.no_grad():
                        layer.kernel.log_A_real[channel_idx] = log_A_real[channel_idx]
                        layer.kernel.A_imag[channel_idx] = A_imag[channel_idx]
                        layer.kernel.C[channel_idx] = C[channel_idx]
                    
                usable_channels[layer_idx] = sorted(
                    range(self.model_configs.d_model), 
                    key = lambda channel_idx: losses[channel_idx], 
                    reverse = True,
                )[:self.peft_configs.select_channels_dim]
                notusable_channels = list(set(range(self.model_configs.d_model)) - set(usable_channels[layer_idx]))
                
                for channel_idx in notusable_channels:
                    with torch.no_grad():
                        layer.kernel.log_A_real[channel_idx] = torch.zeros_like(layer.kernel.log_A_real[channel_idx])
                        layer.kernel.A_imag[channel_idx] = torch.zeros_like(layer.kernel.A_imag[channel_idx])
                        layer.kernel.C[channel_idx] = torch.zeros_like(layer.kernel.C[channel_idx])
                    
                    
                for channel_idx in usable_channels[layer_idx]:
                    losses = {}
                    usable_states[layer_idx][channel_idx] = []
                    for state_idx in range(self.model_configs.frozen_d_state):
                        with torch.no_grad():
                            layer.kernel.log_A_real[channel_idx][state_idx] = torch.zeros_like(layer.kernel.log_A_real[channel_idx][state_idx])
                            layer.kernel.A_imag[channel_idx][state_idx] = torch.zeros_like(layer.kernel.A_imag[channel_idx][state_idx])
                            layer.kernel.C[channel_idx][state_idx] = torch.zeros_like(layer.kernel.C[channel_idx][state_idx])
                        
                        X = self.generate_input()
                        target_Y = self.target_model(X).detach()
                        pred_Y = self.update_model(
                            x = X, 
                            updatable_channels_dim = None, 
                            updatable_states_dim = None, 
                            lp_method = 'full',
                            ssm_method = 'full',
                        )
                        loss = self.criterion(pred_Y, target_Y)
                        losses[state_idx] = loss.cpu().item()
                        
                        with torch.no_grad():
                            layer.kernel.log_A_real[channel_idx][state_idx] = log_A_real[channel_idx][state_idx]
                            layer.kernel.A_imag[channel_idx][state_idx] = A_imag[channel_idx][state_idx]
                            layer.kernel.C[channel_idx][state_idx] = C[channel_idx][state_idx]
                    usable_states[layer_idx][channel_idx] = sorted(
                        range(self.model_configs.frozen_d_state), 
                        key = lambda state_idx: losses[state_idx], 
                        reverse = True,
                    )[:self.peft_configs.select_states_dim]
                    notusable_states = list(set(range(self.model_configs.frozen_d_state)) - set(usable_states[layer_idx][channel_idx]))
                    
                    for state_idx in notusable_states:
                        with torch.no_grad():
                            layer.kernel.log_A_real[channel_idx][state_idx] = torch.zeros_like(layer.kernel.log_A_real[channel_idx][state_idx])
                            layer.kernel.A_imag[channel_idx][state_idx] = torch.zeros_like(layer.kernel.A_imag[channel_idx][state_idx])
                            layer.kernel.C[channel_idx][state_idx] = torch.zeros_like(layer.kernel.C[channel_idx][state_idx])
                    
            self.usable_states, self.usable_channels = usable_states, usable_channels
        else:
            self.usable_states, self.usable_channels = usable_states_default, usable_channels_default
            
        # choose updatable channels and states            
        for layer_idx, layer in enumerate(self.update_model.layers):
            update_channels[layer_idx] = []
            update_states[layer_idx] = dict()
            
        for layer_idx, layer in enumerate(self.update_model.layers):
            # choose updatable channels
            losses = {}
            iter_channels = list(set(self.usable_channels[layer_idx]) & set(update_channels_default[layer_idx]))
            for channel_idx in iter_channels:
                test_update_channels = deepcopy(update_channels)
                test_update_channels[layer_idx] = [channel_idx]
                test_update_states = deepcopy(update_states)
                test_update_states[layer_idx] = {channel_idx: self.usable_states[layer_idx][channel_idx]}
                
                X = self.generate_input()
                target_Y = self.target_model(X).detach()
                pred_Y = self.update_model(
                    x = X, 
                    updatable_channels_dim = test_update_channels, 
                    updatable_states_dim = test_update_states, 
                    lp_method = 'ours',
                    ssm_method = 'ours',
                )
                loss = self.criterion(pred_Y, target_Y)
                losses[channel_idx] = loss.cpu().item()
                
            update_channels[layer_idx] = sorted(
                iter_channels, 
                key = lambda channel_idx: losses[channel_idx]
            )[:self.peft_configs.updatable_channels_dim]
            
            # choose updatable states
            for channel_idx in update_channels[layer_idx]:
                
                test_update_channels = deepcopy(update_channels)
                test_update_channels[layer_idx] = [channel_idx]
                test_update_states = deepcopy(update_states)
                
                losses = {}
                iter_states = list(set(self.usable_states[layer_idx][channel_idx]) & set(update_states_default[layer_idx][channel_idx]))
                for state_idx in iter_states:
                    test_update_states[layer_idx][channel_idx] = [state_idx]
                
                    X = self.generate_input()
                    target_Y = self.target_model(X).detach()
                    pred_Y = self.update_model(
                        x = X, 
                        updatable_channels_dim = test_update_channels, 
                        updatable_states_dim = test_update_states, 
                        lp_method = 'ours',
                        ssm_method = 'ours'
                    )
                    loss = self.criterion(pred_Y, target_Y)
                    losses[state_idx] = loss.cpu().item()
                
                update_states[layer_idx][channel_idx] = sorted(
                    iter_states, 
                    key = lambda state_idx: losses[state_idx], 
                    reverse = True,
                )[:self.peft_configs.updatable_states_dim]
        
        self.update_states, self.update_channels = update_states, update_channels
    
    def get_ssm_our_adapter_hard_grad(
        self,
    ):
        # do not set any channels and states to zero
        self.usable_states, self.usable_channels = self.select_all_dims()
        
        # select the updatable channels and states
        self.update_channels, self.update_states = {}, {}
        
        X = self.generate_input()
        target_Y = self.target_model(X).detach()
        pred_Y = self.update_model(
            x = X, 
            updatable_channels_dim = None, 
            updatable_states_dim = None, 
            lp_method = 'full',
            ssm_method = 'full',
        )
        loss = self.criterion(pred_Y, target_Y)
        loss.backward()
    
        update_channels, update_states = {}, {}
        for layer_idx, layer in enumerate(self.update_model.layers):
            grads = {}
            for channel_idx in self.usable_channels[layer_idx]:
                # compute the gradient of the parameters in this channel and average them
                grads[channel_idx] = torch.norm(torch.cat([
                    layer.kernel.C.grad.detach()[channel_idx].flatten(),
                    torch.exp(layer.kernel.log_A_real.grad.detach()[channel_idx]).flatten(),
                    layer.kernel.A_imag.grad.detach()[channel_idx].flatten(),
                    layer.output_linear.weight.grad.detach()[:,channel_idx].flatten()
                ]))
                
            update_channels[layer_idx] = sorted(
                self.usable_channels[layer_idx], 
                key = lambda channel_idx: grads[channel_idx], 
                reverse = True,
            )[:self.peft_configs.updatable_channels_dim]
            
            update_states[layer_idx] = {}
            for channel_idx in update_channels[layer_idx]:
                grads = {}
                for state_idx in self.usable_states[layer_idx][channel_idx]:
                    grads[state_idx] = torch.norm(torch.cat([
                        layer.kernel.log_A_real.grad.detach()[channel_idx,state_idx].flatten(),
                        layer.kernel.A_imag.grad.detach()[channel_idx,state_idx].flatten(),
                        layer.kernel.C.grad.detach()[channel_idx,state_idx].flatten(),
                    ]))
                    
                update_states[layer_idx][channel_idx] = sorted(
                    self.usable_states[layer_idx][channel_idx], 
                    key = lambda state_idx: grads[state_idx], 
                    reverse = True,
                )[:self.peft_configs.updatable_states_dim]
            
        self.update_channels, self.update_states = update_channels, update_states
    
    
    def get_ssm_our_adapter_hard(
        self,
    ):
        if self.peft_configs.load_ours_config >= 0:
            mode = 'select' if self.peft_configs.select_dim_only else 'update'
            config_dir = f'{root_dir}/configs/s4/ours_config'
            idx = self.peft_configs.load_ours_config
            self.usable_states, self.update_states = load_pickle(f'{config_dir}/{mode}_dim_only_usable_states.pickle')[idx], load_pickle(f'{config_dir}/{mode}_dim_only_update_states.pickle')[idx]
            self.usable_channels, self.update_channels = load_pickle(f'{config_dir}/{mode}_dim_only_usable_channels.pickle')[idx], load_pickle(f'{config_dir}/{mode}_dim_only_update_channels.pickle')[idx]
            return
        
        if self.peft_configs.select_states_dim < self.peft_configs.updatable_states_dim:
            raise ValueError(f"select_states_dim {self.peft_configs.select_states_dim} < updatable_states_dim {self.peft_configs.updatable_states_dim}!")
        if self.peft_configs.select_channels_dim < self.peft_configs.updatable_channels_dim:
            raise ValueError(f"select_channels_dim {self.peft_configs.select_channels_dim} < updatable_channels_dim {self.peft_configs.updatable_channels_dim}!")
        
        if self.peft_configs.ssm_warmup_mode == 'magnitude':
            self.get_ssm_our_adapter_hard_magnitude(
                updatable_states_dim = self.peft_configs.updatable_states_dim,
                updatable_channels_dim = self.peft_configs.updatable_channels_dim,
            )     
                    
        elif self.peft_configs.ssm_warmup_mode == 'loss':
            usable_states_default, usable_channels_default = self.select_all_dims()
            update_channels_default, update_states_default = deepcopy(usable_channels_default), deepcopy(usable_states_default)
            self.get_ssm_our_adapter_hard_loss(
                update_only = False,
                update_states_default = update_states_default,
                update_channels_default = update_channels_default,
                usable_states_default = usable_states_default,
                usable_channels_default = usable_channels_default,
            )
        
        elif self.peft_configs.ssm_warmup_mode == 'grad':
            self.get_ssm_our_adapter_hard_grad()
            
        elif self.peft_configs.ssm_warmup_mode == 'hybrid':
            update_states_default, update_channels_default = self.select_all_dims()
            self.get_ssm_our_adapter_hard_magnitude(
                updatable_states_dim = self.peft_configs.updatable_states_dim+1,
                updatable_channels_dim = self.peft_configs.updatable_channels_dim+1,
            ) 
            self.get_ssm_our_adapter_hard_loss(
                update_only = True,
                update_states_default = update_states_default,
                update_channels_default = update_channels_default,
                usable_states_default = self.usable_states,
                usable_channels_default = self.usable_channels,
            )
        
        if self.peft_configs.select_dim_only:
            self.get_our_dim_acc(mode = 'select')
            
        if self.peft_configs.update_dim_only:
            self.get_our_dim_acc(mode = 'update')
                      
    def get_our_dim_acc(self, mode):
        mode_str = 'usable' if mode == 'select' else 'update'
        
        results = {
            'usable_states': self.usable_states,
            'update_states': self.update_states,
            'usable_channels': self.usable_channels,
            'update_channels': self.update_channels,
        }
        
        # get accuracy: compare with the ground truth
        config_dir = f'{root_dir}/configs/s4/ours_config'
        sorted_idxs = load_pickle(f'{root_dir}/analysis/{mode}_dim_sorted_idxs.pickle')
        best_config_idx = int(sorted_idxs[0])
        
        all_dim_configs = {
            'usable_states': load_pickle(f'{config_dir}/{mode}_dim_only_usable_states.pickle'),
            'update_states': load_pickle(f'{config_dir}/{mode}_dim_only_update_states.pickle'),
            'usable_channels': load_pickle(f'{config_dir}/{mode}_dim_only_usable_channels.pickle'),
            'update_channels': load_pickle(f'{config_dir}/{mode}_dim_only_update_channels.pickle'),
        }
        
        ground_truth = {
            'usable_states': all_dim_configs['usable_states'][best_config_idx],
            'update_states': all_dim_configs['update_states'][best_config_idx],
            'usable_channels': all_dim_configs['usable_channels'][best_config_idx],
            'update_channels': all_dim_configs['update_channels'][best_config_idx],
        }
        # compare the results and ground truth, if the channel is not matched correct, consider all corresponding states are not matched
        corr, tot = 0, self.model_configs.frozen_n_layers * self.peft_configs.select_channels_dim * self.peft_configs.select_states_dim
        for layer_idx in self.usable_channels:
            our_channels = set(results[f'{mode_str}_channels'][layer_idx])  
            gt_channels = set(ground_truth[f'{mode_str}_channels'][layer_idx])
            corr_channels = our_channels & gt_channels
            for channel in corr_channels:
                our_states = set(results[f'{mode_str}_states'][layer_idx][channel])
                gt_states = set(ground_truth[f'{mode_str}_states'][layer_idx][channel])
                corr_states = our_states & gt_states
                corr = corr + len(corr_states)

        print(f"{mode_str.capitalize()} Dimension Selection Accuracy: {corr/tot: .2f}")
        
        # get the rank: compare the resulted dimensions with all possible combinations and find its index, and get the rank
        same_channel_idxs = []
        for idx, config in enumerate(all_dim_configs[f'{mode_str}_channels']):
            same_idx = True
            for layer_idx in config:
                if set(config[layer_idx]) != set(results[f'{mode_str}_channels'][layer_idx]):
                    same_idx = False
                    break
                
            if same_idx: same_channel_idxs.append(idx)
                    
        final_idxs = deepcopy(same_channel_idxs)    
        for idx in same_channel_idxs:
            config = all_dim_configs[f'{mode_str}_states'][idx]
            remove_idx = False
            for layer_idx in config:
                for channel_idx in config[layer_idx]:
                    if set(config[layer_idx][channel_idx]) != set(results[f'{mode_str}_states'][layer_idx][channel_idx]):
                        remove_idx = True
                        break
                if remove_idx: break
            if remove_idx: final_idxs.remove(idx)
            
        if len(final_idxs) != 1: 
            raise ValueError('No configuration matched the current selection. Something is wrong!')
        final_idx = final_idxs[0]
        
        rank = sorted_idxs.index(final_idx)
        print(f"{mode_str.capitalize()} Dimension Selection Rank: {rank} out of {len(sorted_idxs)} (top {100*rank/len(sorted_idxs):.2f}%)")
        
        exit(0)
    
    def get_l1_penalty(
        self,
    ):
        select_states_penalty_term, updatable_states_penalty_term = 0, 0
        for layer in self.update_model.layers:
            select_states_penalty_term += complex_l1_penalty(layer.kernel.C[:,:,0]*torch.exp(layer.kernel.log_A_real + layer.kernel.log_A_real_adapter), layer.kernel.C[:,:,1]*(layer.kernel.A_imag + layer.kernel.A_imag_adapter))
            updatable_states_penalty_term += complex_l1_penalty(torch.exp(layer.kernel.log_A_real_adapter), layer.kernel.A_imag_adapter)
            
        return self.peft_configs.select_states_penalty * select_states_penalty_term + self.peft_configs.updatable_states_penalty * updatable_states_penalty_term
    
    def get_ssm_our_adapter_soft(
        self,
    ):

        # add adapters
        for layer in self.update_model.layers:            
            layer.kernel.log_A_real_adapter = nn.Parameter(torch.zeros_like(layer.kernel.log_A_real))
            layer.kernel.A_imag_adapter = nn.Parameter(torch.zeros_like(layer.kernel.A_imag))
        
        self.warmup()
        
        # check the nonzero dims and updatable dims
        usable_states, update_states = {}, {}
        for layer_idx, layer in enumerate(self.update_model.layers): 
            usable_states[layer_idx], update_states[layer_idx] = {}, {}
            for channel_idx in range(self.model_configs.d_model):
                
                # choose non-zero dims
                C = layer.kernel.C[channel_idx]
                
                A_real = torch.exp(layer.kernel.log_A_real)[channel_idx]
                A_imag = layer.kernel.A_imag[channel_idx]
                A = torch.stack((A_real, A_imag), dim=1)
                
                A_real_adapter = torch.exp(layer.kernel.log_A_real_adapter)[channel_idx]
                A_imag_adapter = layer.kernel.A_imag_adapter[channel_idx]
                A_adapter = torch.stack((A_real_adapter, A_imag_adapter), dim=1)
                
                AC = C*(A + A_adapter)
                nonzero_rows = torch.any(AC != 0, dim=1)
                nonzero_states = torch.nonzero(nonzero_rows, as_tuple=True)[0]
                usable_states[layer_idx][channel_idx] = nonzero_states.cpu().numpy()
                
                # choose the updatable dimensions            
                update_rows = torch.any(A_adapter != 0, dim=1)
                
                update_states[layer_idx][channel_idx] = torch.nonzero(update_rows, as_tuple=True)[0].cpu().numpy()
                
                # find the intersection of update_states and usable_states
                update_states[layer_idx][channel_idx] = list(set(update_states[layer_idx][channel_idx]) & set(usable_states[layer_idx][channel_idx]))
                
        self.usable_states, self.update_states = usable_states, update_states    
        
    def generate_input(self):        
        X = torch.randint(
            0, 
            10, 
            (self.training_configs.batch_size, self.model_configs.d_model, self.training_configs.seq_len)
        ).to(self.training_configs.device)
        
        # X = 10*torch.abs(torch.randn(self.training_configs.batch_size, self.model_configs.d_model, self.training_configs.seq_len)).cuda()
        
        return X 
    
    def train(
        self, 
        n_epochs,
        model,
        lp_method,
        ssm_method,
        pretrain = False,
    ):
        set_seed(self.random_seed)
        
        model.train()
        
        loss_epoch, best_loss, best_accuracy = [], float('inf'), 0.0
            
        tqdm_obj = tqdm(range(n_epochs))
        for _ in tqdm_obj:
            X = self.generate_input()
            
            target_Y = self.target_model(X).detach()
            pred_Y = model(
                x = X, 
                updatable_channels_dim = self.update_channels, 
                updatable_states_dim = self.update_states, 
                lp_method = lp_method,
                ssm_method = ssm_method,
            )
                
            loss = self.criterion(pred_Y, target_Y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            loss_epoch.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                if pretrain and self.model_configs.task_type == 'regression':
                    self.checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'loss': best_loss,
                    }
            
            if self.log_wandb:
                wandb.log({'loss': loss.item()})
                wandb.log({'best_loss': best_loss})
                
            pred_label = torch.argmax(pred_Y, dim=1)
            target_label = torch.argmax(target_Y, dim=1)
            accuracy = ((pred_label == target_label).float().mean() * 100).item()
                
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if pretrain and self.model_configs.task_type == 'classification': 
                    self.checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'acc': best_accuracy,
                    }
                    
            if self.log_wandb:
                wandb.log({'accuracy': accuracy})
                wandb.log({'best_accuracy': best_accuracy})
                
            tqdm_obj.set_description(f"| Train | Loss: {loss.item():.4f} | Acc: {accuracy:.2f}% | Best Acc: {best_accuracy:.2f}%")
                    
                
        print(f"Best Loss: {best_loss}")
        print(f"Best Accuracy: {best_accuracy}")
        
        if pretrain:
            if self.model_configs.task_type == 'regression':
                print(f"Saving the model with best loss {best_loss}...")
            else:
                print(f"Saving the model with best accuracy {best_accuracy}...")
            torch.save(self.checkpoint, self.pretrain_path)
        
        model.eval()
                
if '__main__' == __name__:
    parser = transformers.HfArgumentParser((TrainingConfigs, ModelConfigs, PeftConfigs, DataConfigs))
    all_configs = parser.parse_args_into_dataclasses()
    
    configs, configs_dicts = {}, {}
    for i, mode in enumerate(['training', 'model', 'peft', 'data']):
        configs[mode] = all_configs[i]
        configs_dicts[mode] = asdict(all_configs[i])
        
        # print experiment configurations
        print("########"*3)
        print(f'## {mode} configs:')
        print("########"*3)
        for key, value in configs_dicts[mode].items():
            print(f'| {key}: {value}')
        print("########"*3)
        print()
        
    wandb_configs = dict()
    for mode in configs:
        wandb_configs = {**wandb_configs, **configs_dicts[mode]}
        
    if configs['training'].wandb:
        wandb_config = wandb_configs
        wandb_entity, wandb_proj = WANDB_INFO['entity'], WANDB_INFO['project']
        
        existing_wandb_run = init_wandb(
            wandb_entity = wandb_entity,
            wandb_proj = wandb_proj,
            wandb_config = wandb_config,
            overwrite = configs['training'].overwrite,
        )
    
    sim = simulation_s4d(
        training_configs=configs['training'],
        model_configs=configs['model'],
        peft_configs=configs['peft'],
        data_configs=configs['data']
    )
    
    train = False
    if configs['training'].wandb and existing_wandb_run is None:
        train = True
    elif configs['training'].wandb and configs['training'].overwrite:
        train = True 
    elif not configs['training'].wandb:
        train = True
    
    if train:
        sim.train(
            n_epochs = configs['training'].n_epochs,
            model = sim.update_model,
            lp_method = configs['peft'].lp_method,
            ssm_method = configs['peft'].ssm_method,
            pretrain = False,
        )
    
    if configs['training'].wandb and existing_wandb_run is None:
        wandb.finish()
        
        
        