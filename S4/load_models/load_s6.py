import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# do not delete this line
sys.path.append(os.path.join(root_dir, 'models'))

from torch import nn
import torch
import math
import torch.nn.functional as F
from einops import rearrange, repeat

from helper import adapterLinear

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class S6(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        dt_rank,
        device = 'cuda',
        dtype = torch.float32,
        init_mode = 'random',
        model_mode = 'theory',
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs,
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        
        # S4D real initialization
        if init_mode == 'default':
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_model,
            ).contiguous()
            # D "skip" parameter
            self.D = nn.Parameter(torch.ones(self.d_model, device = device))
        elif init_mode == 'random':
            A = torch.rand(
                self.d_model, 
                self.d_state, 
                dtype = torch.float32,
                device=device
            ) * self.d_state + 1
            # D "skip" parameter
            self.D = nn.Parameter(torch.rand(self.d_model, device = device) * .8)
        else:
            raise ValueError(f"Invalid init_mode {init_mode}!")
        
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        
        self.out_proj = adapterLinear(self.d_model, self.d_model, **factory_kwargs)
        self.hidden_states = None
    
        self.A_log_adapter = None
        self.dt_proj_up_adapter, self.dt_proj_down_adapter = None, None
        self.WB_adapter, self.WC_adapter = None, None
        
        self.model_mode = model_mode
        
    def adapted_params(
        self,
        value,
        value_adapter,
        d_model,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None,
        is_dt = False,
        ssm_method = 'ours',
    ):
        # first dimension should be channel dimension
        cloned_value = value.clone()
        
        if value_adapter is not None:
            if ssm_method == 'ours':
                if updatable_channels_dim is None:
                    channel_iter = list(range(d_model))
                elif updatable_states_dim is not None:
                    channel_iter = updatable_channels_dim[layer_idx]
                else: 
                    # updatable_channels_dim is not None and updatable_states_dim is None
                    indices = updatable_channels_dim[layer_idx]
                    cloned_value[indices] += value_adapter
                    return cloned_value
                
                for i, channel_idx in enumerate(channel_iter):
                    value_adapter_i = value_adapter[i]
                    
                    if updatable_states_dim is not None:
                        if is_dt:
                            indices = updatable_states_dim[layer_idx]
                        else:
                            indices = updatable_states_dim[layer_idx][channel_idx]
                            
                        if indices: cloned_value[channel_idx, indices] += value_adapter_i

                    else:
                        cloned_value[channel_idx] += value_adapter[channel_idx]
                        
            elif ssm_method == 'lora':
                if is_dt:
                    cloned_value += value_adapter
                else:
                    cloned_value += value_adapter[0] @ value_adapter[1]
                    
            
        return cloned_value 
        
    def forward(
        self,
        hidden_states,
        ssm_method = 'ours',
        lp_method = 'ours',
        inference_params = None,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None,
        usable_channels  = None,
        usable_states = None,
    ): 
        self.hidden_states = hidden_states
        batch, seqlen, d_model = hidden_states.shape
        x = rearrange(hidden_states, "b l d -> b d l")
        
        ssm_state = None 
        if inference_params is not None:
            ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _ = self.step(hidden_states, ssm_state, lp_method)
                return out

        A_log = self.adapted_params(
            value = self.A_log,
            value_adapter = self.A_log_adapter,
            d_model = d_model, 
            updatable_channels_dim = updatable_channels_dim,
            updatable_states_dim = updatable_states_dim,
            layer_idx = layer_idx,
            ssm_method = ssm_method,
        )
        A = -torch.exp(A_log.float())  # (d_model, d_state)
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        
        dt_proj_down, WB, WC = torch.split(self.x_proj.weight.T, [self.dt_rank, self.d_state, self.d_state], dim = 1)
        
        WB = self.adapted_params(
            value = WB,
            value_adapter = self.WB_adapter,
            d_model = d_model,
            updatable_channels_dim = updatable_channels_dim,
            updatable_states_dim = updatable_states_dim,
            layer_idx = layer_idx,
            ssm_method = ssm_method,
        )
        
        WC = self.adapted_params(
            value = WC,
            value_adapter = self.WC_adapter,
            d_model = d_model,
            updatable_channels_dim = updatable_channels_dim,
            updatable_states_dim = updatable_states_dim,
            layer_idx = layer_idx,
            ssm_method = ssm_method,
        )
        
        B = F.linear(rearrange(x, "b d l -> (b l) d"), WB.T) # (B*seqlen, d_state)
        C = F.linear(rearrange(x, "b d l -> (b l) d"), WC.T) # (B*seqlen, d_state)
        
        dt_proj_up = self.dt_proj.weight # (d_model, dt_rank)
        dt = dt_proj_up @ dt_proj_down.t() # (d_model, d_model)  
        
        if self.dt_proj_up_adapter is not None and self.dt_proj_down_adapter is not None:
            dt_adapter = self.dt_proj_up_adapter @ self.dt_proj_down_adapter.T # (d_model, d_model)
        else:
            dt_adapter = None
            
        
        dt_weight = self.adapted_params(
             value = dt,
             value_adapter = dt_adapter,
             d_model = d_model,
             updatable_channels_dim = updatable_channels_dim,
             updatable_states_dim = updatable_channels_dim,
             layer_idx = layer_idx,
             is_dt = True,
             ssm_method = ssm_method,
         )
        
        dt = F.linear(rearrange(x, "b d l -> (b l) d"), dt_weight) # (B*seqlen, d_model)
        dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
                
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        if self.model_mode == 'theory':
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                torch.zeros_like(self.D).float(),
                # self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
                
            # if ssm_method == 'ours':
            #     not_usable_channels = list(set(list(range(self.d_model))) - set(usable_channels[layer_idx]))
            #     dB = torch.einsum("bdl,bnl->bdnl", dt, B)
                
            #     for channel_idx in not_usable_channels:
            #         bc_coef = torch.einsum("bdl,bdl->bl", dB[:,channel_idx,:,:], C)
            #         y[:,channel_idx,:] = y[:,channel_idx,:] - torch.einsum("bl,bl->bl", bc_coef, x[:,channel_idx,:])
                    
            #     for channel_idx in usable_channels[layer_idx]:
            #         not_usable_states = list(set(list(range(self.d_state))) - set(usable_states[layer_idx][channel_idx]))
            #         bc_coef = torch.einsum("bdl,bdl->bl", dB[:,channel_idx,not_usable_states,:], C[:,not_usable_states,:])
            #         y[:,channel_idx,:] = y[:,channel_idx,:] - torch.einsum("bl,bl->bl", bc_coef, x[:,channel_idx,:])
                
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y, lp_method)
            out = F.relu(out)
            
            out = out + self.D.to(torch.float32) * hidden_states
            
        elif self.model_mode == 'default':
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            y = F.relu(y)
            out = self.out_proj(y, lp_method)
        return out
     
    # TODO: update this   
    def step(
        self,
        hidden_states,
        ssm_state,
        lp_method,
    ):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        x = hidden_states.squeeze(1) # (B d_model)
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_model)
        A = -torch.exp(self.A_log.float())  # (d_model, d_state) 
        
        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=None, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y, lp_method)
        print('yuchen: please update step')
        return out.unsqueeze(1), ssm_state
    
    def allocate_inference_cache(
        self, 
        batch_size, 
        dtype=None, 
    ):
        device = self.out_proj.weight.device

        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return ssm_state
    
    def _get_states_from_cache(
        self,
        inference_params,
        batch_size,
        initialize_states = False,
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = ssm_state
        else:
            ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                ssm_state.zero_()
        return ssm_state
    
class DeepS6(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_state,
        dt_rank,
        device,
        init_mode,
        model_mode,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            S6(
               d_model = d_model,
               d_state = d_state,
                dt_rank = dt_rank,
                device = device, 
                init_mode = init_mode,
                model_mode = model_mode,
            ) for _ in range(n_layers)
        ])
        
    def forward(
        self,
        x,
        update_channels,
        update_states,
        ssm_method,
        lp_method,
        usable_channels,
        usable_states,
    ):
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                updatable_channels_dim = update_channels,
                updatable_states_dim = update_states,
                layer_idx = layer_idx,
                ssm_method = ssm_method,
                lp_method = lp_method,
                usable_channels = usable_channels,
                usable_states = usable_states, 
            )
        return x
    