import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# do not delete this line
sys.path.append(os.path.join(root_dir, 'models/s4'))

from models.s4.src.models.nn import DropoutNd
from torch import nn
import torch
import math
import torch.nn.functional as F
from einops import repeat
from helper import adapterLinear

import pdb

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
    
class S4DKernel(nn.Module):
    
    """Generate convolution k
    ernel from diagonal SSM
    parameters."""

    def __init__(
        self, 
        d_model, 
        N=64,
        dt_min=0.001,
        dt_max=0.1, 
        lr=None,
        A_init='random',
    ):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        if A_init == 'default':
            log_A_real = torch.log(0.5 * torch.ones(H, N))
            A_imag = math.pi * repeat(torch.arange(N), 'n -> h n', h=H)
        elif A_init == 'random':
            log_A_real = torch.log(torch.rand(H, N))
            A_imag = math.pi * torch.randint(0, N, (H, N))
        else:
            raise ValueError("Invalid A_init")
            
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
        
        self.C_adapter, self.log_A_real_adapter, self.A_imag_adapter = None, None, None 

    def adapted_params(
        self,
        value,
        value_adapter,
        d_model,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None,
        view_as_complex = False,
        ssm_method = 'ours',
    ):
        if view_as_complex:
            cloned_value = torch.view_as_complex(value.clone())
        else:
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
                    if view_as_complex:
                        value_adapter_i = torch.view_as_complex(value_adapter[i])
                    else:
                        value_adapter_i = value_adapter[i]
                    
                    if updatable_states_dim is not None:
                        indices = updatable_states_dim[layer_idx][channel_idx]   
                        if len(indices): cloned_value[channel_idx, indices] += value_adapter_i

                    else:
                        cloned_value[channel_idx] += value_adapter[channel_idx]
                        
            elif ssm_method == 'lora':
                if view_as_complex:
                    value_adapter_0 = torch.view_as_complex(value_adapter[0])
                    value_adapter_1 = torch.view_as_complex(value_adapter[1])
                else:
                    value_adapter_0 = value_adapter[0]
                    value_adapter_1 = value_adapter[1]
                    
                cloned_value += value_adapter_0 @ value_adapter_1
                    
            
        return cloned_value 
        
    def forward(
        self, 
        L,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None,
        ssm_method = 'ours',
    ):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        d_model, d_state, _ = self.C.shape
        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
                
            
        C = self.adapted_params(
            self.C,
            self.C_adapter,
            d_model,
            updatable_channels_dim,
            updatable_states_dim,
            layer_idx,
            view_as_complex = True,
            ssm_method = ssm_method,
        )
        
        log_A_real = self.adapted_params(
            self.log_A_real,
            self.log_A_real_adapter,
            d_model,
            updatable_channels_dim,
            updatable_states_dim,
            layer_idx,
            view_as_complex=False,
            ssm_method = ssm_method,
        )
        
        A_imag = self.adapted_params(
            self.A_imag,
            self.A_imag_adapter,
            d_model,
            updatable_channels_dim,
            updatable_states_dim,
            layer_idx,
            view_as_complex=False,
            ssm_method = ssm_method,
        )
                
        # Compute A with adapters
        A = -torch.exp(log_A_real) + 1j * A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C_final = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C_final, torch.exp(K)).real

        return K
    
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_state=64, 
        dropout=0.0, 
        s4_mode = 'theory',
        A_init = 'random',
        **kernel_args
    ):

        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(
            self.h, 
            N=self.n, 
            A_init = A_init,
            **kernel_args
        )

        
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        # self.output_linear = nn.Sequential(
        #     nn.Conv1d(self.h, 2*self.h, kernel_size=1),
        #     nn.GLU(dim=-2),
        # )
        
        self.s4_mode = s4_mode
        
        if self.s4_mode == 'theory':
            self.output_linear = adapterLinear(self.h, self.h)
        elif self.s4_mode == 'default':
            self.output_linear = nn.Sequential(
                adapterLinear(self.h, 2*self.h),
                nn.GLU(dim=-1), # split based on the last dimension
            )
        else:
            raise ValueError(f"Invalid s4_mode {self.s4_mode}!")

    def forward(
        self,
        u, 
        updatable_channels_dim = None,
        updatable_states_dim = None, 
        lp_method = 'ours',
        layer_idx = None,
        ssm_method = 'ours',
        **kwargs,
    ): # absorbs return_output and transformer src mask

        """ Input and output shape (B, H, L) """
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(
            L=L,
            updatable_channels_dim = updatable_channels_dim,
            updatable_states_dim = updatable_states_dim,
            layer_idx = layer_idx,
            ssm_method = ssm_method,
        ) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        if self.s4_mode == 'theory':
            y = self.output_linear(y.transpose(-1, -2), lp_method)
            y = F.relu(y)
    
        elif self.s4_mode == 'default':
            y = self.dropout(F.gelu(y))
            y = self.output_linear(y.transpose(-1, -2), lp_method)

        else:
            raise ValueError(f"Invalid s4_mode {self.s4_mode}!")

        return y.transpose(-1, -2)
    
class DeepS4D(nn.Module):
    def __init__(
        self, 
        n_layers,
        d_model, 
        d_state=64, 
        dropout=0.0, 
        s4_mode = 'theory',
        A_init = 'random',
        d_input = None,
        d_output = None,
        device = 'cuda',
        **kernel_args
    ):
        super().__init__()
        self.s4_mode = s4_mode
        if s4_mode == 'theory':
            self.layers = nn.ModuleList([
                S4D(
                    d_model = d_model,
                    d_state = d_state,
                    dropout = dropout,
                    s4_mode = s4_mode,
                    A_init = A_init, 
                ) for _ in range(n_layers)
            ])
        elif s4_mode == 'default':
            self.encoder = nn.Linear(d_input, d_model).to(device)
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            
            for _ in range(n_layers):
                self.layers.append(
                    S4D(
                        d_model = d_model, 
                        d_state = d_state, 
                        dropout = dropout, 
                        s4_mode = 'theory',
                        A_init = 'default',
                    )
                )
                self.norms.append(nn.LayerNorm(d_model))
                self.dropouts.append(dropout_fn(dropout))

            # Linear decoder
            self.decoder = nn.Linear(d_model, d_output).to(device)
        
    def forward(
        self,
        x,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        lp_method = 'full',
        ssm_method = 'ours',
    ):
        if self.s4_mode == 'theory':
            for layer_idx, layer in enumerate(self.layers):
                x = layer(
                    x, 
                    updatable_channels_dim,
                    updatable_states_dim,
                    lp_method = lp_method,
                    layer_idx = layer_idx,
                    ssm_method = ssm_method, 
                )
        elif self.s4_mode == 'default':
            x = self.encoder(x)
            x = x.transpose(-1, -2)
            for layer_idx, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
                z = x
                    
                # Apply S4 block: we ignore the state input and output
                z = layer(
                    z, 
                    updatable_channels_dim,
                    updatable_states_dim,
                    lp_method = lp_method,
                    layer_idx = layer_idx,
                    ssm_method = ssm_method,
                )
                
                # Dropout on the output of the S4 block
                z = dropout(z)
                
                # Residual connection
                x = z + x
                    
            x = x.transpose(-1,-2)
            
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)
            
            # Decode the outputs
            x = self.decoder(x) # (B, d_model) -> (B, d_output)
            
        return x
