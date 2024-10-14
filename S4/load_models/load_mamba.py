# Copyright (c) 2023, Tri Dao, Albert Gu.

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
from functools import partial

from helper import adapterLinear
import pdb


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        init_mode = 'default',
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if init_mode == "default":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif init_mode == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
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
                d=self.d_inner,
            ).contiguous()
            # D "skip" parameter
            self.D = nn.Parameter(torch.ones(self.d_inner, device = device))
        elif init_mode == 'random':
            A = torch.rand(
                self.d_inner, 
                self.d_state, 
                dtype = torch.float32,
                device=device,
            ) * self.d_state + 10
            # D "skip" parameter
            self.D = nn.Parameter(torch.rand(self.d_inner, device = device) * .8)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

        self.out_proj = adapterLinear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.A_log_adapter = None 
        self.dt_proj_up_adapter, self.dt_proj_down_adapter = None, None 
        self.WB_adapter, self.WC_adapter = None, None 
        self.in_x_adapter, self.in_z_adapter = None, None 
        self.A, self.B, self.C, self.dt = None, None, None, None
        self.ssm_bias_adapter = None 
    
    def adapted_params(
        self,
        value,
        value_adapter,
        d_inner,
        updatable_dim1 = None,
        updatable_dim2 = None,
        layer_idx = None,
        order = 2,
        method = 'ours',
    ):
        # first dimension should be channel dimension
        cloned_value = value.clone()
        
        if value_adapter is not None:
            if method == 'ours':
                if updatable_dim1 is None:
                    channel_iter = list(range(d_inner))
                elif updatable_dim2 is not None:
                    channel_iter = updatable_dim1[layer_idx]
                else: 
                    # updatable_channels_dim is not None and updatable_states_dim is None
                    indices = updatable_dim1[layer_idx]
                    cloned_value[indices] += value_adapter
                    return cloned_value
                
                for i, channel_idx in enumerate(channel_iter):
                    value_adapter_i = value_adapter[i]
                    
                    if updatable_dim2 is not None:
                        if order == 1:
                            indices = updatable_dim2[layer_idx]
                        elif order == 2:
                            indices = updatable_dim2[layer_idx][channel_idx]
                        else:
                            raise ValueError(f"order {order} is not supported")
                            
                        if indices: cloned_value[channel_idx, indices] += value_adapter_i

                    else:
                        cloned_value[channel_idx] += value_adapter[channel_idx]
                        
            elif method == 'lora':
                if order == 1:
                    cloned_value += value_adapter
                elif order == 2:
                    cloned_value += value_adapter[0] @ value_adapter[1]
                else:
                    raise ValueError(f"order {order} is not supported")
                    
            
        return cloned_value 

    def forward(
        self, 
        hidden_states, 
        ssm_method = 'ours',
        lp_method = 'ours',
        inference_params=None,
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None, 
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        in_x_weight, in_z_weight = self.in_proj.weight.chunk(2, dim=0)
        
        in_x_weight = self.adapted_params(
            value = in_x_weight,
            value_adapter = self.in_x_adapter,
            d_inner = self.d_model,
            updatable_dim1 = updatable_channels_dim,
            updatable_dim2 = None,
            layer_idx = layer_idx,
            order = 2,
            method = lp_method,
        )
        
        in_z_weight = self.adapted_params(
            value = in_z_weight,
            value_adapter = self.in_z_adapter,
            d_inner = self.d_model,
            updatable_dim1 = updatable_channels_dim,
            updatable_dim2 = None,
            layer_idx = layer_idx, 
            order = 2,
            method = lp_method,
        )
        in_proj_weight = torch.cat([in_x_weight, in_z_weight], dim=0)
        
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            in_proj_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
        A_log = self.adapted_params(
            value = self.A_log,
            value_adapter = self.A_log_adapter,
            d_inner = self.d_inner,
            updatable_dim1 = updatable_channels_dim,
            updatable_dim2 = updatable_states_dim,
            layer_idx = layer_idx, 
            order = 2,
            method = ssm_method,
        )

        A = -torch.exp(A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        dt_proj_down, WB, WC = torch.split(self.x_proj.weight.T, [self.dt_rank, self.d_state, self.d_state], dim = 1)
        
        WB = self.adapted_params(
            value = WB,
            value_adapter = self.WB_adapter,
            d_inner = self.d_inner,
            updatable_dim1 = updatable_channels_dim,
            updatable_dim2 = updatable_states_dim,
            layer_idx = layer_idx,
            order = 2,
            method = ssm_method,
        )
        
        WC = self.adapted_params(
            value = WC,
            value_adapter = self.WC_adapter,
            d_inner = self.d_inner,
            updatable_dim1 = updatable_channels_dim,
            updatable_dim2 = updatable_states_dim,
            layer_idx = layer_idx,
            order = 2,
            method = ssm_method,
        )
        
        B = F.linear(rearrange(x, "b d l -> (b l) d"), WB.T) # (B*seqlen, d_state)
        C = F.linear(rearrange(x, "b d l -> (b l) d"), WC.T) # (B*seqlen, d_state)
        
        dt_proj_up = self.dt_proj.weight # (d_model, dt_rank)
        dt_weight = dt_proj_up @ dt_proj_down.t() # (d_model, d_model)  
        
        if self.dt_proj_up_adapter is not None and self.dt_proj_down_adapter is not None:
            dt_adapter = self.dt_proj_up_adapter @ self.dt_proj_down_adapter.T # (d_model, d_model)
        else:
            dt_adapter = None
            
        dt_weight = self.adapted_params(
             value = dt_weight,
             value_adapter = dt_adapter,
             d_inner = self.d_inner,
             updatable_dim1 = updatable_channels_dim,
             updatable_dim2 = updatable_channels_dim,
             layer_idx = layer_idx,
             order = 1,
             method = ssm_method,
         )        
        
        dt = F.linear(rearrange(x, "b d l -> (b l) d"), dt_weight) # (B*seqlen, d_model)
        
        self.A, self.B, self.C, self.dt = A, B, C, dt
                
        dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        assert self.activation in ["silu", "swish"]
        
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
            
        y = rearrange(y, "b d l -> b l d")
        if self.ssm_bias_adapter is not None: y += self.ssm_bias_adapter
        out = self.out_proj(y, lp_method)
        if self.out_proj.bias is not None: out += self.out_proj.bias.to(dtype=out.dtype)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        mixer_cls,         
        init_mode = 'default',
        norm_cls=nn.LayerNorm, 
        fused_add_norm=False, 
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, init_mode = init_mode)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, 
        hidden_states: Tensor, 
        residual: Optional[Tensor] = None, 
        inference_params=None,
        ssm_method = 'full',
        lp_method = 'full',
        updatable_channels_dim = None,
        updatable_states_dim = None,
        layer_idx = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(
            hidden_states, 
            ssm_method,
            lp_method,
            inference_params=inference_params,
            updatable_channels_dim = updatable_channels_dim,
            updatable_states_dim = updatable_states_dim,
            layer_idx = layer_idx,
        )
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    init_mode = 'default',
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        dim = d_model,
        mixer_cls = mixer_cls,
        init_mode = init_mode,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class adaptedEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        **kwargs,
    ):
        super(adaptedEmbedding, self).__init__(
            vocab_size,
            embedding_dim,
            **kwargs,
        )
        
        self.adapter = None 
    
    def forward(
        self, 
        input,
        method = 'lora',
    ):
        output = super().forward(input)
        if not self.adapter is None:
            if method == 'lora':
                # lora
                update =  F.embedding(
                    input, 
                    self.adapter[0] @ self.adapter[1], 
                    self.padding_idx, 
                    self.max_norm,
                    self.norm_type, 
                    self.scale_grad_by_freq, 
                    self.sparse
                )
                output += update
            else:
                raise ValueError(f"method {method} for embedding is not supported")
        return output
        
class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        init_mode = 'default',
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = adaptedEmbedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    init_mode = init_mode,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self, 
        input_ids, 
        inference_params = None,
        ssm_method = 'full',
        lp_method = 'full',
        updatable_channels_dim = None,
        updatable_states_dim = None,
    ):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, 
                residual, 
                inference_params=inference_params,
                ssm_method = ssm_method,
                lp_method = lp_method,
                updatable_channels_dim = updatable_channels_dim,
                updatable_states_dim = updatable_states_dim,
                layer_idx = layer_idx,
            )
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states