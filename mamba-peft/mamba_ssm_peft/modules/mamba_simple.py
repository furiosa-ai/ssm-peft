# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm_peft.ops.selective_scan_module import SelectiveScanModule
from mamba_ssm_peft.peft.mamba_peft_utils import MambaParameterInfo


try:
    from mamba_ssm_peft.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaParameterAdapter(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.shape = param.shape
        self.dtype = param.dtype
    
    def forward(self, param):
        return param


class MultiLinearLayer(nn.Module):
    def __init__(self, linear: nn.Linear, names_dims, cat_output=False) -> None:
        super().__init__()

        assert linear.bias is None

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.names_dims = names_dims

        weights = torch.split(linear.weight, list(names_dims.values()), dim=0)

        for i, (name, dim) in enumerate(names_dims.items()):
            l = nn.Linear(linear.in_features, dim, bias=False, device=linear.weight.device, dtype=linear.weight.dtype)

            with torch.no_grad():
                l.weight[:] = weights[i]
            setattr(self, name, l)

        self.cat_output = cat_output

    @property
    def device(self):
        return getattr(self, next(iter(self.names_dims.keys()))).weight.device
    @property
    def dtype(self):
        return getattr(self, next(iter(self.names_dims.keys()))).weight.dtype

    @property
    def bias(self):
        return None

    def to_linear(self):
        layers = [getattr(self, name) for name in self.names_dims.keys()]
        weights = [l.weight for l in layers]
        weight = torch.cat(weights, dim=0)

        l = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=layers[0].weight.device, dtype=layers[0].weight.dtype)

        with torch.no_grad():
            l.weight[:] = weight

        return l

    def forward(self, x):
        outputs = [getattr(self, name)(x) for name in self.names_dims.keys()]
        if self.cat_output:
            outputs = torch.concat(outputs, -1)
        return outputs


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
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        backend="cuda",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
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

        self.selective_scan_fn = SelectiveScanModule(backend)

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
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
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # Keep A_log in fp32
        self.A_log._no_weight_decay = True
        self.A_log_adapter = MambaParameterAdapter(self.A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.param_info = MambaParameterInfo(self.d_inner, self.d_state, self.d_conv, dtype)

    def split_layers(self):
        if not isinstance(self.x_proj, MultiLinearLayer):
            self.x_proj = MultiLinearLayer(self.x_proj, {
                "x_proj_dt": self.dt_rank,
                "x_proj_B": self.d_state,
                "x_proj_C": self.d_state,
            }, cat_output=False)

        if not isinstance(self.in_proj, MultiLinearLayer):
            self.in_proj = MultiLinearLayer(self.in_proj, {
                "in_proj_x": self.d_inner,
                "in_proj_z": self.d_inner,
            }, cat_output=True)

    def combine_layers(self):
        if isinstance(self.x_proj, MultiLinearLayer):
            self.x_proj = self.x_proj.to_linear()

        if isinstance(self.in_proj, MultiLinearLayer):
            self.in_proj = self.in_proj.to_linear()

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # batch, seqlen, dim = hidden_states.shape
        batch = hidden_states.shape[0]

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = self.in_proj(hidden_states)

        if isinstance(xz, (tuple, list)):
            x, z = xz
            x = rearrange(x, "b l d -> b d l")
            z = rearrange(z, "b l d -> b d l")
        else:
            xz = rearrange(xz, "b l d -> b d l")
            x, z = xz.chunk(2, dim=1)

        A_log = self.A_log_adapter(self.A_log)

        A = -torch.exp(A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        x = self.conv1d(x)
        # x = self.act(x[..., :seqlen])
        x = self.act(x[..., :-(self.d_conv-1)])

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> b l d"))  # (bl d)

        if isinstance(x_dbl, (tuple, list)):
            dt, B, C = x_dbl
        else: 
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt = self.dt_proj.weight @ dt.t()
        dt = self.dt_proj(dt)
        dt = rearrange(dt, "b l d -> b d l", b=batch)
        # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", b=batch).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", b=batch).contiguous()
        D = self.D

        if B.ndim == 4 and C.ndim == 3:
            C = repeat(C, "b n l -> b d n l", d=B.shape[1])
        
        if B.ndim == 3 and C.ndim == 4:
            B = repeat(B, "b n l -> b d n l", d=C.shape[1])

        # add peft cat here, ensure requires grad for columns
        # B D has batch, apply to x_proj instead
        # integrate split in x_proj layer

        assert self.activation in ["silu", "swish"]
        y = self.selective_scan_fn(
            x,
            delta=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            delta_bias=None,  # self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        
        y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)
        
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # assert False, "dont use step() with peft"
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)

        if isinstance(xz, (tuple, list)):
            x, z = xz
        else:
            x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        conv_state[:, :, -1] = x
        if isinstance(self.conv1d, nn.Conv1d):
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
        else:
            x = self.conv1d(conv_state)
            x = x[..., x.shape[-1] // 2]
        
        x = self.act(x).to(dtype=dtype)

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)

        if isinstance(x_db, (tuple, list)):
            dt, B, C = x_db
        else: 
            dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        # dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        dt = self.dt_proj(dt)

        A_log = self.A_log_adapter(self.A_log)

        A = -torch.exp(A_log.float())  # (d_inner, d_state)
        D = self.D

        y, ssm_state = self.selective_scan_fn.step(
            x,
            delta=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            ssm_state=ssm_state,
            delta_bias=None,  # self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=True
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

        if hasattr(self.x_proj, "x_proj_B") and hasattr(self.x_proj.x_proj_B, "out_features"):
            d_state = self.x_proj.x_proj_B.out_features
        else:
            d_state = self.d_state

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
                d_state,
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
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, **kwargs,
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
        self.mixer = mixer_cls(dim, **kwargs)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
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
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, **kwargs,
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
        self.mixer = mixer_cls(dim, **kwargs)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
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
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
