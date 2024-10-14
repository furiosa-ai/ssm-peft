

from dataclasses import dataclass, field
import math
from typing import List, Optional
from peft.config import PeftConfig
import torch
from torch import nn
from einops import repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from mamba_ssm_peft.peft.mamba_base_tuner import MambaBaseTuner
from mamba_ssm_peft.peft.mamba_peft_utils import DropoutTensor, StateMlp


class LoraParam(nn.Module):
    def __init__(self, d1, d2, r, dropout, ratio, device, dtype):
        super().__init__()

        if dropout is None:
            dropout = 0

        if ratio is not None:
            assert d1 % ratio == 0
            d1 = d1 // ratio
            d2 = d2 * ratio

        self.ratio = ratio
        self.lora_A = nn.Parameter(torch.zeros(r, d2, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d1, r, device=device, dtype=dtype))
        self.dropout = DropoutTensor(d2, dropout, device=device, dtype=dtype)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self):
        out = self.lora_B @ self.dropout(self.lora_A)

        if hasattr(self, "ratio") and self.ratio is not None:
            d1, d2 = out.shape
            out = out.reshape(d1 * self.ratio, d2 // self.ratio)

        out = out[None, :, :, None]
        return out


def _create_param(shape, dtype, device, init, hidden_dim=None, r=None, dropout=None, r_ratio=None):
    if r is not None:
        b, d, n, l = shape
        param = LoraParam(d, n, r, dropout, ratio=r_ratio, dtype=dtype, device=device)
    elif hidden_dim is not None:
        b, d, n, l = shape
        param = StateMlp(n, d, hidden_dim=hidden_dim, init=init, dtype=dtype, device=device, output_shape="1 d n 1")
    else:
        param = nn.Parameter(torch.zeros(
                shape, dtype=dtype, device=device))
        match init:
            case "random":
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            case "zero":
                pass
            case "one":
                nn.init.ones_(param)
            case _:
                assert False

    return param


@register_peft_config(MambaPeftType.PARAM_TRANSFORM)
@dataclass
class ParamTransformConfig(PeftConfig):
    # target_modules: Optional[List[str]] = field(default=None)
    parameters: Optional[List[str]] = field(default=None)
    scale_shape: Optional[List[str]] = field(default=None)
    bias_shape: Optional[List[str]] = field(default=None)
    init: Optional[str] = field(default=None)
    hidden_dim: Optional[int] = field(default=None)
    r: int = field(default=None, metadata={"help": "Lora attention dimension"})
    r_ratio: int = field(default=None)
    dropout: float = field(default=None)
    finetune_parameters: List[str] = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.PARAM_TRANSFORM



@register_peft_tuner(MambaPeftType.PARAM_TRANSFORM)
class ParamTransformModel(MambaBaseTuner):
    prefix: str = "paramtf_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        peft_config.target_modules = ["selective_scan_fn"]

        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        finetune_parameters = self.peft_config[self.active_adapter].finetune_parameters

        if finetune_parameters is None:
            finetune_parameters = []

        for n, p in model.named_parameters():
            if self.prefix in n or any(n.endswith("." + fp) for fp in finetune_parameters):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        new_module = None

        hidden_dim = peft_config.hidden_dim
        if isinstance(hidden_dim, dict):
            hidden_dim = hidden_dim[target_name.split(".")[-1]]

        new_module = ParameterTransformSelectiveScanModule(
            target, adapter_name, 
            parameters=peft_config.parameters,
            param_info=self.param_info,
            device=self.device,
            scale_shape=peft_config.scale_shape, bias_shape=peft_config.bias_shape, 
            init=peft_config.init, hidden_dim=hidden_dim, r=peft_config.r, 
            r_ratio=peft_config.r_ratio, dropout=peft_config.dropout)

        return new_module


class ParameterTransformSelectiveScanModule(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, parameters, **kwargs):
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.transforms = nn.ParameterDict({p: ParameterTransform(adapter_name, param_name=p, **kwargs) for p in parameters})

    def transform_params(self, B, C):
        if "B" in self.transforms:
            B = self.transforms["B"](B)

        if "C" in self.transforms:
            C = self.transforms["C"](C)

        if B.ndim == 3 and C.ndim == 4:
            B = repeat(B, "b n l -> b d n l", d=C.shape[1])
        elif B.ndim == 4 and C.ndim == 3:
            C = repeat(C, "b n l -> b d n l", d=B.shape[1])

        return B, C

    def forward(self, u, delta, A, B, C, D, **kwargs):
        B, C = self.transform_params(B, C)

        return self.base_layer(u, delta, A, B, C, D, **kwargs)
    
    def step(self, u, delta, A, B, C, D, **kwargs):
        B, C = B.unsqueeze(-1), C.unsqueeze(-1)  # add L
        B, C = self.transform_params(B, C)
        B, C = B.squeeze(-1), C.squeeze(-1)

        return self.base_layer.step(u, delta, A, B, C, D, **kwargs)


class ParameterTransform(nn.Module, BaseTunerLayer):
    def __init__(self, adapter_name, param_name, param_info, device, hidden_dim=None, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.paramtf_scale = nn.ParameterDict({}) if hidden_dim is None else nn.ModuleDict()  # nn.Parameter(param)
        self.paramtf_bias = nn.ParameterDict({}) if hidden_dim is None else nn.ModuleDict()

        self.dim_names = param_info.get_dim_names(param_name)
        self.dim_sizes = param_info.get_dim_sizes()
        self.dtype = param_info.get_dtype(param_name)
        self.device = device

        self.update_layer(
            adapter_name,
            **kwargs,
            hidden_dim=hidden_dim,
        )

    def update_layer(self, adapter_name, scale_shape=None, bias_shape=None, 
                     init=None, hidden_dim=None, r=None, r_ratio=None, dropout=None):
        fmt = "bdnl"

        if scale_shape is not None:
            shape = [(abs(self.dim_sizes[d]) if d in scale_shape else 1) for d in fmt]
            self.paramtf_scale[adapter_name] = _create_param(shape, self.dtype, self.device, {"zero": "one", "random": "random"}[init], hidden_dim=hidden_dim, r=r, r_ratio=r_ratio, dropout=dropout)

        if bias_shape is not None:
            shape = [(abs(self.dim_sizes[d]) if d in bias_shape else 1) for d in fmt]
            self.paramtf_bias[adapter_name] = _create_param(shape, self.dtype, self.device, init, hidden_dim=hidden_dim, r=r, r_ratio=r_ratio, dropout=dropout)

        self.set_adapter(self.active_adapters)

    def get_scale(self, adapter):
        if adapter in self.paramtf_scale:
            scale = self.paramtf_scale[adapter]
        else:
            scale = None

        return scale

    def get_bias(self, adapter):
        if adapter in self.paramtf_bias:
            bias = self.paramtf_bias[adapter]
        else:
            bias = None

        return bias

    def forward(self, x):
        y = x

        for active_adapter in self.active_adapters:
            scale, bias = self.get_scale(active_adapter), self.get_bias(active_adapter)

            if isinstance(scale, nn.Module):
                scale = scale()

            if isinstance(bias, nn.Module):
                bias = bias()

            if self.dim_names == "bnl":
                y = rearrange(y, "b n l -> b 1 n l")
            elif self.dim_names == "dn":
                y = rearrange(y, "d n l -> 1 d n l")
            else:
                assert False

            if scale is not None:
                y = scale * y

            if bias is not None:
                y = y + bias

            if self.dim_names == "dn":
                y = rearrange(y, "b d n -> (b d) n")

        return y
