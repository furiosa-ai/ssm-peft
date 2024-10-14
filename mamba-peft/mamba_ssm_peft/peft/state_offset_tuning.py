

from dataclasses import dataclass, field
import enum
import math
from typing import List
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from mamba_ssm_peft.peft.mamba_base_tuner import MambaBaseTuner
from mamba_ssm_peft.peft.mamba_peft_utils import DropoutTensor


class StateOffsetTuningMode(str, enum.Enum):
    OUTPUT_TUNING = "OUTPUT_TUNING"
    STATE_OFFSET_TUNING = "STATE_OFFSET_TUNING"


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

        return out


@register_peft_config(MambaPeftType.STATE_OFFSET_TUNING)
@dataclass
class StateOffsetTuningConfig(PeftConfig):
    init: str = field(default="zero")
    tuning_mode: str = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    r: int = field(default=None)
    r_ratio: int = field(default=None)
    dropout: float = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.STATE_OFFSET_TUNING
        self.tuning_mode = StateOffsetTuningMode(self.tuning_mode)


@register_peft_tuner(MambaPeftType.STATE_OFFSET_TUNING)
class StateOffsetTuningModel(MambaBaseTuner):
    prefix: str = "statetune_"

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

        new_module = StateOffsetTuningSelectiveScanModule(
            target, adapter_name=adapter_name,
            param_info=self.param_info,
            device=self.device,
            tuning_mode=peft_config.tuning_mode, 
            init=peft_config.init, 
            r=peft_config.r, dropout=peft_config.dropout, r_ratio=peft_config.r_ratio
        )

        return new_module
    

class StateOffsetTuningSelectiveScanModule(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, **kwargs):
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.m = StateOffsetTuningBiasProcessor(**kwargs)

    def forward(self, u, delta, A, B, C, D, z, **kwargs):
        y = self.base_layer(u, delta, A, B, C, D, z, **kwargs)

        if isinstance(y, (list, tuple)):
            y, ssm_state = y
            return self.m(y, C, z), ssm_state
        else:
            return self.m(y, C, z)
    
    def step(self, u, delta, A, B, C, D, z, **kwargs):
        y, ssm_state = self.base_layer.step(u, delta, A, B, C, D, z, **kwargs)
        return self.m(y, C, z), ssm_state


class StateOffsetTuningBiasProcessor(nn.Module, BaseTunerLayer):
    def __init__(self, adapter_name, param_info, device, r=None, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.statetune_bias = nn.ParameterDict({}) if r is None else nn.ModuleDict({})
        self.statetune_type = {}
        self.dim_sizes = param_info.get_dim_sizes()
        self.dtype = param_info.get_dtype("B")
        self.device = device

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r=r,
            **kwargs
        )

    def _create_param(self, shape, init_type, dtype, device, r=None, dropout=None, r_ratio=None):
        if r is not None:
            d, n, = shape
            return LoraParam(d, n, r, dropout, ratio=r_ratio, dtype=dtype, device=device)
        else:
            match init_type:
                case "random":
                    data = torch.randn(shape, dtype=dtype, device=device) * 0.1
                case "zero":
                    data = torch.zeros(shape, dtype=dtype, device=device)

            return nn.Parameter(data)

    def update_layer(self, adapter_name, tuning_mode, init, r=None, dropout=None, r_ratio=None):
        shape = {
            StateOffsetTuningMode.STATE_OFFSET_TUNING: [self.dim_sizes["d"], self.dim_sizes["n"]],
            StateOffsetTuningMode.OUTPUT_TUNING: [self.dim_sizes["d"]],
        }[tuning_mode]

        param = self._create_param(shape, init, self.dtype, self.device,
                                   r=r, dropout=dropout, r_ratio=r_ratio)

        self.statetune_bias[adapter_name] = param
        self.statetune_type[adapter_name] = tuning_mode

        self.set_adapter(self.active_adapters)

    def forward(self, x, C, z):
        y = x

        for active_adapter in self.active_adapters:
            param = self.statetune_bias[active_adapter]

            if isinstance(param, nn.Module):
                param = param()

            tuning_mode = self.statetune_type[active_adapter]
            
            no_seqlen_dim = z.ndim == 2

            if no_seqlen_dim:
                z = z.unsqueeze(2)
                C = C.unsqueeze(2)

            match tuning_mode:
                case StateOffsetTuningMode.STATE_OFFSET_TUNING:
                    y_add = torch.einsum("bdl,bnl,dn -> bdl", F.silu(z), C, param)
                case StateOffsetTuningMode.OUTPUT_TUNING:
                    y_add = torch.einsum("bdl,d -> bdl", F.silu(z), param)

            if no_seqlen_dim:
                y_add = y_add.squeeze(2)

            y = y + y_add

        return y

