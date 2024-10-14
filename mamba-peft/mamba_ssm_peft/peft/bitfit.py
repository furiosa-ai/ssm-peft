

from dataclasses import dataclass, field
from typing import Optional, Union
from peft.config import PeftConfig
import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from mamba_ssm_peft.peft.mamba_base_tuner import MambaBaseTuner




@register_peft_config(MambaPeftType.BITFIT)
@dataclass
class BitFitConfig(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )

    def __post_init__(self):
        self.peft_type = MambaPeftType.BITFIT


@register_peft_tuner(MambaPeftType.BITFIT)
class BitFitModel(MambaBaseTuner):
    prefix: str = "bitfit_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for n, p in model.named_parameters():
            if self.prefix in n:
                p.requires_grad = True

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        return BitFitLayer(target, adapter_name)


class BitFitLayer(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, **kwargs) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        self.bitfit_bias = nn.ParameterDict({})

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            **kwargs
        )

    def update_layer(self, adapter_name):
        if self.base_layer.bias is not None:
            # take existing bias
            self.bitfit_bias[adapter_name] = self.base_layer.bias
            self.base_layer.bias = None
        else:
            # create new bias
            dim = self.base_layer.out_features if not isinstance(self.base_layer, nn.Conv1d) else self.base_layer.out_channels
            self.bitfit_bias[adapter_name] = nn.Parameter(torch.zeros(
                dim, dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device))

        self.set_adapter(self.active_adapters)

    def forward(self, x):
        y = self.base_layer(x)

        for active_adapter in self.active_adapters:
            bias = self.bitfit_bias[active_adapter]

            if isinstance(self.base_layer, nn.Conv1d):
                bias = {3: bias[None, :, None]}[y.ndim]
            else:
                bias = {2: bias[None, :], 3: bias[None, None, :]}[y.ndim]

            y = y + bias

        return y

