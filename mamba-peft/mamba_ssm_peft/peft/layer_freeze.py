
from dataclasses import dataclass, field
from typing import Optional
from peft.config import PeftConfig
import torch
from torch import nn

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from mamba_ssm_peft.peft.mamba_base_tuner import MambaBaseTuner


@register_peft_config(MambaPeftType.LAYER_FREEZE)
@dataclass
class LayerFreezeConfig(PeftConfig):
    trainable_layers: Optional[int] = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.LAYER_FREEZE



@register_peft_tuner(MambaPeftType.LAYER_FREEZE)
class LayerFreezeModel(MambaBaseTuner):
    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            p.requires_grad = False
            for trainable_layer in self.peft_config[self.active_adapter].trainable_layers:
                if trainable_layer in n:
                    p.requires_grad = True

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    def _create_new_module(self, lora_config, adapter_name, target, target_name):
        return None
