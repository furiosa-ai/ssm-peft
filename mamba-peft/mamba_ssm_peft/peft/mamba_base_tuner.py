

import torch
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists



class MambaBaseTuner(BaseTuner):
    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        super().__init__(model, peft_config, adapter_name)

    
    @property
    def param_info(self):
        return self.get_mamba_blocks()[0].param_info
    
    @property
    def device(self):
        return self.get_mamba_blocks()[0].x_proj.device
    
    @property
    def dtype(self):
        return self.get_mamba_blocks()[0].x_proj.dtype

    def get_mamba_blocks(self):
        return self.model.get_mamba_blocks()

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _replace_module(self, parent, child_name, new_module, child):
        device = next(self.parameters()).device
        new_module = new_module.to(device)
        setattr(parent, child_name, new_module)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        new_module = self._create_new_module(lora_config, adapter_name, target, target_name)
        if adapter_name != self.active_adapter:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)

        if new_module is not None:
            self._replace_module(parent, target_name, new_module, target)

    def split_layers(self):
        self.model.split_layers()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return self.model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
