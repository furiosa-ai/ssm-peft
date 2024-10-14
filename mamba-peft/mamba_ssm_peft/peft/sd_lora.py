
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Dict, Optional, Union, List
from peft.config import PeftConfig
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from einops import einsum, repeat, rearrange

from mamba_ssm_peft.peft import MambaPeftType, register_peft_config, register_peft_tuner
from peft.tuners.tuners_utils import BaseTunerLayer

from peft.tuners.lora import Linear as LoraLinear
from mamba_ssm_peft.peft.mamba_base_tuner import MambaBaseTuner
from utils.utils import find_layer_by_name, find_module_parent


class SelectMode(str, enum.Enum):
    CHANNELS_PER_STATE_CHANNELS = "CHANNELS_PER_STATE_CHANNELS"
    CHANNELS_ALL_STATES = "CHANNELS_ALL_STATES"



@register_peft_config(MambaPeftType.SD_LORA)
@dataclass
class SdLoraConfig(PeftConfig):
    select_mode: SelectMode = field(default=SelectMode.CHANNELS_PER_STATE_CHANNELS)
    proj_select_mode: SelectMode = field(default=SelectMode.CHANNELS_ALL_STATES)
    proj_lora_r: int = field(default=None)
    num_zero: List[int] = field(default=None)
    num_freeze: List[int] = field(default=None)
    num_warmup_it: int = field(default=None)
    reg_scale: float = field(default=0.0)
    target_modules: List[str] = field(default=None)
    finetune_parameters: List[str] = field(default=None)
    sdlora_alpha: List[float] = field(default=None)

    def __post_init__(self):
        self.peft_type = MambaPeftType.SD_LORA


@register_peft_tuner(MambaPeftType.SD_LORA)
class SdLoraModel(MambaBaseTuner):
    prefix: str = "sdlora_"

    def __init__(self, model, peft_config: PeftConfig | dict[str, PeftConfig], adapter_name: str) -> None:
        self.last_mode = None
        super().__init__(model, peft_config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        peft_config.target_modules = [t if t != "A_log" else "A_log_adapter" for t in peft_config.target_modules]

        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        finetune_parameters = self.peft_config[self.active_adapter].finetune_parameters

        if finetune_parameters is None:
            finetune_parameters = []

        for n, p in model.named_parameters():
            if self.prefix in n or any(n.endswith("." + fp) for fp in finetune_parameters) or (self.peft_config["default"].proj_lora_r is not None and "lora_" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _create_new_module(self, peft_config, adapter_name, target, target_name):
        module_name = next(n for n, m in self.model.named_modules() if m is target)

        if target_name in ("in_proj_x", "in_proj_z", "out_proj") and peft_config.proj_lora_r is not None:
            new_module = LoraLinear(target, adapter_name, r=peft_config.proj_lora_r, lora_alpha=peft_config.proj_lora_r, lora_dropout=0.1)
        else:
            block_name = module_name.split(".mixer.")[0] + ".mixer"
            block = find_layer_by_name(self.model, block_name)

            new_module = SdLoraParameter(
                target, adapter_name, module_name, block, peft_config.select_mode, peft_config.proj_select_mode,
                num_zero=peft_config.num_zero, num_freeze=peft_config.num_freeze, num_warmup_it=peft_config.num_warmup_it, 
                sdlora_alpha=(peft_config.sdlora_alpha.get(target_name, 1) * peft_config.sdlora_alpha.get("global", 1)) if peft_config.sdlora_alpha is not None else 1)

        return new_module

    def _get_sdlora_params(self):
        return [m for m in self.model.modules() if isinstance(m, SdLoraParameter)]

    def get_sdlora_mode(self):
        mode = [m.get_sdlora_mode() for m in self.model.modules() if isinstance(m, SdLoraParameter)]
        assert len(set(mode)) == 1
        return mode[0]

    def load_config(self, path):
        for m in self._get_sdlora_params():
            m.load_config(path)

    def save_config(self, path):
        for m in self._get_sdlora_params():
            m.save_config(path)

    @property
    def should_training_stop(self):
        if self.last_mode == "warmup" and self.get_sdlora_mode() == "train":
            self.last_mode = "train"
            res = True
        else:
            res = False

        if self.last_mode is None:
            self.last_mode = self.get_sdlora_mode()

        return res


class SdLoraParameter(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, adapter_name, module_name, block, select_mode, proj_select_mode, num_zero, num_freeze, num_warmup_it, sdlora_alpha=1) -> None:
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer

        # transpose = any(target_name.endswith(p) for p in ("A_log", "out_proj", "in_proj_x", "in_proj_z"))
        self.module_name = module_name.replace(".", "_")
        self.select_mode = select_mode
        self.proj_select_mode = proj_select_mode
        self.num_zero = self._parse_dims(num_zero)  # num_zero
        self.num_freeze = self._parse_dims(num_freeze)  # [num_freeze["state"], num_freeze[]
        self.num_train = (self.get_model_param_info().shape if not self._is_layer_of(("in_proj_x", "in_proj_z")) else self.get_model_param_info().shape[::-1]) - self.num_zero - self.num_freeze
        self.num_warmup_it = num_warmup_it
        self.sdlora_mode = None
        # self.it_counter = 0
        self.train_mask = None
        self.zero_mask = None
        self.sdlora_alpha = sdlora_alpha
        self.get_block = lambda: block

        # save in state dict
        self.register_buffer("it_counter", torch.tensor(0).long())

        self.sdlora_grad = self.create_param(full=True)
        self.sdlora_adapter = self.create_param()

        self.set_sdlora_mode("warmup" if self.training and self.num_warmup_it >=0 else "train")

        self.set_adapter(self.active_adapters)

    def _get_cfg_file(self, path):
        return Path(path) / (self.module_name + ".pkl")

    def load_config(self, path):
        cfg_path = self._get_cfg_file(path)
        if cfg_path.exists():
            if self.sdlora_grad is not None:
                with open(cfg_path, "rb") as f:
                    with torch.no_grad():
                        self.sdlora_grad.data[:] = pickle.load(f)
            print(f"Loaded {cfg_path}")
            self.set_sdlora_mode("train")

    def save_config(self, path):
        cfg_path = self._get_cfg_file(path)
        grad = self.sdlora_grad
        if grad is not None:
            grad = grad.data
        with open(cfg_path, "wb") as f:
            pickle.dump(grad, f)
        print(f"Saved {cfg_path}")

    def get_sdlora_mode(self):
        return self.sdlora_mode

    def get_grads_in_block(self, filter=None):
        block = self.get_block()

        out = {k: v for k, v in block.state_dict().items() if filter is None or ("sdlora_grad" in k and any(f in k for f in filter))}
        assert filter is None or len(out) == len(filter)

        return out

    def _is_layer_of(self, layers):
        if isinstance(layers, str):
            layers = [layers]
        
        return any(n in self.module_name for n in layers)

    def _parse_dims(self, dims):
        param = self.get_model_param_info()

        if self._is_layer_of(("in_proj_x", "in_proj_z")):
            param.shape = [param.shape[1], param.shape[0]]
        # if any(self.module_name.endswith(n) for n in ("out_proj",)):
        #     param.shape = [param.shape[1], param.shape[0]]

        dims = [dims["state"], dims["channel"]]

        for i in range(2):
            if isinstance(dims[i], float):
                dims[i] = int(round(dims[i] * param.shape[i]))

        if not self._is_layer_of(["A_log"]):
            # we freeze states only for A_log and not for B or C
            dims[0] = 0

        dims = np.array(dims)
        return dims

    @property
    def dim_order(self):
        return (0, 1) if not self.transpose else (1, 0)

    @property
    def dim_n(self):
        return self.dim_order[0]
    
    @property
    def dim_d(self):
        return self.dim_order[1]

    # @property
    # def num_train(self):
    #     if self.is_layer:
    #         param = self.base_layer.weight
    #     else:
    #         param = self.base_layer

    #     return param.shape[self.dim_n] - self.num_freeze - self.num_zero

    @property
    def transpose(self):
        return self._is_layer_of("A_log")

    @property
    def is_layer(self):
        return isinstance(self.base_layer, nn.Linear)
    
    def get_model_param_info(self):
        if self.is_layer:
            param = self.base_layer.weight
        else:
            param = self.base_layer
        # param = self.base_layer

        device = getattr(param, "device", "cuda")

        return SimpleNamespace(shape=param.shape if not self.transpose else param.shape[::-1], device=device, dtype=param.dtype)

    def create_param(self, full=False):
        param = self.get_model_param_info()

        if not full:
            if self._is_layer_of(("in_proj_x", "in_proj_z", "out_proj")):
                match self.proj_select_mode:
                    case SelectMode.CHANNELS_ALL_STATES:
                        if self._is_layer_of("out_proj"):
                            shape = np.prod([param.shape[0], self.num_train[1]])
                        else:
                            shape = np.prod([self.num_train[1], param.shape[1]])
                    case _:
                        assert False
            else:
                match self.select_mode:
                    case SelectMode.CHANNELS_PER_STATE_CHANNELS:
                        shape = np.prod(self.num_train)
                    case _:
                        assert False
        else:
            # if self._is_layer_of(("in_proj_x", "in_proj_z", "out_proj")):
            #     # dont create param for proj layers
            #     return None
            if not self._is_layer_of(["A_log"]):
                return None
            
            shape = param.shape

        param = nn.Parameter(torch.zeros(
            shape,
            device=param.device,
            dtype=param.dtype,
        ))

        assert param.numel() > 0

        return param

    def update_layer(self, adapter_name):
        pass

    def set_sdlora_mode(self, sdlora_mode):
        if sdlora_mode != self.sdlora_mode:
            if sdlora_mode == "train":
                if self.sdlora_grad is not None:
                    pass  # need to save those params so keep requires_grad
                    # self.sdlora_grad[self.module_name].requires_grad = False
                # print("Warmup result:", self.get_row_indices())
            else:
                pass

        self.sdlora_mode = sdlora_mode

    def get_importances(self, x, dim, per_row=False):
        norms = x.square().detach()
        if per_row:
            ind = torch.argsort(-norms, dim=dim)
        else:
            dim = 1 - dim
            norms = norms.sum(dim)
            ind = torch.argsort(-norms)
        return ind

    def _dim_name_to_idx(self, dim):
        if isinstance(dim, str):
            dim = {"STATE": 0, "CHANNEL": 1}[dim]

        return dim

    def select_rows(self, x, dim, row_type=None, per_row=False):
        dim = self._dim_name_to_idx(dim)
        imp = self.get_importances(x, dim, per_row=per_row)

        row_types = {
            "train": imp[0:self.num_train[dim]],
            "freeze": imp[self.num_train[dim]:self.num_train[dim]+self.num_freeze[dim]],
            "zero": imp[self.num_train[dim]+self.num_freeze[dim]:self.num_train[dim]+self.num_freeze[dim]+self.num_zero[dim]]
        }

        if row_type is None:
            return row_types
        else:
            return row_types[row_type]

    def get_grad_for_sel(self):
        # TODO: just get A_log grad
        # grads = self.get_grads_in_block(["x_proj_B", "x_proj_C", "A_log"])
        grads = self.get_grads_in_block(["A_log"])

        assert not any(torch.sum(g) == 0 for g in grads.values())
        return torch.prod(torch.stack(list(grads.values())), 0)

    def get_mask(self, mask_type):
        grad = self.get_grad_for_sel()

        param = self.get_model_param_info()
        mask = torch.zeros(param.shape, device=param.device, dtype=torch.bool)

        match self.select_mode:
            case SelectMode.CHANNELS_PER_STATE_CHANNELS:
                channel_indices = self.select_rows(grad, "CHANNEL", mask_type)

                if mask_type == "train":
                    if self._is_layer_of(("in_proj_x", "in_proj_z", "out_proj")):
                        match self.proj_select_mode:
                            case SelectMode.CHANNELS_ALL_STATES:
                                mask.index_fill_(1 if self._is_layer_of("out_proj") else 0, channel_indices, True)
                            case _:
                                assert False
                    # elif self._is_layer_of(("x_proj_B", "x_proj_C")):
                    #     # only select states for A_log
                    #     mask.index_fill_(1, channel_indices, True)
                    else:
                        state_indices_per_row = self.select_rows(grad[:, channel_indices], "STATE", mask_type, per_row=True)

                        n = state_indices_per_row.shape[0]
                        mask.T[channel_indices.repeat(n), state_indices_per_row.reshape(-1)] = True
                elif mask_type == "zero":
                    # set all states in channel to zero
                    assert self._is_layer_of("A_log")
                    mask.index_fill_(1, channel_indices, True)
                else:
                    assert False

        if mask_type == "train":
            # assert torch.sum(mask) == (self.num_train[0] * self.num_train[1])

            # from utils.utils import dump_mask
            # dump_mask(mask, self.module_name)
            pass

        return mask

    def build_train_param(self, param, adapter):
        # return param
        if self.train_mask is None:
            print("Building trainable mask")
            self.train_mask = self.get_mask("train")

        if self._is_layer_of("A_log"):
            if self.zero_mask is None:
                self.zero_mask = self.get_mask("zero")
                # masks should not overlap
                assert torch.sum(self.train_mask & self.zero_mask).item() == 0
            
            param = torch.masked_fill(param, self.zero_mask, 10)  # torch.inf

        bias = torch.masked_scatter(torch.zeros_like(param), self.train_mask, adapter)
        return param + self.sdlora_alpha * bias

    def forward(self, x):
        if not hasattr(self, "sdlora_alpha"):
            # fix for old ckpts
            self.sdlora_alpha = 1

        if self.sdlora_mode == "warmup" and self.num_warmup_it >= 0 and self.it_counter > self.num_warmup_it:
            self.set_sdlora_mode("train")
        
        assert not (self.sdlora_mode == "warmup" and not self.training)

        if self.is_layer:
            param = self.base_layer.weight
            assert not hasattr(self.base_layer, "bias") or self.base_layer.bias is None
        else:
            param, x = x, None

        if self.transpose:
            param = param.T

        if self.sdlora_mode == "warmup":
            param_new = param + self.sdlora_alpha * (self.sdlora_grad if self.sdlora_grad is not None else 0)
        elif self.sdlora_mode == "train":
            param_new = self.build_train_param(param, self.sdlora_adapter)
        else:
            assert False

        self.it_counter += 1

        if self.transpose:
            param_new = param_new.T

        if self.is_layer:
            return F.linear(x, param_new)
        else:
            return param_new
