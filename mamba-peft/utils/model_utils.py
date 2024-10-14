from pathlib import Path
import yaml
import torch

from mamba_ssm_peft import get_mamba_peft_model, load_mamba


class TrainableParamsDb:
    def __init__(self) -> None:
        self.cache_file = Path("out/trainable_params.yaml")
        self.trainable_params = {}

        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                self.trainable_params = yaml.safe_load(f)
        else:
            self.cache_file.parent.mkdir(exist_ok=True, parents=True)

    def _compute_trainable_params(self, model_size, peft): 
        model_kwargs = dict(
            dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}["bf16"], 
            device="cpu",
        )

        # model = load_mamba(args.model, **model_kwargs)
        model = load_mamba(
            "state-spaces/mamba-" + model_size, 
            **model_kwargs
        )["model"]

        model = get_mamba_peft_model(model, peft, return_peft_cfg=False, no_print=True)

        params_dict = {k: v for k, v in model.named_parameters()}
        train_params_dict = {k: v for k, v in params_dict.items() if v.requires_grad}

        train_params = sum(p.numel() for p in train_params_dict.values())
        total_params = sum(p.numel() for p in params_dict.values())

        # print("\n".join(f"{k}" for k, v in train_params_dict.items()))

        return {
            "trainable_param_names": [f"{k}" for k, v in train_params_dict.items()],
            "trainable_params": train_params,
            "total_params": total_params,
            "trainable_params_ratio": train_params / total_params,
        }

    def get_trainable_params(self, model_size, peft):
        if peft is None:
            return {"trainable_params_ratio": 100.0}

        key = "_".join([model_size, peft])

        if key not in self.trainable_params:
            self.trainable_params[key] = self._compute_trainable_params(model_size, peft)
            with open(self.cache_file, "w") as f:
                yaml.safe_dump(self.trainable_params, f)

        return self.trainable_params[key]
