

from mamba_ssm_peft.models.mixer_seq_simple import MambaLMHeadModel

import torch
import json

from transformers import AutoTokenizer
from pathlib import Path
from peft import PeftModelForSeq2SeqLM

from peft import get_peft_model, PeftConfig



def get_checkpoints(path, return_dict=False, local_only=False):
    def _get_it(file):
        try:
            return int(Path(file).stem.split("-")[1])
        except ValueError:
            return 0

    if not Path(path).exists():
        checkpoints = [path]
    else:
        path = Path(path)
        checkpoints = list(path.glob("checkpoint-*"))

        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=_get_it)
        else:
            checkpoints = [path]

    if local_only:
        assert all((c / "model.pt").is_file() for c in checkpoints)

    if return_dict:
        checkpoints = {_get_it(c): str(c) for c in checkpoints}

    return checkpoints


def load_mamba(pretrained, fuse_peft=False, cls=MambaLMHeadModel, **kwargs):
    pretrained = get_checkpoints(pretrained)[-1]

    model_kwargs = kwargs

    trainable_params = 1

    if (Path(pretrained) / "model.pt").exists():
        model = torch.load(Path(pretrained) / "model.pt")
        dtype = next(iter(model.parameters())).dtype

        if dtype != model_kwargs.get("dtype", dtype):
            print(f'Moving model to {model_kwargs["dtype"]}')
            model = model.to(model_kwargs["dtype"])
            assert next(iter(model.parameters())).dtype == model_kwargs["dtype"]

        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, all_params = model.get_nb_trainable_parameters()
            trainable_params = trainable / all_params
        else:
            trainable_params = 1

        if fuse_peft:
            if isinstance(model, PeftModelForSeq2SeqLM):
                model = model.merge_and_unload()

            if isinstance(model, MambaLMHeadModel):
                try:
                    model.combine_layers()
                except AttributeError:
                    print("no method combine_layers")
    else:
        # if (pretrained / "pytorch_model.bin").exists():
        model = cls.from_pretrained(str(pretrained), **model_kwargs)

    tokenizer = load_mamba_tokenizer()

    info = {
        "trainable_params": trainable_params
    }

    return {
        "model": model, 
        "tokenizer": tokenizer,
        "info": info
    }


def load_tokenizer(tokenizer):
    tokenizer = {
        "EleutherAI/gpt-neox-20b": load_mamba_tokenizer
    }[tokenizer]()
    return tokenizer


def load_mamba_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "###"
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template
    return tokenizer


def print_trainable_parameter_names(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def get_mamba_peft_model(model, peft, return_peft_cfg=False, train_embedding=False, no_print=False):
    model.split_layers()

    if isinstance(peft, (str, Path)):
        with open(peft, "r") as f:
            peft = json.load(f)

    if isinstance(peft, list):
        peft = {
            "peft_type": "MULTI_PEFT",
            "configs": peft
        }

    if isinstance(peft, dict):
        peft = PeftConfig.from_peft_type(**peft)

    model = get_peft_model(model, peft)

    if train_embedding:
        model.model.word_embeddings.weight.requires_grad = True

    if not no_print:
        print_trainable_parameter_names(model)

    if return_peft_cfg:
        return model, peft
    return model


def get_trainable_parameters_ratio(model):
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, all_params = model.get_nb_trainable_parameters()
        trainable_params = trainable / all_params
    else:
        trainable_params = 1

    return trainable_params
