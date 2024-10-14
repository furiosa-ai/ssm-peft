import os
from pathlib import Path
from typing import Union
import numpy as np
from torch import nn
import torch


def flatten_dict(dic, sep="_", prefix=""):
    out = {}

    for k, v in dic.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v, sep, prefix + k + sep))
        else:
            out[prefix + k] = v

    return out


def find_module_parent(model: nn.Module, child: nn.Module, n_ancestor=1):
    if n_ancestor > 1:
        return find_module_parent(model, find_module_parent(model, child, n_ancestor-1))

    for c in model.children():
        if c is child:
            return model
        else:
            res = find_module_parent(c, child)

            if res is not None:
                return res

    return None


def find_layer_by_name(model: nn.Module, identifier: Union[str, int]) -> nn.Module:
    """
    Find a layer in a PyTorch model either by its name using dot notation for nested layers or by its index.

    Parameters
    ----------
    model : nn.Module
        Model from which to search for the layer.
    identifier : str or int
        Layer name using dot notation for nested layers or layer index to find in the model.

    Returns
    -------
    nn.Module
        The layer found, or None if no such layer exists.

    Raises
    ------
    ValueError
        If the identifier is neither a string nor an integer.
    """
    # Flatten the model into a list of layers if index is provided
    if isinstance(identifier, int):
        layers = []
        def flatten_model(module):
            for child in module.children():
                if len(list(child.children())) == 0:
                    layers.append(child)
                else:
                    flatten_model(child)
        flatten_model(model)
        if 0 <= identifier < len(layers):
            return layers[identifier]
        return None

    elif isinstance(identifier, str):
        # Access by dot-notated name
        parts = identifier.split('.')
        current_module = model
        try:
            for part in parts:
                current_module = getattr(current_module, part)
            return current_module
        except AttributeError:
            return None
    else:
        raise ValueError(f"Identifier must be either an integer or a string, got {type(identifier)}.")


def dump_mask(mask, name):
    from PIL import Image
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    filename = Path(f"out/{os.getpid()}/{name}.png")
    filename.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(mask).save(filename)

