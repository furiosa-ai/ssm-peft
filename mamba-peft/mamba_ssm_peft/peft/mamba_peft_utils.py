from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as F


def set_peft_params_trainable(model, pattern, enable_train, disable_train):
    for n, p in model.named_parameters():
        if pattern in n:
            if enable_train:
                p.requires_grad = True
        else:
            if disable_train:
                p.requires_grad = False


class MambaParameterInfo:
    def __init__(self, d_inner, d_state, d_conv, dtype) -> None:
        self.dim_names = {
            "A_log": "dn",
            "A": "dn",
            "B": "bnl",
            "C": "bnl",
            "D": "d",
            "dt": "bnl",
        }

        self.dim_sizes = {
            "b": -1,
            "l": -1,
            "d": d_inner,
            "n": d_state,
            "conv": d_conv,
        }

        self.dtypes = {
            "A_log":torch.float32,
            "A": torch.float32,
            "B": dtype,
            "C": dtype,
            "D": torch.float32,
            "dt": dtype,
        }

    @property
    def dtype(self):
        return self.dtypes["B"]

    def get_dim_names(self, param_name):
        return self.dim_names[param_name]
    
    def get_dim_sizes(self):
        return self.dim_sizes

    def get_dtype(self, param_name):
        return self.dtypes[param_name]



class StateMlp(nn.Module):
    shared_embed = {}

    def __init__(self, n, token_dim, hidden_dim, init, dtype, device, output_shape=None, bias=None):
        super().__init__()

        if isinstance(hidden_dim, float):
            hidden_dim = int(hidden_dim * token_dim)
        
        if n not in StateMlp.shared_embed:
            StateMlp.shared_embed[(n, dtype)] = nn.Embedding(n, token_dim, dtype=dtype, device=device)

        self.embed = StateMlp.shared_embed[(n, dtype)]
        self.transform = nn.Sequential(
            nn.Linear(token_dim, hidden_dim, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, token_dim, dtype=dtype, device=device, bias=bias is None),
        )

        if init == "zero":
            nn.init.zeros_(self.transform[2].weight)
            if self.transform[2].bias is not None:
                nn.init.zeros_(self.transform[2].bias)
        elif init == "random":
            pass
        else:
            assert False

        self.prompt_tokens = torch.arange(n, device=device).long()
        self.output_shape = output_shape
        self.bias = nn.Parameter(bias).to(device).to(dtype) if bias is not None else None

    @property
    def shape(self):
        return self.embed.weight.shape

    def forward(self):
        y = self.transform(self.embed(self.prompt_tokens))

        if self.output_shape is None:
            y = y.T
        else:
            y = rearrange(y, "n d -> " + self.output_shape)

        if self.bias is not None:
            y = y + self.bias

        return y
    

class DropoutTensor(nn.Module):
    def __init__(self, d, p, dtype, device):
        super().__init__()
        self.register_buffer("ones", torch.ones(d, dtype=dtype, device=device), persistent=False)
        self.p = p

    def forward(self, x):
        if self.p > 0:
            dropout = self.ones
            dropout = F.dropout(dropout, self.p, self.training, False)
            y = x * dropout[None]
            return y
        else:
            return x
