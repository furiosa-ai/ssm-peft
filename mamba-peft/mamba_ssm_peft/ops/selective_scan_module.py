from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm_peft.ops.selective_scan_cuda import SelectiveScanCuda
from mamba_ssm_peft.ops.selective_scan_split import SelectiveScanSplit
from mamba_ssm_peft.ops.selective_scan_torch import SelectiveScanTorch
from mamba_ssm_peft.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class SelectiveScanModule(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        
        self.scan = None
        self.act = nn.SiLU()
        self.set_mode(mode)

    def set_mode(self, mode):
        self.scan = {
            "cuda": lambda: selective_scan_fn, 
            "cuda_torch": lambda: SelectiveScanCuda(), 
            "ref": lambda: selective_scan_ref, 
            "torch_logcumsumexp": lambda: SelectiveScanTorch("logcumsumexp"),
            "torch_logcumsumexp_compile": lambda: torch.compile(SelectiveScanTorch("logcumsumexp")),
            "split": lambda: SelectiveScanSplit(),
            "split_compile": lambda: torch.compile(SelectiveScanSplit()),
        }[mode]()

    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        assert delta_bias is None
        assert delta_softplus

        return self.scan(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z, 
                         delta_bias=delta_bias, delta_softplus=delta_softplus, 
                         return_last_state=return_last_state)
    
    def step(self, u, delta, A, B, C, D=None, z=None, ssm_state=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        assert delta_bias is None
        assert delta_softplus
        assert return_last_state

        dtype = u.dtype
        
        dt = delta
        dt = F.softplus(dt)  #  + self.dt_proj.bias.to(dtype=dt.dtype)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn" if B.ndim == 2 else "bd,bdn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(u, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd" if C.ndim == 2 else "bdn,bdn->bd", ssm_state.to(dtype), C)
        y = y + D.to(dtype) * u
        y = y * self.act(z)  # (B D)
        return y, ssm_state
