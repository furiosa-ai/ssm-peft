from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm_peft.ops.selective_scan_interface import selective_scan_fn


class SelectiveScanSplit(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        assert delta_bias is None
        assert delta_softplus
        assert not return_last_state

        if delta_softplus:
            delta = F.softplus(delta)

        n = A.shape[1]
        ones = torch.ones_like(C[:, 0:1])

        h = []
        for i in range(n):
            h_i = selective_scan_fn(u, delta, A[:, i:i+1], B[:, i:i+1], C=ones, D=None, z=None, 
                                    delta_bias=None, delta_softplus=False, return_last_state=False)
            h.append(h_i)
        h = torch.stack(h, 1)

        y = torch.einsum("bnl,bndl->bdl", C, h)
        
        y = y + u * D[None, :, None]  # .to(u.dtype)
        y = y * F.silu(z)
        y = y.to(u.dtype)
        
        if return_last_state:
            assert False
        else:
            return y

