import torch
from torch import nn
import torch.nn.functional as F


class SelectiveScanTorch(nn.Module):
    def __init__(self, mode="cumsum") -> None:
        super().__init__()

        self.mode = mode

    # def complex_log(self, input, eps=1e-12):
    def complex_log(self, input, eps=1e-12):
        eps = input.new_tensor(eps)
        real = input.abs().maximum(eps).log()
        imag = (input < 0).to(input.dtype) * torch.pi
        return torch.complex(real.to(torch.float32), imag.to(torch.float32))

    def selective_scan(self, u, dt, A, B, C, D, mode='cumsum'):
        dtype = u.dtype

        dA = torch.einsum('bdl,dn->bldn', dt, A)
        dB_u = torch.einsum('bdl,bdl,bnl->bldn', dt, u, B)
        
        match mode:
            case 'logcumsumexp':
                dB_u_log = self.complex_log(dB_u)
                
                dA_star = F.pad(dA[:, 1:].cumsum(1), (0, 0, 0, 0, 1, 0))
                x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star

                y = torch.einsum('bldn,bnl->bdl', (x_log.real.exp() * torch.cos(x_log.imag)).to(C.dtype), C)
                out = y + u * D[None, :, None]

        return out.to(dtype)

    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        assert delta_bias is None
        assert delta_softplus
        assert not return_last_state

        if delta_softplus:
            delta = F.softplus(delta)
        y = self.selective_scan(u, delta, A, B, C, D, mode=self.mode)
        y = y * F.silu(z)

        if return_last_state:
            assert False
        else:
            return y