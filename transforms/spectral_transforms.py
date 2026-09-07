from __future__ import annotations
import torch
from deepinv.transform.base import Transform, TransformParam

def sample_from(values, shape, device, generator=None, dtype=None):
    N = len(values)
    indices = torch.floor(
        N * torch.rand(shape, dtype=dtype, device=device, generator=generator)
    ).to(torch.long)
    return values[indices]


class SpectralScale(Transform):
    def __init__(self, *args, device='cpu', factors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.factors = factors or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.rng = torch.Generator(device=device)

    def _get_params(self, x):
        b = x.shape[0] * self.n_trans
        factors = torch.tensor(self.factors, device=x.device)
        factor = sample_from(factors, shape=(b,), device=x.device, generator=self.rng)
        return {"factor": TransformParam(factor, neg=lambda x: 1 / x)}

    def _transform(self, x, factor, **kwargs):
        b, c, h, w = x.shape
        out = []
        for i in range(b):
            f = float(factor[i])
            u = torch.arange(c, dtype=torch.float32, device=x.device)
            u_scaled = torch.clamp(u / f, 0, c - 1.001)

            u_low = u_scaled.floor().long()
            u_high = (u_low + 1).clamp(max=c - 1)
            alpha = u_scaled - u_low.float()   
            alpha_inv = 1.0 - alpha             

            x_i = x[i]  # (C, H, W)
            result = x_i[u_low] * alpha_inv.view(-1, 1, 1) + x_i[u_high] * alpha.view(-1, 1, 1)
            out.append(result)
        return torch.stack(out)



