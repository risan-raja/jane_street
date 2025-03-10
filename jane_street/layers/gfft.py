from typing import Optional

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class GaussianFourierFeatureTransform(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    Given an input of size [..., time, dim], returns a tensor of size [..., n_fourier_feats, time].
    """

    def __init__(self, input_dim: int, n_fourier_feats: int, scales: list[int]):
        super().__init__()
        self.input_dim = input_dim
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        n_scale_feats = n_fourier_feats // (2 * len(scales))
        # assert n_scale_feats * 2 * len(scales) == n_fourier_feats, \
        #     f"n_fourier_feats: {n_fourier_feats} must be divisible by 2 * len(scales) = {2 * len(scales)}"
        B_size = (input_dim, n_scale_feats)
        B = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)
        self.register_buffer("B", B)

    def forward(self, x: Tensor) -> Tensor:
        # assert x.dim() >= 2, f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        # _ , dim = x.shape[-2], x.shape[-1]

        # assert dim == self.input_dim, \
        #     f"Expected input to have {self.input_dim} channels (got {dim} channels)"

        x = torch.einsum("... t n, n d -> ... t d", [x, self.B])
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class INRLayer(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, dropout: Optional[float] = 0.01
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        self.gelu = nn.GELU()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self._layer(x)
        return self.norm(out)

    def _layer(self, x: Tensor) -> Tensor:
        return self.dropout(torch.relu(self.linear(x)))


class INR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int,
        n_fourier_feats: int,
        scales: float,
        dropout: Optional[float] = 0.01,
    ):
        super().__init__()
        self.features = (
            nn.Linear(input_dim, hidden_dim)
            if n_fourier_feats == 0
            else GaussianFourierFeatureTransform(input_dim, n_fourier_feats, scales)
        )
        in_size = hidden_dim if n_fourier_feats == 0 else n_fourier_feats
        n_layers = [INRLayer(in_size, hidden_dim, dropout=dropout)] + [
            INRLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers - 1)
        ]
        self.layers = nn.Sequential(*n_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] = 0.0):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(
        self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None
    ) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(
                X.mT, torch.linalg.solve(A, Y.to(dtype=A.dtype, device=A.device))
            )

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
