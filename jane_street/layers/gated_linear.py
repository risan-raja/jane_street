import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        dropout: float | None = 0.1,
    ):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x
