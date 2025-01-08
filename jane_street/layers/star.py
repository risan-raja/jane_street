import torch
from torch import nn
import torch.nn.functional as F


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        """
        STar Aggregate-Redistribute Module

        Args:
            d_series (int): Dimension of the series input.
            d_core (int): Dimension of the core output.

        This module performs a series of transformations on the input tensor,
        including linear transformations, GELU activations, and stochastic pooling
        during training. The final output is a tensor of the same shape as the input.
        """
        super(STAR, self).__init__()

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        """
        Forward pass of the STAR module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, channels, d_series).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, d_series).

        The forward pass involves the following steps:
        1. Apply a linear transformation followed by a GELU activation to the input.
        2. Apply another linear transformation to the result.
        3. During training, perform stochastic pooling by sampling indices based on
           the softmax of the transformed input. During evaluation, use weighted
           averaging based on the softmax of the transformed input.
        4. Concatenate the original input with the pooled/averaged result.
        5. Apply another linear transformation followed by a GELU activation.
        6. Apply a final linear transformation to produce the output.

        Hints for dimensions:
        - Input: (batch_size, channels, d_series)
        - After gen1 and GELU: (batch_size, channels, d_series)
        - After gen2: (batch_size, channels, d_core)
        - After pooling/averaging: (batch_size, channels, d_core)
        - After concatenation: (batch_size, channels, d_series + d_core)
        - After gen3 and GELU: (batch_size, channels, d_series)
        - Final output: (batch_size, channels, d_series)
        """
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            # print(ratio)
            # ratio = torch.nan_to_num(ratio, nan=1e-9)
            indices = torch.multinomial(torch.abs(ratio), 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(
                combined_mean * weight, dim=1, keepdim=True
            ).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat
        return output
