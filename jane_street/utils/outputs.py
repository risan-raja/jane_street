from typing import Dict, List, Tuple, Union
import torch
from .mixins import OutputMixIn


def detach(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
) -> Union[
    Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor],
]:
    """
    Detach object

    Args:
        x: object to detach

    Returns:
        detached object
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        return {name: detach(xi) for name, xi in x.items()}
    elif isinstance(x, OutputMixIn):
        return x.__class__(**{name: detach(xi) for name, xi in x.items()})
    elif isinstance(x, (list, tuple)):
        return [detach(xi) for xi in x]
    else:
        return x


def create_mask(
    size: int, lengths: torch.LongTensor, inverse: bool = False
) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lenghts) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

    Returns:
        torch.BoolTensor: mask
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(-1)


def integer_histogram(
    data: torch.LongTensor, min: Union[None, int] = None, max: Union[None, int] = None
) -> torch.Tensor:
    """
    Create histogram of integers in predefined range

    Args:
        data: data for which to create histogram
        min: minimum of histogram, is inferred from data by default
        max: maximum of histogram, is inferred from data by default

    Returns:
        histogram
    """
    uniques, counts = torch.unique(data, return_counts=True)
    if min is None:
        min = uniques.min()
    if max is None:
        max = uniques.max()
    hist = torch.zeros(max - min + 1, dtype=torch.long, device=data.device).scatter(
        dim=0, index=uniques - min, src=counts
    )
    return hist


def masked_op(
    tensor: torch.Tensor, op: str = "mean", dim: int = 0, mask: torch.Tensor = None
) -> torch.Tensor:
    """Calculate operation on masked tensor.

    Args:
        tensor (torch.Tensor): tensor to conduct operation over
        op (str): operation to apply. One of ["mean", "sum"]. Defaults to "mean".
        dim (int, optional): dimension to average over. Defaults to 0.
        mask (torch.Tensor, optional): boolean mask to apply (True=will take mean, False=ignore).
            Masks nan values by default.

    Returns:
        torch.Tensor: tensor with averaged out dimension
    """
    if mask is None:
        mask = ~torch.isnan(tensor)
    masked = tensor.masked_fill(~mask, 0.0)
    summed = masked.sum(dim=dim)
    if op == "mean":
        return summed / mask.sum(dim=dim)  # Find the average
    elif op == "sum":
        return summed
    else:
        raise ValueError(f"unkown operation {op}")
